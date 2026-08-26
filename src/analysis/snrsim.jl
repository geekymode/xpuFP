# ---------------------------------------------------------------------------
# SNR simulation, with analytical estimates to check it against.
#
# A measured number and a derived number are worth far more together than either alone:
# agreement validates both, and disagreement localises the error.  This file provides
# the Monte-Carlo measurement with a confidence interval, the closed forms where they
# exist, and the rigorous quadrature for block-scaled formats.
# ---------------------------------------------------------------------------

_Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
_φ(x) = exp(-x^2 / 2) / sqrt(2π)

"""
    grid_distance(f, u) -> Float64

Distance from `u` to the nearest representable value of element format `f`, with
saturation: beyond the top code the distance grows without bound.

This is the `err_S(x)` of the element-SNR derivation, the integrand's core."""
function grid_distance(f, u::Real)
    au = abs(Float64(u))
    mx = maxfinite(f)
    au > mx && return au - mx
    abs(quantize(f, au) - au)
end

"""
    estimate_element_snr(bf::BlockFormat; sigma=1.0, rtol=1e-6) -> Float64

The **derived** element SNR of a block-scaled format on i.i.d. Gaussian data, by
quadrature rather than simulation.

Conditioning on the block absolute maximum `m`, whose density is the order statistic
`f(m) = K(2Φ(m)−1)^{K−1}·2φ(m)`, and using the standard fact that the remaining `K−1`
magnitudes are then i.i.d. with the **truncated** folded-normal density
`2φ(x)/(2Φ(m)−1)` on `[0,m]`:

```math
\\mathbb{E}[\\Delta^2] = \\int_0^\\infty f(m)\\left[\\tfrac{1}{K}\\,\\mathrm{err}^2_{S(m)}(m)
  + \\tfrac{K-1}{K}\\,\\mathbb{E}\\!\\left[\\mathrm{err}^2_{S(m)}(X) \\mid |X| \\le m\\right]\\right] dm
```

The truncation is **load-bearing**, not a nicety: dropping it charges phantom saturation
for mass above `m` that cannot exist, because the scale was set *by* `m`. The report
measures that error at 2.1 dB.

```jldoctest
julia> round(estimate_element_snr(MXFP4), digits = 1)
18.8
```
"""
function estimate_element_snr(bf::BlockFormat; sigma::Real = 1.0, rtol::Real = 1e-6)
    K = bf.K
    el = bf.elem
    s = Float64(sigma)
    # inner: E[err²(X) | |X| ≤ m] under the truncated folded normal
    function inner(m, S)
        Z = 2 * _Φ(m / s) - 1
        Z <= 0 && return 0.0
        val, _ = quadgk(x -> grid_distance(el, x / S)^2 * S^2 * 2 * _φ(x / s) / s, 0.0, m;
                        rtol = rtol)
        val / Z
    end
    function integrand(m)
        m <= 0 && return 0.0
        Z = 2 * _Φ(m / s) - 1
        Z <= 0 && return 0.0
        fm = K * Z^(K - 1) * 2 * _φ(m / s) / s
        fm == 0 && return 0.0
        S = block_scale(bf, [m])
        emax2 = (grid_distance(el, m / S) * S)^2
        fm * (emax2 / K + (K - 1) / K * inner(m, S))
    end
    num, _ = quadgk(integrand, 0.0, 12s; rtol = rtol)
    -10 * log10(num / s^2)
end

"""
    SNRSimulation

A measured SNR with its uncertainty, alongside the analytical estimate it is meant to
confirm.

# Fields
`scheme`, `distribution`, `nsamples`, `ntrials`, `measured`, `stderr`, `ci`
(95 % interval), `estimate`, `estimate_method`, `dot_measured`, `dot_predicted`,
`samples` (the per-trial SNRs).
"""
struct SNRSimulation
    scheme::String
    distribution::String
    nsamples::Int
    ntrials::Int
    measured::Float64
    stderr::Float64
    ci::Tuple{Float64,Float64}
    estimate::Union{Float64,Nothing}
    estimate_method::String
    dot_measured::Float64
    dot_predicted::Float64
    samples::Vector{Float64}
end

"""
    analytic_snr(fmt) -> (estimate, method)

The analytical SNR estimate for a format, and the name of the method used.

- [`FloatFormat`](@ref): the closed form `6.02p + 7.44` dB, which is **distribution-free**
  — a float's relative error depends only on the mantissa, never on the magnitude.
- [`BlockFormat`](@ref): [`estimate_element_snr`](@ref), by quadrature, valid for
  Gaussian data.
- Anything else (rotated, integer, fixed): `nothing` — no closed form is claimed rather
  than one invented.
"""
analytic_snr(f::FloatFormat) = (predicted_snr(f), "closed form 6.02p + 7.44 dB (distribution-free)")
analytic_snr(bf::BlockFormat) = (estimate_element_snr(bf), "order-statistic quadrature (Gaussian)")
analytic_snr(::Any) = (nothing, "no closed form available")

"""
    simulate_snr(fmt; n=50_000, trials=20, rng=Random.default_rng(),
                 dist=randn, distname="gaussian", dotlength=4096) -> SNRSimulation

Monte-Carlo the SNR of a format, with a 95 % confidence interval, and compare it against
the analytical estimate.

`dist` is a sampler `(rng, n) -> Vector{Float64}`; the default draws standard normals.
The dot-product SNR is measured too — as the **ensemble ratio**
`10log₁₀(Σy²/Σe²)`, not the median of per-trial ratios — and checked against the
**−3.01 dB law**: two independent noise sources per product cost exactly a factor of two
in power, independent of vector length.

```julia
julia> sim = simulate_snr(MXFP4; n = 40_000, trials = 12, rng = MersenneTwister(1));

julia> round(sim.measured, digits = 1), round(sim.estimate, digits = 1)
(18.8, 18.8)
```
"""
function simulate_snr(fmt; n::Integer = 50_000, trials::Integer = 20,
                      rng::AbstractRNG = Random.default_rng(),
                      dist = (r, m) -> randn(r, m), distname::AbstractString = "gaussian",
                      dotlength::Integer = 4096)
    vals = Float64[]
    for _ in 1:trials
        x = collect(Float64, dist(rng, n))
        push!(vals, measure_snr(fmt, x))
    end
    m = mean(vals)
    se = length(vals) > 1 ? std(vals) / sqrt(length(vals)) : 0.0
    est, how = analytic_snr(fmt)

    # Dot-product SNR, accumulated as an ENSEMBLE ratio 10log₁₀(Σy²/Σe²).
    #
    # Taking the median of per-trial 20log₁₀|y/e| instead is wrong twice over: it is a
    # ratio of two fluctuating quantities, so its median is not the ensemble SNR, and
    # its spread is enormous (the 10th and 90th percentiles differ by ~29 dB), so a
    # handful of trials returns noise.  The report's definition is the ensemble ratio.
    sy = 0.0; se2 = 0.0
    for _ in 1:max(trials, 200)
        a = collect(Float64, dist(rng, dotlength))
        b = collect(Float64, dist(rng, dotlength))
        truth = dot(a, b)
        got = fmt isa BlockFormat ? block_dot(fmt, a, b).value :
              dot(quantize_all(fmt, a), quantize_all(fmt, b))
        sy += truth^2
        se2 += (got - truth)^2
    end
    dotsnr = se2 == 0 ? Inf : 10 * log10(sy / se2)

    SNRSimulation(_fmtname(fmt), String(distname), Int(n), Int(trials), m, se,
                  (m - 1.96se, m + 1.96se), est, how,
                  dotsnr, m - 3.01, vals)
end

"""
    dot_snr_law(element_snr) -> Float64

The dot-product SNR predicted from the element SNR: `SNRₑ − 3.01 dB`.

Writing `â = a(1+δᵃ)`, the per-term error is `δᵃ + δᵇ` — two independent noise sources,
so error power doubles while signal power is unchanged. Both grow linearly in the vector
length, so the ratio is **length-invariant**: the same 3 dB at N = 32 and N = 32768."""
dot_snr_law(element_snr::Real) = Float64(element_snr) - 3.01

function Base.show(io::IO, ::MIME"text/plain", s::SNRSimulation)
    println(io, "SNR simulation: ", s.scheme, " on ", s.distribution)
    println(io, "  ", s.ntrials, " trials × ", s.nsamples, " samples")
    @printf(io, "  measured   : %7.3f dB   ± %.3f (95%% CI %.3f … %.3f)\n",
            s.measured, 1.96s.stderr, s.ci[1], s.ci[2])
    if s.estimate === nothing
        println(io, "  estimate   :    —      ", s.estimate_method)
    else
        d = s.measured - s.estimate
        @printf(io, "  estimate   : %7.3f dB   %s\n", s.estimate, s.estimate_method)
        @printf(io, "  agreement  : %+.3f dB %s\n", d,
                abs(d) < max(2 * 1.96s.stderr, 0.1) ? "— within noise ✓" : "— OUTSIDE the interval")
    end
    @printf(io, "  dot product: %7.3f dB measured, %7.3f dB predicted (SNRₑ − 3.01)\n",
            s.dot_measured, s.dot_predicted)
    @printf(io, "               %+.3f dB from the law", s.dot_measured - s.dot_predicted)
end
