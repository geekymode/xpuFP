# ---------------------------------------------------------------------------
# Simulation and benchmarking harness for block-scaled formats.
#
# One deliberate bias runs through this file: no single number decides a format.  SNR is
# energy-weighted and blind to annihilated coordinates; the zeroed fraction says nothing
# about the survivors' accuracy; the median block SNR hides the unlucky blocks that
# actually break a network.  Every runner here reports several metrics side by side, and
# the plots put them on the same page.
# ---------------------------------------------------------------------------

"""
    test_distributions(n=100_000; rng=Random.default_rng(), outlier_scale=20,
                       outlier_every=40) -> Vector{Pair{String,Vector{Float64}}}

The standard battery of test data for quantization studies, as `label => vector` pairs:

| label | what it models |
|:---|:---|
| `gaussian` | trained weights — near-Gaussian, thin tails, the format's home turf |
| `laplace` | sparser, peakier weights |
| `student-t3` | heavy tails, a stand-in for transformer tensors with outliers |
| `lognormal` | multiplicative spread, no sign structure |
| `outlier` | Gaussian with one large value injected periodically — the activation case |
| `sparse` | 90 % exact zeros, Gaussian remainder |
| `uniform` | platykurtic; the worst case for a quasi-logarithmic grid |

Trained *weights* sit near the optimistic end of this battery; *activations*, with their
documented outlier channels, sample the pessimistic end — which is why activation
quantization is the hard half of the problem.
"""
function test_distributions(n::Integer = 100_000; rng::AbstractRNG = Random.default_rng(),
                            outlier_scale::Real = 20, outlier_every::Integer = 40)
    g = randn(rng, n)
    lap = [(u = rand(rng) - 0.5; -sign(u) * log(1 - 2abs(u)) / sqrt(2)) for _ in 1:n]
    t3 = [randn(rng) / sqrt(sum(abs2, randn(rng, 3)) / 3) for _ in 1:n]
    ln = [exp(randn(rng)) * sign(randn(rng)) for _ in 1:n]
    outl = copy(g)
    for i in 1:outlier_every:n
        outl[i] *= outlier_scale
    end
    sp = [rand(rng) < 0.9 ? 0.0 : randn(rng) for _ in 1:n]
    un = 2 .* rand(rng, n) .- 1
    ["gaussian" => g, "laplace" => lap, "student-t3" => t3, "lognormal" => ln,
     "outlier" => outl, "sparse" => sp, "uniform" => un]
end

"""
    SchemeMetrics

Every metric this package can measure for one (scheme, dataset) pair.

# Fields
- `scheme`, `dataset`, `bits` — identity and storage cost.
- `snr` — energy-weighted SNR in dB.
- `eff_bits` — `snr / 6.02`, the bit-ruler reading.
- `db_per_bit` — coding efficiency, `snr / bits`.
- `zeroed` — fraction of nonzero inputs stored as exact zero. **The acceptance metric.**
- `cosine` — cosine similarity of the whole vector with its reconstruction.
- `worst_rel` — largest per-element relative error among elements that survived.
  **Read this with care for rotated schemes**: a rotation spreads each block's
  quantization noise evenly across all `K` coordinates, so a coordinate that was tiny to
  begin with can acquire a relative error well above 100 % even though nothing was
  annihilated. Rotation converts *annihilation* into *spread* — better for energy and for
  dot products, worse for the relative accuracy of individual small elements.
- `block_snr_median`, `block_snr_p10` — the per-block SNR distribution. Tails threaten
  guarantees, not medians, so `p10` is the number that decides deployability.
"""
struct SchemeMetrics
    scheme::String
    dataset::String
    bits::Float64
    snr::Float64
    eff_bits::Float64
    db_per_bit::Float64
    zeroed::Float64
    cosine::Float64
    worst_rel::Float64
    block_snr_median::Float64
    block_snr_p10::Float64
end

_blocklen(f::BlockFormat) = f.K
_blocklen(f::RotatedBlockFormat) = f.base.K
_blocklen(::Any) = 32

"""
    measure_scheme(fmt, x; dataset="", K=…) -> SchemeMetrics

Run one format over one dataset and report every metric at once.

```julia
using Random
measure_scheme(MXFP4, randn(MersenneTwister(1), 50_000); dataset = "gaussian")
```
"""
function measure_scheme(fmt, x::AbstractVector; dataset::AbstractString = "",
                        K::Integer = _blocklen(fmt))
    xs = collect(Float64, x)
    xh = quantize_all(fmt, xs)
    nz = [i for i in eachindex(xs) if xs[i] != 0]
    zeroed = isempty(nz) ? 0.0 : count(i -> xh[i] == 0, nz) / length(nz)
    surv = [i for i in nz if xh[i] != 0]
    worst = isempty(surv) ? 1.0 : maximum(abs(xh[i] - xs[i]) / abs(xs[i]) for i in surv)
    # per-block SNR, because the tail of this distribution is what breaks networks
    bs = Float64[]
    for i in 1:K:length(xs)
        j = min(i + K - 1, length(xs))
        s = snr_db(@view(xs[i:j]), @view(xh[i:j]))
        isfinite(s) && push!(bs, s)
    end
    s = snr_db(xs, xh)
    b = _storage_bits(fmt)
    SchemeMetrics(_fmtname(fmt), String(dataset), b, s, effective_bits(s), s / b,
                  zeroed, cosine_similarity(xs, xh), worst,
                  isempty(bs) ? NaN : median(bs),
                  isempty(bs) ? NaN : quantile(bs, 0.10))
end

"""
    benchmark_schemes(formats, datasets; rotate=false) -> Vector{SchemeMetrics}

Run every format over every dataset.  `datasets` is a collection of `label => vector`
pairs, e.g. from [`test_distributions`](@ref).

Pass `rotate = true` to wrap each format in a Hadamard rotation first.

```julia
using Random
rng = MersenneTwister(1)
rows = benchmark_schemes((MXFP4, MXFP4_OPT32, NVFP4), test_distributions(40_000; rng))
print_benchmark(rows)
```
"""
function benchmark_schemes(formats, datasets; rotate::Bool = false)
    out = SchemeMetrics[]
    for f0 in formats
        f = rotate ? rotated(f0) : f0
        for (lab, v) in datasets
            push!(out, measure_scheme(f, v; dataset = lab))
        end
    end
    out
end

"""
    print_benchmark(rows; io=stdout, metric=:snr, sortby=:snr)

Render [`benchmark_schemes`](@ref) output as a scheme × dataset table of one metric,
with the storage cost alongside.

`metric` may be `:snr`, `:eff_bits`, `:db_per_bit`, `:zeroed`, `:cosine`, `:worst_rel`,
`:block_snr_median` or `:block_snr_p10`.
"""
function print_benchmark(rows::Vector{SchemeMetrics}; io = stdout,
                         metric::Symbol = :snr, sortby::Symbol = :snr)
    isempty(rows) && return
    schemes = unique(r.scheme for r in rows)
    dsets = unique(r.dataset for r in rows)
    get1(s, d) = (i = findfirst(r -> r.scheme == s && r.dataset == d, rows);
                  i === nothing ? NaN : getfield(rows[i], metric))
    score(s) = mean(getfield(r, sortby) for r in rows if r.scheme == s)
    order = sort(schemes; by = s -> -score(s))
    pct = metric in (:zeroed,)
    println(io, "metric: ", metric, pct ? "  (%)" : "")
    @printf(io, "%-16s %6s", "scheme", "b/val")
    for d in dsets
        @printf(io, " %11s", first(d, 11))
    end
    println(io)
    println(io, "-"^(23 + 12 * length(dsets)))
    for s in order
        b = rows[findfirst(r -> r.scheme == s, rows)].bits
        @printf(io, "%-16s %6.3f", s, b)
        for d in dsets
            v = get1(s, d)
            pct ? @printf(io, " %10.2f%%", 100v) : @printf(io, " %11.3f", v)
        end
        println(io)
    end
end

"""
    dot_snr_curve(fmt, lengths; trials=24, rng=Random.default_rng(), accumulate=FP32)
        -> Vector{NamedTuple}

Dot-product SNR against vector length for one scheme, as the **ensemble ratio**
`10log₁₀(Σy²/Σe²)` with a bootstrap 10–90 % band.

The expected result is a **flat line**: all the noise enters at the encoder and the block
arithmetic is exact, so signal and noise power grow together and their ratio is
independent of `N`.  A curve that sags with length would mean the arithmetic is
contributing error, which for these formats it must not.
"""
function dot_snr_curve(fmt, lengths; trials::Integer = 200,
                       rng::AbstractRNG = Random.default_rng(),
                       accumulate::FloatFormat = FP32)
    out = NamedTuple{(:N, :snr, :lo, :hi),NTuple{4,Float64}}[]
    for N in lengths
        # the ensemble ratio Σy²/Σe², not the median of per-trial ratios: the latter is
        # a ratio of fluctuating quantities whose median is a different (and far noisier)
        # statistic than the SNR
        ys = Float64[]; es = Float64[]
        # y² is χ²-distributed, so a single large draw dominates Σy²; the ensemble
        # ratio needs a few hundred trials before the flatness of the law is visible
        # rather than buried in sampling noise
        for _ in 1:max(trials, 200)
            a = randn(rng, N); b = randn(rng, N)
            truth = dot(a, b)
            got = fmt isa Union{BlockFormat,RotatedBlockFormat} ?
                  _dot_via(fmt, a, b, accumulate) : dot(quantize_all(fmt, a), quantize_all(fmt, b))
            push!(ys, truth^2); push!(es, (got - truth)^2)
        end
        snr = 10 * log10(sum(ys) / max(sum(es), 1e-300))
        # bootstrap the ensemble ratio for an honest band
        bs = Float64[]
        for _ in 1:200
            idx = rand(rng, eachindex(ys), length(ys))
            push!(bs, 10 * log10(sum(ys[idx]) / max(sum(es[idx]), 1e-300)))
        end
        push!(out, (N = Float64(N), snr = snr,
                    lo = quantile(bs, 0.1), hi = quantile(bs, 0.9)))
    end
    out
end

_dot_via(bf::BlockFormat, a, b, acc) = block_dot(bf, a, b; accumulate = acc).value
# a rotation is orthogonal, so rotating both operands leaves the dot product invariant
_dot_via(f::RotatedBlockFormat, a, b, acc) = dot(quantize_all(f, a), quantize_all(f, b))

"""
    error_profile(fmt, ts=exp10.(range(-3, 0; length=400)); M=1.0) -> Vector{Float64}

Per-element relative representation error against the element's size **relative to its
block maximum**, `t = |xᵢ|/M` — the like-for-like comparison curve.

A float format is a flat line at its machine epsilon. A block-scaled format is a ripple
inside its window, a wall through the twilight, and 100 % beyond.
"""
function error_profile(fmt, ts = exp10.(range(-3, 0; length = 400)); M::Real = 1.0)
    if fmt isa BlockFormat
        S = block_scale(fmt, [Float64(M)])
        return [(x = t * M; abs(quantize(fmt.elem, x / S) * S - x) / x) for t in ts]
    else
        return [(x = t * M; abs(quantize(fmt, x) - x) / x) for t in ts]
    end
end

"""
    pareto_frontier(rows::Vector{SchemeMetrics}; dataset="gaussian") -> Vector{SchemeMetrics}

The schemes that are not dominated on (bits, SNR): nothing else achieves both fewer bits
*and* higher SNR.  The honest shortlist for a storage budget."""
function pareto_frontier(rows::Vector{SchemeMetrics}; dataset::AbstractString = "gaussian")
    sel = filter(r -> r.dataset == dataset, rows)
    sort!(sel; by = r -> (r.bits, -r.snr))
    out = SchemeMetrics[]
    best = -Inf
    for r in sel
        if r.snr > best
            push!(out, r); best = r.snr
        end
    end
    out
end

function Base.show(io::IO, ::MIME"text/plain", m::SchemeMetrics)
    println(io, m.scheme, "  on  ", m.dataset, "   (", round(m.bits, digits = 3), " bits/value)")
    @printf(io, "  SNR            : %8.3f dB   (%.2f effective bits, %.3f dB/bit)\n",
            m.snr, m.eff_bits, m.db_per_bit)
    @printf(io, "  per-block SNR  : median %.2f dB,  10th pct %.2f dB\n",
            m.block_snr_median, m.block_snr_p10)
    @printf(io, "  zeroed         : %8.3f %%   ← the acceptance metric\n", 100 * m.zeroed)
    @printf(io, "  cosine         : %8.5f\n", m.cosine)
    @printf(io, "  worst survivor : %8.3f %% relative error", 100 * m.worst_rel)
end
