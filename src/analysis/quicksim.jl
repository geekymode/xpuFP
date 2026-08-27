# ---------------------------------------------------------------------------
# Quick SNR simulation for FP4 block formats.
#
# `measure_snr` is the reference path: it builds a `QuantizedBlock` per block, with
# its codes, its element vector and its copy of the input, and it reaches the bit
# level through `quantize` for every single value.  That is the right shape for
# *watching* arithmetic and the wrong shape for *sweeping* it — exploring a design
# space means quantizing tens of millions of values, and allocation dominates.
#
# This file is the sweep path.  Two observations make it fast:
#
#   1. A 4-bit element format has **eight** positive levels.  Rounding onto it is a
#      table lookup, not an exponent extraction: precompute the levels and their
#      midpoints once, then every value costs one branch-free binary search over a
#      seven-element vector that lives in cache.
#   2. The SNR is a ratio of two sums.  Nothing needs to be stored — signal power and
#      noise power accumulate in two scalars as the data streams past, so a block of
#      32 values needs one stack buffer and no heap at all.
#
# The result is the same number the reference path returns — `test/runtests.jl` pins
# them together to 1e-9 dB, and in practice they agree bit for bit — at roughly 10^8
# values a second, which is 30x the reference path on MXFP4.
# ---------------------------------------------------------------------------

# ---- the lookup table ------------------------------------------------------

"""
    GridLUT

A small format's rounding grid, flattened into arrays for repeated lookup.

# Fields
- `levels` — the non-negative representable values, ascending, `levels[1] == 0`.
- `mids` — the `length(levels)-1` midpoints between consecutive levels.
- `tie` — for each midpoint, the index of the level that *exact* ties round to.
  Ties-to-even is a statement about the code, not the value, so the direction differs
  from midpoint to midpoint (E2M1 sends `1.25 → 1` but `3.5 → 4`); the table is built
  by asking the reference [`quantize`](@ref) and storing what it says.
- `maxpos`, `maxneg` — the saturation magnitudes. They differ for two's-complement
  integers, where `INT4` reaches `-8` but only `+7`.
- `negtie` — the level magnitude an exact `(maxpos+maxneg)/2` rounds to, for that
  asymmetric top step.
- `ftz` — the flush-to-zero floor. A format with no subnormal ramp (E8M0) does not
  round its smallest values to the smallest level, it **truncates the binade** and
  returns zero, so the bottom boundary sits at `minnormal`, not at a midpoint. `0.0`
  for every format with subnormals, where the ordinary midpoint is already correct.
- `zeroval` — what an exact zero maps to. `0.0` everywhere except E8M0, which has no
  encoding for zero at all and returns its smallest normal instead.
- `signed`, `negzero` — how negatives are handled. E8M0 is an unsigned *float* and
  returns the magnitude; an unsigned *integer* clamps to zero instead. The two
  disagree, so the flag records which one this format is.
- `saturates`, `ovf` — the overflow behaviour. A saturating format clamps everything
  above the top code; an IEEE-style one rounds first and escapes to `±Inf` only past
  `maxfinite + ulp/2`, so the two differ over exactly one half-ulp band.

Built by [`gridlut`](@ref); consumed by [`lut_quantize`](@ref).
"""
struct GridLUT
    levels::Vector{Float64}
    mids::Vector{Float64}
    tie::Vector{Int}
    maxpos::Float64
    maxneg::Float64
    negtie::Float64
    ftz::Float64
    zeroval::Float64
    signed::Bool
    negzero::Bool
    saturates::Bool
    ovf::Float64
end

"""
    gridlut(f) -> GridLUT

Tabulate an element or scale format's grid for fast repeated rounding.

Only worth building for narrow formats — the table has one entry per code, so E2M1
costs 8 levels and E4M3 costs 128, while FP32 would cost 2^31 and must keep using
[`quantize`](@ref). [`lut_quantize`](@ref) reproduces `quantize` exactly on every
format this package uses as an element or a scale.
"""
function gridlut(f::Union{FloatFormat,IntFormat})
    lv = sort!(unique!(filter(v -> isfinite(v) && v >= 0, posgrid(f))))
    isempty(lv) && error("gridlut: $(_fmtname(f)) has no finite non-negative grid")
    lv[1] == 0.0 || pushfirst!(lv, 0.0)     # E8M0 has no zero; the LUT still needs a floor
    mids = [(lv[i] + lv[i+1]) / 2 for i in 1:length(lv)-1]
    # ask the reference quantizer which side each exact midpoint falls on
    tie = [findfirst(==(quantize(f, m)), lv)::Int for m in mids]
    mxp = Float64(maxfinite(f))
    mxn = f isa IntFormat ? abs(intmin(f)) : mxp
    ntie = mxn == mxp ? mxp : abs(quantize(f, -(mxp + mxn) / 2))
    # a missing subnormal ramp is a hard floor, not a rounding boundary
    ftz = (f isa FloatFormat && !f.subnormals) ? Float64(minnormal(f)) : 0.0
    sat = !(f isa FloatFormat) || f.saturate
    GridLUT(lv, mids, tie, mxp, mxn, ntie, ftz, Float64(quantize(f, 0.0)),
            f.signed, !f.signed && f isa IntFormat, sat,
            sat ? Inf : mxp + ulp(f, mxp) / 2)
end

"""
    lut_quantize(L::GridLUT, u::Real) -> Float64

Round `u` onto the tabulated grid — the fast equivalent of `quantize(f, u)`.

One binary search over `L.mids` locates the interval; the exact-midpoint case reads
the precomputed tie direction. Out-of-range magnitudes saturate, which is what every
FP4 element format does (there are no infinities to escape to).
"""
@inline function lut_quantize(L::GridLUT, u::Real)
    x = Float64(u)
    isnan(x) && return NaN
    x == 0 && return L.zeroval
    x < 0 && L.negzero && return 0.0         # unsigned integers clamp, they do not fold
    neg = x < 0 && L.signed
    ax = x < 0 ? -x : x
    ax < L.ftz && return neg ? -0.0 : 0.0    # no subnormal ramp: the binade truncates
    if neg && L.maxneg > L.maxpos && ax > L.maxpos   # two's complement's extra low step
        ax >= L.maxneg && return -L.maxneg
        m = (L.maxpos + L.maxneg) / 2
        return -(ax > m ? L.maxneg : ax < m ? L.maxpos : L.negtie)
    end
    if ax >= L.maxpos
        (!L.saturates && ax >= L.ovf) && return neg ? -Inf : Inf
        return neg ? -L.maxpos : L.maxpos
    end
    i = searchsortedfirst(L.mids, ax)
    v = @inbounds i > length(L.mids) ? L.levels[end] :
                  L.mids[i] == ax ? L.levels[L.tie[i]] : L.levels[i]
    neg ? -v : v
end

# ---- the fast block encoder ------------------------------------------------

"""
    BlockKernel

A [`BlockFormat`](@ref) with its element grid, its scale grid and its rule constants
resolved once, ready to be run over a stream of blocks.

Constructing one is the setup cost of a quick simulation; running it is the loop.
"""
struct BlockKernel
    fmt::BlockFormat
    elem::GridLUT
    scale::GridLUT
    emax::Int
    maxel::Float64
    phistar::Float64
    cap::Float64
    npts::Int
end

"""
    blockkernel(bf::BlockFormat; npts=200) -> BlockKernel

Precompute everything about `bf` that does not change block to block.

`npts` is the candidate count for the `MSE_OPTIMAL` scale search, which is the only
rule whose cost is not `O(K)` per block.
"""
function blockkernel(bf::BlockFormat; npts::Integer = 200)
    BlockKernel(bf, gridlut(bf.elem), gridlut(bf.scale), elem_emax(bf),
                Float64(maxfinite(bf.elem)),
                bf.rule == OPT_SHIFT ? opt_shift_threshold(bf.K) : NaN,
                bf.rule == MSE_OPTIMAL ? 16.0 : 8.0, Int(npts))
end

# squared error of a block under a trial scale, LUT-rounded
@inline function _sse(k::BlockKernel, blk, S::Float64)
    S <= 0 && return Inf
    s = 0.0
    @inbounds for j in eachindex(blk)
        d = lut_quantize(k.elem, blk[j] / S) * S - blk[j]
        s += d * d
    end
    s
end

# absolute error of a block under a trial scale — BO2_BRACKET's metric
@inline function _sae(k::BlockKernel, blk, S::Float64)
    S <= 0 && return Inf
    s = 0.0
    @inbounds for j in eachindex(blk)
        s += abs(lut_quantize(k.elem, blk[j] / S) * S - blk[j])
    end
    s
end

# the block scale, matching `block_scale` rule for rule (including its range guard)
function _scale(k::BlockKernel, blk, M::Float64)
    M == 0 && return 1.0
    r = k.fmt.rule
    S = if r == MX_FLOOR_POW2
        exp2(binade_exponent(M) - k.emax)
    elseif r == OPT_SHIFT
        mx_scale_opt(M; phistar = k.phistar, emax = k.emax)
    elseif r == NV_MAXDIV
        lut_quantize(k.scale, M / k.maxel)
    elseif r == BEST_POW2
        e = binade_exponent(M) - k.emax
        a, b = exp2(e), exp2(e + 1)
        _sse(k, blk, a) <= _sse(k, blk, b) ? a : b
    elseif r == BO2_BRACKET
        # ceil(M/emax) on the scale grid, then BO2_NCAND-1 steps down, chosen on L1.
        # Mirrors `_bo2_bracket_scale` including its first-strict-minimum tie-break.
        c = _scale_grid_ceil(k.fmt.scale, M / k.maxel)
        best, bestS = Inf, c
        for _ in 1:BO2_NCAND
            (c <= 0 || !isfinite(c)) && break
            e = _sae(k, blk, c)
            e < best && (best = e; bestS = c)
            c = _scale_grid_down(k.fmt.scale, c)
        end
        bestS
    else  # MSE_OPTIMAL
        # The 1.5-octave search window is scanned at `npts` points, but the scale format
        # is *discrete*: E4M3 offers eight steps per binade, so those 200 trial exponents
        # collapse onto roughly a dozen distinct scales. Skipping a candidate identical
        # to its predecessor evaluates the same set of scales in the same ascending
        # order — so it keeps `_mse_optimal_scale`'s first-strict-minimum tie-break, and
        # its answer, while doing an order of magnitude less block SSE.
        anchor = M / k.maxel
        best, bestS = Inf, anchor
        prev = NaN
        lo, hi = log2(anchor) - 1.0, log2(anchor) + 0.5
        @inbounds for t in range(lo, hi; length = k.npts)
            c = lut_quantize(k.scale, exp2(t))
            c == prev && continue
            prev = c
            e = _sse(k, blk, c)
            if e < best
                best, bestS = e, c
            end
        end
        bestS
    end
    # same fallback as `block_scale`: a bounded scale format (E4M3) can round the ideal
    # ratio out of its own range, and power-of-two alignment always exists
    (isfinite(S) && S > 0 && M / S <= k.cap) ? S : exp2(binade_exponent(M) - k.emax)
end

"""
    QuickSNR

The result of one [`quick_snr`](@ref) run.

# Fields
- `scheme`, `dataset`, `n`, `K` — what was run on what.
- `snr` — energy-weighted element SNR in dB, identical to [`measure_snr`](@ref).
- `eff_bits` — `snr / 6.02`, the reading on the bit ruler.
- `bits` — storage cost per value, scale amortised.
- `db_per_bit` — `snr / bits`, coding efficiency.
- `dot_snr` — the dot-product SNR implied by the `−3.01 dB` law, not re-measured;
  [`simulate_snr`](@ref) is where that law gets checked against real dot products.
- `zeroed` — fraction of nonzero inputs stored as exact zero. Read this next to `snr`,
  never after it: SNR is energy-weighted and cannot see an annihilated coordinate.
- `clipped` — fraction of inputs whose magnitude exceeded the block's top code, and so
  came back saturated. Zero for no rule in practice, which is why it is worth printing:
  `MX_FLOOR_POW2` only promises `M/S ∈ [4,8)` against a grid that stops at 6, so a block
  whose maximum lands high in its binade has that maximum clamped; `NV_MAXDIV` aims the
  maximum straight at the top code but must round `M/6` onto E4M3, and rounding *down*
  overloads. Measured on Gaussian data these run 2–4 %. A dial, not an alarm.
- `blocks`, `seconds` — block count and wall-clock time.
"""
struct QuickSNR
    scheme::String
    dataset::String
    n::Int
    K::Int
    snr::Float64
    eff_bits::Float64
    bits::Float64
    db_per_bit::Float64
    dot_snr::Float64
    zeroed::Float64
    clipped::Float64
    blocks::Int
    seconds::Float64
    qsnr_median::Float64
    qsnr_p10::Float64
    qsnr_min::Float64
    qsnr_gap::Float64
end

"""
    quick_data(name, n; rng) -> Vector{Float64}

One dataset from the standard battery, generated on its own rather than alongside the
other six — [`test_distributions`](@ref) builds all seven, which is wasted work when a
sweep wants only Gaussians.

Accepts `:gaussian`, `:laplace`, `:student_t3`, `:lognormal`, `:uniform`, `:sparse`,
`:outlier`, and takes the same shape and scaling as the battery's entries.
"""
function quick_data(name::Symbol, n::Integer; rng::AbstractRNG = Random.default_rng(),
                    outlier_scale::Real = 20, outlier_every::Integer = 40)
    m = Int(n)
    if name === :gaussian
        randn(rng, m)
    elseif name === :uniform
        2 .* rand(rng, m) .- 1
    elseif name === :laplace
        [(u = rand(rng) - 0.5; -sign(u) * log(1 - 2abs(u)) / sqrt(2)) for _ in 1:m]
    elseif name === :student_t3
        [randn(rng) / sqrt(sum(abs2, randn(rng, 3)) / 3) for _ in 1:m]
    elseif name === :lognormal
        [exp(randn(rng)) * sign(randn(rng)) for _ in 1:m]
    elseif name === :sparse
        [rand(rng) < 0.9 ? 0.0 : randn(rng) for _ in 1:m]
    elseif name === :outlier
        g = randn(rng, m)
        for i in 1:outlier_every:m
            g[i] *= outlier_scale
        end
        g
    else
        throw(ArgumentError("quick_data: unknown distribution :$name — one of " *
                            ":gaussian, :uniform, :laplace, :student_t3, :lognormal, " *
                            ":sparse, :outlier"))
    end
end

"""
    quick_snr(bf::BlockFormat; n=1_000_000, dist=:gaussian, seed=0, npts=200) -> QuickSNR
    quick_snr(bf::BlockFormat, x::AbstractVector; dataset="") -> QuickSNR

Estimate a block format's SNR by streaming simulation — the sweep tool for FP4 design
questions ("what does MXFP4 lose at K=64?", "is the E4M3 scale worth its half bit?").

Returns the same SNR as `measure_snr(bf, x)` on the same data, plus the zeroed and
clipped fractions that SNR alone cannot show, and reports its own wall-clock so a
sweep's cost is visible rather than guessed at.

```julia
julia> quick_snr(MXFP4)
MXFP4 on gaussian — 1000000 values in 31250 blocks of 32
  SNR          :  18.790 dB   (3.12 effective bits)
  storage      :   4.250 bits/value   →  4.42 dB per bit
  dot product  :  15.780 dB   (element SNR − 3.01)
  zeroed       :    8.78 % of nonzero inputs
  clipped      :    2.34 % above the top code
  elapsed      :   0.012 s

julia> quick_snr(NVFP4).snr - quick_snr(MXFP4).snr    # what the finer scale buys
1.6501282995808744
```

For a single format on a single dataset this replaces the whole
[`simulate_snr`](@ref) ritual; use `simulate_snr` when you want the confidence
interval, the analytical cross-check and the measured dot product instead.
"""
function quick_snr(bf::BlockFormat, x::AbstractVector; dataset::AbstractString = "custom",
                   npts::Integer = 200)
    t0 = time()
    xs = x isa Vector{Float64} ? x : collect(Float64, x)
    k = blockkernel(bf; npts = npts)
    K = bf.K
    n = length(xs)
    n == 0 && throw(ArgumentError("quick_snr: empty data"))

    # The tensor scale is a power of two applied identically to signal and
    # reconstruction, so it cancels out of the SNR — but not out of *representability*,
    # which is the whole reason NVFP4 carries one. Divide once, up front.
    sg = tensor_scale(bf, xs)

    sig = 0.0; noi = 0.0
    nz = 0; zeroed = 0; clipped = 0; nblk = 0
    blk = Vector{Float64}(undef, K)
    bq = Float64[]                       # per-block QSNR, the distribution behind `snr`
    sizehint!(bq, cld(n, K))

    @inbounds for i in 1:K:n
        j = min(i + K - 1, n)
        len = j - i + 1
        M = 0.0
        for t in 1:len
            v = sg == 1.0 ? xs[i+t-1] : xs[i+t-1] / sg
            blk[t] = v
            av = abs(v)
            av > M && (M = av)
        end
        S = _scale(k, view(blk, 1:len), M)
        top = k.elem.maxpos * S
        bsig = 0.0; bnoi = 0.0
        for t in 1:len
            v = blk[t]
            q = lut_quantize(k.elem, v / S) * S
            xt = v * sg                       # back in the original units for the ratio
            d = q * sg - xt
            bsig += xt * xt
            bnoi += d * d
            if v != 0
                nz += 1
                q == 0 && (zeroed += 1)
                abs(v) > top && (clipped += 1)
            end
        end
        sig += bsig; noi += bnoi
        # an all-zero block was empty, not damaged — skip it rather than score it 0 dB
        bsig > 0 && push!(bq, bnoi == 0 ? Inf : 10 * log10(bsig / bnoi))
        nblk += 1
    end

    snr = sig == 0 ? 0.0 : noi == 0 ? Inf : 10 * log10(sig / noi)
    b = bits_per_element(bf)
    fin = filter(isfinite, bq)
    if isempty(fin)
        qmed = qp10 = qmin = Inf; qgap = 0.0
    else
        sort!(fin)
        pick(p) = fin[clamp(ceil(Int, p * length(fin)), 1, length(fin))]
        qmed = pick(0.50); qp10 = pick(0.10); qmin = fin[1]; qgap = snr - qp10
    end
    QuickSNR(bf.name, String(dataset), n, K, snr, effective_bits(snr), b, snr / b,
             dot_snr_law(snr), nz == 0 ? 0.0 : zeroed / nz, nz == 0 ? 0.0 : clipped / nz,
             nblk, time() - t0, qmed, qp10, qmin, qgap)
end

function quick_snr(bf::BlockFormat; n::Integer = 1_000_000, dist::Symbol = :gaussian,
                   seed::Integer = 0, npts::Integer = 200, kwargs...)
    x = quick_data(dist, n; rng = MersenneTwister(seed), kwargs...)
    quick_snr(bf, x; dataset = String(dist), npts = npts)
end

function Base.show(io::IO, ::MIME"text/plain", q::QuickSNR)
    println(io, q.scheme, " on ", q.dataset, " — ", q.n, " values in ", q.blocks,
            " blocks of ", q.K)
    @printf(io, "  SNR (pooled) : %7.3f dB   (%.2f effective bits)\n", q.snr, q.eff_bits)
    @printf(io, "  QSNR / block : median %.3f, p10 %.3f, min %.3f dB\n",
            q.qsnr_median, q.qsnr_p10, q.qsnr_min)
    @printf(io, "  gap          : %+7.3f dB   (pooled − p10)\n", q.qsnr_gap)
    @printf(io, "  storage      : %7.3f bits/value   →  %.2f dB per bit\n",
            q.bits, q.db_per_bit)
    @printf(io, "  dot product  : %7.3f dB   (element SNR − 3.01)\n", q.dot_snr)
    @printf(io, "  zeroed       : %7.2f %% of nonzero inputs\n", 100 * q.zeroed)
    @printf(io, "  clipped      : %7.2f %% above the top code\n", 100 * q.clipped)
    @printf(io, "  elapsed      : %7.3f s", q.seconds)
end

# ---- naming the variants ---------------------------------------------------

"""
    fp4_variant(; K=32, scale=E8M0, rule=MX_FLOOR_POW2, elem=E2M1, name=nothing)
                -> BlockFormat

Build a modified FP4 format for a sweep, named after what was modified.

The FP4 design space is four choices wide — block length, scale format, scale rule,
element format — and `MXFP4` and `NVFP4` are two corners of it. This constructor is
for the rest:

```julia
julia> fp4_variant(K = 16)                              # MXFP4's rule, NVFP4's block
BlockFormat(FP4-K16-E8M0-MX_FLOOR_POW2, K=16, E2M1 × E8M0, 4.5 b/elem)

julia> fp4_variant(rule = OPT_SHIFT)                    # the one-comparison improvement
BlockFormat(FP4-K32-E8M0-OPT_SHIFT, K=32, E2M1 × E8M0, 4.25 b/elem)

julia> quick_snr(fp4_variant(rule = OPT_SHIFT)).snr - quick_snr(MXFP4).snr
0.25261160975338015
```
"""
function fp4_variant(; K::Integer = 32, scale::ElementFormat = E8M0,
                     rule::ScaleRule = MX_FLOOR_POW2, elem::ElementFormat = E2M1,
                     name::Union{Nothing,AbstractString} = nothing)
    nm = name === nothing ?
         "FP4-K$(K)-$(_fmtname(scale))-$(rule)" * (elem === E2M1 ? "" : "-$(_fmtname(elem))") :
         String(name)
    BlockFormat(nm, Int(K), elem, scale, rule)
end

# ---- sweeps ----------------------------------------------------------------

"""
    quick_compare(formats; n=1_000_000, dist=:gaussian, seed=0, io=stdout) -> Vector{QuickSNR}

Run [`quick_snr`](@ref) over several formats on **one shared draw** of data and print
the comparison table.

Sharing the sample is the point: format-to-format differences here are differences in
the *format*, not in the luck of two independent draws. At `n = 10⁶` the sampling
noise on a single SNR is about ±0.02 dB, but the noise on a *difference* over common
data is far smaller, which is what makes a 0.1 dB improvement legible.

```julia
julia> quick_compare([MXFP4, NVFP4, fp4_variant(rule = OPT_SHIFT), XPFP4_32]);
scheme                        bits    SNR dB  eff bits  dB/bit  zeroed%  clip%     s
────────────────────────────────────────────────────────────────────────────────────
MXFP4                         4.25    18.790      3.12    4.42     8.78   2.34  0.01
NVFP4                         4.50    20.440      3.40    4.54     6.78   3.47  0.03
FP4-K32-E8M0-OPT_SHIFT        4.25    19.043      3.16    4.48     9.48   0.91  0.01
XPFP4-32                      4.25    20.768      3.45    4.89     8.18   2.02  0.38
```
"""
function quick_compare(formats; n::Integer = 1_000_000, dist::Symbol = :gaussian,
                       seed::Integer = 0, io::IO = stdout, npts::Integer = 200)
    x = quick_data(dist, n; rng = MersenneTwister(seed))
    rows = [quick_snr(bf, x; dataset = String(dist), npts = npts) for bf in formats]
    @printf(io, "%-28s %5s %9s %9s %8s %7s %7s %8s %6s\n",
            "scheme", "bits", "SNR dB", "QSNR p10", "QSNR min", "gap", "dB/bit",
            "zeroed%", "clip%")
    println(io, "─"^96)
    for r in rows
        @printf(io, "%-28s %5.2f %9.3f %9.3f %8.3f %+7.3f %7.2f %8.2f %6.2f\n",
                r.scheme, r.bits, r.snr, r.qsnr_p10, r.qsnr_min, r.qsnr_gap,
                r.db_per_bit, 100 * r.zeroed, 100 * r.clipped)
    end
    println(io, "\nSNR dB is pooled (energy-weighted, the usual published QSNR).")
    println(io, "QSNR p10/min are per-block, every block weighted equally; gap = pooled − p10.")
    rows
end

"""
    quick_sweep(; Ks=(8,16,32,64,128), scale=E8M0, rule=MX_FLOOR_POW2, elem=E2M1,
                n=500_000, dist=:gaussian, seed=0) -> Vector{Pair{Int,QuickSNR}}

Sweep block length at fixed rule and scale format — the cheapest question in the design
space, and the one whose answer is most often assumed rather than measured.

The assumption is that shorter blocks buy SNR, a shared scale fitting fewer values
better, paid for in bits (`nbits(scale)/K` per value). Under `MX_FLOOR_POW2` on Gaussian
data the SNR column runs the other way — 18.40 dB at `K=8` rising to 18.96 dB at
`K=128` — because the rule sees nothing but the block maximum, and a longer block's
maximum lands higher in its binade, where less of the element grid goes unused. Short
blocks cost more bits *and* measure slightly worse.

Which relocates the credit for NVFP4's 1.65 dB over MXFP4: it is the E4M3 scale format,
not the halved block. The sweep prints bits beside SNR so both columns are read rather
than argued about.
"""
function quick_sweep(; Ks = (8, 16, 32, 64, 128), scale::ElementFormat = E8M0,
                     rule::ScaleRule = MX_FLOOR_POW2, elem::ElementFormat = E2M1,
                     n::Integer = 500_000, dist::Symbol = :gaussian, seed::Integer = 0,
                     io::IO = stdout)
    x = quick_data(dist, n; rng = MersenneTwister(seed))
    out = Pair{Int,QuickSNR}[]
    @printf(io, "%5s %7s %9s %9s %8s\n", "K", "bits", "SNR dB", "eff bits", "dB/bit")
    println(io, "─"^42)
    for K in Ks
        bf = fp4_variant(; K = K, scale = scale, rule = rule, elem = elem)
        q = quick_snr(bf, x; dataset = String(dist))
        push!(out, Int(K) => q)
        @printf(io, "%5d %7.3f %9.3f %9.2f %8.2f\n", K, q.bits, q.snr, q.eff_bits,
                q.db_per_bit)
    end
    out
end
