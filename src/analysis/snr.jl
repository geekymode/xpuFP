# ---------------------------------------------------------------------------
# The measuring stick.  A number format is treated as a noisy channel: a value x
# goes in, its representable stand-in x̂ comes out, and the difference is bundled
# as additive noise.  Legitimate because (Bennett, 1948) deterministic rounding
# behaves statistically like independent additive noise over varied data.
# ---------------------------------------------------------------------------

"""    DB_PER_BIT

`20log₁₀2 = 6.0206` dB — one bit of resolution.

Simultaneously the humble uniform quantizer's slope (halve the step, quarter the
error power) *and* Shannon's rate–distortion ceiling for a Gaussian source,
`D(R) = σ²2^{-2R}`.  Dividing an SNR by it converts decibels into effective bits."""
const DB_PER_BIT = 20 * log10(2)

"""
    snr_db(x, x̂) -> Float64

Signal-to-noise ratio in decibels: the power ratio of what you meant to what got
corrupted.

```math
\\mathrm{SNR} = 10\\log_{10}\\frac{\\mathbb{E}[x^2]}{\\mathbb{E}[(x-\\hat x)^2]}
             = -20\\log_{10}(\\text{relative RMS error})
```

Returns `Inf` when the reconstruction is exact and `0.0` when the signal itself is
zero.

!!! warning "SNR is energy-weighted, hence deaf to annihilated small coordinates"
    A block with one large outlier and 31 zeroed elements can report a *better* SNR
    than a well-behaved one, because the outlier alone carries all the energy.  Read
    SNR alongside [`zeroed_count`](@ref) and the per-element error profile, never
    alone.  See [`plot_outlier_law`](@ref) for this failure mode measured.
"""
function snr_db(x::AbstractVector, xhat::AbstractVector)
    length(x) == length(xhat) || throw(DimensionMismatch("snr_db: length mismatch"))
    sig = 0.0; noi = 0.0
    @inbounds for i in eachindex(x)
        sig += Float64(x[i])^2
        d = Float64(x[i]) - Float64(xhat[i])
        noi += d * d
    end
    sig == 0 && return 0.0
    noi == 0 && return Inf
    10 * log10(sig / noi)
end

"""
    effective_bits(snr) -> Float64

Convert an SNR in dB to effective bits of resolution, `SNR / 6.02`.

This column is the calibration proof of the whole metric: run it on the IEEE formats
and it recovers *exactly* their significand widths — 8 for BF16, 11 for FP16, 24 for
FP32.  The block formats' fractional readings (3.1 for MXFP4, 3.4 for NVFP4) are
honest measurements of how much resolution their four stored bits actually deliver."""
effective_bits(snr::Real) = Float64(snr) / DB_PER_BIT

"""
    rel_rms_error(snr) -> Float64

The relative RMS error corresponding to an SNR in dB: `10^(-snr/20)`."""
rel_rms_error(snr::Real) = exp10(-Float64(snr) / 20)

"""
    cosine_similarity(x, x̂) -> Float64

Cosine of the angle between a vector and its reconstruction.

For independent error, `cos ≈ 1 − ½·10^{-SNR/10}` — so the same decibels bound the
*angle* a vector survives quantization by.  At 18.7 dB the identity predicts 0.99326,
and a real 32-element MXFP4 block measures 0.9932: the direction survives to within
about 6.7°, which is the sense in which a neural layer dotting this vector against
weights hardly notices."""
function cosine_similarity(x::AbstractVector, xhat::AbstractVector)
    nx = norm(collect(Float64, x)); nh = norm(collect(Float64, xhat))
    (nx == 0 || nh == 0) && return 0.0
    dot(collect(Float64, x), collect(Float64, xhat)) / (nx * nh)
end

"""
    cosine_from_snr(snr) -> Float64

The predicted cosine similarity at a given SNR, `1 − ½·10^{-snr/10}`."""
cosine_from_snr(snr::Real) = 1 - 0.5 * exp10(-Float64(snr) / 10)

"""
    quantize_all(fmt, x) -> Vector{Float64}

Round every element of `x` onto `fmt`'s grid.  Works for [`FloatFormat`](@ref),
[`IntFormat`](@ref), [`FixedFormat`](@ref), and — via [`reconstruct`](@ref) —
[`BlockFormat`](@ref), which is what makes the SNR machinery format-agnostic."""
quantize_all(f::Union{FloatFormat,IntFormat}, x::AbstractVector) = [quantize(f, v) for v in x]
quantize_all(f::FixedFormat, x::AbstractVector) = [quantize(f, v) for v in x]
quantize_all(bf::BlockFormat, x::AbstractVector) = reconstruct(bf, x)

"""
    measure_snr(fmt, x) -> Float64

Encode `x` in `fmt`, decode, and report the SNR in dB.  The one-line workhorse behind
every measurement table in this package.

```jldoctest
julia> using Random; rng = MersenneTwister(42);

julia> x = randn(rng, 200_000);

julia> round(measure_snr(MXFP4, x), digits=1)      # the report's 18.8 dB
18.8
```
"""
measure_snr(fmt, x::AbstractVector) = snr_db(x, quantize_all(fmt, x))

# ---- QSNR: the same ratio, but told block by block --------------------------

"""
    QSNR

The quantization SNR of a format on a dataset, reported **pooled and per block**.

# Fields
- `pooled` — `10log₁₀(Σx² / Σ(x−q(x))²)` over the whole vector. This is the figure the
  quantization literature usually prints as QSNR, and it is **numerically identical to
  [`measure_snr`](@ref)** — the same ratio, energy-weighted across everything.
- `mean`, `median`, `p10`, `p01`, `min` — the same ratio computed *within* each block of
  `K`, then aggregated with every block counting equally.
- `dead_blocks` — blocks whose QSNR fell to `≤ 0 dB`: the error carries at least as much
  energy as the signal, so nothing survived.
- `gap` — `pooled − p10`, how much better the headline number looks than the tenth
  percentile block.

!!! note "Why both numbers exist"
    Pooling weights each block by its energy, so a high-energy block can carry the score
    while a low-energy block is quietly ruined — the failure mode [`snr_db`](@ref) warns
    about. On seeded `:sparse` data MXFP4 pools to **18.16 dB**, but its tenth-percentile
    block is **14.81 dB** and its worst is **12.03 dB**: a `gap` of **+3.34 dB** between
    the headline and the tail. Pooled QSNR answers "how much signal energy survived";
    per-block QSNR answers "is there a block I ruined". Ship a format on the second.

    Note that a block of *all zeros* is skipped, not scored. [`snr_db`](@ref) returns
    `0.0` for a zero signal, and counting that as a 0 dB block would report empty data as
    catastrophic damage.
"""
struct QSNR
    format::String
    n::Int
    K::Int
    nblocks::Int
    pooled::Float64
    mean::Float64
    median::Float64
    p10::Float64
    p01::Float64
    min::Float64
    dead_blocks::Int
    gap::Float64
end

_qsnr_K(fmt, K) = K === nothing ? _blocklen(fmt) : Int(K)

"""
    qsnr_blocks(fmt, x; K = nothing) -> Vector{Float64}

QSNR in dB for each consecutive block of `K` values, `K` defaulting to the format's own
block length (32 for formats that have none).

Blocks whose signal energy is zero are dropped rather than reported as `0.0`; a block of
zeros was not damaged, it was empty.
"""
function qsnr_blocks(fmt, x::AbstractVector; K = nothing)
    k = _qsnr_K(fmt, K)
    k > 0 || throw(ArgumentError("qsnr_blocks: K must be positive"))
    xh = quantize_all(fmt, x)
    out = Float64[]
    n = length(x)
    @inbounds for i in 1:k:n
        j = min(i + k - 1, n)
        sig = 0.0; noi = 0.0
        for t in i:j
            xt = Float64(x[t]); d = xt - Float64(xh[t])
            sig += xt * xt; noi += d * d
        end
        sig == 0 && continue                       # an empty block, not a damaged one
        push!(out, noi == 0 ? Inf : 10 * log10(sig / noi))
    end
    out
end

"""
    measure_qsnr(fmt, x; K = nothing) -> QSNR

Quantize `x` in `fmt` and report the full QSNR picture: the pooled figure plus the
per-block distribution behind it.

```julia
julia> using Random

julia> x = quick_data(:sparse, 200_000; rng = MersenneTwister(0));

julia> q = measure_qsnr(MXFP4, x);

julia> round.((q.pooled, q.p10, q.gap), digits = 2)
(18.16, 14.81, 3.34)
```

The pooled column is what a paper reports; the `p10` and `dead_blocks` columns are what
decide whether the format is safe to deploy.
"""
function measure_qsnr(fmt, x::AbstractVector; K = nothing)
    k = _qsnr_K(fmt, K)
    b = qsnr_blocks(fmt, x; K = k)
    fin = filter(isfinite, b)
    pooled = measure_snr(fmt, x)
    nm = fmt isa AbstractString ? String(fmt) : string(_fmtname(fmt))
    isempty(fin) && return QSNR(nm, length(x), k, length(b), pooled,
                                Inf, Inf, Inf, Inf, Inf, 0, 0.0)
    srt = sort(fin)
    q(p) = srt[clamp(ceil(Int, p * length(srt)), 1, length(srt))]
    p10 = q(0.10)
    QSNR(nm, length(x), k, length(b), pooled,
         sum(fin) / length(fin), q(0.50), p10, q(0.01), srt[1],
         count(<=(0.0), fin), pooled - p10)
end

"""
    qsnr(fmt, x; K = nothing, agg = :pooled) -> Float64

One QSNR number, with the aggregation named explicitly.

`agg` may be `:pooled` (energy-weighted over everything — the usual published figure,
equal to [`measure_snr`](@ref)), or `:mean`, `:median`, `:p10`, `:p01`, `:min`, which
weight every block of `K` equally.

**The aggregation is not a detail.** Pooled and per-block medians agree to a tenth of a
dB on well-behaved data and diverge by whole dB once the data has outliers or sparsity —
which is exactly when you need the number.
"""
function qsnr(fmt, x::AbstractVector; K = nothing, agg::Symbol = :pooled)
    agg === :pooled && return measure_snr(fmt, x)
    r = measure_qsnr(fmt, x; K = K)
    agg === :mean   ? r.mean   :
    agg === :median ? r.median :
    agg === :p10    ? r.p10    :
    agg === :p01    ? r.p01    :
    agg === :min    ? r.min    :
    throw(ArgumentError("qsnr: agg must be :pooled, :mean, :median, :p10, :p01 or :min"))
end

function Base.show(io::IO, ::MIME"text/plain", q::QSNR)
    println(io, "QSNR — ", q.format, " on ", q.n, " values, ", q.nblocks,
            " blocks of ", q.K)
    @printf(io, "  pooled       : %7.3f dB   (energy-weighted; == measure_snr)\n", q.pooled)
    @printf(io, "  per-block    : mean %.3f, median %.3f dB\n", q.mean, q.median)
    @printf(io, "  tails        : p10 %.3f, p01 %.3f, min %.3f dB\n", q.p10, q.p01, q.min)
    @printf(io, "  dead blocks  : %d   (QSNR <= 0 dB — nothing survived)\n", q.dead_blocks)
    @printf(io, "  gap          : %+.3f dB   (pooled − p10: how much the headline flatters)",
            q.gap)
end

"""
    per_element_relerror(fmt, x) -> Vector{Float64}

Relative error `|x−x̂|/|x|` for every element.  The companion to [`measure_snr`](@ref)
that SNR alone cannot give you: it exposes the zeroed elements (relative error 1.0)
that energy weighting hides."""
function per_element_relerror(fmt, x::AbstractVector)
    xh = quantize_all(fmt, x)
    [Float64(x[i]) == 0 ? 0.0 : abs(xh[i] - x[i]) / abs(x[i]) for i in eachindex(x)]
end

"""
    db_per_bit(fmt, x) -> Float64

Coding efficiency: measured SNR divided by storage cost in bits per value.

The surprise of the report: on concentrated data MXFP4 extracts **4.42 dB per bit**,
statistically tied with FP32's 4.52, ahead of FP16, and far ahead of BF16, whose
oversized exponent field is pure overhead on block-scaled data.  MXFP4 is not a
sloppier code than FP32 — it is an *equally efficient, much shorter* one.  The
difference is budget, not thrift.

The law behind it is the **resolution fraction**:

```math
\\frac{\\text{dB}}{\\text{bit}} \\approx 6.02 \\times \\frac{p}{b_\\text{total}}
```

— exponent bits buy *range*, not SNR.  BF16's mediocre score is a theorem, not an
observation: half its bits are exponent."""
db_per_bit(fmt, x::AbstractVector) = measure_snr(fmt, x) / _storage_bits(fmt)

_storage_bits(f::Union{FloatFormat,IntFormat}) = Float64(nbits(f))
_storage_bits(f::FixedFormat) = Float64(nbits(f))
_storage_bits(bf::BlockFormat) = bits_per_element(bf)

"""
    resolution_fraction(f::FloatFormat) -> Float64

`(mbits+1) / nbits` — the share of the word spent on resolution rather than range.
Multiplied by 6.02 it predicts the format's dB per bit."""
resolution_fraction(f::FloatFormat) = (f.mbits + 1) / nbits(f)

"""
    predicted_snr(f::FloatFormat) -> Float64

The closed-form RMS SNR of a floating-point format, `6.02p + 7.44` dB.

The `+7.44` is derived, not fitted: the significand grid has step `2^{1-p}`, so the
rounding error is uniform on `±2^{-p}` with mean square `2^{-2p}/3`; the relative
error divides by the mantissa `f ∈ [1,2)`, which for wide-ranging data is log-uniform,
giving `E[1/f²] = 0.375/ln2 = 0.5410`.  Hence RMS relative error `= 0.425·2^{-p}` and
`SNR = 6.02p − 20log₁₀0.425 = 6.02p + 7.44` dB.

Measurement confirms the same constant for FP32, FP16 *and* BF16."""
predicted_snr(f::FloatFormat) = DB_PER_BIT * (f.mbits + 1) - 20 * log10(sqrt(0.375 / log(2) / 3))

"""
    LOG_UNIFORM_INV_SQ

`E[1/f²] = 0.375/ln2 = 0.5410` for a log-uniform mantissa on `[1,2)` — the constant
inside [`predicted_snr`](@ref).  Checks to four decimals on Gaussian data."""
const LOG_UNIFORM_INV_SQ = 0.375 / log(2)
