# ---------------------------------------------------------------------------
# Block-scaled ("microscaling") formats: a shared scale over a small block of
# narrow elements.  Floating point wrapped around fixed point's oldest idea.
# ---------------------------------------------------------------------------

"""
    ScaleRule

How a block's shared scale is chosen from its absolute maximum `M`.

- `MX_FLOOR_POW2`: `S = 2^(⌊log₂M⌋ − e_max)`, the OCP MX rule.  The unique power of
  two that lands `M` in the element grid's top binade, so `M/S ∈ [4,8)` for E2M1.
- `NV_MAXDIV`: `S = R_scale(M / maxfinite(elem))`, the NVFP4 rule — the exact real
  ratio, snapped onto the scale format.  Pins the block maximum at the top code.
- `BEST_POW2`: try both `2^e` and `2^(e+1)` and keep whichever gives lower MSE.  One
  comparison, no multiplies, and worth ~0.25–0.5 dB over `MX_FLOOR_POW2`.
- `MSE_OPTIMAL`: a 1-D search for the scale minimising block MSE, accepting a little
  clamping on the largest element in exchange for finer resolution everywhere else.
  Worth ~0.7–0.85 dB over `NV_MAXDIV`, and entirely inside the encoder — the wire
  format, decoder, and MAC path are untouched.
- `OPT_SHIFT`: the **optimized shift rule** — the floor scale unless the block maximum's
  mantissa fraction exceeds a fixed threshold, in which case shift one binade up.  One
  comparison against a precomputed constant, no block MSE evaluation, and it reaches the
  power-of-two class ceiling.  See [`mx_scale_opt`](@ref).
"""
@enum ScaleRule MX_FLOOR_POW2 NV_MAXDIV BEST_POW2 MSE_OPTIMAL OPT_SHIFT

"""
    BlockFormat

A two-level code: `K` narrow elements sharing one scale.  Each stored value is
`S × eᵢ` — a coarse shared exponent times a fine element — which is how a 4-bit
payload recovers a usable dynamic range.

# Fields
- `name::String`
- `K::Int` — block length, along the *reduction* axis of the matrix multiply.
- `elem::ElementFormat` — the per-element format (E2M1 for both shipped FP4 schemes).
- `scale::ElementFormat` — the block-scale format (E8M0 for MX, E4M3 for NV).
- `rule::ScaleRule` — how the scale is chosen.

# Why blocks must run along the reduction axis
`S` factors out of `∑ᵢ S eᵢ xᵢ` only if every element under the sum shares it.
Blocked along the output axis instead, the scales would be trapped inside the sum and
the cheap `y = S_a S_b ∑ eᵃᵢeᵇᵢ` factorisation — hence the cheap hardware — would
evaporate.
"""
struct BlockFormat
    name::String
    K::Int
    elem::ElementFormat
    scale::ElementFormat
    rule::ScaleRule
end

"""
    BlockFormat(name, K, elem, scale, rule)
    BlockFormat(; name, K, elem, scale, rule)
"""
BlockFormat(; name::AbstractString, K::Integer, elem::ElementFormat,
            scale::ElementFormat, rule::ScaleRule) =
    BlockFormat(String(name), Int(K), elem, scale, rule)

"""    MXFP4

The OCP Microscaling 4-bit format: blocks of 32 E2M1 elements sharing one E8M0
(power-of-two) scale.  `32×4 + 8 = 136` bits per block — **4.25 bits per value**.
Applying the scale is an exponent add, never a multiplication.

Measured element SNR on Gaussian data: **18.8 dB** (≈3.1 effective bits)."""
const MXFP4 = BlockFormat("MXFP4", 32, E2M1, E8M0, MX_FLOOR_POW2)

"""    NVFP4

NVIDIA's Blackwell-native variant: blocks of 16 E2M1 elements sharing one E4M3
mini-float scale, with a per-tensor FP32 factor above it.  `16×4 + 8 = 72` bits per
block — **4.5 bits per value**.

The finer scale places the block maximum within ~6% of the top of the range instead
of somewhere in a whole binade, which is nearly the entire 18.8 → **20.4 dB** gap."""
const NVFP4 = BlockFormat("NVFP4", 16, E2M1, E4M3, NV_MAXDIV)

"""    MXINT4

INT4 elements with an MX-style block scale — the uniform-grid rival.  Measures
16.8 dB on Gaussian data against MXFP4's 18.8: the uniform grid wastes its levels on
the Gaussian's thin tails, which is the relative-versus-absolute-error argument in
measured form."""
const MXINT4 = BlockFormat("MXINT4", 32, INT4, E8M0, MX_FLOOR_POW2)

"""    ALL_BLOCK_FORMATS"""
const ALL_BLOCK_FORMATS = (MXFP4, NVFP4, MXINT4)

"""
    elem_emax(bf::BlockFormat) -> Int

The binade of the element format's largest representable value — `⌊log₂ 6⌋ = 2` for
E2M1, `⌊log₂ 448⌋ = 8` for E4M3.  This is the `−2` in the MX scale formula: the rule
aligns the data's top octave with the *grid's* top octave, and the constant is
nothing but `−e_max`."""
elem_emax(bf::BlockFormat) = binade_exponent(maxfinite(bf.elem))

"""
    bits_per_element(bf::BlockFormat) -> Float64

Storage cost per value including the amortised block scale: `nbits(elem) +
nbits(scale)/K`.  4.25 for MXFP4, 4.5 for NVFP4."""
bits_per_element(bf::BlockFormat) = nbits(bf.elem) + nbits(bf.scale) / bf.K

"""    bits_per_block(bf) -> Int"""
bits_per_block(bf::BlockFormat) = bf.K * nbits(bf.elem) + nbits(bf.scale)

# ---- scale selection -------------------------------------------------------

"""
    block_scale(bf::BlockFormat, x) -> Float64

Choose the shared scale for block `x` according to `bf.rule`, already quantized onto
the scale format.

!!! note "Narrow scale formats need a tensor scale first"
    E4M3 spans only about `0.002 … 448`.  Applied to raw data outside that window its
    scale would round to zero or saturate, so this function falls back to power-of-two
    alignment in that case.  [`reconstruct`](@ref) avoids the situation entirely by
    dividing through by [`tensor_scale`](@ref) first, which is what deployed NVFP4 does.

For `MX_FLOOR_POW2` this implements `S = 2^(⌊log₂ max|xᵢ|⌋ − e_max)`, which is
provably the *unique* power of two landing the block maximum in the element grid's
top binade — one step larger strands the maximum an octave low (wasting the top codes
and doubling the effective step for every element), one step smaller guarantees the
block's most energetic value saturates.
"""
function block_scale(bf::BlockFormat, x::AbstractVector)
    M = maximum(abs, x)
    M == 0 && return 1.0
    S = _raw_block_scale(bf, x, M)
    # A scale format with a bounded range (E4M3 spans only ~0.002 … 448) can round the
    # ideal ratio to zero for a small block, or saturate for a large one — either way
    # the block is left unrepresentable.  Fall back to the power-of-two alignment, which
    # always exists, rather than returning a scale that divides to Inf or strands the
    # maximum far outside the element grid.  `tensor_scale` is the proper cure; this is
    # the guard for callers who use `block_scale` directly on raw data.
    #
    # Each rule gets its own ceiling, so the guard enforces that rule's contract:
    # the alignment rules promise the maximum lands inside the element grid (M/S < 8),
    # while MSE_OPTIMAL deliberately overloads, bounded at 2× by its search window.
    cap = bf.rule == MSE_OPTIMAL ? 16.0 : 8.0
    ok = isfinite(S) && S > 0 && M / S <= cap
    ok || return exp2(binade_exponent(M) - elem_emax(bf))
    S
end

function _raw_block_scale(bf::BlockFormat, x::AbstractVector, M::Float64)
    if bf.rule == MX_FLOOR_POW2
        return exp2(binade_exponent(M) - elem_emax(bf))
    elseif bf.rule == NV_MAXDIV
        return quantize(bf.scale, M / maxfinite(bf.elem))
    elseif bf.rule == BEST_POW2
        e = binade_exponent(M) - elem_emax(bf)
        cands = (exp2(e), exp2(e + 1))
        return argmin(S -> _block_sse(bf, x, S), cands)
    elseif bf.rule == OPT_SHIFT
        return mx_scale_opt(M; phistar = opt_shift_threshold(bf.K),
                            emax = elem_emax(bf))
    elseif bf.rule == MSE_OPTIMAL
        return _mse_optimal_scale(bf, x)
    end
    error("unreachable scale rule")
end

"""
    needs_tensor_scale(bf::BlockFormat) -> Bool

Whether the block-scale format has too narrow a range to cover arbitrary data on its
own, and therefore needs a per-tensor factor above it.

E8M0 spans `2^±127` and covers everything alone. E4M3 spans only about `0.002 … 448`, so
a tensor whose blocks fall outside that window would have its scales rounded to zero or
saturated — which is exactly why deployed NVFP4 is a **three-level** scheme."""
needs_tensor_scale(bf::BlockFormat) = !(bf.scale === E8M0)

"""
    tensor_scale(bf::BlockFormat, x) -> Float64

The single per-tensor factor that shifts every block ratio into the scale format's
usable window — "range insurance for the scale format", not a precision mechanism.

Chosen as a power of two so it is exact, and applied identically to signal and
reconstruction, so it **cancels out of every SNR ratio**: it changes what is
representable, never what the error is.

Returns `1.0` for formats that do not need one."""
function tensor_scale(bf::BlockFormat, x::AbstractVector)
    needs_tensor_scale(bf) || return 1.0
    M = maximum(abs, x)
    (M == 0 || !isfinite(M)) && return 1.0
    ideal = M / maxfinite(bf.elem)          # the largest per-block scale we will need
    # park it near the upper-middle of the scale format's normal range
    exp2(binade_exponent(ideal) - binade_exponent(maxfinite(bf.scale)) + 1)
end

# squared error of block x under scale S
function _block_sse(bf::BlockFormat, x::AbstractVector, S::Float64)
    S <= 0 && return Inf
    s = 0.0
    @inbounds for v in x
        d = quantize(bf.elem, v / S) * S - v
        s += d * d
    end
    s
end

# 1-D search for the MSE-minimising scale, then snapped onto the scale format
function _mse_optimal_scale(bf::BlockFormat, x::AbstractVector; npts::Int = 200)
    M = maximum(abs, x)
    M == 0 && return 1.0
    anchor = M / maxfinite(bf.elem)
    best, bestS = Inf, anchor
    # search a window around the anchor: overloading (S below anchor) trades a small
    # clamp on the largest element for finer resolution on everyone else
    # overloading below the anchor buys resolution but clips the largest element;
    # one octave is enough to find the optimum and bounds the clip at 2×
    for t in range(log2(anchor) - 1.0, log2(anchor) + 0.5; length = npts)
        S = quantize(bf.scale, exp2(t))
        e = _block_sse(bf, x, S)
        if e < best
            best, bestS = e, S
        end
    end
    bestS
end

# ---- the quantized object --------------------------------------------------

"""
    QuantizedBlock

One encoded block: the shared scale (as both a value and its stored code), the
element codes, and the reconstruction `x̂ᵢ = S · eᵢ`.

Display it for a cell-by-cell table of `x → x/S → code → x̂ → error`.
"""
struct QuantizedBlock
    fmt::BlockFormat
    scale::Float64
    scale_code::UInt64
    codes::Vector{UInt64}
    elements::Vector{Float64}   # the decoded element values eᵢ (pre-scale)
    values::Vector{Float64}     # the reconstruction x̂ᵢ = S·eᵢ
    original::Vector{Float64}
end

"""
    quantize_block(bf::BlockFormat, x) -> QuantizedBlock

Encode one block: choose the shared scale, divide every element by it, round onto the
element grid, and store.

```jldoctest
julia> qb = quantize_block(MXFP4, [0.11, -0.35, 0.02, 0.24]);

julia> qb.scale, qb.elements
(0.0625, [2.0, -6.0, 0.5, 4.0])

julia> qb.values
4-element Vector{Float64}:
  0.125
 -0.375
  0.03125
  0.25
```
"""
function quantize_block(bf::BlockFormat, x::AbstractVector)
    xs = collect(Float64, x)
    S = block_scale(bf, xs)
    elems = [quantize(bf.elem, v / S) for v in xs]
    codes = [_elemcode(bf.elem, e) for e in elems]
    vals = elems .* S
    QuantizedBlock(bf, S, _elemcode(bf.scale, S), codes, elems, vals, xs)
end

_elemcode(f::FloatFormat, v::Real) = encode(f, v)
_elemcode(f::IntFormat, v::Real) = reinterpret(UInt64, Int64(encode(f, v))) &
                                   ((UInt64(1) << f.bits) - 1)

"""    dequantize(qb::QuantizedBlock) -> Vector{Float64}

The reconstruction `x̂ᵢ = S · eᵢ`."""
dequantize(qb::QuantizedBlock) = copy(qb.values)

"""
    quantize_blocked(bf::BlockFormat, x) -> Vector{QuantizedBlock}

Split a long vector into consecutive blocks of `bf.K` and encode each.  A trailing
partial block is encoded at its own (shorter) length."""
function quantize_blocked(bf::BlockFormat, x::AbstractVector)
    n = length(x)
    [quantize_block(bf, @view x[i:min(i + bf.K - 1, n)]) for i in 1:bf.K:n]
end

"""
    reconstruct(bf::BlockFormat, x) -> Vector{Float64}

Block-quantize `x` and immediately decode: the round trip `D(E(x))`, which is what
every SNR measurement in this package compares against the original."""
function reconstruct(bf::BlockFormat, x::AbstractVector)
    xs = collect(Float64, x)
    out = similar(xs)
    n = length(xs)
    sg = tensor_scale(bf, xs)               # 1.0 unless the scale format needs it
    for i in 1:bf.K:n
        j = min(i + bf.K - 1, n)
        seg = sg == 1.0 ? xs[i:j] : (xs[i:j] ./ sg)
        qb = quantize_block(bf, seg)
        out[i:j] .= sg == 1.0 ? qb.values : (qb.values .* sg)
    end
    out
end

# ---- the window theorem ----------------------------------------------------

"""
    dead_zone_threshold(qb::QuantizedBlock) -> Float64

Elements below this magnitude are stored as **exact zero** — information annihilated.

From the alignment proof `S ∈ (M/8, M/4]`, so the round-to-zero threshold `S/4` lies
in `(M/32, M/16]`.  Any element with `|xᵢ| ≤ M/32` therefore lies below *every*
possible threshold and cannot survive."""
dead_zone_threshold(qb::QuantizedBlock) = qb.scale * _first_positive(qb.fmt.elem) / 2

_first_positive(f::FloatFormat) = minsubnormal(f)
_first_positive(f::IntFormat) = 1.0

"""
    good_zone_threshold(qb::QuantizedBlock) -> Float64

Elements above this magnitude reconstruct to within 1/3 relative error (the sawtooth's
worst midpoint).  Equals `3M/32` for MXFP4."""
good_zone_threshold(qb::QuantizedBlock) = 3 * maximum(abs, qb.original) / 32

"""
    zeroed_count(qb::QuantizedBlock) -> Int

How many elements came back as exact zero having gone in nonzero.

**This is the acceptance test for MXFP4**, not the block ℓ₂ error.  Three very
different blocks — Gaussian, Gaussian with one 64× outlier, and a 4096× log-spread —
all report a comfortable ~10% ℓ₂ error while silently zeroing 3, 31, and 21 of their
32 elements.  Judge blocks by this count and the per-element profile; keep ℓ₂ for
comparing formats, never for verdicts."""
zeroed_count(qb::QuantizedBlock) =
    count(i -> qb.values[i] == 0 && qb.original[i] != 0, eachindex(qb.original))

function Base.show(io::IO, ::MIME"text/plain", qb::QuantizedBlock)
    bf = qb.fmt
    println(io, bf.name, " block of ", length(qb.original), "  —  scale S = ", qb.scale,
            "  (code 0x", string(qb.scale_code, base=16), ")")
    println(io, "  ", bits_per_block(bf), " bits per block, ",
            round(bits_per_element(bf), digits=3), " bits per value")
    println(io, "  ┌────────────┬────────────┬────────┬────────────┬────────────┐")
    println(io, "  │      value │      ÷ S   │   code │  reconstr. │      error │")
    println(io, "  ├────────────┼────────────┼────────┼────────────┼────────────┤")
    for i in eachindex(qb.original)
        @printf(io, "  │ %10.5g │ %10.5g │ %6s │ %10.5g │ %10.4g │\n",
                qb.original[i], qb.original[i] / qb.scale, string(qb.elements[i]),
                qb.values[i], qb.values[i] - qb.original[i])
    end
    println(io, "  └────────────┴────────────┴────────┴────────────┴────────────┘")
    z = zeroed_count(qb)
    @printf(io, "  ℓ₂ error %.3g%%   SNR %.2f dB   elements zeroed: %d",
            100 * norm(qb.values .- qb.original) / max(norm(qb.original), eps()),
            snr_db(qb.original, qb.values), z)
end

Base.show(io::IO, bf::BlockFormat) =
    print(io, "BlockFormat(", bf.name, ", K=", bf.K, ", ", bf.elem.name,
          " × ", bf.scale.name, ", ", round(bits_per_element(bf), digits=3), " b/elem)")

function Base.show(io::IO, ::MIME"text/plain", bf::BlockFormat)
    println(io, bf.name, "  —  block-scaled format")
    println(io, "  block size K      : ", bf.K)
    println(io, "  element format    : ", bf.elem.name, " (", nbits(bf.elem), " bits)")
    println(io, "  scale format      : ", bf.scale.name, " (", nbits(bf.scale), " bits)")
    println(io, "  scale rule        : ", bf.rule)
    println(io, "  bits per block    : ", bits_per_block(bf))
    print(io,   "  bits per element  : ", round(bits_per_element(bf), digits=4))
end
