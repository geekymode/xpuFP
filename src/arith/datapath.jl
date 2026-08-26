# ---------------------------------------------------------------------------
# Floating-point arithmetic, simulated stage by stage.
#
# Every operation returns a DatapathTrace: not just the answer, but the unpacked
# operands, the alignment shift, the raw significand result, the normalisation
# shift, and where the single rounding landed.  That is the whole point — the
# arithmetic is meant to be watched.
#
# Correctness note.  All formats in this package have at most 23 mantissa bits, so
# Float64 (53 bits) satisfies the classical no-double-rounding condition
# `53 ≥ 2p + 2` for every one of them.  Computing in Float64 and rounding once to
# the target format therefore yields exactly the IEEE-mandated correctly rounded
# result.  FP64 itself is the sole exception and is reference-only.
# ---------------------------------------------------------------------------

"""
    FPParts

An unpacked floating-point value: the fields the hardware actually works on once the
packed word has been split and the hidden bit reattached.

# Fields
- `sign::Int` — 0 or 1.
- `E::Int` — the stored, biased exponent field.
- `e::Int` — the effective unbiased exponent (`E - bias`, or `emin` for subnormals).
- `M::UInt64` — the stored fraction field, as an integer.
- `significand::Float64` — `1.f` for normals, `0.f` for subnormals.
- `value::Float64` — the value denoted.
- `issubnormal`, `iszero`, `isinf`, `isnan` — classification.
"""
struct FPParts
    sign::Int
    E::Int
    e::Int
    M::UInt64
    significand::Float64
    value::Float64
    issubnormal::Bool
    iszero::Bool
    isinf::Bool
    isnan::Bool
end

"""
    unpack(f::FloatFormat, x::Real) -> FPParts

Split `x` (first quantized onto `f`'s grid) into sign / exponent / significand, the
way the unpack stage of an FPU does before any arithmetic happens.

```jldoctest
julia> p = unpack(FP32, 12.0);

julia> p.sign, p.E, p.e, p.significand
(0, 130, 3, 1.5)
```
"""
function unpack(f::FloatFormat, x::Real)
    v = quantize(f, x)
    if isnan(v)
        return FPParts(0, nexpcodes(f) - 1, emax(f) + 1, UInt64(1), NaN, NaN,
                       false, false, false, true)
    end
    s = signbit(v) ? 1 : 0
    av = abs(v)
    if isinf(av)
        return FPParts(s, nexpcodes(f) - 1, emax(f) + 1, UInt64(0),
                       Inf, v, false, false, true, false)
    end
    if av == 0
        return FPParts(s, 0, emin(f), UInt64(0), 0.0, v, false, true, false, false)
    end
    code = encode(f, v)
    mmask = (UInt64(1) << f.mbits) - 1
    M = f.mbits == 0 ? UInt64(0) : (code & mmask)
    E = Int((code >> f.mbits) & ((UInt64(1) << f.ebits) - 1))
    sub = (E == 0 && f.zero_exp == SUBNORMAL_ZERO)
    e = sub ? emin(f) : E - f.bias
    sig = sub ? M / nmancodes(f) : 1 + M / nmancodes(f)
    FPParts(s, E, e, M, sig, v, sub, false, false, false)
end

"""
    Stage

One labelled step of a datapath walkthrough: what the stage is called, a
human-readable rendering of what it holds, and (where meaningful) the numeric value.
"""
struct Stage
    label::String
    detail::String
    value::Union{Float64,Nothing}
end
Stage(label, detail) = Stage(label, detail, nothing)

"""
    DatapathTrace

The complete record of one floating-point operation: the operands, every pipeline
stage, the delivered result, and the exact real-number answer it was rounded from.

Display it (`show`) for a formatted stage-by-stage walkthrough.

# Fields
- `op::Symbol` — `:add`, `:mul`, `:div`, `:fma`, `:mul_then_add`.
- `fmt::FloatFormat`
- `inputs::Vector{Float64}` — operands, already quantized onto the format.
- `stages::Vector{Stage}`
- `result::Float64` — what the format delivers.
- `exact::Float64` — the infinitely precise answer.
- `info::NamedTuple` — operation-specific numeric detail (alignment shift,
  normalisation shift, rounding residual in ulps, …).
"""
struct DatapathTrace
    op::Symbol
    fmt::FloatFormat
    inputs::Vector{Float64}
    stages::Vector{Stage}
    result::Float64
    exact::Float64
    info::NamedTuple
end

"""    relerror(t::DatapathTrace) -> Float64

Relative error of the delivered result against the exact answer."""
relerror(t::DatapathTrace) = t.exact == 0 ? abs(t.result) : abs(t.result - t.exact) / abs(t.exact)

"""    rounding_ulps(t::DatapathTrace) -> Float64

How far the single rounding moved the answer, in units of the result's ulp.  Bounded
by 0.5 for a correctly rounded operation."""
rounding_ulps(t::DatapathTrace) = get(t.info, :round_ulps, NaN)

# ---- helpers ---------------------------------------------------------------

_sigstr(f::FloatFormat, sig::Float64; bits::Int = -1) = begin
    b = bits < 0 ? f.mbits : bits
    ip = sig >= 1 ? "1" : "0"
    frac = sig - floor(sig)
    s = ip * "."
    for _ in 1:b
        frac *= 2
        d = frac >= 1 ? 1 : 0
        s *= string(d)
        frac -= d
    end
    # trim trailing zeros for legibility, keeping at least one fraction digit
    while endswith(s, "0") && !endswith(s, ".0")
        s = s[1:end-1]
    end
    s
end

_expstr(e::Int) = "2^" * (e < 0 ? "($e)" : string(e))

"""    binade_exponent(x) -> Int

The `e` with `2^e ≤ |x| < 2^(e+1)`, computed robustly (no reliance on `log2`
landing on the correct side of a power of two)."""
function binade_exponent(x::Real)
    ax = abs(Float64(x))
    ax == 0 && return 0
    e = floor(Int, log2(ax))
    exp2(e) > ax && (e -= 1)
    exp2(e + 1) <= ax && (e += 1)
    e
end

# how far the rounding moved the answer, in ulps of the result
function _round_ulps(f::FloatFormat, exact::Float64, result::Float64)
    (!isfinite(exact) || !isfinite(result)) && return NaN
    u = ulp(f, result == 0 ? exact : result)
    u == 0 ? 0.0 : (result - exact) / u
end
