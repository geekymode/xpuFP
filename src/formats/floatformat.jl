# ---------------------------------------------------------------------------
# Generic IEEE-754-style binary floating-point formats.
#
# One struct covers FP32/FP16/BF16, the OCP FP8 pair, the FP4 family, and the
# mantissa-less E8M0 scale format, because they differ only in field widths and
# in what the reserved codes mean.  Every value of every format defined here is
# exactly representable in Float64 (the widest has 23 mantissa bits), so Float64
# is used throughout as the "exact real" working type.
# ---------------------------------------------------------------------------

"""
    NaNStyle

How a format spends its top exponent code.

- `IEEE_NAN`  : `E = all ones` is reserved; `M = 0` is ±Inf, `M ≠ 0` is NaN (FP32, FP16, BF16, E5M2).
- `E4M3_NAN`  : only the single code `E = all ones, M = all ones` is NaN; every other
                pattern is an ordinary finite number.  This is what buys OCP E4M3 its
                maximum of 448 (`2^8 × 1.75`) instead of stopping at 240.
- `E8M0_NAN`  : `E = all ones` is NaN, and there are no other reserved codes (E8M0).
- `NO_SPECIAL`: every code is a finite number; overflow saturates (E2M1, E1M2, E3M0).
"""
@enum NaNStyle IEEE_NAN E4M3_NAN E8M0_NAN NO_SPECIAL

"""
    ZeroExponent

What the `E = 0` code means.

- `SUBNORMAL_ZERO`: the IEEE convention — `E = 0` drops the hidden bit and pins the
  scale at `2^(1-bias)`, giving zero and the subnormal ramp.
- `NORMAL_ZERO`: `E = 0` is an ordinary normal code with a hidden 1.  Used by E8M0,
  where every code is a plain power of two and there is no encoding for zero.
"""
@enum ZeroExponent SUBNORMAL_ZERO NORMAL_ZERO

"""
    FloatFormat

A binary floating-point format: 1 optional sign bit, `ebits` exponent bits, `mbits`
mantissa (fraction) bits.  A normal value decodes as

```math
x = (-1)^s \\, 2^{E-\\mathrm{bias}} \\left(1 + M/2^{mbits}\\right)
```

and, when `zero_exp == SUBNORMAL_ZERO`, the code `E = 0` decodes instead as
`(-1)^s 2^{1-bias} (M / 2^{mbits})`.

Construct these with [`FloatFormat`](@ref) or take one from the registry
([`FP32`](@ref), [`E2M1`](@ref), …).

# Fields
- `name::String` — display name, e.g. `"FP32"`.
- `ebits`, `mbits` — field widths in bits.
- `bias::Int` — exponent bias; defaults to `2^(ebits-1) - 1`.
- `signed::Bool` — whether a sign bit is present.
- `subnormals::Bool` — whether the subnormal ramp is populated.
- `nan_style::NaNStyle`, `zero_exp::ZeroExponent` — reserved-code semantics.
- `saturate::Bool` — on overflow, clamp to the largest finite value instead of
  producing ±Inf.  Formats without infinities must saturate.
"""
struct FloatFormat
    name::String
    ebits::Int
    mbits::Int
    bias::Int
    signed::Bool
    subnormals::Bool
    nan_style::NaNStyle
    zero_exp::ZeroExponent
    saturate::Bool
end

"""
    FloatFormat(name, ebits, mbits; kwargs...)

Build a [`FloatFormat`](@ref).  Keyword defaults follow the IEEE conventions:
`bias = 2^(ebits-1) - 1`, signed, with subnormals, `IEEE_NAN`, `SUBNORMAL_ZERO`,
and overflow to infinity.
"""
function FloatFormat(name::AbstractString, ebits::Integer, mbits::Integer;
                     bias::Integer = (1 << (ebits - 1)) - 1,
                     signed::Bool = true,
                     subnormals::Bool = true,
                     nan_style::NaNStyle = IEEE_NAN,
                     zero_exp::ZeroExponent = SUBNORMAL_ZERO,
                     saturate::Bool = (nan_style == NO_SPECIAL))
    ebits >= 1 || throw(ArgumentError("ebits must be ≥ 1"))
    mbits >= 0 || throw(ArgumentError("mbits must be ≥ 0"))
    FloatFormat(String(name), Int(ebits), Int(mbits), Int(bias), signed,
                subnormals, nan_style, zero_exp, saturate)
end

# ---- basic geometry --------------------------------------------------------

"""    nbits(f) -> Int

Total width of the format in bits (sign + exponent + mantissa)."""
nbits(f::FloatFormat) = (f.signed ? 1 : 0) + f.ebits + f.mbits

"Number of distinct exponent codes, `2^ebits`."
nexpcodes(f::FloatFormat) = 1 << f.ebits

"Number of distinct mantissa codes, `2^mbits`."
nmancodes(f::FloatFormat) = 1 << f.mbits

"""    max_normal_E(f) -> Int

Largest exponent code that still encodes finite numbers."""
function max_normal_E(f::FloatFormat)
    top = nexpcodes(f) - 1
    if f.nan_style == IEEE_NAN || f.nan_style == E8M0_NAN
        return top - 1          # the all-ones code is fully reserved
    else
        return top              # E4M3_NAN loses one *mantissa* code, not the exponent
    end
end

"""    emin(f) -> Int

Exponent of the smallest normal value (`1 - bias` in the IEEE convention)."""
emin(f::FloatFormat) = (f.zero_exp == SUBNORMAL_ZERO ? 1 : 0) - f.bias

"""    emax(f) -> Int

Exponent of the largest normal binade."""
emax(f::FloatFormat) = max_normal_E(f) - f.bias

"""    machine_eps(f) -> Float64

Machine epsilon `2^-mbits`: the gap between 1.0 and the next representable value."""
machine_eps(f::FloatFormat) = exp2(-f.mbits)

"""    maxfinite(f) -> Float64

The largest finite magnitude the format can represent.

For `E4M3_NAN` formats the top mantissa code is stolen by NaN, which is exactly why
OCP E4M3 tops out at `448 = 2^8 × 1.75` rather than `480`."""
function maxfinite(f::FloatFormat)
    topM = nmancodes(f) - 1
    if f.nan_style == E4M3_NAN
        topM -= 1
    end
    exp2(emax(f)) * (1 + topM / nmancodes(f))
end

"""    minnormal(f) -> Float64

Smallest positive *normal* magnitude, `2^emin`."""
minnormal(f::FloatFormat) = exp2(emin(f))

"""    minsubnormal(f) -> Float64

Smallest positive magnitude of any kind.  Equals [`minnormal`](@ref) when the format
has no subnormal ramp."""
minsubnormal(f::FloatFormat) =
    f.subnormals && f.zero_exp == SUBNORMAL_ZERO ? exp2(emin(f) - f.mbits) : minnormal(f)

"""    dynamic_range(f) -> Float64

Ratio of the largest finite value to the smallest positive one."""
dynamic_range(f::FloatFormat) = maxfinite(f) / minsubnormal(f)

"""    decimal_digits(f) -> Float64

Approximate number of significant decimal digits, `(mbits+1) * log10(2)`."""
decimal_digits(f::FloatFormat) = (f.mbits + 1) * log10(2)

# ---- decode ----------------------------------------------------------------

"""
    decode(f::FloatFormat, code::Integer) -> Float64

Decode a raw bit pattern into the real value it denotes, applying the format's
reserved-code rules.  The inverse of [`encode`](@ref).

```jldoctest
julia> decode(FP32, 0x41C80000)
25.0

julia> decode(E2M1, 0b0111)      # the top FP4 code
6.0
```
"""
function decode(f::FloatFormat, code::Integer)
    code = UInt64(code) & ((UInt64(1) << nbits(f)) - 1)
    mmask = (UInt64(1) << f.mbits) - 1
    emask = (UInt64(1) << f.ebits) - 1
    M = f.mbits == 0 ? UInt64(0) : (code & mmask)
    E = (code >> f.mbits) & emask
    s = f.signed ? ((code >> (f.ebits + f.mbits)) & 1) : UInt64(0)
    neg = (s == 1)

    topE = emask
    if f.nan_style == IEEE_NAN && E == topE
        return M == 0 ? (neg ? -Inf : Inf) : NaN
    elseif f.nan_style == E4M3_NAN && E == topE && M == mmask
        return NaN
    elseif f.nan_style == E8M0_NAN && E == topE
        return NaN
    end

    v = if E == 0 && f.zero_exp == SUBNORMAL_ZERO
        f.subnormals ? exp2(emin(f)) * (M / nmancodes(f)) : 0.0
    else
        exp2(Int(E) - f.bias) * (1 + M / nmancodes(f))
    end
    return neg ? -v : v
end

# ---- encode / quantize -----------------------------------------------------

"""
    quantize(f::FloatFormat, x::Real) -> Float64

Round `x` to the nearest value representable in `f`, ties to even — the same
`roundTiesToEven` that IEEE 754 makes the default.  Out-of-range magnitudes either
saturate to [`maxfinite`](@ref) or become ±Inf, per the format's `saturate` flag.

This is the *only* lossy step in any encode pipeline; everything else is bookkeeping.

```jldoctest
julia> quantize(E2M1, 2.25)      # rounds to 2, an 11% error
2.0

julia> quantize(E2M1, 3.5)       # exact midpoint: ties-to-even picks 4 (M=0)
4.0

julia> quantize(E2M1, 100.0)     # no infinities in FP4 — it clamps
6.0
```
"""
function quantize(f::FloatFormat, x::Real)
    xf = Float64(x)
    isnan(xf) && return NaN
    neg = signbit(xf)
    ax = abs(xf)

    if isinf(ax)
        f.saturate && return neg ? -maxfinite(f) : maxfinite(f)
        return neg ? -Inf : Inf
    end

    if ax == 0
        # Formats with NORMAL_ZERO (E8M0) have no encoding for zero at all.
        f.zero_exp == NORMAL_ZERO && return minnormal(f)
        return neg && f.signed ? -0.0 : 0.0
    end

    # Locate the binade, then clamp into the pinned subnormal scale if we fell off
    # the bottom.  `e` is the exponent whose ulp we will round against.
    e = floor(Int, log2(ax))
    # guard against log2 landing a hair on the wrong side of a power of two
    if exp2(e) > ax
        e -= 1
    elseif exp2(e + 1) <= ax
        e += 1
    end
    lo = emin(f)
    if e < lo
        f.subnormals || return neg && f.signed ? -0.0 : 0.0
        e = lo
    end

    q = exp2(e - f.mbits)                 # ulp of this binade
    n = round(ax / q, RoundNearest)       # Julia's RoundNearest is ties-to-even
    v = n * q

    if v > maxfinite(f)
        if f.saturate
            v = maxfinite(f)
        else
            return neg ? -Inf : Inf
        end
    end
    return neg && (f.signed || v == 0) ? -v : v
end

"""
    encode(f::FloatFormat, x::Real) -> UInt64

Quantize `x` onto `f`'s grid and return the raw bit pattern.  Round-trips with
[`decode`](@ref): `decode(f, encode(f, x)) == quantize(f, x)`.

```jldoctest
julia> string(encode(FP32, -6.375), base=16)
"c0cc0000"
```
"""
function encode(f::FloatFormat, x::Real)
    xf = Float64(x)
    mmask = (UInt64(1) << f.mbits) - 1
    emask = (UInt64(1) << f.ebits) - 1

    if isnan(xf)
        f.nan_style == NO_SPECIAL && throw(ArgumentError("$(f.name) has no NaN encoding"))
        if f.nan_style == E4M3_NAN
            return (emask << f.mbits) | mmask
        elseif f.nan_style == E8M0_NAN
            return emask << f.mbits
        else
            return (emask << f.mbits) | UInt64(1)
        end
    end

    v = quantize(f, xf)
    sbit = (f.signed && signbit(v)) ? UInt64(1) << (f.ebits + f.mbits) : UInt64(0)
    av = abs(v)

    if isinf(av)
        return sbit | (emask << f.mbits)
    end
    if av == 0 && f.zero_exp == SUBNORMAL_ZERO
        return sbit
    end

    e = floor(Int, log2(av))
    if exp2(e) > av
        e -= 1
    elseif exp2(e + 1) <= av
        e += 1
    end
    if e < emin(f)                       # subnormal
        M = round(UInt64, av / exp2(emin(f) - f.mbits))
        return sbit | M
    end
    E = UInt64(e + f.bias)
    M = round(UInt64, (av / exp2(e) - 1) * nmancodes(f))
    if M == nmancodes(f)                 # rounding carried into the next binade
        M = UInt64(0)
        E += 1
    end
    return sbit | (E << f.mbits) | M
end

# ---- exhaustive enumeration ------------------------------------------------

"""
    grid(f::FloatFormat; finite_only=true) -> Vector{Float64}

Every distinct value the format can represent, sorted ascending.  Practical only for
narrow formats — it refuses above 20 bits.  For FP4 this is the whole number system
on one line, which is what makes the format so pleasant to reason about:

```jldoctest
julia> grid(E2M1)
15-element Vector{Float64}:
 -6.0
 -4.0
 -3.0
 -2.0
 -1.5
 -1.0
 -0.5
  0.0
  0.5
  1.0
  1.5
  2.0
  3.0
  4.0
  6.0
```
"""
function grid(f::FloatFormat; finite_only::Bool = true)
    n = nbits(f)
    n <= 20 || throw(ArgumentError("grid() refuses $(n)-bit formats ($(1<<n) codes); " *
                                   "use posgrid() or sample the format instead"))
    vals = Float64[]
    for c in 0:((1 << n) - 1)
        v = decode(f, c)
        (finite_only && !isfinite(v)) && continue
        push!(vals, v == 0 ? 0.0 : v)   # collapse -0.0 onto +0.0: same *value*
    end
    unique!(sort!(vals))
    return vals
end

"""
    posgrid(f::FloatFormat) -> Vector{Float64}

The non-negative half of [`grid`](@ref); the negative half mirrors it."""
posgrid(f::FloatFormat) = filter(>=(0.0), grid(f))

"""
    midpoints(f::FloatFormat) -> Vector{Float64}

The rounding-cell boundaries of the positive grid: the midpoints between adjacent
representable values.  Landing exactly on one of these invokes ties-to-even.

For E2M1 these are `0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0` — the report's tie table.
"""
function midpoints(f::FloatFormat)
    g = posgrid(f)
    [(g[i] + g[i+1]) / 2 for i in 1:length(g)-1]
end

"""
    ulp(f::FloatFormat, x::Real) -> Float64

The spacing of the grid in the neighbourhood of `x` (its "unit in the last place").
Constant within a binade, doubling at every power of two."""
function ulp(f::FloatFormat, x::Real)
    ax = abs(Float64(x))
    ax == 0 && return exp2(emin(f) - f.mbits)
    e = floor(Int, log2(ax))
    e = max(e, emin(f))
    exp2(e - f.mbits)
end

# ---- pretty printing -------------------------------------------------------

function Base.show(io::IO, f::FloatFormat)
    print(io, "FloatFormat(", f.name, ": E", f.ebits, "M", f.mbits,
          ", bias=", f.bias, ", ", nbits(f), " bits)")
end

function Base.show(io::IO, ::MIME"text/plain", f::FloatFormat)
    println(io, f.name, "  —  ", nbits(f), "-bit format (",
            f.signed ? "1 sign, " : "no sign, ", f.ebits, " exponent, ", f.mbits, " mantissa)")
    println(io, "  bias              : ", f.bias)
    println(io, "  normal exponents  : E = ", f.zero_exp == SUBNORMAL_ZERO ? 1 : 0,
            " … ", max_normal_E(f), "   ⇒  e = ", emin(f), " … ", emax(f))
    println(io, "  significand bits  : ", f.mbits + 1, "  (≈ ",
            round(decimal_digits(f), digits=2), " decimal digits)")
    println(io, "  machine epsilon   : 2^-", f.mbits, " = ", machine_eps(f))
    println(io, "  largest finite    : ", maxfinite(f))
    println(io, "  smallest normal   : ", minnormal(f))
    println(io, "  smallest subnormal: ", minsubnormal(f))
    println(io, "  dynamic range     : ", round(dynamic_range(f), sigdigits=4), " : 1")
    print(io,   "  specials          : ",
          f.nan_style == NO_SPECIAL ? "none (overflow saturates)" :
          f.nan_style == E4M3_NAN   ? "NaN only (one code); no ±Inf" :
          f.nan_style == E8M0_NAN   ? "NaN at E=all-ones" : "±Inf and NaN")
end
