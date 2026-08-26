# ---------------------------------------------------------------------------
# Fixed point: an ordinary two's-complement integer with an agreed binary point.
# ---------------------------------------------------------------------------

"""
    FixedFormat

A signed Q`m`.`n` fixed-point format: 1 sign bit, `m` integer bits, `n` fraction
bits, `1 + m + n` bits in total.  The stored integer `I` denotes the real value
`I / 2^n`.

Unlike floating point the spacing is *uniform*: the step is `2^-n` everywhere, so
absolute error is bounded but relative error explodes for small values.  That single
difference is the whole fixed-vs-float trade-off.

# Fields
- `name::String`
- `m::Int` — integer bits, excluding the sign bit.
- `n::Int` — fraction bits.
- `signed::Bool` — whether a sign bit is present (unsigned Q formats set this false).
"""
struct FixedFormat
    name::String
    m::Int
    n::Int
    signed::Bool
end

"""
    FixedFormat(m, n; signed=true, name="Q\$m.\$n")

Build a Q`m`.`n` fixed-point format.

```jldoctest
julia> f = FixedFormat(7, 8);   # the report's 16-bit Q7.8

julia> nbits(f), resolution(f), fxmax(f)
(16, 0.00390625, 127.99609375)
```
"""
FixedFormat(m::Integer, n::Integer; signed::Bool = true,
            name::AbstractString = "Q$(m).$(n)") =
    FixedFormat(String(name), Int(m), Int(n), signed)

"Total width in bits."
nbits(f::FixedFormat) = (f.signed ? 1 : 0) + f.m + f.n

"""    resolution(f) -> Float64

The constant step between adjacent representable values, `2^-n`."""
resolution(f::FixedFormat) = exp2(-f.n)

"""    fxmax(f) -> Float64

Largest representable value, `2^m - 2^-n`."""
fxmax(f::FixedFormat) = exp2(f.m) - exp2(-f.n)

"""    fxmin(f) -> Float64

Smallest (most negative) representable value: `-2^m` if signed, else `0`."""
fxmin(f::FixedFormat) = f.signed ? -exp2(f.m) : 0.0

"""    fxrange(f) -> Tuple{Float64,Float64}"""
fxrange(f::FixedFormat) = (fxmin(f), fxmax(f))

"Largest storable integer code."
imax(f::FixedFormat) = f.signed ? (1 << (f.m + f.n)) - 1 : (1 << (f.m + f.n)) - 1
"Smallest storable integer code."
imin(f::FixedFormat) = f.signed ? -(1 << (f.m + f.n)) : 0

"""
    OverflowMode

What to do when a fixed-point result leaves the representable range.

- `SATURATE`: clamp to the nearest endpoint (the safe default in DSP).
- `WRAP`: two's-complement wraparound — fast, and catastrophic when it happens.
- `THROW`: raise an `OverflowError`.
"""
@enum OverflowMode SATURATE WRAP THROW

"""
    encode(f::FixedFormat, x::Real; mode=SATURATE) -> Int

Scale, round (ties to even), and store `x` as the underlying integer.

```jldoctest
julia> encode(FixedFormat(7,8), -3.65)      # the report's worked example
-934
```
"""
function encode(f::FixedFormat, x::Real; mode::OverflowMode = SATURATE)
    scaled = Float64(x) * exp2(f.n)
    I = round(Int, scaled, RoundNearest)
    return _clampcode(f, I, mode)
end

function _clampcode(f::FixedFormat, I::Integer, mode::OverflowMode)
    lo, hi = imin(f), imax(f)
    if I > hi || I < lo
        if mode == SATURATE
            return I > hi ? hi : lo
        elseif mode == THROW
            throw(OverflowError("$(I) does not fit in $(f.name)"))
        else
            width = 1 << nbits(f)
            J = mod(I - lo, width) + lo
            return J
        end
    end
    return Int(I)
end

"""
    decode(f::FixedFormat, I::Integer) -> Float64

Recover the real value `I / 2^n` from a stored integer."""
decode(f::FixedFormat, I::Integer) = Float64(I) * exp2(-f.n)

"""
    quantize(f::FixedFormat, x::Real; mode=SATURATE) -> Float64

Round `x` onto the fixed-point grid and return the real value actually stored.

```jldoctest
julia> quantize(FixedFormat(7,8), -3.65)
-3.6484375
```
"""
quantize(f::FixedFormat, x::Real; mode::OverflowMode = SATURATE) =
    decode(f, encode(f, x; mode))

"""
    grid(f::FixedFormat) -> Vector{Float64}

Every representable value, ascending.  Refuses formats wider than 20 bits."""
function grid(f::FixedFormat)
    nbits(f) <= 20 || throw(ArgumentError("grid() refuses $(nbits(f))-bit fixed formats"))
    [decode(f, I) for I in imin(f):imax(f)]
end

function Base.show(io::IO, f::FixedFormat)
    print(io, "FixedFormat(", f.name, ", ", nbits(f), " bits)")
end

function Base.show(io::IO, ::MIME"text/plain", f::FixedFormat)
    println(io, f.name, "  —  ", nbits(f), "-bit fixed point (",
            f.signed ? "1 sign, " : "unsigned, ", f.m, " integer, ", f.n, " fraction)")
    println(io, "  resolution (step) : 2^-", f.n, " = ", resolution(f))
    println(io, "  range             : [", fxmin(f), ", ", fxmax(f), "]")
    print(io,   "  max abs error     : ", resolution(f) / 2, " (uniform, everywhere)")
end
