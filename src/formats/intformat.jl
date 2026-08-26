# ---------------------------------------------------------------------------
# Plain integer grids — the fixed-point rival to the tiny floats.
# ---------------------------------------------------------------------------

"""
    IntFormat

A uniform signed-integer grid, the fixed-point competitor to the narrow floats.
`INT4` is the grid `{-8, …, 7}`: constant *absolute* error against E2M1's roughly
constant *relative* error — the report's opening trade-off, decided in E2M1's favour
by the heavy-tailed weight distributions of neural networks, where representing a few
large outliers matters more than covering the middle uniformly.

Set `symmetric = true` for the `{-(2^(b-1)-1), …, 2^(b-1)-1}` variant that many
quantization toolchains prefer, because it makes the grid symmetric about zero.
"""
struct IntFormat
    name::String
    bits::Int
    signed::Bool
    symmetric::Bool
end

"""
    IntFormat(bits; signed=true, symmetric=false, name="INT\$bits")
"""
IntFormat(bits::Integer; signed::Bool = true, symmetric::Bool = false,
          name::AbstractString = "INT$(bits)") =
    IntFormat(String(name), Int(bits), signed, symmetric)

nbits(f::IntFormat) = f.bits

"Largest representable value."
function maxfinite(f::IntFormat)
    f.signed || return Float64((1 << f.bits) - 1)
    Float64((1 << (f.bits - 1)) - 1)
end

"Smallest (most negative) representable value."
function intmin(f::IntFormat)
    f.signed || return 0.0
    f.symmetric ? -maxfinite(f) : -Float64(1 << (f.bits - 1))
end

minsubnormal(f::IntFormat) = 1.0
minnormal(f::IntFormat) = 1.0
machine_eps(f::IntFormat) = 1.0

"""
    quantize(f::IntFormat, x) -> Float64

Round to the nearest integer (ties to even) and clamp into range."""
quantize(f::IntFormat, x::Real) =
    clamp(round(Float64(x), RoundNearest), intmin(f), maxfinite(f))

encode(f::IntFormat, x::Real) = Int(quantize(f, x))
decode(f::IntFormat, I::Integer) = Float64(I)

"""    grid(f::IntFormat) -> Vector{Float64}"""
grid(f::IntFormat) = collect(Float64, intmin(f):maxfinite(f))
posgrid(f::IntFormat) = collect(Float64, 0:maxfinite(f))
ulp(::IntFormat, ::Real) = 1.0

"""    INT4

The 4-bit uniform grid `{-8, …, 7}`."""
const INT4 = IntFormat(4)

"""    INT8

The 8-bit uniform grid `{-128, …, 127}`."""
const INT8 = IntFormat(8)

Base.show(io::IO, f::IntFormat) = print(io, "IntFormat(", f.name, ", ", f.bits, " bits)")
function Base.show(io::IO, ::MIME"text/plain", f::IntFormat)
    println(io, f.name, "  —  ", f.bits, "-bit uniform integer grid")
    println(io, "  range      : [", intmin(f), ", ", maxfinite(f), "]")
    print(io,   "  step       : 1 (constant — uniform absolute error)")
end

"""
    ElementFormat

Any format usable as the *element* of a block-scaled format: a narrow
[`FloatFormat`](@ref) or an [`IntFormat`](@ref)."""
const ElementFormat = Union{FloatFormat,IntFormat}
