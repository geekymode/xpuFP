# ---------------------------------------------------------------------------
# One conversion surface across every format family in the package.
#
# The pipeline is always the same three steps: decode the source to an exact real,
# round once onto the target's grid, store.  Decoding is exact in every scheme, so all
# the loss lives in the middle step — which is what makes "where is the loss?" a
# well-posed question to ask of any pair of formats.
# ---------------------------------------------------------------------------

"""
    AnyFormat

Every format family this package can convert between: [`FloatFormat`](@ref),
[`FixedFormat`](@ref), [`IntFormat`](@ref), [`DigitFormat`](@ref) and
[`BlockFormat`](@ref)."""
const AnyFormat = Union{FloatFormat,FixedFormat,IntFormat,DigitFormat,BlockFormat}

"""
    realvalue(x) -> Real

The exact real number a stored object denotes, whatever its container.

`SignedDigits` decode exactly (to an `Integer` or a `Rational`); floats and fixed-point
values are already reals; a [`QuantizedBlock`](@ref) decodes to its reconstruction
vector."""
realvalue(x::Real) = x
realvalue(sd::SignedDigits) = value(sd)
realvalue(qb::QuantizedBlock) = dequantize(qb)
realvalue(v::AbstractVector) = collect(Float64, v)

"""
    represent(fmt, x; kwargs...)

Store `x` in `fmt`, returning the format's **native container**:

| Format | Container |
|:---|:---|
| [`FloatFormat`](@ref), [`FixedFormat`](@ref), [`IntFormat`](@ref) | `Float64` (the quantized value) |
| [`DigitFormat`](@ref) | [`SignedDigits`](@ref) |
| [`BlockFormat`](@ref) | [`QuantizedBlock`](@ref) (vector input) |

```jldoctest
julia> represent(FP32, 0.1)
0.10000000149011612

julia> digit_string(represent(RR4, 16.15625))
"101.1̄1̄2"
```
"""
represent(f::Union{FloatFormat,FixedFormat,IntFormat}, x; kwargs...) =
    quantize(f, realvalue(x); kwargs...)
represent(f::DigitFormat, x; kwargs...) = to_digits(f, realvalue(x); kwargs...)
represent(bf::BlockFormat, x::AbstractVector; kwargs...) = quantize_block(bf, realvalue(x))
represent(f::Union{FloatFormat,FixedFormat,IntFormat}, x::AbstractVector; kwargs...) =
    [quantize(f, v; kwargs...) for v in x]

"""
    convert_format(to, from, x; kwargs...)
    convert_format(to, x; kwargs...)

Convert a value from one format to another, returning the target's native container.

The two-argument form infers the source from the value itself — use it when `x` is
already a stored object (a `SignedDigits`, or a `Float64` you know is exact).  The
three-argument form first rounds `x` onto `from`'s grid, which is what you want when
modelling a real pipeline: *this value was already living in FP32 before we narrowed it*.

Any pair of families is allowed, so this spans float → digit, digit → fixed,
block → float and everything between.

```jldoctest
julia> convert_format(E2M1, FP32, 2.25)
2.0

julia> digit_string(convert_format(RR4, FP32, 6.5))
"12.2"

julia> convert_format(FP32, RR4, to_digits(RR4, 6.5))
6.5

julia> convert_format(FixedFormat(7, 8), FP32, -3.65)
-3.6484375
```
"""
convert_format(to::AnyFormat, from::AnyFormat, x; kwargs...) =
    represent(to, realvalue(represent(from, x)); kwargs...)
convert_format(to::AnyFormat, x; kwargs...) = represent(to, realvalue(x); kwargs...)

convert_format(to::AnyFormat, from::AnyFormat, x::AbstractVector; kwargs...) =
    represent(to, realvalue(represent(from, x)); kwargs...)

"""
    ConversionReport

What one format-to-format conversion did: the value in, the value out, the error, and
whether it hit an edge (overflow, underflow, the subnormal ramp, saturation)."""
struct ConversionReport
    from::String
    to::String
    input::Float64
    output::Float64
    abserror::Float64
    relerror::Float64
    exact::Bool
    overflowed::Bool
    underflowed::Bool
    subnormal::Bool
end

"""
    conversion_report(to, from, x) -> ConversionReport

Convert and *audit*: how far the value moved, and which edge of the target format it
met, if any.

```jldoctest
julia> r = conversion_report(E2M1, FP32, 2.25);

julia> r.output, round(r.relerror, digits=4), r.exact
(2.0, 0.1111, false)

julia> conversion_report(FP16, FP32, 1.0e30).overflowed
true
```
"""
function conversion_report(to::AnyFormat, from::AnyFormat, x::Real)
    src = Float64(realvalue(represent(from, x)))
    outc = represent(to, src)
    out = Float64(realvalue(outc))
    ae = abs(out - src)
    ConversionReport(_fmtname(from), _fmtname(to), src, out, ae,
                     src == 0 ? abs(out) : ae / abs(src),
                     out == src,
                     _overflowed(to, src), src != 0 && out == 0, _subnormal(to, out))
end

_overflowed(f::FloatFormat, v::Real) = abs(v) > maxfinite(f)
_overflowed(f::FixedFormat, v::Real) = v > fxmax(f) || v < fxmin(f)
_overflowed(f::IntFormat, v::Real) = v > maxfinite(f) || v < intmin(f)
_overflowed(::DigitFormat, ::Real) = false          # digit strings simply grow
_overflowed(bf::BlockFormat, v::Real) = false
_subnormal(f::FloatFormat, v::Real) = v != 0 && abs(v) < minnormal(f)
_subnormal(::Any, ::Real) = false

_fmtname(f::DigitFormat) = f.name

"""
    conversion_matrix(formats, x) -> Matrix{Float64}

Round-trip `x` through every ordered pair of `formats`, returning the relative error of
each conversion.  A quick way to see which pairs are lossless (digit systems, and any
widening) and which are not.

```julia
julia> conversion_matrix((FP32, BF16, E2M1, RR4), 2.25)
```
"""
function conversion_matrix(formats, x::Real)
    n = length(formats)
    M = zeros(Float64, n, n)
    for (i, f) in enumerate(formats), (j, t) in enumerate(formats)
        M[i, j] = conversion_report(t, f, x).relerror
    end
    M
end

function Base.show(io::IO, ::MIME"text/plain", r::ConversionReport)
    println(io, r.from, "  →  ", r.to)
    @printf(io, "  in      : %.17g\n", r.input)
    @printf(io, "  out     : %.17g\n", r.output)
    @printf(io, "  abs err : %.6g\n", r.abserror)
    @printf(io, "  rel err : %.6g%s\n", r.relerror, r.exact ? "   (exact — no loss)" : "")
    flags = String[]
    r.overflowed && push!(flags, "OVERFLOW")
    r.underflowed && push!(flags, "UNDERFLOW to zero")
    r.subnormal && push!(flags, "landed in the subnormal ramp")
    print(io, "  edges   : ", isempty(flags) ? "none" : join(flags, ", "))
end
