# ---------------------------------------------------------------------------
# Digit-string formats.
#
# `FloatFormat` and `FixedFormat` describe how bits are packed; `DigitFormat`
# describes a positional digit system — radix, alphabet, canonical form.  Making it a
# format object (rather than leaving RR4 as a bare value type) is what lets RR4, CSD
# and plain binary sit in the same conversion and plotting machinery as FP32.
# ---------------------------------------------------------------------------

"""
    DigitFormat

A positional digit system: the pair (radix `r`, digit alphabet `D`), plus an optional
canonical form.

Ordinary binary is `(2, {0,1})` and ordinary radix 4 is `(4, {0,1,2,3})` — *minimal*
digit sets, exactly `r` values, and that minimality is what forces uniqueness.  Enlarge
the alphabet and the same value formula happily accepts many strings for one value;
that surplus is all "redundant" means.

# Fields
- `name::String`
- `radix::Int`
- `maxdigit::Int` — alphabet half-width `a`.
- `signed::Bool` — digits run `-a…a` when true, `0…a` when false.
- `canonical::Symbol` — `:none`, or `:nonadjacent` for the CSD/NAF normal form.

Values of a `DigitFormat` are [`SignedDigits`](@ref).
"""
struct DigitFormat
    name::String
    radix::Int
    maxdigit::Int
    signed::Bool
    canonical::Symbol
end

"""
    DigitFormat(name, radix, maxdigit; signed=true, canonical=:none)
"""
DigitFormat(name::AbstractString, radix::Integer, maxdigit::Integer;
            signed::Bool = true, canonical::Symbol = :none) =
    DigitFormat(String(name), Int(radix), Int(maxdigit), signed, canonical)

"""    RR4

Redundant radix 4 over the **minimally redundant** alphabet `{-2..2}` — Booth's
alphabet, and the one every shipping multiplier and SRT divider recodes into."""
const RR4 = DigitFormat("RR4", 4, 2)

"""    RR4_MAX

Redundant radix 4 over the **maximally redundant** alphabet `{-3..3}`, whose extra
slack permits the fully local one-column carry-free addition rule."""
const RR4_MAX = DigitFormat("RR4max", 4, 3)

"""    CSD

Canonical signed digit: radix 2 over `{-1,0,1}`, in non-adjacent form — unique, and
provably minimal in nonzero count."""
const CSD = DigitFormat("CSD", 2, 1; canonical = :nonadjacent)

"""    SIGNED_BINARY

Radix 2 over `{-1,0,1}` with no canonical form imposed — redundant, and the alphabet
Booth's original 1951 recoding emits."""
const SIGNED_BINARY = DigitFormat("signed-binary", 2, 1)

"""    BINARY

Ordinary unsigned binary: radix 2 over `{0,1}`.  Non-redundant, hence unique."""
const BINARY = DigitFormat("binary", 2, 1; signed = false)

"""    RADIX4

Conventional non-redundant radix 4 over `{0,1,2,3}` — the canonical form redundant
values convert *back* to at the exit of an arithmetic unit."""
const RADIX4 = DigitFormat("radix-4", 4, 3; signed = false)

"""    ALL_DIGIT_FORMATS"""
const ALL_DIGIT_FORMATS = (BINARY, SIGNED_BINARY, CSD, RADIX4, RR4, RR4_MAX)

"""
    alphabet(f::DigitFormat) -> UnitRange{Int}

The digit alphabet as a range."""
alphabet(f::DigitFormat) = f.signed ? (-f.maxdigit:f.maxdigit) : (0:f.maxdigit)

"""    nbits(f::DigitFormat) -> Int

Bits per digit cell, `⌈log₂|D|⌉`.

The hardware coincidence that seals the maximally redundant set's popularity:
`⌈log₂5⌉ = ⌈log₂7⌉ = 3`, so at radix 4 both alphabets occupy the same 3-bit cell —
the extra redundancy is **free at the storage level**."""
nbits(f::DigitFormat) = ceil(Int, log2(length(alphabet(f))))

"""    is_redundant(f::DigitFormat) -> Bool

Whether the alphabet is larger than the radix, i.e. whether multiple spellings exist."""
is_redundant(f::DigitFormat) = length(alphabet(f)) > f.radix

"""    redundancy(f::DigitFormat) -> Int

The surplus `|D| − r`.  Radix 4 offers exactly two redundant settings: `ρ = 1`
(minimally redundant) and `ρ = 3` (maximally redundant)."""
redundancy(f::DigitFormat) = length(alphabet(f)) - f.radix

"""
    radix_parts(x, radix; fracdigits=nothing) -> (N::BigInt, exponent::Int)

Decompose `x` into a scaled integer, `x ≈ N · radix^exponent`, exactly where possible.

A value has a finite expansion in base `r` exactly when its denominator divides some
power of `r`.  Every `Float64` is a dyadic rational, so it is exact in base 2 and base
4; `1//3` is exact in neither and must be rounded via `fracdigits`.
"""
function radix_parts(x, radix::Integer; fracdigits = nothing)
    r = Rational{BigInt}(x)
    R = BigInt(radix)
    if fracdigits !== nothing
        f = Int(fracdigits)
        return (round(BigInt, r * R^f), -f)
    end
    q = denominator(r)
    m = 0
    qq = q
    while qq != 1
        g = gcd(qq, R)
        g == 1 && throw(ArgumentError(
            "$(x) has no finite radix-$(radix) expansion (denominator $(q) does not " *
            "divide any power of $(radix)); pass fracdigits = n to round"))
        qq ÷= g
        m += 1
        m > 8192 && throw(ArgumentError("radix_parts: expansion did not terminate"))
    end
    (numerator(r) * R^m ÷ q, -m)
end

"""
    to_digits(f::DigitFormat, x; fracdigits=nothing, ndigits=nothing, exponent=nothing)
        -> SignedDigits

Convert any value into the digit system `f`.

Handles `Integer`, `AbstractFloat`, `Rational` and `SignedDigits` alike.  Unsigned
formats reject negative values; the `:nonadjacent` canonical form runs the greedy CSD
scan instead of the plain conversion.

```jldoctest
julia> digit_string(to_digits(RR4, 49))
"11̄01"

julia> digit_string(to_digits(RADIX4, 49))
"301"

julia> digit_string(to_digits(CSD, 231))
"1001̄01001̄"

julia> value(to_digits(RR4, 16.15625))
517//32
```
"""
function to_digits(f::DigitFormat, x; fracdigits = nothing, ndigits = nothing,
                   exponent = nothing)
    N, e = radix_parts(x isa SignedDigits ? value(x) : x, f.radix; fracdigits)
    if exponent !== nothing
        shift = e - Int(exponent)
        shift >= 0 || throw(ArgumentError(
            "exponent $(exponent) is finer than the value supports (natural $(e)); " *
            "pass fracdigits to add precision first"))
        N *= BigInt(f.radix)^shift
        e = Int(exponent)
    end
    (!f.signed && N < 0) && throw(ArgumentError(
        "$(f.name) is unsigned and cannot represent the negative value $(x)"))

    ds = if f.canonical === :nonadjacent
        f.radix == 2 || throw(ArgumentError("the non-adjacent form is defined for radix 2"))
        csd(N).digits
    else
        _signed_radix_digits(N, f.radix, f.maxdigit)
    end
    if ndigits !== nothing
        n = Int(ndigits)
        length(ds) <= n || throw(ArgumentError(
            "$(x) needs $(length(ds)) digits in $(f.name) at exponent $(e); cannot fit in $(n)"))
        while length(ds) < n
            push!(ds, 0)
        end
    end
    SignedDigits(ds, f.radix, f.maxdigit, e)
end

"""
    quantize(f::DigitFormat, x; kwargs...)

The value `f` actually stores.  Exact whenever `x` has a finite expansion in the radix,
which is the usual case — digit systems are *re-spellings*, not approximations, so
unlike a float format they lose nothing unless you force `fracdigits`."""
quantize(f::DigitFormat, x; kwargs...) = value(to_digits(f, x; kwargs...))

"""    encode(f::DigitFormat, x; kwargs...) -> SignedDigits"""
encode(f::DigitFormat, x; kwargs...) = to_digits(f, x; kwargs...)

"""    decode(::DigitFormat, sd::SignedDigits)"""
decode(::DigitFormat, sd::SignedDigits) = value(sd)

"""
    conforms(f::DigitFormat, sd::SignedDigits) -> Bool

Whether a digit string is legal in the format: right radix, inside the alphabet, and —
for `:nonadjacent` formats — free of adjacent nonzeros."""
function conforms(f::DigitFormat, sd::SignedDigits)
    sd.radix == f.radix || return false
    all(d -> d in alphabet(f), sd.digits) || return false
    f.canonical === :nonadjacent && has_adjacent_nonzeros(sd) && return false
    true
end

Base.show(io::IO, f::DigitFormat) =
    print(io, "DigitFormat(", f.name, ", radix ", f.radix, ", digits ",
          f.signed ? "±$(f.maxdigit)" : "0…$(f.maxdigit)", ")")

function Base.show(io::IO, ::MIME"text/plain", f::DigitFormat)
    println(io, f.name, "  —  positional digit system")
    println(io, "  radix            : ", f.radix)
    println(io, "  alphabet         : {", join(alphabet(f), ", "), "}   (",
            length(alphabet(f)), " values)")
    println(io, "  status           : ", is_redundant(f) ?
            "redundant, surplus ρ = $(redundancy(f)) ⇒ many spellings per value" :
            "non-redundant ⇒ unique spelling")
    println(io, "  bits per digit   : ", nbits(f))
    print(io,   "  canonical form   : ", f.canonical === :none ? "none imposed" :
          "non-adjacent (minimal weight)")
end
