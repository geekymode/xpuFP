# ---------------------------------------------------------------------------
# Signed-digit representations.
#
# A positional representation is a digit string valued by one fixed formula,
# x = Σ dᵢ rⁱ; the *system* is the pair (radix r, digit set D).  Ordinary binary
# is (2,{0,1}); ordinary radix 4 is (4,{0,1,2,3}) — minimal digit sets, exactly r
# values, and that minimality is what forces uniqueness.
#
# Nothing in the value formula requires minimality.  Enlarge D and the same formula
# happily accepts many strings for one value.  That surplus of spellings is all
# "redundant" means — and these are *working* representations used inside an
# arithmetic circuit, not storage formats.
# ---------------------------------------------------------------------------

"""
    SignedDigits

A positional number in radix `radix` over the symmetric digit alphabet
`{-a, …, a}` where `a = maxdigit`.

`digits[1]` is the **least** significant digit, so the value is
`Σ digits[i] · radix^(i-1)`.  Print it and you get the conventional
most-significant-first rendering with `1̄` for `-1`.

# Redundancy in one line
The alphabet holds `2a+1` values; representing every integer needs at least `radix`
of them, and the system is *redundant* once it has more.  The surplus
`ρ = 2a+1 − radix` is exactly a count of the spelling freedom, and that freedom is
what a carry-free addition rule spends.

```jldoctest
julia> sd = SignedDigits([-1, 0, 0, 1], 2, 1);   # LSB first: 8 − 1

julia> value(sd)
7

julia> sd
100̄1̄ … wait, printed MSB-first: 1 0 0 1̄  (radix 2, digits in [-1,1]) = 7
```
"""
struct SignedDigits
    digits::Vector{Int}
    radix::Int
    maxdigit::Int
    exponent::Int

    # Inner, so there is no unvalidated path: a plain outer constructor would be
    # shadowed by the compiler-generated `SignedDigits(::Vector{Int}, ::Int, ::Int)`
    # for exactly the commonest call, silently admitting out-of-alphabet digits.
    function SignedDigits(digits::AbstractVector{<:Integer}, radix::Integer,
                          maxdigit::Integer, exponent::Integer = 0)
        maxdigit >= 0 || throw(ArgumentError("maxdigit must be ≥ 0, got $(maxdigit)"))
        radix >= 2 || throw(ArgumentError("radix must be ≥ 2, got $(radix)"))
        for d in digits
            abs(d) <= maxdigit || throw(ArgumentError(
                "digit $(d) is outside the alphabet {-$(maxdigit) … $(maxdigit)}"))
        end
        new(collect(Int, digits), Int(radix), Int(maxdigit), Int(exponent))
    end
end

"""
    value(sd::SignedDigits)

Evaluate the represented number, **exactly**:

```math
x = \\sum_i d_i \\, r^{\\,i-1+\\mathrm{exponent}}
```

This is the invariant every redundant algorithm in this module preserves:
*representation free, value invariant*.

The return type follows the value, not the container: an `Integer` when the number is
integral (which is every `exponent ≥ 0` case), otherwise an exact
`Rational{BigInt}`.  Use [`float_value`](@ref) if you want a `Float64` regardless.

```jldoctest
julia> value(SignedDigits([-1, 0, 0, 1], 2, 1))        # 8 − 1
7

julia> value(SignedDigits([1, 1], 4, 2, -1))           # 1/4 + 1
5//4
```
"""
function value(sd::SignedDigits)
    r = BigInt(sd.radix)
    acc = zero(Rational{BigInt})
    for (i, d) in enumerate(sd.digits)
        d == 0 && continue
        p = sd.exponent + i - 1
        acc += p >= 0 ? Rational{BigInt}(BigInt(d) * r^p) :
                        Rational{BigInt}(BigInt(d), r^(-p))
    end
    if isinteger(acc)
        n = numerator(acc)
        return typemin(Int) <= n <= typemax(Int) ? Int(n) : n
    end
    return acc
end

"""
    scaled_integer(sd::SignedDigits) -> BigInt

The digit string read as a plain integer, ignoring the exponent: `N = Σ dᵢ·rⁱ⁻¹`.

Together with the scale this is the whole number, `value = N · radix^exponent` — the
same "integer plus an agreed scale" decomposition that fixed point uses, which is why
a redundant digit string can carry fractions at no structural cost.

```jldoctest
julia> sd = to_rr4(49.25);

julia> scaled_integer(sd), sd.exponent      # 197 × 4^-1 = 49.25
(197, -1)
```
"""
function scaled_integer(sd::SignedDigits)
    r = BigInt(sd.radix)
    N = BigInt(0)
    p = BigInt(1)
    for d in sd.digits
        N += d * p
        p *= r
    end
    N
end

"""
    digit_parts(sd::SignedDigits) -> NamedTuple

Unpack a digit string into its pieces, for when you want the raw arrays rather than the
container.

# Returned fields
- `digits::Vector{Int}` — least significant first, the field order.
- `digits_msb::Vector{Int}` — most significant first, the *printed* order.
- `scale::Int` — the integer exponent: the least significant digit has weight
  `radix^scale`.
- `scaled_integer::BigInt` — `N`, with `value == N · radix^scale`.
- `radix`, `maxdigit`, `alphabet`, `nonzeros`, `value`, `string`.

!!! note "Two orders, deliberately opposite"
    `digits` is indexed by *significance* (`digits[1]` is the least significant, so
    `digits[i]` has weight `radix^(i-1+scale)`), while `digits_msb` matches how the
    number reads on the page.  Mixing them up is the easiest indexing error here, so
    both are provided by name.

```jldoctest
julia> p = digit_parts(to_rr4(49));

julia> p.digits, p.scale, p.scaled_integer
([1, 0, -1, 1], 0, 49)

julia> p.digits_msb
4-element Vector{Int64}:
  1
 -1
  0
  1
```
"""
function digit_parts(sd::SignedDigits)
    (digits = copy(sd.digits),
     digits_msb = reverse(sd.digits),
     scale = sd.exponent,
     scaled_integer = scaled_integer(sd),
     radix = sd.radix,
     maxdigit = sd.maxdigit,
     alphabet = (-sd.maxdigit):(sd.maxdigit),
     ndigits = length(sd.digits),
     nonzeros = weight(sd),
     value = value(sd),
     string = digit_string(sd))
end

"""
    float_value(sd::SignedDigits) -> Float64

The represented number as a `Float64`, whatever its exponent.  Convenient for
plotting and comparison; use [`value`](@ref) when you need exactness."""
float_value(sd::SignedDigits) = Float64(value(sd))

"""
    scale(sd::SignedDigits) -> Rational{BigInt}

The weight of the least significant digit, `radix^exponent`."""
function scale(sd::SignedDigits)
    r = BigInt(sd.radix)
    sd.exponent >= 0 ? Rational{BigInt}(r^sd.exponent) : Rational{BigInt}(1, r^(-sd.exponent))
end

"""
    nfracdigits(sd::SignedDigits) -> Int

How many digits sit to the right of the radix point, `max(0, -exponent)`."""
nfracdigits(sd::SignedDigits) = max(0, -sd.exponent)

"""
    isintegral(sd::SignedDigits) -> Bool

Whether the represented value is an integer."""
isintegral(sd::SignedDigits) = isinteger(value(sd))

"""    ndigits_sd(sd) -> Int"""
ndigits_sd(sd::SignedDigits) = length(sd.digits)

"""
    weight(sd::SignedDigits) -> Int

Number of **nonzero** digits.  For a constant multiplier this is the count that
matters: one adder per nonzero digit beyond the first, with `-1` digits free because
subtraction costs the same as addition in two's complement."""
weight(sd::SignedDigits) = count(!=(0), sd.digits)

"""
    adders(sd::SignedDigits) -> Int

Adders needed to multiply by this constant: `weight − 1`."""
adders(sd::SignedDigits) = max(weight(sd) - 1, 0)

"""
    is_redundant(sd::SignedDigits) -> Bool

Whether the alphabet is larger than the radix, i.e. whether multiple spellings of a
value exist."""
is_redundant(sd::SignedDigits) = 2 * sd.maxdigit + 1 > sd.radix

"""
    redundancy(sd::SignedDigits) -> Int

The surplus `ρ = 2a + 1 − r`.  Radix 4 offers exactly two redundant settings:
`ρ = 1` (minimally redundant, `{-2..2}` — Booth's alphabet) and `ρ = 3` (maximally
redundant, `{-3..3}` — the fully local carry-free adder's alphabet)."""
redundancy(sd::SignedDigits) = 2 * sd.maxdigit + 1 - sd.radix

"""
    has_adjacent_nonzeros(sd::SignedDigits) -> Bool

Whether any two neighbouring digits are both nonzero — the property the canonical
signed-digit (non-adjacent) form forbids."""
function has_adjacent_nonzeros(sd::SignedDigits)
    for i in 1:length(sd.digits)-1
        sd.digits[i] != 0 && sd.digits[i+1] != 0 && return true
    end
    false
end

"""
    digit_string(sd::SignedDigits; msb_first=true, point=true) -> String

Render the digits with an overbar-style `1̄` for negatives, most significant first by
convention, inserting a radix point when the exponent is negative.

```jldoctest
julia> digit_string(SignedDigits([-1, 0, 0, 1], 2, 1))
"1001̄"

julia> digit_string(to_rr4(49.25))
"301.1"
```
"""
function digit_string(sd::SignedDigits; msb_first::Bool = true, point::Bool = true)
    nf = nfracdigits(sd)
    chars = _digitchar.(sd.digits)                      # LSB first
    if !msb_first
        return join(chars)
    end
    if !point || nf == 0
        # a positive exponent means implied trailing zeros; show them for honesty
        pad = sd.exponent > 0 ? repeat("0", sd.exponent) : ""
        return join(reverse(chars)) * pad
    end
    intpart = length(chars) > nf ? join(reverse(chars[nf+1:end])) : "0"
    fracpart = join(reverse(chars[1:min(nf, length(chars))]))
    length(chars) < nf && (fracpart = repeat("0", nf - length(chars)) * fracpart)
    return intpart * "." * fracpart
end

function _digitchar(d::Integer)
    d == 0 && return "0"
    d > 0 && return string(d)
    return string(-d) * "̄"      # combining macron: 1̄
end

Base.length(sd::SignedDigits) = length(sd.digits)
Base.getindex(sd::SignedDigits, i) = sd.digits[i]

function Base.show(io::IO, sd::SignedDigits)
    print(io, digit_string(sd), " (radix ", sd.radix, ", digits ±", sd.maxdigit,
          sd.exponent == 0 ? "" : ", exp $(sd.exponent)", ") = ", value(sd))
end

function Base.show(io::IO, ::MIME"text/plain", sd::SignedDigits)
    n = length(sd.digits)
    e = sd.exponent
    println(io, "SignedDigits  radix ", sd.radix, ", alphabet {-", sd.maxdigit,
            " … ", sd.maxdigit, "}", is_redundant(sd) ? "  [redundant, ρ=$(redundancy(sd))]" :
                                                        "  [non-redundant, unique]")
    println(io, "  digits (MSB→LSB) : ", digit_string(sd), "     (", n, " digits)")
    # Print the array in the SAME order as the line above, so the two can actually be
    # read against each other; the `.digits` field is the reverse, and says so.
    println(io, "  same, as an array: ", _msb_array_string(sd))
    println(io, "  ↑ MSB→LSB.  The .digits FIELD is the reverse (LSB→MSB): ",
            "digits[i] has weight ", sd.radix, "^(i-1", e == 0 ? "" : (e > 0 ? "+$(e)" : "$(e)"), ")")
    println(io, "  scale (exponent) : ", e, "   ⇒ digits[1] (least significant) has weight ",
            sd.radix, "^", e, e == 0 ? " = 1" : " = $(scale(sd))")
    N = scaled_integer(sd)
    v = value(sd)
    checked = v == N * (e >= 0 ? Rational{BigInt}(BigInt(sd.radix)^e) :
                                 Rational{BigInt}(1, BigInt(sd.radix)^(-e)))
    println(io, "  scaled integer   : ", N, "   ⇒ value = ", N, " × ", sd.radix, "^", e,
            checked ? "   ✓" : "   ✗ MISMATCH")
    println(io, "  value            : ", v, isintegral(sd) ? "" :
            "  ≈ $(round(float_value(sd), sigdigits = 10))")
    println(io, "  nonzero digits   : ", weight(sd), "  ⇒ ", adders(sd),
            " adders as a constant multiplier")
    print(io,   "  adjacent nonzeros: ", has_adjacent_nonzeros(sd) ? "yes" :
          (sd.radix == 2 ? "no — this is the canonical (non-adjacent) form" :
                           "no — but at radix $(sd.radix) non-adjacency is not the canonical rule"))
end

# Render the digits MSB→LSB with a radix point, compactly, eliding the middle of long
# strings so the head and tail stay checkable against the printed digit string.
# Deliberately space-free: the whole purpose is to fit as many digits on the line as
# possible for cross-checking, and a comma alone separates them unambiguously.
function _msb_array_string(sd::SignedDigits; maxshow::Int = 44)
    n = length(sd.digits)
    nf = nfracdigits(sd)
    msb = reverse(sd.digits)                       # index 1 is now most significant
    pointat = n - nf                               # radix point after this many entries

    toks = String[]
    if nf > 0 && pointat <= 0
        # the value is below 1: the point sits outside the array, and the printed
        # string carries implied leading zeros — show them so the two still align
        push!(toks, "·")
        for _ in 1:(-pointat)
            push!(toks, "0")
        end
    end
    for k in 1:n
        (nf > 0 && pointat > 0 && k == pointat + 1) && push!(toks, "·")
        push!(toks, string(msb[k]))
    end

    ndig = count(!=("·"), toks)
    if ndig > maxshow
        head = cld(maxshow, 2)
        tail = maxshow - head
        # keep whole tokens at each end, counting only digits toward the budget
        hi = 0; seen = 0
        while seen < head
            hi += 1
            toks[hi] != "·" && (seen += 1)
        end
        lo = length(toks) + 1; seen = 0
        while seen < tail
            lo -= 1
            toks[lo] != "·" && (seen += 1)
        end
        toks = vcat(toks[1:hi], ["…($(ndig - maxshow) more)…"], toks[lo:end])
    end
    "{" * _joincompact(toks) * "}"
end

# comma between digits, but never beside the radix point
function _joincompact(toks::Vector{String})
    io = IOBuffer()
    for (i, t) in enumerate(toks)
        if i > 1 && t != "·" && toks[i-1] != "·"
            print(io, ",")
        end
        print(io, t)
    end
    String(take!(io))
end

# ---- plain conversions -----------------------------------------------------

"""
    to_binary(x::Integer, n=0) -> Vector{Int}

The unsigned binary digits of `x`, least significant first, zero-padded to `n`
positions (`n = 0` means "as many as needed")."""
function to_binary(x::Integer, n::Integer = 0)
    x >= 0 || throw(ArgumentError("to_binary expects a non-negative integer"))
    bits = Int[]
    v = x
    while v > 0
        push!(bits, v & 1); v >>= 1
    end
    isempty(bits) && push!(bits, 0)
    while length(bits) < n
        push!(bits, 0)
    end
    bits
end

"""
    twos_complement_bits(B::Integer, n::Integer) -> Vector{Int}

The `n`-bit two's-complement pattern of `B`, least significant bit first.  Valid for
`-2^(n-1) ≤ B ≤ 2^(n-1) − 1`."""
function twos_complement_bits(B::Integer, n::Integer)
    lo, hi = -(1 << (n - 1)), (1 << (n - 1)) - 1
    lo <= B <= hi || throw(ArgumentError("$B does not fit in $n-bit two's complement"))
    u = B >= 0 ? B : B + (1 << n)
    [Int((u >> i) & 1) for i in 0:n-1]
end

"""
    twos_complement_value(bits) -> Int

Read a two's-complement bit vector (LSB first) back as a signed integer:
`−b_{n-1}2^{n-1} + Σ_{i<n-1} bᵢ2ⁱ`."""
function twos_complement_value(bits::AbstractVector{<:Integer})
    n = length(bits)
    v = 0
    for i in 1:n-1
        v += bits[i] << (i - 1)
    end
    v - (bits[n] << (n - 1))
end

"""
    to_radix(x::Integer, r::Integer) -> Vector{Int}

The conventional non-redundant digits of `x ≥ 0` in radix `r`, least significant
first — the canonical `{0 … r−1}` form that redundant values convert *back* to at the
exit of an arithmetic unit."""
function to_radix(x::Integer, r::Integer)
    x >= 0 || throw(ArgumentError("to_radix expects a non-negative integer"))
    ds = Int[]
    v = x
    while v > 0
        push!(ds, v % r); v ÷= r
    end
    isempty(ds) && push!(ds, 0)
    ds
end
