# ---------------------------------------------------------------------------
# Arithmetic operators on digit strings.
#
# These make `to_rr4(a) + to_rr4(b)` just work.  They are exact — redundant systems
# trade cost, never accuracy — and for radix 4 the addition genuinely runs the
# carry-free rule rather than detouring through the value.
#
# When you want the *cost* as well as the answer, use `rr4_add` (which returns a full
# column-by-column trace) or `compare_adders` / `compare_multipliers`.
# ---------------------------------------------------------------------------

function _check_same_radix(x::SignedDigits, y::SignedDigits)
    x.radix == y.radix || throw(ArgumentError(
        "cannot combine radix-$(x.radix) and radix-$(y.radix) digit strings; " *
        "convert one first, e.g. to_digits(RR4, y)"))
end

"""
    promote_alphabet(sd::SignedDigits, maxdigit::Integer) -> SignedDigits

Re-label a digit string as belonging to a wider alphabet.  Free and lossless — the
alphabets are strictly nested, so every legal string in `{-a..a}` is already legal in
`{-b..b}` for `b ≥ a`."""
function promote_alphabet(sd::SignedDigits, maxdigit::Integer)
    Int(maxdigit) >= sd.maxdigit || throw(ArgumentError(
        "promote_alphabet only widens; use maximal_to_minimal to narrow"))
    SignedDigits(sd.digits, sd.radix, Int(maxdigit), sd.exponent)
end

# re-encode an exact value back into the same digit system
function _reencode(x::SignedDigits, v)
    N, e = radix_parts(v, x.radix)
    SignedDigits(_signed_radix_digits(N, x.radix, x.maxdigit), x.radix, x.maxdigit, e)
end

"""
    +(x::SignedDigits, y::SignedDigits) -> SignedDigits

Add two digit strings.  **Exact**, and for radix 4 it runs the genuine carry-free
addition rule — operands are aligned to the finer exponent, then every column resolves
in parallel with transfers that travel one position and stop.

Operands may use different alphabets and different exponents; the result takes the
wider alphabet and the finer exponent.

```jldoctest
julia> s = to_rr4(712.3) + to_rr4(12.3);

julia> Float64(value(s))
724.6

julia> value(s) == Rational{BigInt}(712.3) + Rational{BigInt}(12.3)
true
```

!!! note "This returns the answer, not the cost"
    For the column-by-column trace — transfers, interim digits, depth — call
    [`rr4_add`](@ref) instead, which returns an [`RR4AddTrace`](@ref).  For a
    side-by-side cost comparison against ripple and prefix adders, use
    [`compare_adders`](@ref).
"""
function Base.:+(x::SignedDigits, y::SignedDigits)
    _check_same_radix(x, y)
    a = max(x.maxdigit, y.maxdigit)
    if x.radix == 4 && 2 <= a <= 3
        alp = a >= 3 ? MAX_REDUNDANT : MIN_REDUNDANT
        return rr4_add(promote_alphabet(x, a), promote_alphabet(y, a); alphabet = alp).result
    end
    _reencode(promote_alphabet(x, a), value(x) + value(y))
end

"""
    -(x::SignedDigits) -> SignedDigits

Negate — in a signed-digit system this is **free**: flip the sign of every digit, no
borrow, no carry, no conversion.  It is the same fact that makes `-1` digits cost the
same as `+1` ones in a CSD constant multiplier."""
Base.:-(x::SignedDigits) = SignedDigits(.-x.digits, x.radix, x.maxdigit, x.exponent)

"""    -(x::SignedDigits, y::SignedDigits) -> SignedDigits

Subtraction, i.e. `x + (-y)`.  Costs exactly what addition costs, because negation is
free."""
Base.:-(x::SignedDigits, y::SignedDigits) = x + (-y)

"""
    *(x::SignedDigits, y::SignedDigits) -> SignedDigits

Multiply two digit strings.  Exponents add and the scaled integers multiply, so the
result is exact.

```jldoctest
julia> p = to_rr4(49.25) * to_rr4(4);

julia> value(p)
197//1

julia> to_rr4(12) * to_rr4(12) |> value
144
```

!!! note "The interesting multiplication is elsewhere"
    This gives the product.  For the *structure* — partial products, the carry-save
    reduction tree, the single exit CPA — use [`rr4_multiply`](@ref),
    [`wallace_multiply`](@ref) or [`booth_multiply`](@ref), which report rows, tree
    levels, cells and depth.
"""
function Base.:*(x::SignedDigits, y::SignedDigits)
    _check_same_radix(x, y)
    a = max(x.maxdigit, y.maxdigit)
    N = scaled_integer(x) * scaled_integer(y)
    e = x.exponent + y.exponent
    SignedDigits(_signed_radix_digits(N, x.radix, a), x.radix, a, e)
end

# --- mixed arithmetic with ordinary numbers ---------------------------------
# The scalar is converted into the digit string's own system first, so
# `to_rr4(712.3) + 12.3` means what it looks like.

for op in (:+, :-, :*)
    @eval begin
        Base.$op(x::SignedDigits, y::Real) = $op(x, _reencode(x, y))
        Base.$op(x::Real, y::SignedDigits) = $op(_reencode(y, x), y)
    end
end

"""
    ==(x::SignedDigits, y::SignedDigits) -> Bool

Compare by **value**, not by spelling.

This is the right semantics for a redundant system: `(1,0,3)₄` and `(1,0,-1,1)₄` are
different strings denoting the same 49, and the whole design principle is
*representation free, value invariant*.  Use [`same_spelling`](@ref) when you need
digit-for-digit identity.

```jldoctest
julia> to_rr4(49) == to_rr4_maximal(49)      # different alphabets, different digits
true

julia> same_spelling(to_rr4(49), to_rr4_maximal(49))
false
```
"""
Base.:(==)(x::SignedDigits, y::SignedDigits) = value(x) == value(y)
Base.:(==)(x::SignedDigits, y::Real) = value(x) == y
Base.:(==)(x::Real, y::SignedDigits) = x == value(y)
Base.hash(x::SignedDigits, h::UInt) = hash(value(x), h)

"""
    same_spelling(x::SignedDigits, y::SignedDigits) -> Bool

Digit-for-digit identity, including alphabet and exponent — the stricter comparison
that `==` deliberately does not perform."""
same_spelling(x::SignedDigits, y::SignedDigits) =
    x.radix == y.radix && x.maxdigit == y.maxdigit && x.exponent == y.exponent &&
    x.digits == y.digits

Base.isless(x::SignedDigits, y::SignedDigits) = value(x) < value(y)
Base.isless(x::SignedDigits, y::Real) = value(x) < y
Base.isless(x::Real, y::SignedDigits) = x < value(y)

Base.zero(x::SignedDigits) = SignedDigits([0], x.radix, x.maxdigit, x.exponent)
Base.iszero(x::SignedDigits) = all(iszero, x.digits)
Base.abs(x::SignedDigits) = value(x) < 0 ? -x : x
Base.sign(x::SignedDigits) = sign(value(x))
Base.Float64(x::SignedDigits) = float_value(x)

"""
    sum_rr4(xs; alphabet=MIN_REDUNDANT) -> SignedDigits

Accumulate many values in the redundant domain, converting **once** at the end.

This is the shape the whole redundancy argument is about: `k` additions cost
`Θ(k)` constant-depth steps plus one exit conversion, against `Θ(k log n)` for a
conventional chain that re-canonicalises after every term.

```jldoctest
julia> value(sum_rr4([1, 2, 3, 4, 5]))
15

julia> Float64(value(sum_rr4([712.3, 12.3, 0.4])))
725.0
```
"""
function sum_rr4(xs; alphabet::RR4Alphabet = MIN_REDUNDANT)
    isempty(xs) && return to_rr4(0; alphabet)
    acc = to_rr4(first(xs); alphabet)
    for v in Iterators.drop(xs, 1)
        acc = acc + to_rr4(v; alphabet)
    end
    acc
end

"""
    dot_rr4(xs, ys; alphabet=MIN_REDUNDANT) -> SignedDigits

A dot product computed entirely in the redundant domain — exact, with one conversion at
the exit rather than one per term."""
function dot_rr4(xs, ys; alphabet::RR4Alphabet = MIN_REDUNDANT)
    length(xs) == length(ys) || throw(DimensionMismatch("dot_rr4: length mismatch"))
    isempty(xs) && return to_rr4(0; alphabet)
    acc = to_rr4(first(xs); alphabet) * to_rr4(first(ys); alphabet)
    for i in 2:length(xs)
        acc = acc + to_rr4(xs[i]; alphabet) * to_rr4(ys[i]; alphabet)
    end
    acc
end
