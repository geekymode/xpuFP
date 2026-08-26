# ---------------------------------------------------------------------------
# Booth recoding — redundant number systems entering hardware, 1951.
#
# Andrew D. Booth's observation was the cashier's trick industrialised: while the
# multiplier streams past a run of 1s, do nothing but shift, and pay only at the
# run's two boundaries.
# ---------------------------------------------------------------------------

"""
    booth_radix2(bits) -> SignedDigits

Booth's original 1951 recoding: `dᵢ = b_{i-1} − bᵢ` with `b₋₁ = 0`, over the signed
binary alphabet `{-1,0,1}`.

The algebra is a one-line telescope — the recoding literally computes `2x − x` digit
by digit:

```math
\\sum_i d_i 2^i = \\sum_i b_{i-1}2^i - \\sum_i b_i 2^i = 2x - x = x
```

so a run of 1s cancels internally, leaving `+1` above its top and `-1` at its foot.
The digits are the **discrete derivative** of the bit string — nonzero only at
`0↔1` edges — which is why runs of equal bits are free.

Its failure mode is data-dependent: an alternating multiplier like `01010101` makes
every pair a boundary and recodes to *eight* nonzero digits against plain binary's
four.  That is exactly why MacSorley's 1961 modified Booth moved to radix 4.
"""
function booth_radix2(bits::AbstractVector{<:Integer})
    n = length(bits)
    d = Vector{Int}(undef, n)
    for i in 1:n
        prev = i == 1 ? 0 : bits[i-1]
        d[i] = prev - bits[i]
    end
    SignedDigits(d, 2, 1)
end

booth_radix2(B::Integer, n::Integer) = booth_radix2(twos_complement_bits(B, n))

"""
    booth_radix4(B::Integer, n::Integer) -> SignedDigits

Radix-4 (modified) Booth recoding of the `n`-bit two's-complement value `B`, over the
**minimally redundant** alphabet `{-2..2}`:

```math
d_j = b_{2j-1} + b_{2j} - 2\\,b_{2j+1}, \\qquad b_{-1} = 0
```

# Theorem 1
(i) `dⱼ ∈ {-2,…,2}` for every `j`; (ii) `Σⱼ dⱼ4ʲ = B` — *exactly*, for arbitrary
two's-complement inputs, negative values included, with **no correction step**.  That
uniform signed multiplication is what Booth's 1951 title advertised.

Three readings of the same formula, one per proof it powers:

- **It is a derivative.** As `2t_{2j+1} + t_{2j}` over the radix-2 Booth digits
  `tᵢ = b_{i-1} − bᵢ`, the window reads two edge-digits at once — modified Booth is
  original Booth read two digits at a time.
- **It is a pre-paid borrow.** As `dⱼ = (uⱼ + cⱼ) − 4c_{j+1}` with `uⱼ = 2b_{2j+1}+b_{2j}`
  and `cⱼ = b_{2j-1}`, every "carry" is an *input bit*, already written in the
  operand — nothing to propagate, hence the `O(1)` recode.
- **It is a packer.** That is the entire content of "radix 4": same telescope, half
  the rows.

```jldoctest
julia> d = booth_radix4(13, 6);        # 13 = 16 − 4 + 1

julia> d.digits
3-element Vector{Int64}:
  1
 -1
  1

julia> value(d)
13

julia> value(booth_radix4(-74, 8))     # negatives need no correction
-74
```
"""
function booth_radix4(B::Integer, n::Integer)
    iseven(n) || (n += 1)                        # sign-extension makes the width even
    bits = twos_complement_bits(B, n)
    m = n ÷ 2
    d = Vector{Int}(undef, m)
    for j in 0:m-1
        bm1 = (2j - 1) < 0 ? 0 : bits[2j]        # b_{2j-1}, 1-based
        b0 = bits[2j+1]                          # b_{2j}
        b1 = (2j + 2) <= n ? bits[2j+2] : bits[n]  # b_{2j+1}
        d[j+1] = bm1 + b0 - 2 * b1
    end
    SignedDigits(d, 4, 2)
end

"""
    booth_radix4_unsigned(B::Integer, n::Integer) -> SignedDigits

The same device run with zero-extension instead of sign-extension, which recodes
**unsigned** operands.  This is the origin of the "+1 guard digit": zero-extending a
24-bit unsigned significand to even width 26 yields 13 digits reconstructing it
exactly.

Theorem 1 is really about *bit strings with declared boundary behaviour* — `b₋₁ = 0`
always, and the top extension chosen to encode the signedness convention.  Evenness,
odd widths, and guard digits are all the one theorem wearing different boundary
conditions.

Returns exactly `⌈(n+1)/2⌉` digits so the count reflects the declared **width**, which
is what a hardware row count needs; pass `trim = true` for the shortest string that
happens to spell this particular value.

```jldoctest
julia> length(booth_radix4_unsigned(12345, 24).digits)     # FP32 significand width
13

julia> value(booth_radix4_unsigned(12345, 24))
12345
```
"""
function booth_radix4_unsigned(B::Integer, n::Integer; trim::Bool = false)
    B >= 0 || throw(ArgumentError("booth_radix4_unsigned expects B ≥ 0"))
    n = Int(n)
    B < (BigInt(1) << n) || throw(ArgumentError("$(B) does not fit in $(n) unsigned bits"))
    # An n-bit unsigned operand is zero-extended by one bit, so the window rule yields
    # ⌈(n+1)/2⌉ digits — 13 for FP32's 24-bit significand.  That extra digit is the
    # "guard digit", and it is a boundary condition, not a special case.
    m = cld(n + 1, 2)
    bits = [Int((BigInt(B) >> i) & 1) for i in 0:(2m)]      # zero-extended
    d = Vector{Int}(undef, m)
    for j in 0:m-1
        bm1 = (2j - 1) < 0 ? 0 : bits[2j]
        b0 = bits[2j+1]
        b1 = bits[2j+2]
        d[j+1] = bm1 + b0 - 2 * b1
    end
    if trim
        while length(d) > 1 && d[end] == 0
            pop!(d)
        end
    end
    SignedDigits(d, 4, 2)
end

"""
    BoothProduct

A Booth multiplication worked end to end: the recoded digits, the partial-product
rows they select, and the product.

Each digit selects a trivially generated row from `{0, ±A, ±2A}` — all free shifts and
negations.  `dⱼ = ±2` absorbs its factor of two into *one more* shift; each negative
sign is two's-complement negation, whose pending `+1` is deposited into the summation
tree as one extra input bit rather than performed as an addition.
"""
struct BoothProduct
    A::Int
    B::Int
    n::Int
    digits::SignedDigits
    rows::Vector{Int}
    product::Int
    plain_rows::Vector{Int}
end

"""
    booth_multiply(A::Integer, B::Integer, n::Integer) -> BoothProduct

Multiply `A × B` by radix-4 Booth recoding of the multiplier `B`.

```math
A \\times B = A \\times \\sum_j d_j 4^j = \\sum_j d_j\\,(A \\ll 2j)
```

The index `j` runs over `n/2` values, so at most `⌈n/2⌉` rows — against `n` for the
schoolbook method.

```jldoctest
julia> bp = booth_multiply(11, 13, 6);

julia> bp.rows, bp.product        # three rows instead of six
([11, -44, 176], 143)

julia> bp = booth_multiply(93, -74, 8);

julia> bp.rows, bp.product
([-186, 744, -1488, -5952], -6882)
```
"""
function booth_multiply(A::Integer, B::Integer, n::Integer)
    d = booth_radix4(B, n)
    rows = [d.digits[j] * (Int(A) << (2 * (j - 1))) for j in eachindex(d.digits)]
    bits = twos_complement_bits(B, iseven(n) ? n : n + 1)
    plain = [bits[i] * (Int(A) << (i - 1)) for i in eachindex(bits)]
    BoothProduct(Int(A), Int(B), Int(n), d, rows, sum(rows), plain)
end

"""
    booth_rows(n::Integer) -> Tuple{Int,Int}

Partial-product row counts, `(plain, booth)`, for an `n`-bit **unsigned** multiplier:
`(n, ⌈(n+1)/2⌉)` — the `+1` inside the ceiling is the zero-extension that the unsigned
convention requires, i.e. the guard digit.  For FP32's 24-bit significand that is
`(24, 13)`; for FP64's 53-bit significand, `(53, 27)`.

A **signed** `n`-bit multiplier needs only `n/2` rows, since two's complement needs no
extension — see [`booth_radix4`](@ref).

**Theorem 2 — why redundancy is the enabling trick, not an incidental one.**  Plain
radix-4 digits `{0,1,2,3}` would also halve the rows, but the multiple `3A = 2A + A`
requires a full carry-propagate addition *before summation begins*.  The signed set
caps the digit magnitude at `r/2 = 2` instead of `r−1 = 3`, and `{0,±A,±2A}` are all
shifts.  Generally, radix `2ᵏ` recoding needs multiples up to `2^{k-1}A`, shift-only
iff `2^{k-1} ≤ 2` — so **radix 4 is the unique sweet spot**, and radix 8 exists only
at the price of a precomputed `3A`."""
booth_rows(n::Integer) = (Int(n), cld(Int(n) + 1, 2))

"""
    verify_booth_exhaustive(n::Integer) -> Bool

Verify Theorem 1 over **all** `2ⁿ` two's-complement values of width `n`: the recoded
digits stay in `{-2..2}` and reconstruct the value exactly.

```jldoctest
julia> verify_booth_exhaustive(8)
true

julia> verify_booth_exhaustive(10)
true
```
"""
function verify_booth_exhaustive(n::Integer)
    lo, hi = -(1 << (n - 1)), (1 << (n - 1)) - 1
    for B in lo:hi
        d = booth_radix4(B, n)
        all(x -> abs(x) <= 2, d.digits) || return false
        value(d) == B || return false
    end
    true
end

"""
    booth_pairing(B::Integer, n::Integer) -> NamedTuple

The second proof of Theorem 1, digit by digit: radix-2 Booth digits `tᵢ` (the discrete
derivative of the bit string), then the shaded pairs packed into radix-4 digits
`dⱼ = 2t_{2j+1} + t_{2j}`.

Modified Booth adds no new mathematics to original Booth — it reads the same telescope
two digits at a time, halving the rows.

```jldoctest
julia> p = booth_pairing(-74, 8);

julia> p.radix4.digits == p.packed
true
```
"""
function booth_pairing(B::Integer, n::Integer)
    iseven(n) || (n += 1)
    bits = twos_complement_bits(B, n)
    t = booth_radix2(bits)
    packed = [2 * t.digits[2j+2] + t.digits[2j+1] for j in 0:(n ÷ 2)-1]
    (bits = bits, radix2 = t, packed = packed, radix4 = booth_radix4(B, n))
end

"""
    booth_borrow_form(B::Integer, n::Integer) -> Vector{NamedTuple}

The sharper range proof as a physical borrow scheme: `dⱼ = (uⱼ + cⱼ) − 4c_{j+1}`,
where `uⱼ = 2b_{2j+1} + b_{2j}` is window `j`'s own 2-bit field, `cⱼ = b_{2j-1}` its
carry-in, and `c_{j+1} = b_{2j+1}` its carry-out.

Each window is a teller holding `uⱼ` plus a `+1` handed up from below; whenever the
holdings reach 2 the teller pays a single `+1` into the next radix-4 place and is
charged `−4` locally.  The physical punchline: unlike addition's carries, which are
data-dependent signals that must ripple, **these "carries" are plain input bits** —
the operand arrives with its entire borrow schedule pre-written, which is exactly why
all `n/2` digits emerge in parallel at constant depth."""
function booth_borrow_form(B::Integer, n::Integer)
    iseven(n) || (n += 1)
    bits = twos_complement_bits(B, n)
    m = n ÷ 2
    out = NamedTuple{(:j, :u, :c_in, :c_out, :d),NTuple{5,Int}}[]
    for j in 0:m-1
        b0 = bits[2j+1]
        b1 = (2j + 2) <= n ? bits[2j+2] : bits[n]
        cin = (2j - 1) < 0 ? 0 : bits[2j]
        u = 2 * b1 + b0
        cout = b1
        push!(out, (j = j, u = u, c_in = cin, c_out = cout, d = (u + cin) - 4 * cout))
    end
    out
end

function Base.show(io::IO, ::MIME"text/plain", bp::BoothProduct)
    println(io, "─"^70)
    println(io, "  Radix-4 Booth multiplication:  ", bp.A, " × ", bp.B, " = ", bp.product,
            bp.product == bp.A * bp.B ? "  ✓" : "  ✗")
    println(io, "─"^70)
    println(io, "  multiplier bits (LSB→MSB): ",
            join(twos_complement_bits(bp.B, iseven(bp.n) ? bp.n : bp.n + 1)))
    println(io, "  Booth digits    (LSB→MSB): ", bp.digits.digits,
            "   [alphabet {-2..2}, minimally redundant]")
    println(io, "  partial products:")
    for (j, r) in enumerate(bp.rows)
        @printf(io, "    d_%d = %+2d  ⇒  %+2d × (A ≪ %d) = %+d\n",
                j - 1, bp.digits.digits[j], bp.digits.digits[j], 2(j - 1), r)
    end
    println(io, "─"^70)
    println(io, "  rows: ", length(bp.rows), " (Booth) vs ", length(bp.plain_rows), " (schoolbook)")
    print(io,   "  the row halving is what shrinks the Wallace tree")
end
