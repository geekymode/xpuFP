# ---------------------------------------------------------------------------
# Redundant radix 4.
#
# Why do ordinary carries *chain*?  Because minimal digits have no headroom: in
# radix 4 with digits 0..3, a position that receives a carry while already holding 3
# overflows and must pass the problem on — and on, and on.  RR4's digit set builds in
# spare room, and the addition rule is designed to bank that slack.
# ---------------------------------------------------------------------------

"""
    RR4Alphabet

Which radix-4 signed-digit alphabet an adder or conversion works in.

- `MAX_REDUNDANT` — `{-3..3}`, surplus `ρ = 3`.  One unit of slack on each flank,
  which is exactly the room needed to absorb a ±1 transfer unconditionally, so the
  transfer decision reads **one column only**: pure locality.
- `MIN_REDUNDANT` — `{-2..2}`, surplus `ρ = 1`.  Booth's alphabet, and the one that
  actually ships in silicon.  Zero slack, so the rule buys safety with *information*
  instead: a one-column peek at the right neighbour's sign.

Redundancy and lookahead are exchangeable currencies — turn the dial down one click
and the rule pays for it with one column of sight.

!!! note "Why not `MIN_REDUNDANT` / `MAX_REDUNDANT`"
    These name how *redundant* the alphabet is, which is a different axis from how
    *short* a representation is.  [`minimal_rr4`](@ref) finds the fewest-digit
    spelling and works in either alphabet, so the old names invited exactly the wrong
    reading.  `MIN_REDUNDANT` is the default everywhere, because it is what hardware
    uses.

One hardware coincidence seals the maximal set's popularity in textbooks:
`⌈log₂5⌉ = ⌈log₂7⌉ = 3`, so at radix 4 both alphabets occupy the same 3-bit digit
cell — the extra redundancy is free at the storage level.
"""
@enum RR4Alphabet MAX_REDUNDANT MIN_REDUNDANT

alphabet_halfwidth(a::RR4Alphabet) = a == MAX_REDUNDANT ? 3 : 2

"""
    RR4Column

One column of a carry-free addition, with every intermediate exposed.

# Fields
- `i::Int` — position (0 = least significant).
- `x`, `y::Int` — the input digits.
- `s::Int` — the raw column sum `xᵢ + yᵢ`.
- `t_out::Int` — the transfer this column *ships* to position `i+1`.
- `t_in::Int` — the transfer *received* from position `i−1`.
- `w::Int` — the interim digit kept locally, `s − 4·t_out`.
- `z::Int` — the final digit, `w + t_in`.
- `peeked::Bool` — whether the transfer decision consulted the right neighbour
  (`MIN_REDUNDANT` alphabet, borderline sum `|s| = 2`).
"""
struct RR4Column
    i::Int
    x::Int
    y::Int
    s::Int
    t_out::Int
    t_in::Int
    w::Int
    z::Int
    peeked::Bool
end

"""
    RR4AddTrace

The complete record of one carry-free addition: every column, the result digits, and
the verification that value was conserved.

The `depth` field is the point of the whole exercise — it is **3 at any word length**
(sums, then transfers, then digits), against one step per position for a ripple adder.
"""
struct RR4AddTrace
    alphabet::RR4Alphabet
    x::SignedDigits
    y::SignedDigits
    columns::Vector{RR4Column}
    result::SignedDigits
    carry_out::Int
    depth::Int
    exact::Bool
end

# ---------------------------------------------------------------------------
# Conversion INTO redundant radix 4, from anything.
# ---------------------------------------------------------------------------

"""
    dyadic_parts(x; fracdigits=nothing) -> (N::BigInt, exponent::Int)

Decompose `x` into a scaled integer, `x ≈ N · 4^exponent`, exactly where possible.

Integers are trivially exact.  Every `Float64` is a dyadic rational `m·2^k`, and since
`4 = 2²` every dyadic rational has a **finite** radix-4 expansion — so floats convert
exactly, without rounding.  A `Rational` whose denominator is not a power of two (say
`1//3`) has no finite expansion in any power-of-two radix and must be rounded; pass
`fracdigits` to say how far.

Passing `fracdigits` always forces rounding to that many radix-4 fraction digits, even
when an exact expansion exists.

```jldoctest
julia> dyadic_parts(49)
(49, 0)

julia> dyadic_parts(49.25)          # 49.25 = 197/4, exact in one fraction digit
(197, -1)

julia> dyadic_parts(1//3; fracdigits = 3)
(21, -3)
```
"""
dyadic_parts(x; fracdigits = nothing) = radix_parts(x, 4; fracdigits)

# integer → signed radix-4 digits inside an alphabet of half-width `a`
function _signed_radix_digits(N::Integer, radix::Integer, a::Integer)
    v = BigInt(N)
    neg = v < 0
    ds = Int[]
    u = abs(v)
    while u > 0
        push!(ds, Int(u % radix)); u ÷= radix
    end
    isempty(ds) && push!(ds, 0)
    neg && (ds .= .-ds)
    # one bounded pass: rewrite any |d| > a using the identity r = (1,0)
    carry = 0
    out = Int[]
    for d in ds
        w = d + carry
        carry = 0
        while w > a
            w -= radix; carry += 1
        end
        while w < -a
            w += radix; carry -= 1
        end
        push!(out, w)
    end
    while carry != 0
        w = carry; carry = 0
        while w > a
            w -= radix; carry += 1
        end
        while w < -a
            w += radix; carry -= 1
        end
        push!(out, w)
    end
    out
end

"""
    to_rr4(x; alphabet=MIN_REDUNDANT, fracdigits=nothing, ndigits=nothing, exponent=nothing)

Convert **anything** into a legal redundant-radix-4 digit string.

Accepts `Integer`, `AbstractFloat`, `Rational`, and `SignedDigits` (which re-spells an
existing string into the target alphabet, radix 4 or otherwise).  Floats convert
*exactly* — every `Float64` is a dyadic rational and therefore has a finite radix-4
expansion — so no precision is silently invented.

The default alphabet is `MIN_REDUNDANT` (`{-2..2}`), the one hardware actually uses.
For the fully local carry-free adder's `{-3..3}` use [`to_rr4_maximal`](@ref), which is
named rather than flagged so the two can never be confused.

# Keywords
- `alphabet` — target alphabet.
- `fracdigits` — round to this many radix-4 fraction digits.  Required for rationals
  with no finite expansion (`1//3`); optional elsewhere.
- `ndigits` — pad (or demand) exactly this many digits; see [`rr4_with_length`](@ref).
- `exponent` — force this radix-4 exponent instead of the natural one.

Because the system is redundant this returns *a* spelling, not *the* spelling.  The rule
is a single borrow pass over the non-redundant `{0..3}` digits: cheap, deterministic and
reproducible, but **not minimum weight** — it exceeds the minimum for about 28% of
values.  For the cheapest spelling use [`canonical_rr4`](@ref); for the whole set see
[`all_rr4_representations`](@ref); for the shortest, [`minimal_rr4`](@ref).

```jldoctest
julia> value(to_rr4(49))
49

julia> value(to_rr4(49.25))          # exact: 49.25 = 197/4
197//4

julia> digit_string(to_rr4(49.25))
"301.1"

julia> value(to_rr4(-7//8))
-7//8

julia> value(to_rr4(1//3; fracdigits = 6))   # no finite expansion — rounded
1365//4096
```
"""
function to_rr4(x; alphabet::RR4Alphabet = MIN_REDUNDANT, fracdigits = nothing,
                ndigits = nothing, exponent = nothing)
    a = alphabet_halfwidth(alphabet)
    N, e4 = dyadic_parts(x; fracdigits)
    if exponent !== nothing
        shift = e4 - Int(exponent)
        shift >= 0 || throw(ArgumentError(
            "exponent $(exponent) is finer than the value supports (natural exponent $(e4)); " *
            "pass fracdigits to add precision first"))
        N *= BigInt(4)^shift
        e4 = Int(exponent)
    end
    ds = _signed_radix_digits(N, 4, a)
    if ndigits !== nothing
        n = Int(ndigits)
        length(ds) <= n || throw(ArgumentError(
            "$(x) needs $(length(ds)) digits in {-$(a)..$(a)} at exponent $(e4); " *
            "cannot fit in $(n)"))
        while length(ds) < n
            push!(ds, 0)
        end
    end
    SignedDigits(ds, 4, a, e4)
end

function to_rr4(sd::SignedDigits; alphabet::RR4Alphabet = MIN_REDUNDANT, kwargs...)
    to_rr4(value(sd); alphabet, kwargs...)
end

"""
    to_rr4_parts(x; alphabet=MIN_REDUNDANT, kwargs...) -> NamedTuple

Convert to RR4 and return the **unpacked pieces** rather than the container: the digit
array and the integer scale, plus the scaled integer they multiply out to.

Same keywords as [`to_rr4`](@ref).  Equivalent to `digit_parts(to_rr4(x; ...))`, and
provided because the digit vector and the exponent are usually what you actually want
to index into or feed to a circuit model.

```jldoctest
julia> p = to_rr4_parts(49);

julia> p.digits, p.scale
([1, 0, -1, 1], 0)

julia> p.scaled_integer, p.value
(49, 49)

julia> q = to_rr4_parts(49.25);

julia> q.digits, q.scale, q.scaled_integer      # 197 × 4^-1 = 49.25
([1, 1, 0, -1, 1], -1, 197)

julia> q.value
197//4
```
"""
to_rr4_parts(x; kwargs...) = digit_parts(to_rr4(x; kwargs...))

"""
    to_digits_parts(f::DigitFormat, x; kwargs...) -> NamedTuple

The [`to_rr4_parts`](@ref) equivalent for any [`DigitFormat`](@ref).

```jldoctest
julia> to_digits_parts(CSD, 231).digits
9-element Vector{Int64}:
 -1
  0
  0
  1
  0
 -1
  0
  0
  1
```
"""
to_digits_parts(f::DigitFormat, x; kwargs...) = digit_parts(to_digits(f, x; kwargs...))

"""
    to_rr4_minimal(x; kwargs...) -> SignedDigits

Convert into the **minimally redundant** `{-2..2}` alphabet — Booth's alphabet, and
what every shipping multiplier and SRT divider recodes into.  Same as the `to_rr4`
default, spelled out."""
to_rr4_minimal(x; kwargs...) = to_rr4(x; alphabet = MIN_REDUNDANT, kwargs...)

"""
    to_rr4_maximal(x; kwargs...) -> SignedDigits

Convert into the **maximally redundant** `{-3..3}` alphabet — the one whose extra
slack permits the fully local, one-column carry-free addition rule.

Deliberately a separate function rather than a keyword: "maximal alphabet" and
"minimal *length*" are different axes, and conflating them is the easiest mistake to
make in this corner of the subject.

```jldoctest
julia> sd = to_rr4_maximal(49);

julia> sd.maxdigit, value(sd)
(3, 49)
```
"""
to_rr4_maximal(x; kwargs...) = to_rr4(x; alphabet = MAX_REDUNDANT, kwargs...)

# ---------------------------------------------------------------------------
# Enumerating and reshaping representations.
# ---------------------------------------------------------------------------

"""
    max_representable(ndigits, alphabet) -> BigInt

The largest magnitude an `ndigits`-digit string over the alphabet can spell,
`a·(4ⁿ − 1)/3`."""
function max_representable(ndigits::Integer, alphabet::RR4Alphabet = MIN_REDUNDANT)
    a = alphabet_halfwidth(alphabet)
    BigInt(a) * (BigInt(4)^Int(ndigits) - 1) ÷ 3
end

"""
    min_ndigits(x; alphabet=MIN_REDUNDANT, fracdigits=nothing) -> Int

The fewest digits any spelling of `x` can use in this alphabet.

More redundancy means a larger per-digit range, so the maximally redundant alphabet can
sometimes spell a value one digit shorter than the minimally redundant one."""
function min_ndigits(x; alphabet::RR4Alphabet = MIN_REDUNDANT, method = nothing,
                     fracdigits = nothing)
    alphabet = method === nothing ? alphabet : _method_alphabet(method)
    N, _ = dyadic_parts(x; fracdigits)
    N == 0 && return 1
    n = 1
    while max_representable(n, alphabet) < abs(N)
        n += 1
    end
    n
end

"""
    minimal_rr4(x; alphabet=MIN_REDUNDANT, fracdigits=nothing, by=:weight) -> SignedDigits

The **shortest** spelling of `x`: the fewest digits the alphabet allows.

Among the shortest, ties are broken by `by`:
- `:weight` (default) — fewest nonzero digits, i.e. cheapest as a constant multiplier;
- `:onepass` — whatever the one-pass conversion emits, no search.  (Note this is *not*
  a canonical form in the CSD sense; see [`canonical_rr4`](@ref).)

Note this is minimal *length*, an entirely different axis from the minimally redundant
*alphabet*; both are available independently.

```jldoctest
julia> sd = minimal_rr4(49);

julia> length(sd), value(sd)
(3, 49)

julia> weight(minimal_rr4(63)) <= weight(to_rr4(63))
true
```
"""
function minimal_rr4(x; alphabet::RR4Alphabet = MIN_REDUNDANT, fracdigits = nothing,
                     by::Symbol = :weight)
    n = min_ndigits(x; alphabet, fracdigits)
    (by === :onepass || by === :canonical) &&
        return to_rr4(x; alphabet, fracdigits, ndigits = n)
    by === :weight || throw(ArgumentError("by must be :weight or :onepass"))
    reps = all_rr4_representations(x, n; alphabet, fracdigits)
    isempty(reps) && return to_rr4(x; alphabet, fracdigits, ndigits = n)
    argmin(weight, reps)
end

"""
    all_rr4_representations(x, ndigits=min_ndigits(x); alphabet=MIN_REDUNDANT,
                            fracdigits=nothing, exponent=nothing) -> Vector{SignedDigits}

**Every** `ndigits`-digit spelling of `x` in the given alphabet, sorted by nonzero
count.

This is the redundancy made concrete: the count is the spelling freedom the local
addition rule spends, and it grows fast with both the alphabet and the length.

!!! note "Why a width is needed at all"
    The set of spellings is only finite once the digit count is fixed — a value can
    always be padded with leading zeros, or re-spelled longer. Omitting `ndigits`
    defaults to [`min_ndigits`](@ref), the *shortest* width the alphabet allows, which
    is usually a small set; widen it to see the freedom grow (see
    [`rr4_representation_counts`](@ref)).

```jldoctest
julia> length(all_rr4_representations(10, 4))                        # {-2..2}
3

julia> length(all_rr4_representations(10, 4; alphabet = MAX_REDUNDANT))
6

julia> value(first(all_rr4_representations(49, 4)))
49
```
"""
function all_rr4_representations(x, ndigits::Union{Integer,Nothing} = nothing;
                                 alphabet::RR4Alphabet = MIN_REDUNDANT,
                                 method = nothing,
                                 fracdigits = nothing, exponent = nothing,
                                 limit::Union{Integer,Nothing} = nothing)
    alphabet = method === nothing ? alphabet : _method_alphabet(method)
    a = alphabet_halfwidth(alphabet)
    N, e4 = dyadic_parts(x; fracdigits)
    ndigits === nothing && (ndigits = min_ndigits(x; alphabet, fracdigits))
    if exponent !== nothing
        shift = e4 - Int(exponent)
        shift >= 0 || throw(ArgumentError("exponent $(exponent) is finer than the value supports"))
        N *= BigInt(4)^shift
        e4 = Int(exponent)
    end
    n = Int(ndigits)
    out = SignedDigits[]
    dig = zeros(Int, n)
    R = BigInt(4)
    # Work from the LEAST significant digit, carrying the remainder in units of the
    # current position's weight.  Two prunes make this tractable:
    #   (1) divisibility — position `pos` can only hold a digit `d ≡ rem (mod 4)`,
    #       which leaves at most two candidates out of the alphabet, not all of it;
    #   (2) reach — the positions still to come can represent at most
    #       a·(4^(remaining) − 1)/3 in these units.
    # Without (1) the search is 5^n and a 27-digit value (any Float64) never returns.
    cap = limit === nothing ? typemax(Int) : Int(limit)
    function rec(pos::Int, rem::BigInt)
        length(out) >= cap && return
        if pos > n
            rem == 0 && push!(out, SignedDigits(copy(dig), 4, a, e4))
            return
        end
        reach = BigInt(a) * (R^(n - pos + 1) - 1) ÷ 3
        abs(rem) > reach && return
        for d in -a:a
            r2 = rem - d
            mod(r2, R) == 0 || continue
            dig[pos] = d
            rec(pos + 1, r2 ÷ R)
        end
        dig[pos] = 0
    end
    rec(1, N)
    sort!(out; by = weight)
    out
end

"""
    RR4Complexity

A cost preview for enumerating a value's redundant spellings: how many there are, how
big the naive search would have been, and whether listing them is a good idea.

The representation count is **exact**, not an estimate — it comes from a dynamic
program over the carry that costs well under a millisecond even at 600 digits, so it is
always worth computing before deciding to enumerate.

# Fields
`value`, `method`, `maxdigit`, `ndigits`, `scale`, `total`, `naive_search`,
`pruned_search`, `bytes`, `listable`, `limit`, `seconds`.
"""
struct RR4Complexity
    value::Any
    method::Symbol
    maxdigit::Int
    ndigits::Int
    scale::Int
    total::BigInt
    naive_search::BigInt
    pruned_search::BigInt
    bytes::BigInt
    listable::Bool
    limit::Int
    seconds::Float64
end

"""
    rr4_complexity(x; method=:minimally_redundant, ndigits=nothing, fracdigits=nothing,
                   limit=10_000) -> RR4Complexity

Count a value's redundant spellings and report what enumerating them would cost, without
enumerating anything.

Print it for a readable summary, or read the fields.  Worth calling before
[`rr4_representations`](@ref) on any value you have not seen before — every `Float64`
needs about 27 radix-4 digits, and at that width the spelling count ranges from 1 to
hundreds of thousands depending on the value.

```jldoctest
julia> c = rr4_complexity(6.3);

julia> c.ndigits, c.total, c.listable
(27, 2, true)

julia> c = rr4_complexity(0.1);

julia> c.total, c.listable
(317811, false)
```
"""
function rr4_complexity(x; method = :minimally_redundant, ndigits = nothing,
                        fracdigits = nothing, limit::Integer = 10_000)
    alp = _method_alphabet(method)
    a = alphabet_halfwidth(alp)
    n = ndigits === nothing ? min_ndigits(x; alphabet = alp, fracdigits) : Int(ndigits)
    _, e = dyadic_parts(x; fracdigits)
    t = @elapsed total = BigInt(count_rr4_representations(x, n; alphabet = alp, fracdigits))
    naive = BigInt(2a + 1)^n
    pruned = BigInt(2)^n                     # divisibility leaves ≤2 digits per position
    bytes = total * (8 * n + 64)             # a Vector{Int} per spelling, plus overhead
    RR4Complexity(x, _alphabet_method(alp), a, n, e, total, naive, pruned, bytes,
                  total <= limit, Int(limit), t)
end

_si(n::BigInt) = n < 1_000_000 ? string(n) : Printf.@sprintf("%.3g", Float64(n))
_bytes_h(b::BigInt) = b < 1024 ? "$(b) B" :
                      b < 1024^2 ? Printf.@sprintf("%.1f KB", Float64(b) / 1024) :
                      b < 1024^3 ? Printf.@sprintf("%.1f MB", Float64(b) / 1024^2) :
                                   Printf.@sprintf("%.2f GB", Float64(b) / 1024^3)

function Base.show(io::IO, ::MIME"text/plain", c::RR4Complexity)
    println(io, "RR4 representation complexity for ", c.value)
    println(io, "  method          : ", c.method, "   alphabet {-", c.maxdigit, "…", c.maxdigit, "}")
    println(io, "  width           : ", c.ndigits, " digits, scale exponent ", c.scale)
    println(io, "  representations : ", c.total, "   (exact, by dynamic programming in ",
            Printf.@sprintf("%.1f", c.seconds * 1e3), " ms)")
    println(io, "  naive search    : ", 2c.maxdigit + 1, "^", c.ndigits, " ≈ ", _si(c.naive_search),
            " digit strings")
    println(io, "  pruned search   : ≈ 2^", c.ndigits, " ≈ ", _si(c.pruned_search),
            "   (only digits d ≡ rem mod 4 are viable, ≤2 per position)")
    println(io, "  listing cost    : ", _bytes_h(c.bytes), " for all ", c.total, " spellings")
    if c.listable
        print(io, "  verdict         : fine — below the limit of ", c.limit,
              "; rr4_representations will list them all")
    else
        println(io, "  verdict         : TOO MANY — above the limit of ", c.limit)
        print(io, "  advice          : use count_rr4_representations(x) for the number, ",
              "or narrow with ndigits=…, or raise limit= deliberately")
    end
end

"""
    RR4Representations

Every spelling of one value in one redundant radix-4 system, in tabular form.

# Fields
- `value` — the exact value all rows denote.
- `method::Symbol` — `:minimally_redundant` or `:maximally_redundant`.
- `alphabet::RR4Alphabet`, `maxdigit::Int`
- `ndigits::Int` — digits per row.
- `scale::Int` — the shared radix-4 exponent: every row means
  `Σ dᵢ·4^(i-1+scale)`.
- `digits::Matrix{Int}` — one **row per representation**, columns most significant
  first (so a row reads left to right exactly as the number prints).
- `reps::Vector{SignedDigits}` — the same spellings as objects, sorted by nonzero count.

Display it for the table; hand it to [`plot_rr4_representations`](@ref) for the colour
map.
"""
struct RR4Representations
    value::Any
    method::Symbol
    alphabet::RR4Alphabet
    maxdigit::Int
    ndigits::Int
    scale::Int
    digits::Matrix{Int}
    reps::Vector{SignedDigits}
    total::Int
    truncated::Bool
end

_method_alphabet(a::RR4Alphabet) = a
function _method_alphabet(m::Symbol)
    m in (:minimally_redundant, :min_redundant, :min, :minimal) && return MIN_REDUNDANT
    m in (:maximally_redundant, :max_redundant, :max, :maximal) && return MAX_REDUNDANT
    throw(ArgumentError("method must be :minimally_redundant or :maximally_redundant, got :$(m)"))
end
_alphabet_method(a::RR4Alphabet) =
    a == MIN_REDUNDANT ? :minimally_redundant : :maximally_redundant

"""
    rr4_representations(x; method=:minimally_redundant, ndigits=nothing,
                        fracdigits=nothing, exponent=nothing) -> RR4Representations

List **every** redundant radix-4 spelling of `x`, returning the digits as a matrix and
the shared scaling exponent.

`method` selects the digit alphabet:

| `method` | alphabet | note |
|:---|:---|:---|
| `:minimally_redundant` (default) | `{-2..2}` | Booth's alphabet; what ships in silicon |
| `:maximally_redundant` | `{-3..3}` | the slack that permits fully local carry-free addition |

`ndigits` defaults to [`min_ndigits`](@ref) — the shortest width the alphabet allows.
Widen it to see the spelling freedom grow, since the set is only finite once the width
is fixed.

The true total is computed first by a cheap dynamic program, so it is always reported
even when only `limit` rows (default 10 000) are materialised; `truncated` says which
happened.  Any `Float64` needs ~27 radix-4 digits, and at that width some values have
very many spellings — [`count_rr4_representations`](@ref) answers "how many" without
building them.

```jldoctest
julia> r = rr4_representations(12.5);

julia> r.ndigits, r.scale, size(r.digits)
(4, -1, (2, 4))

julia> r.digits                      # one row per spelling, MSB first
2×4 Matrix{Int64}:
 1  -1  0   2
 1  -1  1  -2

julia> r.value
25//2

julia> rr4_representations(25; method = :maximally_redundant).digits |> size
(4, 3)
```

Each row means `Σᵢ dᵢ·4^(i-1+scale)` once read back least-significant-first; the
[`digit_matrix_with_scale`](@ref) helper appends the scale as a final column, which is
what [`plot_rr4_representations`](@ref) draws.
"""
function rr4_representations(x; method = :minimally_redundant, ndigits = nothing,
                             fracdigits = nothing, exponent = nothing,
                             limit::Integer = 10_000, verbose::Bool = true)
    alp = _method_alphabet(method)
    n = ndigits === nothing ? min_ndigits(x; alphabet = alp, fracdigits) : Int(ndigits)
    # counting is a cheap DP, so learn the true total BEFORE materialising anything
    total = count_rr4_representations(x, n; alphabet = alp, fracdigits)
    if verbose && total > limit
        @warn "rr4_representations: $(total) spellings exist at $(n) digits; " *
              "listing the first $(limit). Use count_rr4_representations(x) for just " *
              "the number, rr4_complexity(x) for a cost preview, or raise limit=." total ndigits=n limit
    elseif verbose && total > 1_000
        @info "rr4_representations: $(total) spellings at $(n) digits — " *
              "listing all of them (≈$(_bytes_h(BigInt(total) * (8n + 64))))."
    end
    reps = all_rr4_representations(x, n; alphabet = alp, fracdigits, exponent,
                                   limit = min(total, Int(limit)))
    e = isempty(reps) ? (exponent === nothing ? dyadic_parts(x; fracdigits)[2] : Int(exponent)) :
                        reps[1].exponent
    M = Matrix{Int}(undef, length(reps), n)
    for (i, r) in enumerate(reps)
        M[i, :] = reverse(r.digits)          # MSB first, so rows read like the number
    end
    RR4Representations(isempty(reps) ? (x isa Integer ? x : Rational{BigInt}(x)) : value(reps[1]),
                       _alphabet_method(alp), alp, alphabet_halfwidth(alp), n, e, M, reps,
                       total, total > length(reps))
end

"""
    digit_matrix(r::RR4Representations) -> Matrix{Int}

The digits alone, one row per spelling, most significant first."""
digit_matrix(r::RR4Representations) = copy(r.digits)

"""
    digit_matrix_with_scale(r::RR4Representations) -> Matrix{Int}

The digits with the **scaling exponent appended as a final column** — the layout
[`plot_rr4_representations`](@ref) renders, where the last cell of each row is the
scale rather than a digit.

```jldoctest
julia> digit_matrix_with_scale(rr4_representations(12.5))
2×5 Matrix{Int64}:
 1  -1  0   2  -1
 1  -1  1  -2  -1
```
"""
digit_matrix_with_scale(r::RR4Representations) =
    hcat(r.digits, fill(r.scale, Base.size(r.digits, 1)))

Base.length(r::RR4Representations) = length(r.reps)
Base.isempty(r::RR4Representations) = isempty(r.reps)

function Base.show(io::IO, ::MIME"text/plain", r::RR4Representations)
    println(io, "RR4Representations of ", r.value, "   (", r.method,
            ", alphabet {-", r.maxdigit, "…", r.maxdigit, "})")
    println(io, "  ", r.total, " spelling", r.total == 1 ? "" : "s",
            r.truncated ? " ($(length(r.reps)) listed — raise `limit` for more)" : "",
            " at ", r.ndigits, " digits, shared scale exponent ", r.scale,
            "   ⇒ value = Σ dᵢ·4^(i-1", r.scale == 0 ? "" :
            (r.scale > 0 ? "+$(r.scale)" : "$(r.scale)"), ")")
    isempty(r.reps) && return print(io, "  (no spelling fits in ", r.ndigits, " digits)")
    nf = max(0, -r.scale)
    pointat = r.ndigits - nf
    header = IOBuffer()
    print(header, "     ")
    for k in 1:r.ndigits
        (nf > 0 && pointat > 0 && k == pointat + 1) && print(header, "  ")
        print(header, lpad("4^$(r.scale + r.ndigits - k)", 6))
    end
    print(header, "  |  scale   weight")
    println(io, String(take!(header)))
    for i in axes(r.digits, 1)
        print(io, "  ", lpad(i, 2), " ")
        for k in 1:r.ndigits
            (nf > 0 && pointat > 0 && k == pointat + 1) && print(io, " ·")
            print(io, lpad(r.digits[i, k], 6))
        end
        @printf(io, "  |  %5d   %6d\n", r.scale, weight(r.reps[i]))
    end
    print(io, "  rows sorted by nonzero count; row 1 is the cheapest as a constant multiplier")
end

"""
    count_rr4_representations(x, ndigits; alphabet=MIN_REDUNDANT, fracdigits=nothing) -> Int

How many `ndigits`-digit spellings of `x` exist, without materialising them.  Counted
by dynamic programming over the carry, so it stays fast where enumeration would not."""
function count_rr4_representations(x, ndigits::Union{Integer,Nothing} = nothing;
                                   alphabet::RR4Alphabet = MIN_REDUNDANT,
                                   method = nothing, fracdigits = nothing)
    alphabet = method === nothing ? alphabet : _method_alphabet(method)
    a = alphabet_halfwidth(alphabet)
    N, _ = dyadic_parts(x; fracdigits)
    ndigits === nothing && (ndigits = min_ndigits(x; alphabet, fracdigits))
    n = Int(ndigits)
    counts = Dict{BigInt,Int}(N => 1)
    for _ in 1:n
        nxt = Dict{BigInt,Int}()
        for (rem, c) in counts, d in -a:a
            t = rem - d
            iszero(mod(t, 4)) || continue
            k = t ÷ 4
            nxt[k] = get(nxt, k, 0) + c
        end
        counts = nxt
    end
    get(counts, BigInt(0), 0)
end

"""
    rr4_representation_counts(x; method=:minimally_redundant, ndigits=nothing,
                              fracdigits=nothing) -> Int
    rr4_representation_counts(x, widths; ...) -> Vector{Pair{Int,Int}}

**How many** ways `x` can be spelled in redundant radix 4, for a fixed method and a
fixed scale.

Called with just a value, it answers the obvious question with a single number, using
the default alphabet and the shortest width that fits ([`min_ndigits`](@ref)):

```jldoctest
julia> rr4_representation_counts(6)
2

julia> rr4_representation_counts(10)
1

julia> rr4_representation_counts(10; method = :maximally_redundant)
2

julia> rr4_representation_counts(25; method = :maximally_redundant)
4
```

!!! note "The comparison across methods is not like-for-like"
    Each method is counted at *its own* minimum width, and the wider alphabet often
    needs fewer digits — 49 takes 4 digits in `{-2..2}` but only 3 in `{-3..3}`. Pass an
    explicit `ndigits` to compare the alphabets at a fixed width.

Called with a range of `widths`, it sweeps them and returns `width => count` pairs —
the spelling freedom as a function of how much room you give it. A width below
[`min_ndigits`](@ref) yields `0`, since the value does not fit.

```jldoctest
julia> rr4_representation_counts(25, 3:6)
4-element Vector{Pair{Int64, Int64}}:
 3 => 2
 4 => 3
 5 => 3
 6 => 3

julia> rr4_representation_counts(25, 3:6; method = :maximally_redundant)
4-element Vector{Pair{Int64, Int64}}:
 3 => 4
 4 => 8
 5 => 12
 6 => 16

julia> rr4_representation_counts(49, 3:5)      # 49 needs 4 digits in {-2..2}
3-element Vector{Pair{Int64, Int64}}:
 3 => 0
 4 => 1
 5 => 1
```

Counting is a dynamic program over the carry and costs well under a millisecond even at
600 digits, so it is always cheaper than enumerating — see [`rr4_complexity`](@ref) for
a full cost preview and [`rr4_representations`](@ref) to actually list them.
"""
rr4_representation_counts(x; method = :minimally_redundant, ndigits = nothing,
                          fracdigits = nothing) =
    count_rr4_representations(x, ndigits; method, fracdigits)

rr4_representation_counts(x, widths; method = :minimally_redundant,
                          fracdigits = nothing) =
    [Int(n) => count_rr4_representations(x, n; method, fracdigits) for n in widths]

"""
    rr4_branch_residues(method=:minimally_redundant) -> Vector{Pair{Int,Vector{Int}}}

Which running remainders admit more than one digit — the sole source of spelling
freedom, and therefore the whole explanation of how the representation counts behave.

Working from the least significant end, position `k` can only hold a digit
`d ≡ rem (mod 4)`.  In the **minimally redundant** alphabet exactly one residue offers a
choice:

| `rem mod 4` | viable digits | choices |
|---:|:---|---:|
| 0 | `0` | 1 |
| 1 | `1` | 1 |
| 2 | `-2, 2` | **2** |
| 3 | `-1` | 1 |

The **maximally redundant** alphabet branches at three residues out of four, which is
why its counts grow so much faster.

```jldoctest
julia> rr4_branch_residues()
4-element Vector{Pair{Int64, Vector{Int64}}}:
 0 => [0]
 1 => [1]
 2 => [-2, 2]
 3 => [-1]
```
"""
function rr4_branch_residues(method = :minimally_redundant)
    a = alphabet_halfwidth(_method_alphabet(method))
    [res => [d for d in -a:a if mod(d - res, 4) == 0] for res in 0:3]
end

"""
    rr4_max_count(ndigits; method=:minimally_redundant) -> BigInt

The **largest** number of spellings any value can have at this width — a closed form,
and a sharp bound.

- Minimally redundant `{-2..2}`: exactly the Fibonacci number `F(n+1)`, so counts grow
  like `φⁿ` with `φ ≈ 1.618`.  Only one residue in four branches, and branches
  recombine, which is precisely what produces a Fibonacci recurrence rather than `2ⁿ`.
- Maximally redundant `{-3..3}`: exactly `2^(n-1)`.

Verified against exhaustive search over all values for `n ≤ 11`.

```jldoctest
julia> [Int(rr4_max_count(n)) for n in 1:8]
8-element Vector{Int64}:
  1
  2
  3
  5
  8
 13
 21
 34

julia> [Int(rr4_max_count(n; method = :maximally_redundant)) for n in 1:8]
8-element Vector{Int64}:
   1
   2
   4
   8
  16
  32
  64
 128

julia> rr4_max_count(27) == count_rr4_representations(0.1, 27)   # 0.1 attains it
true
```
"""
function rr4_max_count(ndigits::Integer; method = :minimally_redundant)
    n = Int(ndigits)
    n <= 0 && return big(1)
    if _method_alphabet(method) == MAX_REDUNDANT
        return big(2)^(n - 1)
    end
    a, b = big(1), big(1)                 # F(1), F(2)
    for _ in 3:(n + 1)
        a, b = b, a + b
    end
    n + 1 <= 2 ? big(1) : b
end

"""
    rr4_with_length(x, ndigits; alphabet=MIN_REDUNDANT, exponent=nothing,
                    fracdigits=nothing) -> SignedDigits

Spell `x` using **exactly** `ndigits` digits, at an optionally forced `exponent`.

Padding above the minimum is free — leading zeros cost nothing and change no value —
which is how a redundant datapath keeps every operand the same width regardless of
magnitude.  Asking for fewer digits than the alphabet can carry raises an error
naming the shortfall.

```jldoctest
julia> sd = rr4_with_length(49, 6);

julia> length(sd), value(sd), digit_string(sd)
(6, 49, "000301")

julia> sd = rr4_with_length(49, 6; exponent = -2);   # two fraction digits

julia> value(sd), digit_string(sd)
(49, "0301.00")
```
"""
rr4_with_length(x, ndigits::Integer; alphabet::RR4Alphabet = MIN_REDUNDANT,
                exponent = nothing, fracdigits = nothing) =
    to_rr4(x; alphabet, exponent, fracdigits, ndigits = ndigits)

"""
    rr4_rescale(sd::SignedDigits, exponent::Integer) -> SignedDigits

Re-express a digit string at a different (finer or equal) exponent, padding with
low-order zeros.  The value is invariant; only the alignment changes.

Used internally by [`rr4_add`](@ref) to line up operands before the column-wise add."""
function rr4_rescale(sd::SignedDigits, exponent::Integer)
    e = Int(exponent)
    e <= sd.exponent || throw(ArgumentError(
        "cannot rescale from exponent $(sd.exponent) up to $(e) without discarding digits"))
    shift = sd.exponent - e
    SignedDigits(vcat(zeros(Int, shift), sd.digits), sd.radix, sd.maxdigit, e)
end

# --- the maximal-set transfer rule -----------------------------------------

"""
    rr4_transfer(s::Integer) -> Int

The maximal-set transfer rule: ship ±1 when `|s| ≥ 3`, otherwise nothing.

Equivalently — and this is the slicker route — `t = round(s/4)` with ties toward
zero, so the interim digit `u = s − 4t` is nothing but a **rounding remainder**: the
signed distance from `s` to the nearest multiple of 4.  Any real number lies within
half a grid step of the nearest grid point, so `|u| ≤ ½·4 = 2` in one line — the very
same half-interval bound that governs round-to-nearest floating point everywhere else,
wearing integer clothes.

The threshold is load-bearing: waiting for `|s| ≥ 4` would permit `u = 3`, an incoming
`+1` would make `w = 4` — illegal — forcing a fresh transfer that *depends on* the
incoming one.  A chain, the very disease being cured."""
rr4_transfer(s::Integer) = s >= 3 ? 1 : (s <= -3 ? -1 : 0)

"""
    rr4_split_table() -> Vector{NamedTuple}

The complete split table: all thirteen possible column sums `s ∈ [-6,6]`, each with
its transfer `t` and interim digit `u = s − 4t`.  Every `u` lands in `[-2,2]`."""
rr4_split_table() = [(s = s, t = rr4_transfer(s), u = s - 4 * rr4_transfer(s)) for s in -6:6]

# --- the minimal-set transfer rule (one-column peek) -----------------------

"""
    rr4_transfer_minimal(s::Integer, s_right::Integer) -> Int

The minimal-set (`{-2..2}`) transfer rule.  The maximal formulation provably cannot
survive here — with `a = 2`, absorbing ±1 would demand `|u| ≤ 1`, yet the column sum
`s = 2` has no compliant split.  What replaces it is a rule with a **one-column peek**:

```math
t_{i+1} = \\begin{cases}
0 & |s_i| \\le 1\\\\
\\pm1 & |s_i| \\ge 3 \\ (\\text{sign of } s_i)\\\\
1 \\text{ if } s_{i-1} \\ge 0,\\ \\text{else } 0 & s_i = 2\\\\
-1 \\text{ if } s_{i-1} \\le 0,\\ \\text{else } 0 & s_i = -2
\\end{cases}
```

The borderline sums `±2` — exactly the cases that broke pure locality — are resolved
by the neighbour's sign, which *determines the sign of the incoming transfer before it
arrives*.

The relationship between the two transfers is not a message but a **common cause**:
both `t_{i+1} = f(sᵢ, s_{i-1})` and `tᵢ = f(s_{i-1}, s_{i-2})` read the shared column
`s_{i-1}`, and the peek was chosen so that exactly when the interim is pushed to its
rim, the sign of that shared column forbids the incoming transfer from pushing it
over."""
function rr4_transfer_minimal(s::Integer, s_right::Integer)
    abs(s) <= 1 && return 0
    s >= 3 && return 1
    s <= -3 && return -1
    s == 2 && return s_right >= 0 ? 1 : 0
    s == -2 && return s_right <= 0 ? -1 : 0
    return 0
end

"""
    rr4_add(x, y; alphabet=MAX_REDUNDANT) -> RR4AddTrace

Carry-free addition in redundant radix 4.

Every position forms its digit sum independently; sums past the threshold shed a
transfer of ±1 to the next column and keep an interim digit in `[-2,2]`, so absorbing
an incoming transfer can never push a digit out of range — **no chain can form**.
Addition depth is `O(1)` in the word length, against `Θ(log n)` for any
non-redundant representation.

Accepts `SignedDigits` or plain integers; integers are converted with
[`to_rr4`](@ref) first.

```jldoctest
julia> tr = rr4_add(6, 43);          # the report's worked example

julia> value(tr.result)
49

julia> tr.depth                       # constant, whatever the width
3

julia> tr = rr4_add(7, -1; alphabet=MIN_REDUNDANT);   # the peek earns its keep

julia> value(tr.result), any(c -> c.peeked, tr.columns)
(6, true)
```
"""
function rr4_add(x::SignedDigits, y::SignedDigits; alphabet::RR4Alphabet = MIN_REDUNDANT)
    a = alphabet_halfwidth(alphabet)
    (x.radix == 4 && y.radix == 4) || throw(ArgumentError("rr4_add expects radix-4 operands"))
    (x.maxdigit <= a && y.maxdigit <= a) ||
        throw(ArgumentError("operand digits exceed the $(alphabet) alphabet"))

    # operands may carry different exponents; align to the finer of the two first,
    # exactly as an FP adder aligns significands before it may add them
    e = min(x.exponent, y.exponent)
    xa = x.exponent == e ? x : rr4_rescale(x, e)
    ya = y.exponent == e ? y : rr4_rescale(y, e)
    n = max(length(xa.digits), length(ya.digits))
    xd = [i <= length(xa.digits) ? xa.digits[i] : 0 for i in 1:n]
    yd = [i <= length(ya.digits) ? ya.digits[i] : 0 for i in 1:n]
    s = xd .+ yd

    # Stage 1: all column sums, simultaneously.
    # Stage 2: all transfers, simultaneously — each reads only column sums.
    tout = Vector{Int}(undef, n)
    peek = falses(n)
    for i in 1:n
        if alphabet == MAX_REDUNDANT
            tout[i] = rr4_transfer(s[i])
        else
            sr = i == 1 ? 0 : s[i-1]
            tout[i] = rr4_transfer_minimal(s[i], sr)
            peek[i] = abs(s[i]) == 2
        end
    end
    # Stage 3: all digits, simultaneously.
    cols = Vector{RR4Column}(undef, n)
    z = Vector{Int}(undef, n)
    for i in 1:n
        tin = i == 1 ? 0 : tout[i-1]
        w = s[i] - 4 * tout[i]
        z[i] = w + tin
        cols[i] = RR4Column(i - 1, xd[i], yd[i], s[i], tout[i], tin, w, z[i], peek[i])
    end

    carry_out = tout[n]
    zdigits = copy(z)
    carry_out != 0 && push!(zdigits, carry_out)
    res = SignedDigits(zdigits, 4, a, e)
    exact = value(res) == value(x) + value(y)
    RR4AddTrace(alphabet, x, y, cols, res, carry_out, 3, exact)
end

"""
    rr4_add(x, y; alphabet=MIN_REDUNDANT)

Any two values — integers, floats, rationals, or existing digit strings — are converted
with [`to_rr4`](@ref) and added carry-free."""
rr4_add(x, y; alphabet::RR4Alphabet = MIN_REDUNDANT) =
    rr4_add(to_rr4(x; alphabet), to_rr4(y; alphabet); alphabet)

"""
    conventional_add_trace(x::Integer, y::Integer; radix=4) -> Vector{NamedTuple}

The same addition in the *conventional*, non-redundant system, for contrast: the
canonical spellings are forced, and so is the sequencing — each position waits on the
carry from the one below.  Depth grows with the digit count.

```jldoctest
julia> tr = conventional_add_trace(6, 43);

julia> [(c.i, c.sum, c.digit, c.carry_out) for c in tr]
3-element Vector{Tuple{Int64, Int64, Int64, Int64}}:
 (0, 5, 1, 1)
 (1, 4, 0, 1)
 (2, 1, 1, 0)
```
"""
function conventional_add_trace(x::Integer, y::Integer; radix::Integer = 4)
    xd = to_radix(x, radix); yd = to_radix(y, radix)
    n = max(length(xd), length(yd))
    out = NamedTuple{(:i, :x, :y, :carry_in, :sum, :digit, :carry_out),NTuple{7,Int}}[]
    c = 0
    for i in 1:n
        a = i <= length(xd) ? xd[i] : 0
        b = i <= length(yd) ? yd[i] : 0
        t = a + b + c
        d = t % radix
        cout = t ÷ radix
        push!(out, (i = i - 1, x = a, y = b, carry_in = c, sum = t, digit = d, carry_out = cout))
        c = cout
    end
    c != 0 && push!(out, (i = n, x = 0, y = 0, carry_in = c, sum = c, digit = c, carry_out = 0))
    out
end

"""
    ripple_depth(x::Integer, y::Integer; radix=4) -> Int

How many *sequential* carry stages a conventional adder needs for this pair — the
length of the dependency chain that redundant arithmetic removes."""
ripple_depth(x::Integer, y::Integer; radix::Integer = 4) =
    length(conventional_add_trace(x, y; radix))

# --- verification -----------------------------------------------------------

"""
    absorption_table() -> Matrix{Int}

The absorption lemma by total enumeration: the final digit
`w = (s − 4t(s)) + t_in` for every one of the `13 × 3` possible
(column sum, incoming transfer) pairs.

All entries lie in `[-3,3]` — legal digits, no case needing a second transfer — and
the extremes `|w| = 3` occur precisely at the rounding ties `s ∈ {-6,-2,2,6}` combined
with an aligned incoming transfer, confirming the half-step bound is tight.

Rows index `s ∈ -6:6`, columns index `t_in ∈ -1:1`."""
function absorption_table()
    [(s - 4 * rr4_transfer(s)) + t for s in -6:6, t in -1:1]
end

"""
    verify_absorption() -> Bool

Check that every entry of [`absorption_table`](@ref) is a legal maximal-set digit.

```jldoctest
julia> verify_absorption()
true
```
"""
verify_absorption() = all(w -> abs(w) <= 3, absorption_table())

"""
    verify_minimal_closure() -> Bool

Exhaustively verify claim (ii) of the minimal-set theorem — *digit closure* — over
all `9³ = 729` three-column windows `(s_{i-2}, s_{i-1}, sᵢ)`: every output digit
`zᵢ = sᵢ − 4f(sᵢ,s_{i-1}) + f(s_{i-1},s_{i-2})` lands inside `{-2..2}`.

The engine is the sign-coupling lemma: a column's sign already determines which
transfers it can emit, before its peek is even consulted.

```jldoctest
julia> verify_minimal_closure()
true
```
"""
function verify_minimal_closure()
    for s2 in -4:4, s1 in -4:4, s0 in -4:4
        tout = rr4_transfer_minimal(s0, s1)
        tin = rr4_transfer_minimal(s1, s2)
        z = s0 - 4 * tout + tin
        abs(z) <= 2 || return false
    end
    true
end

"""
    verify_sign_coupling() -> Bool

The lemma that makes closure work: for all arguments, `s ≥ 0 ⟹ f(s,·) ∈ {0,1}` and
`s ≤ 0 ⟹ f(s,·) ∈ {-1,0}`.  Checked over all 81 pairs."""
function verify_sign_coupling()
    for s in -4:4, sr in -4:4
        t = rr4_transfer_minimal(s, sr)
        s >= 0 && !(t in (0, 1)) && return false
        s <= 0 && !(t in (-1, 0)) && return false
    end
    true
end

"""
    verify_rr4_random(ntrials=10_000; alphabet=MAX_REDUNDANT, ndig=6, rng=default_rng()) -> Bool

Random-input verification that the adder is a **value homomorphism**: for every pair
of legal spellings, `val(z) + t_out·4ⁿ = val(x) + val(y)`.

The correctness quantifies over *all* legal digit strings, not canonical ones — so
feeding a different spelling of the same number may change the output *string* but
cannot change its *value*.  Representation freedom is the resource the arithmetic
spends; value is the invariant it guards.

```jldoctest
julia> verify_rr4_random(2000)
true
```
"""
function verify_rr4_random(ntrials::Integer = 10_000; alphabet::RR4Alphabet = MIN_REDUNDANT,
                           ndig::Integer = 6, rng::AbstractRNG = Random.default_rng())
    a = alphabet_halfwidth(alphabet)
    for _ in 1:ntrials
        xd = rand(rng, -a:a, ndig)
        yd = rand(rng, -a:a, ndig)
        x = SignedDigits(xd, 4, a); y = SignedDigits(yd, 4, a)
        tr = rr4_add(x, y; alphabet)
        value(tr.result) == value(x) + value(y) || return false
        all(c -> abs(c.z) <= a, tr.columns) || return false
        all(c -> abs(c.t_out) <= 1, tr.columns) || return false
    end
    true
end

"""
    spelling_census(v::Integer, ndig::Integer, a::Integer) -> Int

How many `ndig`-digit radix-4 strings over `{-a..a}` spell the value `v`.

The surplus is what the local addition rule spends: measured over 4-digit strings the
value 10 has 3 minimal spellings and 6 maximal; 25 has 3 and 8; −7 has 2 and 6."""
function spelling_census(v::Integer, ndig::Integer, a::Integer)
    cnt = 0
    dig = zeros(Int, ndig)
    function rec(pos::Int, rem::Int)
        if pos > ndig
            rem == 0 && (cnt += 1)
            return
        end
        maxreach = 0
        for k in pos:ndig
            maxreach += a * 4^(k - 1)
        end
        abs(rem) > maxreach && return
        for d in -a:a
            rec(pos + 1, rem - d * 4^(pos - 1))
        end
    end
    rec(1, Int(v))
    cnt
end

function Base.show(io::IO, ::MIME"text/plain", tr::RR4AddTrace)
    println(io, "─"^78)
    println(io, "  RR4 carry-free addition   alphabet ",
            tr.alphabet == MAX_REDUNDANT ? "{-3..3} (maximally redundant, ρ=3)" : "{-2..2} (minimally redundant, ρ=1, one-column peek)")
    println(io, "  ", digit_string(tr.x), " (", value(tr.x), ")  +  ",
            digit_string(tr.y), " (", value(tr.y), ")  =  ",
            digit_string(tr.result), " (", value(tr.result), ")")
    println(io, "─"^78)
    println(io, "   i │  x   y │   s │ t_out │ t_in │   w │   z │ peek")
    println(io, "  ───┼────────┼─────┼───────┼──────┼─────┼─────┼──────")
    for c in reverse(tr.columns)
        @printf(io, "  %2d │ %2d  %2d │ %3d │  %+2d   │  %+2d  │ %3d │ %3d │ %s\n",
                c.i, c.x, c.y, c.s, c.t_out, c.t_in, c.w, c.z, c.peeked ? "yes" : "")
    end
    tr.carry_out != 0 && println(io, "  carry-out digit: ", tr.carry_out)
    println(io, "─"^78)
    println(io, "  depth: ", tr.depth, " stages (sums ▸ transfers ▸ digits) — constant at ANY word length")
    print(io,   "  value conserved: ", tr.exact ? "yes ✓" : "NO ✗")
end
