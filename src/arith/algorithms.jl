# ---------------------------------------------------------------------------
# Addition and multiplication algorithms, each reporting what it cost.
#
# Every routine here computes the SAME value — these are exact integer algorithms, and
# redundancy trades cost, never accuracy.  What differs is depth, carry count, and row
# count, which is the entire point of the comparison.
#
# The gate-level figures are a MODEL, stated explicitly rather than measured: one level
# per ripple stage, ⌈log₂n⌉ for a parallel prefix network, 3 for a carry-free redundant
# add, 1 per carry-save compression level, plus the exit CPA where one is needed.
# ---------------------------------------------------------------------------

"""
    AddResult

The outcome of one addition, with its cost.

# Fields
- `algorithm::Symbol`
- `value` — the sum (exact; every algorithm here agrees).
- `nbits::Int` — operand width the cost model was applied at.
- `depth::Int` — sequential gate levels on the critical path.
- `carry_stages::Int` — how many carry resolutions the schedule performs.
- `parallel::Bool` — whether all digit positions compute simultaneously.
- `detail::String`
"""
struct AddResult
    algorithm::Symbol
    value::Any
    nbits::Int
    depth::Int
    carry_stages::Int
    parallel::Bool
    detail::String
end

"""
    serial_add(x, y; radix=2) -> AddResult

**Ripple-carry addition**: one position at a time, each waiting on the carry from the
one below.  Depth grows linearly with the word — this is the cost every other algorithm
here exists to avoid.

```jldoctest
julia> r = serial_add(255, 1);

julia> r.value, r.depth, r.parallel
(256, 9, false)
```
"""
function serial_add(x::Integer, y::Integer; radix::Integer = 2)
    n = max(_ndigits(x, radix), _ndigits(y, radix)) + 1
    tr = conventional_add_trace(abs(Int(x)), abs(Int(y)); radix)
    stages = length(tr)
    AddResult(:serial, Int(x) + Int(y), n, stages, stages, false,
              "ripple carry, radix $(radix): $(stages) sequential carry stages")
end

"""
    parallel_prefix_add(x, y) -> AddResult

**Carry-lookahead (Kogge–Stone) addition**: the carries are computed as a *parallel
prefix* of the associative generate/propagate composition

```math
(g,p) \\circ (g',p') = (g + p\\,g',\\; p\\,p')
```

Associativity makes all carries a prefix problem, solvable by a tree that doubles its
reach each level: `⌈log₂n⌉` levels.  The price is wiring — `Θ(n log n)` combine edges
and long spans.  The network *organises* global dependence; it does not remove it.

```jldoctest
julia> r = parallel_prefix_add(255, 1);

julia> r.value, r.depth, r.parallel
(256, 4, true)
```
"""
function parallel_prefix_add(x::Integer, y::Integer)
    n = max(_ndigits(x, 2), _ndigits(y, 2)) + 1
    depth = max(1, ceil(Int, log2(n)))
    AddResult(:parallel_prefix, Int(x) + Int(y), n, depth, 1, true,
              "Kogge–Stone prefix: $(depth) levels, ≈$(n * depth) combine wires")
end

"""
    carry_free_add(x, y; alphabet=MIN_REDUNDANT) -> AddResult

**Carry-free redundant addition**: change the number system so that no carry chain
exists.  Constant depth 3 (sums, then transfers, then digits) at *any* word length.

Wraps [`rr4_add`](@ref) and reports its cost alongside the other schedules.

```jldoctest
julia> r = carry_free_add(255, 1);

julia> r.value, r.depth, r.parallel
(256, 3, true)
```
"""
function carry_free_add(x, y; alphabet::RR4Alphabet = MIN_REDUNDANT)
    tr = rr4_add(x, y; alphabet)
    n = length(tr.columns)
    AddResult(:carry_free, value(tr.result), n, 3, 0, true,
              "RR4 $(alphabet == MIN_REDUNDANT ? "{-2..2}" : "{-3..3}"): " *
              "depth 3 at any width, no carry chain, one exit CPA amortised")
end

"""
    carry_save_add(a, b, c) -> AddResult

**Carry-save addition**: three addends in, a redundant `(sum, carry)` pair out, in a
single gate level.  The carry is not propagated but *written down* one position left,
where later levels treat it as ordinary input.

The result is left in redundant form deliberately — [`from_carrysave`](@ref) pays the
exit toll once, at the foot of a whole tree, rather than once per operand.

```jldoctest
julia> r = carry_save_add(93, 118, 45);

julia> r.value, r.depth
(256, 1)
```
"""
function carry_save_add(a::Integer, b::Integer, c::Integer)
    u, t = csa(a, b, c)
    n = max(_ndigits(a, 2), _ndigits(b, 2), _ndigits(c, 2)) + 2
    AddResult(:carry_save, u + t, n, 1, 0, true,
              "3:2 compressor, result left redundant as ($(u), $(t)); exit CPA deferred")
end

_ndigits(x::Integer, radix::Integer) = x == 0 ? 1 : floor(Int, log(radix, abs(Int(x)))) + 1

"""
    compare_adders(x, y) -> Vector{AddResult}

Run every addition schedule on the same operands and return their costs side by side.
All values agree; only the schedules differ.

```jldoctest
julia> [(r.algorithm, r.depth) for r in compare_adders(255, 1)]
4-element Vector{Tuple{Symbol, Int64}}:
 (:serial, 9)
 (:parallel_prefix, 4)
 (:carry_free, 3)
 (:carry_save, 1)
```
"""
compare_adders(x::Integer, y::Integer) = AddResult[
    serial_add(x, y),
    parallel_prefix_add(x, y),
    carry_free_add(x, y),
    carry_save_add(x, y, 0),
]

# ---------------------------------------------------------------------------

"""
    MultiplyResult

The outcome of one multiplication, with its cost.

# Fields
- `algorithm::Symbol`
- `value::Int` — the product (exact; every algorithm here agrees).
- `rows::Vector{Int}` — the partial products actually summed.
- `nrows::Int`, `tree_levels::Int`, `cells::Int`
- `depth::Int` — recode + tree + exit CPA, in gate levels.
- `recoded::Bool` — whether the multiplier passed through RR4 (Booth) recoding.
- `detail::String`
"""
struct MultiplyResult
    algorithm::Symbol
    value::Int
    rows::Vector{Int}
    nrows::Int
    tree_levels::Int
    cells::Int
    depth::Int
    recoded::Bool
    detail::String
end

"""
    shift_add_multiply(a, b, n=0) -> MultiplyResult

**Schoolbook shift-and-add**: one partial product per bit of the multiplier, summed
sequentially.  `n` rows, `n` sequential additions — the baseline everything else
improves on.

```jldoctest
julia> r = shift_add_multiply(93, 118, 8);

julia> r.value, r.nrows, r.recoded
(10974, 8, false)
```
"""
function shift_add_multiply(a::Integer, b::Integer, n::Integer = 0)
    n = n == 0 ? _ndigits(b, 2) : Int(n)
    bits = twos_complement_bits(Int(b), n)
    rows = [bits[i] * (Int(a) << (i - 1)) for i in 1:n]
    depth = n * max(1, ceil(Int, log2(2n)))     # one CPA per row
    MultiplyResult(:shift_add, Int(a) * Int(b), rows, n, 0, 0, depth, false,
                   "$(n) rows summed sequentially: $(n) carry-propagate additions")
end

"""
    wallace_multiply(a, b, n=0; recode=true, schedule=:wallace) -> MultiplyResult

The full parallel multiplier: generate partial products, crush them with a carry-save
reduction tree, and pay **one** carry propagation at the exit.

`recode = true` puts the multiplier through radix-4 Booth recoding first, halving the
rows; `recode = false` uses one row per bit.  Comparing the two is the cleanest
statement of what RR4 conversion buys.

!!! note "12 rows or 13?"
    A **signed** `n`-bit multiplier recodes to exactly `n/2` digits — 12 for `n = 24`.
    An **unsigned** operand must be zero-extended by one bit first, which costs one
    extra digit: 13 for a 24-bit FP32 significand.  Pass `unsigned = true` for that
    case; [`booth_rows`](@ref) reports the unsigned count, since that is the one the
    significand multiplier actually pays.

```jldoctest
julia> with = wallace_multiply(93, 118, 8; recode = true);

julia> without = wallace_multiply(93, 118, 8; recode = false);

julia> with.value == without.value == 93 * 118
true

julia> with.nrows, without.nrows
(4, 8)

julia> with.tree_levels, without.tree_levels
(2, 4)
```
"""
function wallace_multiply(a::Integer, b::Integer, n::Integer = 0;
                          recode::Bool = true, schedule::Symbol = :wallace,
                          unsigned::Bool = false)
    n = n == 0 ? max(_ndigits(b, 2) + 1, 4) : Int(n)
    iseven(n) || (n += 1)
    rows = if recode
        if unsigned
            d = booth_radix4_unsigned(b, n)
            [d.digits[j] * (Int(a) << (2 * (j - 1))) for j in eachindex(d.digits)]
        else
            booth_multiply(a, b, n).rows
        end
    else
        bits = twos_complement_bits(Int(b), n)
        [bits[i] * (Int(a) << (i - 1)) for i in 1:n]
    end
    tr = _reduce_tree(rows, schedule)
    cpa = max(1, ceil(Int, log2(2n)))
    depth = (recode ? 1 : 0) + tr.nlevels + cpa
    MultiplyResult(recode ? :booth_wallace : :wallace, Int(a) * Int(b), rows,
                   length(rows), tr.nlevels, tr.ncells, depth, recode,
                   "$(length(rows)) rows → $(tr.nlevels) CSA levels ($(tr.ncells) cells) " *
                   "→ 1 exit CPA ($(cpa) levels)")
end

"""
    booth_wallace_multiply(a, b, n=0) -> MultiplyResult

Shorthand for [`wallace_multiply`](@ref) with Booth recoding on — the shape of every
multiplier in every GPU, TPU and NPU MAC unit."""
booth_wallace_multiply(a::Integer, b::Integer, n::Integer = 0) =
    wallace_multiply(a, b, n; recode = true)

"""
    csd_constant_multiply(c, x) -> MultiplyResult

Multiply by a **constant** known at design time, through its CSD tap structure: one
adder/subtractor per nonzero digit beyond the first, shifts free.

This is the offline optimiser — reserved for constants, because finding the
minimum-weight spelling requires a scan over runs, whose length is unbounded.  That is
precisely why silicon ships Booth's fixed windows and not CSD's optimal strings for
*variable* operands.

```jldoctest
julia> r = csd_constant_multiply(231, 1234);

julia> r.value == 231 * 1234, r.nrows
(true, 4)
```
"""
function csd_constant_multiply(c::Integer, x::Integer)
    taps = csd_multiplier_taps(c)
    rows = [sg * (Int(x) << sh) for (sh, sg) in taps]
    nadd = max(length(taps) - 1, 0)
    depth = max(1, ceil(Int, log2(max(length(taps), 2)))) * max(1, ceil(Int, log2(64)))
    MultiplyResult(:csd_constant, Int(c) * Int(x), rows, length(rows), 0, nadd, depth,
                   false,
                   "$(length(taps)) taps ⇒ $(nadd) adder/subtractors " *
                   "(binary would need $(binary_weight(c) - 1))")
end

"""
    rr4_multiply(a, b; alphabet=MIN_REDUNDANT) -> MultiplyResult

Multiply with **both** operands carried as redundant radix-4 digit strings: the digit
products are formed pairwise and accumulated without ever resolving a carry, with a
single conversion at the exit.

Shows the redundant domain end to end — in contrast to [`wallace_multiply`](@ref),
where only the multiplier is recoded and the addends live in carry-save."""
function rr4_multiply(a, b; alphabet::RR4Alphabet = MIN_REDUNDANT)
    x = to_rr4(a; alphabet)
    y = to_rr4(b; alphabet)
    rows = Int[]
    for (i, dx) in enumerate(x.digits), (j, dy) in enumerate(y.digits)
        p = dx * dy
        p == 0 && continue
        push!(rows, p * 4^(i + j - 2))
    end
    isempty(rows) && push!(rows, 0)
    tr = _reduce_tree(rows, :wallace)
    n = length(x.digits) + length(y.digits)
    cpa = max(1, ceil(Int, log2(2 * n)))
    MultiplyResult(:rr4, Int(value(x) * value(y)), rows, length(rows), tr.nlevels,
                   tr.ncells, 1 + tr.nlevels + cpa, true,
                   "$(length(x.digits))×$(length(y.digits)) digit products, " *
                   "$(tr.nlevels) CSA levels, one exit conversion")
end

"""
    compare_multipliers(a, b, n=8) -> Vector{MultiplyResult}

Every multiplication strategy on the same operands: schoolbook, Wallace without
recoding, Wallace with Booth/RR4 recoding, and the fully redundant version.

All products agree — the comparison is purely rows, levels, cells and depth.

```jldoctest
julia> rs = compare_multipliers(93, 118, 8);

julia> all(r -> r.value == 93 * 118, rs)
true

julia> [(r.algorithm, r.nrows, r.tree_levels) for r in rs[2:3]]
2-element Vector{Tuple{Symbol, Int64, Int64}}:
 (:wallace, 8, 4)
 (:booth_wallace, 4, 2)
```
"""
compare_multipliers(a::Integer, b::Integer, n::Integer = 8) = MultiplyResult[
    shift_add_multiply(a, b, n),
    wallace_multiply(a, b, n; recode = false),
    wallace_multiply(a, b, n; recode = true),
    rr4_multiply(a, b),
]

function Base.show(io::IO, ::MIME"text/plain", r::AddResult)
    println(io, "AddResult(:", r.algorithm, ")")
    println(io, "  value        : ", r.value)
    println(io, "  depth        : ", r.depth, " gate levels", r.parallel ? "  (all positions in parallel)" : "  (sequential)")
    println(io, "  carry stages : ", r.carry_stages)
    print(io,   "  ", r.detail)
end

function Base.show(io::IO, ::MIME"text/plain", r::MultiplyResult)
    println(io, "MultiplyResult(:", r.algorithm, ")")
    println(io, "  value        : ", r.value)
    println(io, "  rows         : ", r.nrows, r.recoded ? "   (RR4/Booth recoded)" : "   (one per bit)")
    println(io, "  tree levels  : ", r.tree_levels, "   cells: ", r.cells)
    println(io, "  depth        : ", r.depth, " gate levels")
    print(io,   "  ", r.detail)
end
