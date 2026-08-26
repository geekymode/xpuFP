# ---------------------------------------------------------------------------
# Carry-save arithmetic and the Wallace/Dadda reduction trees.
#
# Carry-save form is redundant radix 2: each position holds a (sum, carry) bit pair —
# digit values {0,1,2}, asymmetric, not a signed-digit system at all — whose 3:2
# compressor cell is a handful of gates.  It is the representation that actually owns
# every accumulation tree in every AI accelerator.
# ---------------------------------------------------------------------------

"""
    csa(a::Integer, b::Integer, c::Integer) -> Tuple{Int,Int}

One 3:2 compressor (a full adder per bit position), returning the `(sum, carry)` pair:

```math
a + b + c = (a \\oplus b \\oplus c) + 2\\,\\mathrm{maj}(a,b,c)
```

Per bit position, three bits sum to a two-bit number whose low bit is exactly the
parity and whose high bit is exactly the majority — eight cases, all checked.

This is the base-2 sibling of RR4's split rule `s = 4t + u`: the "carry" is not
propagated but simply **written down** in a second vector, one position to the left,
where later tree levels treat it as ordinary input.  Transfers that travel one hop and
stop, deferred wholesale rather than absorbed one at a time.

```jldoctest
julia> u, t = csa(93, 118, 45);

julia> u + t == 93 + 118 + 45
true
```
"""
function csa(a::Integer, b::Integer, c::Integer)
    u = xor(xor(Int(a), Int(b)), Int(c))
    m = (Int(a) & Int(b)) | (Int(a) & Int(c)) | (Int(b) & Int(c))
    (u, m << 1)
end

"""
    verify_csa_identity(trials=10_000; rng=default_rng()) -> Bool

Check `a + b + c = u + t` on random integers, negatives included (the identity holds
on two's-complement patterns).

```jldoctest
julia> verify_csa_identity(5000)
true
```
"""
function verify_csa_identity(trials::Integer = 10_000; rng::AbstractRNG = Random.default_rng())
    for _ in 1:trials
        a, b, c = rand(rng, -10^6:10^6, 3)
        u, t = csa(a, b, c)
        u + t == a + b + c || return false
    end
    true
end

"""
    ReductionTrace

A carry-save reduction run: the row multiset at every level, the surviving pair, and
the invariant check.

# The theorem, in one sentence
A Wallace tree is an arbitrary composition of one sum-preserving gadget, and its
correctness is the equation `σ ∘ C = σ`; everything else is scheduling.

# Fields
- `levels::Vector{Vector{Int}}` — `levels[1]` is the input rows, then each reduction.
- `pair::Tuple{Int,Int}` — the surviving `(u*, t*)`.
- `total::Int` — `u* + t*`, which equals the sum of the input rows.
- `nlevels::Int`, `ncells::Int` — depth and full-adder count.
- `schedule::Symbol` — `:wallace` (greedy) or `:dadda` (lazy).
- `exact::Bool`
"""
struct ReductionTrace
    levels::Vector{Vector{Int}}
    pair::Tuple{Int,Int}
    total::Int
    nlevels::Int
    ncells::Int
    schedule::Symbol
    exact::Bool
end

"""
    wallace_reduce(rows) -> ReductionTrace

Wallace's **greedy** schedule (C. S. Wallace, 1964): every level feeds as many
disjoint triples of rows as possible into 3:2 compressors, turning `r` rows into
`⌈2r/3⌉`, until exactly two rows remain for the single exit carry-propagate adder.

The *tree* is the ancestry structure inside that narrowing: pick either final wire and
trace it upward — its provenance branches three ways at every box it passed through, a
ternary tree whose leaves are the original rows.  Depth is `Θ(log_{3/2} r)`, which is
where the report's `24 → 16 → 11 → 8 → 6 → 4 → 3 → 2`, seven levels, comes from.

```jldoctest
julia> tr = wallace_reduce([1, 2, 4, 8, 16, 32, 64, 128, 256]);

julia> [length(l) for l in tr.levels]
5-element Vector{Int64}:
 9
 6
 4
 3
 2

julia> tr.total, tr.exact
(511, true)
```
"""
function wallace_reduce(rows::AbstractVector{<:Integer})
    _reduce_tree(rows, :wallace)
end

"""
    dadda_reduce(rows) -> ReductionTrace

Dadda's **lazy** 1965 variant: compress only enough at each level to land on the
precomputed targets `…, 19, 13, 9, 6, 4, 3, 2`.

Identical depth to Wallace, fewer compressor cells — work done earlier than necessary
buys no depth.  Real multipliers ship Dadda-style counts under the Wallace name.

```jldoctest
julia> w = wallace_reduce(collect(1:24)); d = dadda_reduce(collect(1:24));

julia> w.nlevels == d.nlevels, d.ncells <= w.ncells
(true, true)
```
"""
function dadda_reduce(rows::AbstractVector{<:Integer})
    _reduce_tree(rows, :dadda)
end

"""
    dadda_targets(m::Integer) -> Vector{Int}

The Dadda target sequence at or below `m`: `2, 3, 4, 6, 9, 13, 19, …`, each term
`⌊3/2 ×⌋` the previous."""
function dadda_targets(m::Integer)
    ts = [2]
    while ts[end] < m
        push!(ts, (ts[end] * 3) ÷ 2)
    end
    reverse(ts[1:end-1])
end

function _reduce_tree(rows::AbstractVector{<:Integer}, schedule::Symbol)
    cur = collect(Int, rows)
    total = sum(cur)
    levels = [copy(cur)]
    cells = 0
    targets = schedule === :dadda ? dadda_targets(length(cur)) : Int[]
    ti = 1
    while length(cur) > 2
        if schedule === :wallace
            nxt = Int[]
            i = 1
            while i + 2 <= length(cur)
                u, t = csa(cur[i], cur[i+1], cur[i+2])
                push!(nxt, u, t); cells += 1
                i += 3
            end
            while i <= length(cur)
                push!(nxt, cur[i]); i += 1
            end
            cur = nxt
        else
            # lazy: compress only down to the next target
            while ti <= length(targets) && targets[ti] >= length(cur)
                ti += 1
            end
            tgt = ti <= length(targets) ? targets[ti] : 2
            ti += 1
            nxt = copy(cur)
            # each 3:2 reduces the row count by exactly one
            ncomp = length(cur) - tgt
            work = Int[]
            keep = Int[]
            need = 3 * ncomp
            for (k, v) in enumerate(nxt)
                (k <= need) ? push!(work, v) : push!(keep, v)
            end
            out = Int[]
            i = 1
            while i + 2 <= length(work)
                u, t = csa(work[i], work[i+1], work[i+2])
                push!(out, u, t); cells += 1
                i += 3
            end
            while i <= length(work)
                push!(out, work[i]); i += 1
            end
            cur = vcat(out, keep)
        end
        push!(levels, copy(cur))
    end
    while length(cur) < 2
        push!(cur, 0)
    end
    pair = (cur[1], cur[2])
    ReductionTrace(levels, pair, pair[1] + pair[2], length(levels) - 1, cells,
                   schedule, pair[1] + pair[2] == total)
end

"""
    reduction_schedule(nrows::Integer; schedule=:wallace) -> Vector{Int}

The row counts level by level, without doing any arithmetic — the shape of the tree.

```jldoctest
julia> reduction_schedule(24)          # plain binary, FP32 significand
8-element Vector{Int64}:
 24
 16
 11
  8
  6
  4
  3
  2

julia> reduction_schedule(13)          # after Booth recoding: two levels shallower
6-element Vector{Int64}:
 13
  9
  6
  4
  3
  2
```
"""
function reduction_schedule(nrows::Integer; schedule::Symbol = :wallace)
    out = [Int(nrows)]
    if schedule === :wallace
        while out[end] > 2
            push!(out, cld(2 * out[end], 3))
        end
    else
        for t in dadda_targets(nrows)
            t < out[end] && push!(out, t)
        end
        out[end] != 2 && push!(out, 2)
    end
    out
end

"""
    compressor_cells(nrows::Integer) -> Int

Total 3:2 cells in a reduction from `nrows` to 2.  Each cell eliminates exactly one
row, so the count is simply `nrows − 2` per bit column."""
compressor_cells(nrows::Integer) = max(Int(nrows) - 2, 0)

"""
    verify_tree_invariant(rows; schedule=:wallace) -> Bool

The load-bearing check: the total of the live rows never changes at any level.

Note that `u*` is **not** a closed-form "XOR of everything" — the recursion interleaves
parity and majority products at every level, and no formula like `u* = ⊕ᵢrᵢ` survives
it.  The theorem's content is precisely that no closed form is needed: the *invariant*
is the whole proof.

```jldoctest
julia> verify_tree_invariant(collect(1:24))
true
```
"""
function verify_tree_invariant(rows::AbstractVector{<:Integer}; schedule::Symbol = :wallace)
    tr = _reduce_tree(rows, schedule)
    total = sum(rows)
    all(l -> sum(l) == total, tr.levels) && tr.total == total
end

"""
    booth_tree_saving(n::Integer) -> NamedTuple

What Booth/RR4 buys, quantified for an `n`-bit multiplier: rows, tree levels, and
compressor cells, plain versus recoded.

```jldoctest
julia> booth_tree_saving(24)
(plain_rows = 24, booth_rows = 13, plain_levels = 7, booth_levels = 5, plain_cells = 22, booth_cells = 11, level_saving = 2, cell_ratio = 0.5)
```
"""
function booth_tree_saving(n::Integer)
    pr, br = booth_rows(n)
    ps = reduction_schedule(pr); bs = reduction_schedule(br)
    (plain_rows = pr, booth_rows = br,
     plain_levels = length(ps) - 1, booth_levels = length(bs) - 1,
     plain_cells = compressor_cells(pr), booth_cells = compressor_cells(br),
     level_saving = (length(ps) - 1) - (length(bs) - 1),
     cell_ratio = compressor_cells(br) / compressor_cells(pr))
end

function Base.show(io::IO, ::MIME"text/plain", tr::ReductionTrace)
    println(io, "─"^70)
    println(io, "  Carry-save reduction  (", tr.schedule == :wallace ? "Wallace, greedy" : "Dadda, lazy", ")")
    println(io, "─"^70)
    for (k, l) in enumerate(tr.levels)
        @printf(io, "  level %d: %3d rows   Σ = %d\n", k - 1, length(l), sum(l))
    end
    println(io, "─"^70)
    println(io, "  surviving pair : ", tr.pair)
    println(io, "  u* + t*        : ", tr.total, tr.exact ? "  ✓ invariant held at every level" : "  ✗")
    println(io, "  levels         : ", tr.nlevels, "   cells: ", tr.ncells)
    print(io,   "  the exit CPA is the ONLY carry chain in the whole structure")
end
