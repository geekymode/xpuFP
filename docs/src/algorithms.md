# Arithmetic algorithms

Every routine here computes the **same value** — these are exact integer algorithms, and
redundancy trades cost, never accuracy. What differs is depth, carry count and row
count, which is the entire point of the comparison.

!!! note "The gate figures are a stated model, not a measurement"
    One level per ripple stage, `⌈log₂n⌉` for a parallel prefix network, 3 for a
    carry-free redundant add, 1 per carry-save compression level, plus the exit CPA
    where one is needed.

## Addition

```@docs
AddResult
serial_add
parallel_prefix_add
carry_free_add
carry_save_add
compare_adders
```

Adding `255 + 1` — the worst case for a ripple, since the carry must visit every
position:

| Algorithm | Depth | Carry stages | Parallel? |
|:---|---:|---:|:---|
| `serial` (ripple) | 9 | 9 | no — strictly sequential |
| `parallel_prefix` (Kogge–Stone) | 4 | 1 | yes |
| `carry_free` (RR4) | 3 | 0 | yes |
| `carry_save` (3:2) | 1 | 0 | yes, result left redundant |

The two escapes from the chain are visible here as numbers: the prefix network
*organises* the global dependence into `⌈log₂n⌉` levels at `Θ(n log n)` wires; the
redundant alphabet *removes* it, at constant depth 3 and linear wiring. Carry-save is
cheaper still, but leaves the result in redundant form — the exit toll is deferred to
the foot of a whole tree rather than paid per operand.

Crucially, `carry_free_add` has the **same depth at any width**, while `serial_add` does
not:

```jldoctest
julia> using xpuFP

julia> carry_free_add(2^40 - 1, 1).depth == carry_free_add(3, 1).depth
true

julia> serial_add(2^40 - 1, 1).depth > serial_add(3, 1).depth
true
```

## Multiplication

```@docs
MultiplyResult
shift_add_multiply
wallace_multiply
booth_wallace_multiply
rr4_multiply
csd_constant_multiply
compare_multipliers
```

### With and without RR4 conversion

The cleanest statement of what Booth/RR4 recoding buys is the same multiplier run both
ways:

```jldoctest
julia> using xpuFP

julia> without = wallace_multiply(93, 118, 8; recode = false);

julia> with = wallace_multiply(93, 118, 8; recode = true);

julia> with.value == without.value == 93 * 118
true

julia> (without.nrows, without.tree_levels), (with.nrows, with.tree_levels)
((8, 4), (4, 2))
```

At FP32's significand width the effect is the report's headline: **24 rows and 7 tree
levels** become **13 rows and 5 levels**, with about half the compressor cells.

```jldoctest
julia> using xpuFP

julia> p = wallace_multiply(12345, 9999, 24; recode = false);

julia> b = wallace_multiply(12345, 9999, 24; recode = true, unsigned = true);

julia> (p.nrows, p.tree_levels), (b.nrows, b.tree_levels)
((24, 7), (13, 5))
```

!!! warning "12 rows or 13?"
    A **signed** `n`-bit multiplier recodes to exactly `n/2` digits — 12 for `n = 24`.
    An **unsigned** operand must be zero-extended by one bit first, costing one extra
    digit: 13 for an FP32 significand. Pass `unsigned = true` for that case.
    [`booth_rows`](@ref) reports the unsigned count, `⌈(n+1)/2⌉`, since that is the one
    the significand multiplier actually pays.

### Where the redundancy enters

`wallace_multiply` puts redundancy in **twice** — the multiplier's digits become
minimally redundant RR4 (halving the rows), and every intermediate sum lives in
carry-save form — so the only true carry chain in the whole multiplication runs once, at
the exit.

`rr4_multiply` goes further and carries *both* operands as digit strings, forming the
full digit×digit outer product. That costs more rows, each trivial; it is the honest
picture of the fully redundant domain rather than the shipping compromise.

`csd_constant_multiply` is the offline optimiser: one adder per nonzero CSD digit, and
exact, but only for constants known at design time — finding the minimum-weight spelling
needs a scan over runs whose length is unbounded, which is precisely why silicon ships
Booth's fixed windows for variable operands.
