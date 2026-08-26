# Redundant number systems

Ordinary binary gives every value exactly one representation, and that uniqueness is
what makes addition slow: a carry born at bit 0 may propagate all the way to bit 47.
**Redundant** number systems deliberately allow multiple representations per value; the
slack lets each digit position absorb what would have been a travelling carry.

!!! note "Redundant systems are exact"
    Conventional binary, CSD and RR4 all represent integers perfectly; converting
    between them loses nothing, ever. Redundancy trades **cost**, never accuracy — the
    opposite deal from quantization. The comparison is purely about time, gates and bits.

```@docs
SignedDigits
value
weight
adders
redundancy
is_redundant
digit_string
```

## Canonical signed digit

The cashier's trick: paying a 999 bill by handing over 1000 and receiving 1 is writing
``999 = 1000 - 1``. Binary CSD is the identical move on runs of ones.

```@docs
csd
csd_trace
all_signed_spellings
verify_csd_minimal
min_signed_weight
csd_multiplier_taps
csd_multiply
csd_density
binary_weight
```

```@example red
using xpuFP, CairoMakie # hide
plot_csd_digits(231)
```

The greedy ``\bmod 4`` scan emits the **provably minimum-weight** spelling with no
search at all. Here is the whole landscape it is choosing from:

```@example red
using xpuFP, CairoMakie # hide
plot_csd_census(231, 9)
```

Cost varies wildly — three adders up to eight for spellings of the very same number —
and even the *minimum* is not unique. Two spellings achieve weight 4; the non-adjacency
rule is what breaks the tie canonically.

## Redundant radix 4

### Representation

A radix-4 signed-digit number is a [`SignedDigits`](@ref) with `radix = 4`: a digit
vector (least significant first), the alphabet half-width, and a **radix-4 exponent**
giving the weight of the least significant digit. The value is always

```math
x = \sum_i d_i \, 4^{\,i-1+\mathrm{exponent}}
```

The exponent is what makes fractions representable, and it is exact for every `Float64`:
a float is a dyadic rational ``m \cdot 2^k``, and since ``4 = 2^2`` every dyadic
rational has a **finite** radix-4 expansion. Nothing is rounded on the way in.

```@docs
dyadic_parts
scale
scaled_integer
nfracdigits
isintegral
float_value
```

### Getting at the pieces

`to_rr4` returns a container; when you want the raw arrays — to index into, or to feed a
circuit model — ask for the parts instead:

```@docs
to_rr4_parts
to_digits_parts
digit_parts
```

```jldoctest
julia> using xpuFP

julia> p = to_rr4_parts(49.25);

julia> p.digits, p.scale, p.scaled_integer
([1, 1, 0, -1, 1], -1, 197)

julia> p.value == p.scaled_integer * Rational{BigInt}(4)^p.scale
true
```

The identity `value = scaled_integer × radix^scale` is the same "integer plus an agreed
scale" decomposition fixed point uses — which is why a redundant digit string carries
fractions at no structural cost.

!!! warning "Two digit orders, deliberately opposite"
    `digits` is indexed by **significance**: `digits[1]` is the *least* significant, so
    `digits[i]` carries weight `radix^(i-1+scale)`. `digits_msb` is the *printed* order.
    Mixing them up is the easiest indexing error in this module, so both are named.

### Converting in, from anything

```@docs
to_rr4
to_rr4_minimal
to_rr4_maximal
```

!!! warning "Two different senses of *minimal*"
    `MIN_REDUNDANT` describes the **alphabet** (`{-2..2}`, small surplus).
    [`minimal_rr4`](@ref) finds the **shortest spelling** (fewest digits), and works in
    either alphabet. They are orthogonal axes, which is why the alphabet constants are
    named `MIN_REDUNDANT`/`MAX_REDUNDANT` rather than `MINIMAL`/`MAXIMAL`, and why the
    maximally redundant conversion has its own function name instead of a flag.

### Arithmetic on digit strings

`to_rr4(a) + to_rr4(b)` just works, and so do `-`, `*`, comparison and mixing with
ordinary numbers.

```@docs
Base.:+(::SignedDigits, ::SignedDigits)
Base.:-(::SignedDigits)
Base.:*(::SignedDigits, ::SignedDigits)
Base.:(==)(::SignedDigits, ::SignedDigits)
same_spelling
promote_alphabet
sum_rr4
dot_rr4
```

```jldoctest
julia> using xpuFP

julia> s = to_rr4(712.3) + to_rr4(12.3);

julia> value(s) == Rational{BigInt}(712.3) + Rational{BigInt}(12.3)
true

julia> value(to_rr4(49.25) * to_rr4(4))
197//1

julia> value(sum_rr4([1, 2, 3, 4, 5]))
15
```

!!! tip "The addition is exact where floating point is not"
    `value(to_rr4(712.3) + to_rr4(12.3))` is the **exact** sum of the two `Float64`s.
    `712.3 + 12.3` in `Float64` rounds, and misses it by ``4.6\times10^{-14}``. This is
    the redundancy bargain stated plainly: cost is traded, accuracy never is.

    Note the corollary — `Float64(value(...))` prints `724.5999999999999`, not
    `724.6`, precisely *because* the exact answer is not a `Float64`.

!!! note "Operators give the answer; the trace functions give the cost"
    `+` returns a [`SignedDigits`](@ref). When you want the column-by-column story —
    transfers, interim digits, constant depth — call [`rr4_add`](@ref), which returns an
    [`RR4AddTrace`](@ref). For multiplication structure use [`rr4_multiply`](@ref),
    [`wallace_multiply`](@ref) or [`booth_multiply`](@ref); for a cost table across
    strategies use [`compare_adders`](@ref) and [`compare_multipliers`](@ref).

Negation is **free** in a signed-digit system — flip every digit, no borrow, no carry —
which is the same fact that makes `-1` digits cost nothing extra in a CSD constant
multiplier.

Equality compares **values, not spellings**, because the whole design principle is
*representation free, value invariant*:

```jldoctest
julia> using xpuFP

julia> to_rr4(49) == to_rr4_maximal(49)
true

julia> same_spelling(to_rr4(49), to_rr4_maximal(49))
false
```

### Enumerating and reshaping

Because the system is redundant, `to_rr4` returns *a* spelling, not *the* spelling.

#### As a table, with the digits and the scale

```@docs
rr4_representations
RR4Representations
digit_matrix
digit_matrix_with_scale
plot_rr4_representations
```

```jldoctest
julia> using xpuFP

julia> r = rr4_representations(12.5);

julia> r.ndigits, r.scale
(4, -1)

julia> r.digits
2×4 Matrix{Int64}:
 1  -1  0   2
 1  -1  1  -2

julia> digit_matrix_with_scale(r)
2×5 Matrix{Int64}:
 1  -1  0   2  -1
 1  -1  1  -2  -1
```

#### Check the cost before you enumerate

```@docs
rr4_complexity
RR4Complexity
```

The representation count is **exact**, not an estimate — a dynamic program over the
carry, costing well under a millisecond even at 600 digits. So it is always worth
running before deciding to enumerate.

```jldoctest
julia> using xpuFP

julia> c = rr4_complexity(0.1);

julia> c.ndigits, c.total, c.listable
(27, 317811, false)

julia> rr4_complexity(6.3).total
2
```

Printed, it lays out the whole picture:

```
RR4 representation complexity for 0.1
  method          : minimally_redundant   alphabet {-2…2}
  width           : 27 digits, scale exponent -28
  representations : 317811   (exact, by dynamic programming in 0.0 ms)
  naive search    : 5^27 ≈ 7.45e+18 digit strings
  pruned search   : ≈ 2^27 ≈ 1.34e+08   (only digits d ≡ rem mod 4 are viable, ≤2 per position)
  listing cost    : 84.9 MB for all 317811 spellings
  verdict         : TOO MANY — above the limit of 10000
  advice          : use count_rr4_representations(x) for the number, or narrow with ndigits=…
```

The two search figures are the story of why enumeration is feasible at all: the naive
tree has ``5^{27} \approx 7\times10^{18}`` branches, while the divisibility rule
(only digits ``d \equiv \mathrm{rem} \bmod 4`` can occur) leaves at most two per
position — about ``2^{27}``, ten orders of magnitude smaller.

[`rr4_representations`](@ref) runs this check itself and warns before listing anything
when the total exceeds `limit`; pass `verbose = false` to silence it.

!!! warning "Any Float64 needs ~27 radix-4 digits"
    `6.3` is not `6.3` — the nearest `Float64` has a full 53-bit significand, so its
    exact radix-4 form runs to 27 digits. At that width some values have very many
    spellings (`0.1` has **317 811**). `rr4_representations` therefore counts first by
    a cheap dynamic program and lists at most `limit` rows (default 10 000), reporting
    the true `total` and setting `truncated` either way. Use
    [`count_rr4_representations`](@ref) when you only want the number.

`method` picks the alphabet — `:minimally_redundant` (default, Booth's `{-2..2}`) or
`:maximally_redundant` (`{-3..3}`). Rows are most significant first, so each reads left
to right like the number, and are sorted cheapest-first by nonzero count.

```@example reps
using xpuFP, CairoMakie # hide
plot_rr4_representations(12.5)
```

```@example reps
using xpuFP, CairoMakie # hide
plot_rr4_representations(25; method = :maximally_redundant)
```

Every row denotes the same value; the differences are pure spelling. The final cell is
the **scaling exponent**, drawn in a flat light blue rather than on the viridis scale —
it is an exponent, not a digit, and sharing a colour scale would imply a comparison that
does not exist.

There is no colorbar by default: an RR4 alphabet holds at most seven values and each is
printed in its own cell, so a legend would spend width restating the labels. Pass
`colorbar = true` if you want one.

Widen the field and the freedom grows:

```@example reps
using xpuFP, CairoMakie # hide
plot_rr4_representations(10; method = :maximally_redundant, ndigits = 4)
```

#### Minimal weight: the cheapest spelling

The *count* of spellings is a curiosity; the **minimum nonzero count** is the
engineering quantity, because a constant multiplier costs one adder/subtractor per
nonzero digit.

```@docs
minimal_weight
weight_distribution
weight_stats
WeightStats
weight_scaling
plot_weight_scaling
```

Both come from a dynamic program over the carry — enumerating up to `F(n+1)` spellings
to find the cheapest is hopeless past a dozen digits.

```jldoctest
julia> using xpuFP

julia> minimal_weight(231), weight(to_rr4(231))
(4, 5)

julia> minimal_weight(1000), weight(to_rr4(1000))
(3, 4)
```

##### How it scales

Sampling random values at each width and taking the mean minimal weight per digit
position gives a density that converges cleanly:

| system | minimal-weight density |
|:---|---:|
| radix-2 `{-1,0,1}` (CSD / NAF) | ``1/3`` — the classical result |
| radix-4 `{-2..2}` minimally redundant | ``\to 2/3 \approx 0.667`` |
| radix-4 `{-3..3}` maximally redundant | ``\to 3/5 = 0.600`` |

```@example wsc
using xpuFP, CairoMakie, Random # hide
plot_weight_scaling([8, 16, 32, 64]; samples = 120, rng = MersenneTwister(9))
```

More redundancy buys a lower density, as it must: the alphabets are nested, so the
minimum over the larger set cannot exceed the minimum over the smaller. And searching
for the minimum is worth roughly **10% of the nonzero digits** over the one-pass
conversion, at every width — the one-pass form sits near 0.75 in both alphabets.

!!! warning "Sample from the representable range"
    A width-`n` string over `{-2..2}` reaches only ``2(4^n-1)/3``, not ``4^n``. Sampling
    beyond it mixes in values that do not fit, and [`minimal_weight`](@ref) returns `-1`
    for those — which will quietly wreck an average if you do not filter it.
    [`weight_scaling`](@ref) samples the correct range for you.

#### Canonical forms — and why radix 4 has no obvious one

```@docs
canonical_rr4
nonadjacent_rr4
has_nonadjacent_form
```

"Canonical" means a rule that selects exactly **one** spelling per value, restoring the
uniqueness redundancy gave away. At radix 2 the non-adjacent form does this in a single
stroke, because it has three properties at once:

1. it **exists** for every value,
2. it is **unique**, and
3. it is **minimum weight**.

That is what makes *the* CSD well defined, and why [`csd`](@ref) needs no options.

**At radix 4, property (1) fails.** Measured over the values 0–600, only about **15%**
have any non-adjacent spelling at all, in either alphabet — unsurprising once you know
the minimal-weight density is ``\approx 2/3``, well above the ``1/2`` that
non-adjacency would demand. Most radix-4 values simply cannot be spread out that thinly.

Properties (2) and (3) do survive: where a non-adjacent form exists it is unique and is
minimum weight (verified exhaustively below 300, both alphabets).

So a radix-4 canonical form has to *state its rule*:

| `rule` | what it selects | unique? | minimum weight? |
|:---|:---|:---|:---|
| `:minweight` (default) | fewest nonzeros, ties to the smaller digit at the less significant end | yes | **yes** |
| `:nonadjacent` | no two adjacent nonzeros | yes | yes — but exists for only ~15% of values |
| `:onepass` | what [`to_rr4`](@ref) emits | yes | **no** — exceeds the minimum for ~28% of values |

```jldoctest
julia> using xpuFP

julia> weight(canonical_rr4(231)), weight(to_rr4(231))
(4, 5)

julia> nonadjacent_rr4(231) === nothing
true
```

!!! warning "`to_rr4` is not a canonical form in the CSD sense"
    It applies one borrow pass to the non-redundant `{0..3}` digits — deterministic and
    reproducible, so it is *a* well-defined selection, but it is **not** minimum weight.
    Use [`canonical_rr4`](@ref) when the adder count matters.

There is a third, quite different sense of "canonical" worth keeping separate: the
unique **non-redundant** `{0..3}` form, which is what [`rr4_to_canonical`](@ref)
produces. That is not a redundant representation at all — it is the exit conversion, the
carry-propagate that a redundant datapath pays once on the way out.

#### Lower-level accessors

```@docs
all_rr4_representations
count_rr4_representations
min_ndigits
max_representable
minimal_rr4
rr4_with_length
rr4_rescale
```

The simplest question — *how many ways can this be spelled?* — is one call:

```jldoctest
julia> using xpuFP

julia> rr4_representation_counts(6)
2

julia> rr4_representation_counts(10; method = :maximally_redundant)
2
```

That uses the default alphabet and the shortest width that fits. Give it a range of
widths instead and it sweeps them, which is where the freedom becomes visible.

The spelling count is the resource the local addition rule spends, and it grows with
both the alphabet and the length. At a **fixed** 4 digits:

| value | `{-2..2}` | `{-3..3}` |
|---:|---:|---:|
| 10 | 3 | 6 |
| 25 | 3 | 8 |
| −7 | 2 | 6 |

#### Why the counts jump around

The counts look erratic — `rr4_representation_counts.([26, 26.3, 26.35, 26.4, 26.5])`
gives `3, 3, 253732, 271443, 4` — but the mechanism is simple and the bound is exact.

```@docs
rr4_branch_residues
rr4_max_count
```

Working from the least significant end, position `k` can only hold a digit
`d ≡ rem (mod 4)`. In `{-2..2}` **exactly one residue in four offers a choice**:

| `rem mod 4` | viable digits | choices |
|---:|:---|---:|
| 0 | `0` | 1 |
| 1 | `1` | 1 |
| 2 | `-2, 2` | **2** |
| 3 | `-1` | 1 |

So a value's spelling count is decided by how often its remainder trajectory lands on
`≡ 2`. That is why the two neighbours differ so violently:

| value | scaled integer `N` | `N mod 4` | levels that branch | count |
|:---|---:|---:|---:|---:|
| `26.3` | 7402791887490253 | 1 (odd) | 3 of 27 | 3 |
| `26.35` | 7416865636325786 | 2 (even) | **27 of 27** | 253 732 |

`26.3` is odd, so its first step is forced, and its trajectory then avoids `≡ 2` for 24
consecutive levels. `26.35` starts at `≡ 2` and branches at *every* level.

The maximum is a closed form, verified against exhaustive search for `n ≤ 11`:

```math
\max_x \#\text{spellings}(x, n) =
\begin{cases} F(n+1) & \{-2..2\} \\ 2^{\,n-1} & \{-3..3\} \end{cases}
```

Counts in the minimally redundant alphabet therefore grow like ``\varphi^n``
(``\varphi \approx 1.618``), **not** ``2^n`` — because branches *recombine*: distinct
digit choices often lead to the same remainder, which is exactly what turns a binary
tree into a Fibonacci recurrence. The maximally redundant alphabet branches at three
residues out of four and does reach ``2^{\,n-1}``.

`0.1` attains the bound exactly, at both width 27 (`F(28) = 317811`) and width 28
(`F(29) = 514229`).

!!! note "Typical values sit far below the maximum"
    Over 400 random floats near 26 the median count is about 150, the quartiles 54 and
    432, against a theoretical ceiling of 317 811. So `26.3`'s count of 3 is unusually
    *low* and `26.35`'s quarter-million unusually high — both are tails of a very
    wide distribution.

!!! warning "Comparing methods at their own minimum widths is not like-for-like"
    The wider alphabet often needs *fewer* digits — 49 takes 4 in `{-2..2}` but only 3
    in `{-3..3}` — so `rr4_representation_counts(49)` and its `:maximally_redundant`
    counterpart are measured at different widths. Pass an explicit `ndigits` to compare
    the alphabets on equal terms.

```@docs
RR4Alphabet
rr4_transfer
rr4_transfer_minimal
rr4_add
RR4AddTrace
RR4Column
rr4_split_table
conventional_add_trace
ripple_depth
spelling_census
```

### The maximal alphabet

With digits ``\{-3..3\}`` there is one unit of slack on each flank — exactly the room
needed to absorb a ``\pm 1`` transfer unconditionally, so the transfer decision reads
**one column only**.

```@example red
using xpuFP, CairoMakie # hide
plot_rr4_addition(rr4_add(6, 43))
```

### The minimal alphabet, and its one-column peek

``\{-2..2\}`` is Booth's alphabet and the one that actually ships. It has zero slack, so
the rule buys safety with *information* instead: a peek at the right neighbour's sign,
which determines the sign of the incoming transfer **before it arrives**.

```@example red
using xpuFP, CairoMakie # hide
plot_rr4_addition(rr4_add(121, 101; alphabet = MIN_REDUNDANT))
```

Redundancy and lookahead are exchangeable currencies: turn the dial down one click, and
the rule pays for it with one column of sight.

### Verification

Every claim is machine-checked rather than asserted.

```@docs
absorption_table
verify_absorption
verify_sign_coupling
verify_minimal_closure
verify_rr4_random
```

```@example red
using xpuFP, CairoMakie # hide
plot_absorption_table()
```

The complete minimal-set rule as data — note that **seven of the nine rows of the
transfer map are constant**, and that panel (d) has exactly two forbidden cells:

```@example red
using xpuFP, CairoMakie # hide
plot_minimal_maps()
```

## Why no chain can form

```@example red
using xpuFP, CairoMakie # hide
plot_dependency_graph(5)
```

An arrow into a node means "argument of". On the left, `c_{k+1} = maj(x_k, y_k, c_k)`
has the previous carry among its arguments, so the graph contains a horizontal path by
construction — that path *is* the ripple. On the right the transfer row has **no
internal edges at all**; the crossed dashed arrows mark edges that would have to exist
for a carry to travel two steps, and the rule gives them nothing to be.

The same fact in Forney normal factor-graph form, where variables live on edges and
constraints are boxes:

```@example red
using xpuFP, CairoMakie # hide
plot_factor_graph(4)
```

Binary addition is a **state-space model** — the carry variables form a rail threading
every box, the trellis of a system with memory. Minimal RR4 admits two *horizontal*
full-width cuts, so the whole graph clocks in three stages at any width. Carry
propagation is memory; a redundant alphabet buys the model down from "chain with state"
to "sliding window, order two".

## The two escapes from the carry chain

```@example red
using xpuFP, CairoMakie # hide
plot_butterfly_vs_planes(16)
```

Keep binary and compute all carries as a **parallel prefix** — the Kogge–Stone
network's diagonals are exactly the wings of a radix-2 FFT butterfly, because prefix
networks and butterflies belong to the same graph family. Or change the number system,
and there is no dependence left to organise.

The FFT connection is therefore a shared naming convention (radix as stage economics),
not a shared topology: butterflies organise computations where every output needs every
input, while a redundant alphabet builds one where no output needs more than its
neighbourhood.

## Booth recoding

```@docs
booth_radix2
booth_radix4
booth_radix4_unsigned
booth_multiply
BoothProduct
booth_rows
booth_pairing
booth_borrow_form
verify_booth_exhaustive
```

```@example red
using xpuFP, CairoMakie # hide
plot_booth_windows(-74, 8)
```

Each brace shares one bit with its neighbour — **the overlap is what makes the recoding
correct**. Theorem 1 says the digits stay in ``\{-2..2\}`` and reconstruct the
two's-complement value exactly, negatives included, with no correction step.

!!! tip "Why radix 4 is the unique sweet spot"
    Plain radix-4 digits ``\{0,1,2,3\}`` would also halve the rows, but the multiple
    ``3A = 2A + A`` needs a real addition *before summation begins*. The signed set caps
    the digit magnitude at ``r/2 = 2``, and ``\{0, \pm A, \pm 2A\}`` are all shifts.
    Generally radix ``2^k`` needs multiples up to ``2^{k-1}A``, shift-only iff
    ``2^{k-1} \le 2``.

## Carry-save and the reduction trees

```@docs
csa
verify_csa_identity
wallace_reduce
dadda_reduce
ReductionTrace
reduction_schedule
dadda_targets
compressor_cells
verify_tree_invariant
booth_tree_saving
```

```@example red
using xpuFP, CairoMakie # hide
plot_wallace_tree([1, 2, 4, 8, 16, 32, 64, 128, 256])
```

The narrowing bundle **is** the tree: trace any final wire upward and its ancestry is a
ternary tree of boxes with the rows as leaves.

```@example red
using xpuFP, CairoMakie # hide
plot_reduction_pyramids(24)
```

!!! note "A Wallace tree is one equation"
    ``\sigma \circ C = \sigma``. Everything else is scheduling. Note that ``u^*`` is
    **not** a closed-form "XOR of everything" — the recursion interleaves parity and
    majority at every level. The theorem's content is precisely that no closed form is
    needed: the *invariant* is the whole proof.
