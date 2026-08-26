# ---------------------------------------------------------------------------
# What redundancy actually costs and saves, in three currencies at once.
#
# The rest of the package reports DEPTH, because depth is what redundancy buys.  This
# file adds the two currencies that decide whether the trade is worth taking:
#
#   STATE  — flip-flops held between cycles.  Redundancy LOSES here, always: a
#            redundant digit needs more bits than a non-redundant one, and carry-save
#            needs two words where binary needs one.  Any honest account leads with it.
#   CELLS  — combinational cells burned.  Booth wins big (half the partial-product
#            rows), carry-free adders lose modestly, CSD wins on constants.
#   DEPTH  — gate levels on the critical path.  This is redundancy's home turf.
#
# ---------------------------------------------------------------------------
# THE MODEL, stated so it can be argued with
# ---------------------------------------------------------------------------
# Encoding, bits per digit:
#   binary radix-4 digit  {0..3}   → 2 bits          (4 symbols)
#   RR4 minimal           {-2..2}  → 3 bits          (5 symbols, one wasted code)
#   RR4 maximal           {-3..3}  → 3 bits          (7 symbols, one wasted code)
#   carry-save                     → 2 bits/bit      (a sum word and a carry word)
#
# Adder cells at width w bits:
#   ripple            w full adders
#   Kogge–Stone       w·⌈log₂w⌉ prefix cells + w XOR
#   carry-save 3:2    w full adders, and no exit
#   RR4 carry-free    3 small cells per radix-4 digit ⇒ 1.5w
#
# Adder depth at width w bits:
#   ripple w+1 · prefix ⌈log₂w⌉ · carry-save 1 · RR4 3
#
# Multiplier n×n:
#   rows          n plain, ⌈(n+1)/2⌉ Booth radix-4
#   PP cells      rows × n select/AND cells (Booth's selects are wider than an AND;
#                 charged at 1.3× to keep the comparison from flattering it)
#   CSA cells     (rows − 2) × 2n
#   recode cells  Booth only, 5 per digit
#   PP storage    rows × 2n bits held in flight.  A sequential shift-add holds ONE row
#                 plus the accumulator (4n); a tree materialises every row at once.
#                 This is the multiplier's "memory", and it is what Booth halves.
#
# Everything here is FIRST ORDER.  It is calibrated to reproduce the two textbook
# facts — Booth halves the rows, carry-free is depth-3 at any width — and to make the
# state penalty visible.  It is not a substitute for synthesis.
# ---------------------------------------------------------------------------

"""    RR4_BITS_PER_DIGIT

3 bits to hold one `{-2..2}` digit — five symbols in a three-bit field, one code wasted.
Against 2 bits for a non-redundant radix-4 digit, this is the **1.5× state penalty**
that every carry-free datapath pays for its depth."""
const RR4_BITS_PER_DIGIT = 3

"""    BINARY_BITS_PER_RADIX4_DIGIT

2 bits for a `{0..3}` digit — the baseline `RR4_BITS_PER_DIGIT` is measured against."""
const BINARY_BITS_PER_RADIX4_DIGIT = 2

"""
    CostBudget

One implementation of one operation, priced in all three currencies.

# Fields
- `op` — `:add`, `:accumulate`, or `:multiply`.
- `method` — the scheme being priced.
- `n` — terms accumulated, or operand width in bits for `:add` / `:multiply`.
- `width` — datapath width in bits.
- `state_bits` — bits held in flight: the accumulator register for `:add`/`:accumulate`,
  the partial products for `:multiply`. **Redundancy's cost on the first two, Booth's
  saving on the third.**
- `cells` — combinational cells.
- `depth` — gate levels on the critical path.
- `redundant::Bool` — whether the scheme keeps a redundant representation.
- `detail::String`

See the source header for the full cost model. Ratios between rows are the intended
reading; absolute numbers are first-order.
"""
struct CostBudget
    op::Symbol
    method::Symbol
    n::Int
    width::Int
    state_bits::Int
    cells::Int
    depth::Int
    redundant::Bool
    detail::String
end

_prefix_cells(w) = w * max(1, ceil(Int, log2(max(2, w)))) + w
_rr4_digits(w) = cld(w, 2)

"""
    add_costs(w::Integer) -> Vector{CostBudget}

Price a **single** `w`-bit addition every way the package implements it.

The headline is that carry-free is the only scheme whose depth does not move with `w`
— and that it pays for that with 1.5× the state and, at small `w`, more cells than a
ripple adder. One addition is also the case where redundancy looks *worst*, because the
exit conversion cannot be amortised over anything.

```julia
julia> [(b.method, b.depth) for b in add_costs(64)]
4-element Vector{Tuple{Symbol, Int64}}:
 (:ripple, 65)
 (:prefix, 6)
 (:carry_save, 1)
 (:rr4_carry_free, 3)
```
"""
function add_costs(w::Integer)
    W = Int(w)
    W > 0 || throw(ArgumentError("add_costs: width must be positive"))
    d = _rr4_digits(W)
    [CostBudget(:add, :ripple, W, W, W, W, W + 1, false,
                "$(W) full adders, carry chain the full width"),
     CostBudget(:add, :prefix, W, W, W, _prefix_cells(W),
                max(1, ceil(Int, log2(max(2, W)))), false,
                "Kogge–Stone: $(_prefix_cells(W)) cells for $(max(1,ceil(Int,log2(max(2,W))))) levels"),
     CostBudget(:add, :carry_save, W, W, 2W, W, 1, true,
                "3:2 compressor, result left redundant in two words"),
     CostBudget(:add, :rr4_carry_free, W, W, d * RR4_BITS_PER_DIGIT, 3d, 3, true,
                "$(d) radix-4 digits × 3 bits = $(d*RR4_BITS_PER_DIGIT) bits of state")]
end

"""
    accumulate_costs(n::Integer, w::Integer; exit=:prefix) -> Vector{CostBudget}

Price accumulating `n` terms into a `w`-bit register.

This is where redundancy earns its keep, because the exit conversion is paid **once**
across `n` terms while the conventional schemes pay a carry chain `n` times. `depth`
here is the whole accumulation, `cells` is the one adder that is reused every cycle,
and `state_bits` is the accumulator register.

Compare against [`accumulate_rr4`](@ref) and [`accumulate_carry`](@ref), which run the
arithmetic for real rather than costing a model of it.
"""
function accumulate_costs(n::Integer, w::Integer; exit::Symbol = :prefix)
    N = Int(n); W = Int(w)
    (N > 0 && W > 0) || throw(ArgumentError("accumulate_costs: n and w must be positive"))
    d = _rr4_digits(W)
    pfx = max(1, ceil(Int, log2(max(2, W))))
    ecpa = exit === :ripple ? W + 1 : pfx
    [CostBudget(:accumulate, :ripple, N, W, W, W, N * (W + 1), false,
                "$(N) × full-width carry chain"),
     CostBudget(:accumulate, :prefix, N, W, W, _prefix_cells(W), N * pfx, false,
                "$(N) × $(pfx) prefix levels"),
     CostBudget(:accumulate, :rr4_carry_free, N, W, d * RR4_BITS_PER_DIGIT, 3d,
                3N + ecpa, true,
                "$(N) × 3 levels + one $(ecpa)-level exit, amortised over $(N) terms")]
end

"""
    multiply_costs(n::Integer) -> Vector{CostBudget}

Price an `n × n` bit multiply: schoolbook, Wallace without recoding, and Wallace with
radix-4 Booth recoding.

Booth is the one place in this whole file where redundancy **saves in every currency at
once**: halving the partial-product rows removes cells and tree levels together, and the
recoded multiplier is consumed immediately rather than stored, so the 3-bits-per-digit
penalty never reaches a flip-flop.

```julia
julia> b = multiply_costs(24);      # an FP32 significand

julia> [(x.method, x.cells) for x in b]
```
"""
function multiply_costs(n::Integer)
    N = Int(n)
    N > 1 || throw(ArgumentError("multiply_costs: n must exceed 1"))
    plain, booth = booth_rows(N)
    pp(rows, wide) = round(Int, rows * N * (wide ? 1.3 : 1.0))   # Booth selects cost more
    csa(rows) = compressor_cells(rows) * 2N
    lv(rows) = length(reduction_schedule(rows)) - 1
    cpa = max(1, ceil(Int, log2(2N)))
    # one row live at a time plus the accumulator, versus every row materialised at once
    [CostBudget(:multiply, :shift_add, N, 2N, 4N, pp(plain, false) + N * (2N),
                plain * cpa, false,
                "$(plain) rows summed sequentially, 1 row live: $(plain) carry-propagate adds"),
     CostBudget(:multiply, :wallace_plain, N, 2N, plain * 2N,
                pp(plain, false) + csa(plain), lv(plain) + cpa, true,
                "$(plain) rows in flight → $(lv(plain)) CSA levels → 1 exit CPA"),
     CostBudget(:multiply, :booth_wallace, N, 2N, booth * 2N,
                pp(booth, true) + csa(booth) + 5 * booth,
                1 + lv(booth) + cpa, true,
                "$(booth) rows in flight (Booth radix-4) → $(lv(booth)) CSA levels → 1 exit CPA")]
end

"""
    constant_multiply_costs(c::Integer, w::Integer=32) -> NamedTuple

Price multiplying a `w`-bit variable by the **constant** `c`, binary shift-add against
its CSD spelling: one adder per nonzero digit beyond the first.

Constants are the case where the optimal signed spelling is affordable, because the
search happens once at design time — which is exactly why silicon ships Booth's fixed
windows for variable operands and CSD's minimal strings for coefficients.

```julia
julia> constant_multiply_costs(231).adders_saved
2
```
"""
function constant_multiply_costs(c::Integer, w::Integer = 32)
    W = Int(w)
    bw = binary_weight(Int(c))
    cw = length(filter(!iszero, csd(Int(c)).digits))
    (constant = Int(c), binary_weight = bw, csd_weight = cw,
     binary_adders = max(bw - 1, 0), csd_adders = max(cw - 1, 0),
     adders_saved = max(bw - 1, 0) - max(cw - 1, 0),
     binary_cells = max(bw - 1, 0) * W, csd_cells = max(cw - 1, 0) * W,
     cell_ratio = bw <= 1 ? 1.0 : max(cw - 1, 0) / max(bw - 1, 1))
end

# ---- reporting -------------------------------------------------------------

function _print_budgets(budgets::Vector{CostBudget}, baseline::Symbol; io::IO = stdout)
    b = budgets[findfirst(x -> x.method === baseline, budgets)]
    @printf(io, "%-18s %8s %10s %8s   %9s %9s %9s\n",
            "method", "state b", "cells", "depth", "state ×", "cells ×", "depth ×")
    println(io, "─"^80)
    for x in budgets
        ratio(v, base) = base == 0 ? nothing : v / base
        mark(v) = v === nothing ? "        —" :
                  v < 0.995 ? @sprintf("%8.2f↓", v) : v > 1.005 ? @sprintf("%8.2f↑", v) :
                  @sprintf("%8.2f ", v)
        @printf(io, "%-18s %8d %10d %8d   %s %s %s\n",
                String(x.method), x.state_bits, x.cells, x.depth,
                mark(ratio(x.state_bits, b.state_bits)),
                mark(ratio(x.cells, b.cells)), mark(ratio(x.depth, b.depth)))
    end
    println(io, "(↓ cheaper than $(baseline), ↑ dearer; state is flip-flops, cells combinational)")
    budgets
end

"""
    cost_report(; w=64, n=1024, mulbits=24, io=stdout) -> NamedTuple

The whole savings picture in one call: addition, accumulation and multiplication priced
side by side against their conventional baselines.

Read it as three separate verdicts, because they genuinely differ:

- **Addition** — carry-free wins depth and loses state. On a single add there is
  nothing to amortise the exit conversion against, so it is the weakest case.
- **Accumulation** — carry-free wins depth by a wide margin *because* the exit is paid
  once across `n` terms. Still 1.5× the state.
- **Multiplication** — Booth recoding wins state, cells and depth simultaneously. It is
  the only unambiguous win in the file, and it is why every shipping MAC uses it.
"""
function cost_report(; w::Integer = 64, n::Integer = 1024, mulbits::Integer = 24,
                     io::IO = stdout)
    println(io, "="^80)
    println(io, "  ADDITION — one $(w)-bit add (baseline: ripple)")
    println(io, "="^80)
    a = _print_budgets(add_costs(w), :ripple; io)

    println(io, "\n", "="^80)
    println(io, "  ACCUMULATION — $(n) terms into $(w) bits (baseline: ripple)")
    println(io, "="^80)
    b = _print_budgets(accumulate_costs(n, w), :ripple; io)

    println(io, "\n", "="^80)
    println(io, "  MULTIPLICATION — $(mulbits)×$(mulbits) bits (baseline: shift-add)")
    println(io, "="^80)
    c = _print_budgets(multiply_costs(mulbits), :shift_add; io)

    println(io, "\n", "="^80)
    println(io, "  THE VERDICT, per currency")
    println(io, "="^80)
    rr4a = a[findfirst(x -> x.method === :rr4_carry_free, a)]
    rip  = a[1]
    rr4b = b[findfirst(x -> x.method === :rr4_carry_free, b)]
    ripb = b[1]
    bw   = c[findfirst(x -> x.method === :booth_wallace, c)]
    wp   = c[findfirst(x -> x.method === :wallace_plain, c)]
    @printf(io, "  state   : carry-free costs %.2f× a binary register — redundancy never saves memory\n",
            rr4a.state_bits / rip.state_bits)
    @printf(io, "  cells   : Booth cuts the multiplier array to %.2f× of plain Wallace\n",
            bw.cells / wp.cells)
    @printf(io, "  rows    : Booth halves partial products, %d → %d, so PP storage drops %.2f×\n",
            wp.state_bits ÷ (2 * mulbits), bw.state_bits ÷ (2 * mulbits),
            wp.state_bits / bw.state_bits)
    @printf(io, "  depth   : accumulation %.1f× shallower carry-free (%d vs %d levels)\n",
            ripb.depth / rr4b.depth, ripb.depth, rr4b.depth)
    (add = a, accumulate = b, multiply = c)
end

"""
    format_memory_costs(nvalues::Integer=4096*4096; io=stdout) -> Vector{NamedTuple}

Where the memory savings in this package actually come from — **not** from redundancy,
which costs state, but from narrowing the *format*.

Prints storage for one weight matrix across the FP formats, so the two effects can be
seen at their true relative sizes: block-scaled FP4 saves ~7.5× of memory, while a
carry-free datapath spends 1.5× more state on the accumulator that consumes it.
"""
function format_memory_costs(nvalues::Integer = 4096 * 4096; io::IO = stdout)
    rows = NamedTuple[]
    @printf(io, "  %-12s %10s %12s %10s\n", "format", "bits/val", "MB", "vs FP32")
    println(io, "  " * "─"^48)
    for f in (FP32, FP16, BF16, E4M3, MXFP4, NVFP4, MXINT4)
        bits = f isa BlockFormat ? bits_per_element(f) : Float64(nbits(f))
        mb = storage_bytes(f, nvalues) / 1e6
        ratio = 32 / bits
        @printf(io, "  %-12s %10.3f %12.2f %9.2f×\n", _fmtname(f), bits, mb, ratio)
        push!(rows, (format = _fmtname(f), bits = bits, megabytes = mb, vs_fp32 = ratio))
    end
    rows
end
