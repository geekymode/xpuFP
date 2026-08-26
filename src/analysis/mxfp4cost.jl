# ---------------------------------------------------------------------------
# What RR4 is worth once the weights are already MXFP4.
#
# The redundant-arithmetic chapters of this package are written at general word
# widths, where carry-free addition's constant depth 3 beats a ripple's w+1 by a mile.
# MXFP4 changes the question, because it changes the widths:
#
#   E2M1 element   ×2 ⇒ integer in {0,±1,±2,±3,±4,±6,±8,±12}   4 magnitude bits
#   one product    m_a·m_b ∈ ±144                               9 bits signed
#   K=32 core sum  |Σ| ≤ 32·144 = 4608                          14 bits signed
#
# Every operand in the MXFP4 inner loop is 4 to 14 bits wide.  Two consequences run
# through this whole file, and both cut against RR4:
#
#   1. A Kogge-Stone adder at 14 bits is ⌈log₂14⌉ = 4 levels deep.  Carry-free is 3.
#      The margin is 1.33x, not the 20x that a 64-bit sequential accumulator shows.
#   2. Reduction trees do not want carry-free addition at all.  A 3:2 compressor is
#      ONE gate level per reduction level; an RR4 add is three.  For K = 32 products
#      that is 8 levels against 15 — carry-save wins by 1.9x, and Wallace trees have
#      used it since 1964.
#
#      This holds for SEQUENTIAL accumulation too, and it is the finding that decides
#      the question: a carry-save accumulator absorbs one term per gate level —
#      (S,C) + x = csa(S,C,x) — where RR4 needs three.  Carry-save dominates RR4 on
#      depth in every schedule.  RR4's remaining advantages over it are real but
#      narrow: 1.5w bits of state against carry-save's 2w, and a genuine positional
#      representation with bounded digits, which can be shifted, truncated and
#      indexed, where a (sum, carry) pair cannot.
#
# So this file exists to say where RR4 does and does not pay on MXFP4 data, with the
# arithmetic widths derived from the format rather than assumed.  The honest headline:
# on DEPTH there is no configuration measured here in which RR4 beats carry-save.  Its
# best case is against a canonical adder in a wide sequential accumulator (~1.9x), and
# even there carry-save is ahead.  What RR4 genuinely offers is 1.5w state against
# carry-save's 2w, and a positional representation that a (sum, carry) pair is not.
#
# Cost model as in `analysis/costmodel.jl`: ripple w+1, prefix ⌈log₂w⌉, RR4 3,
# 3:2 compressor 1 per level.  Sign detection is modelled below, and is the reason
# soft-max resists redundancy.
# ---------------------------------------------------------------------------

"""
    mxfp4_widths(bf::BlockFormat = MXFP4) -> NamedTuple

The bit widths MXFP4 arithmetic actually needs, derived from the format rather than
assumed.

E2M1's grid is `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`; doubling it gives the integers
`{0, ±1, ±2, ±3, ±4, ±6, ±8, ±12}`, so an element is a **4-bit magnitude plus sign**,
a product fits in **9 bits**, and a `K`-element core sum fits in
`⌈log₂(K·144)⌉ + 1` bits — **14 for K = 32**.

Everything downstream follows from these being small.

```jldoctest
julia> w = mxfp4_widths();

julia> w.elem_mag_bits, w.product_bits, w.core_bits
(4, 9, 14)
```
"""
function mxfp4_widths(bf::BlockFormat = MXFP4)
    g = filter(isfinite, grid(bf.elem))
    scaled = Int.(round.(g .* 2))
    all(abs.(g .* 2 .- scaled) .< 1e-12) ||
        throw(ArgumentError("mxfp4_widths: $(bf.elem.name) is not a half-integer grid"))
    mmax = maximum(abs, scaled)
    pmax = mmax^2
    csum = bf.K * pmax
    (format = bf.name, K = bf.K,
     elem_ints = sort(unique(scaled)),
     elem_mag_bits = ndigits(mmax; base = 2),
     elem_bits = ndigits(mmax; base = 2) + 1,
     product_max = pmax, product_bits = ndigits(pmax; base = 2) + 1,
     core_sum_max = csum, core_bits = ndigits(csum; base = 2) + 1,
     distinct_products = length(unique(a * b for a in scaled, b in scaled)),
     lut_entries = length(unique(abs, scaled))^2)
end

_csa_levels(rows) = length(reduction_schedule(Int(rows))) - 1
_rr4_tree_depth(rows) = 3 * max(1, ceil(Int, log2(max(2, Int(rows)))))

"""
    sign_detect_depth(w::Integer; redundant::Bool) -> Int

Gate levels to learn the **sign** of a `w`-bit value.

Conventional two's complement: **0** — the sign *is* the top bit, already there.

Redundant signed-digit: the sign is that of the most significant **non-zero** digit
(the bound `|Σ_{i<k} d_i r^i| < r^k` makes lower digits unable to overturn it), but
finding that digit is a priority encode over all `⌈w/2⌉` digits, so
`⌈log₂⌈w/2⌉⌉` levels.

This asymmetry is why redundancy helps addition and **not** comparison — and therefore
not soft-max, whose first phase is a max reduction. A redundant datapath removes the
carry chain from `+` and leaves it in `<`.
"""
sign_detect_depth(w::Integer; redundant::Bool) =
    redundant ? max(1, ceil(Int, log2(max(2, cld(Int(w), 2))))) : 0

"""
    mxfp4_multiply_options(bf::BlockFormat = MXFP4) -> Vector{NamedTuple}

Price one E2M1 × E2M1 element product four ways: a plain array, Booth radix-4, a
lookup table, and RR4 digit-serial.

At this size the table wins outright. E2M1 has 8 distinct magnitudes, so the whole
multiplier is a **64-entry ROM** — and there are only 37 distinct signed products in
the entire format. Booth recoding a 5-bit operand saves two rows and spends recode
logic doing it; RR4 recoding costs more than the multiply.

**Redundancy has nothing to sell at 4 bits.** The interesting question for MXFP4 is
what happens *after* the products, which is [`mxfp4_reduction_options`](@ref).
"""
function mxfp4_multiply_options(bf::BlockFormat = MXFP4)
    w = mxfp4_widths(bf)
    n = w.elem_bits
    plain, booth = booth_rows(n)
    [(method = :array, rows = plain, depth = _csa_levels(plain) + 3,
      cells = plain * n, note = "$(plain)-row array, no recoding"),
     (method = :booth_r4, rows = booth, depth = 1 + _csa_levels(booth) + 3,
      cells = round(Int, booth * n * 1.3) + 5booth,
      note = "$(booth) rows + recode; the recode costs as much as it saves here"),
     (method = :lut, rows = 0, depth = 2, cells = w.lut_entries,
      note = "$(w.lut_entries)-entry ROM over 8 magnitudes; $(w.distinct_products) distinct products exist"),
     (method = :rr4_digits, rows = cld(n, 2)^2, depth = 1 + _rr4_tree_depth(cld(n, 2)^2) + 3,
      cells = cld(n, 2)^2 * 4,
      note = "$(cld(n,2))×$(cld(n,2)) digit products — recoding a 4-bit operand is pure overhead")]
end

"""
    mxfp4_reduction_options(K::Integer = 32; acc_bits = nothing) -> Vector{NamedTuple}

Reducing `K` element products to one sum — the heart of a block MAC — priced in
**gate levels**. `levels` counts reduction stages; `depth` is gate delays, which is
`levels` times the cost of one stage plus the exit conversion.

This is the comparison that decides RR4's fate inside MXFP4, and it goes against it:

| scheme | levels | why |
|:---|---:|:---|
| 3:2 carry-save (Wallace) | `1` per level | a compressor is one gate deep |
| RR4 carry-free | `3` per level | sum, transfer, digit |
| sequential prefix adds | `⌈log₂w⌉` per level | full carry resolution each time |

Carry-save **is** a redundant representation — it just happens to be the cheapest one
for a reduction, because it never needs its digits back in range until the exit. RR4
buys bounded digits, and a reduction tree has no use for them.
"""
function mxfp4_reduction_options(K::Integer = 32; acc_bits = nothing)
    Kk = Int(K)
    w = acc_bits === nothing ? mxfp4_widths(MXFP4).core_bits : Int(acc_bits)
    lv = _csa_levels(Kk)
    pfx = max(1, ceil(Int, log2(max(2, w))))
    exitcpa = pfx
    [(method = :carry_save_tree, levels = lv, depth = lv + exitcpa,
      note = "$(lv) 3:2 levels @ 1 gate + one $(exitcpa)-level exit CPA"),
     (method = :rr4_tree, levels = ceil(Int, log2(Kk)), depth = _rr4_tree_depth(Kk) + exitcpa,
      note = "$(ceil(Int,log2(Kk))) levels @ 3 gates + one exit conversion"),
     (method = :prefix_tree, levels = ceil(Int, log2(Kk)),
      depth = ceil(Int, log2(Kk)) * pfx,
      note = "$(ceil(Int,log2(Kk))) levels @ $(pfx) gates, canonical throughout"),
     (method = :sequential_prefix, levels = Kk - 1, depth = (Kk - 1) * pfx,
      note = "no tree: $(Kk-1) dependent adds")]
end

"""
    mxfp4_dot_costs(N::Integer; bf = MXFP4, acc_bits = 32, schedule = :tree) -> NamedTuple

**Gate levels on the critical path** for a length-`N` MXFP4 dot product, broken into the
four stages that actually happen, so the RR4-addressable share can be read off rather
than guessed.

Every number returned is a **depth**, not a count. The dot product performs `N`
multiplies and `N-1` additions whichever datapath is chosen; what changes is how many
gate delays separate the inputs from the answer. For counts see
[`mxfp4_multiply_options`](@ref) (`rows`, `cells`) and [`CostBudget`](@ref).

```
  products      N element multiplies, fully parallel
  block reduce  N/K independent K-product reductions
  scale         one E8M0 exponent add per block — free, it is an integer add on exponents
  cross-block   combine N/K block results at `acc_bits`  ← the only RR4-addressable stage
```

`schedule` picks how the cross-block stage runs: `:tree` (the usual) or `:sequential`
(a streaming accumulator with a loop-carried dependence).

Returns per-stage depths for **three genuinely different** cross-block datapaths —
`canonical` (no redundancy anywhere), `rr4`, and `carry_save` — plus
`rr4_addressable`, the fraction of total depth RR4 can even touch, which is the Amdahl
ceiling on any speedup it could deliver.

`speedup` is RR4 against canonical, and it exceeds 1: carry-free genuinely beats a
prefix adder here. `rr4_vs_cs` is the number that actually decides the question, and it
does not: carry-save absorbs one term per gate level against RR4's three, so it leads in
**both** schedules. Real hardware uses carry-save, so that is the comparison that
matters.
"""
function mxfp4_dot_costs(N::Integer; bf::BlockFormat = MXFP4, acc_bits::Integer = 32,
                         schedule::Symbol = :tree)
    Nn = Int(N); K = bf.K
    nblocks = cld(Nn, K)
    w = mxfp4_widths(bf)
    aw = Int(acc_bits)
    pfx = max(1, ceil(Int, log2(max(2, aw))))

    d_prod = 2                                    # the 64-entry LUT
    d_block = _csa_levels(K) + max(1, ceil(Int, log2(max(2, w.core_bits))))
    d_scale = 1                                   # E8M0 is an exponent add

    # three genuinely different cross-block datapaths:
    #   canonical  — no redundancy anywhere, a prefix adder at every combine
    #   rr4        — carry-free, 3 levels per combine, one exit conversion
    #   carry-save — 1 level per combine, one exit CPA; what real hardware does
    d_cross_conv, d_cross_rr4, d_cross_cs = if schedule === :tree
        lv = max(1, ceil(Int, log2(max(2, nblocks))))
        (lv * pfx, _rr4_tree_depth(nblocks) + pfx, _csa_levels(nblocks) + pfx)
    elseif schedule === :sequential
        ((nblocks - 1) * pfx, 3 * (nblocks - 1) + pfx, (nblocks - 1) + pfx)
    else
        throw(ArgumentError("mxfp4_dot_costs: schedule must be :tree or :sequential"))
    end

    conv = d_prod + d_block + d_scale + d_cross_conv
    rr4  = d_prod + d_block + d_scale + d_cross_rr4
    cs   = d_prod + d_block + d_scale + d_cross_cs
    (N = Nn, K = K, blocks = nblocks, schedule = schedule, acc_bits = aw,
     products = d_prod, block_reduce = d_block, scale = d_scale,
     cross_conv = d_cross_conv, cross_rr4 = d_cross_rr4, cross_cs = d_cross_cs,
     total_conv = conv, total_rr4 = rr4, total_cs = cs,
     rr4_addressable = d_cross_conv / conv,
     speedup = conv / rr4, speedup_cs = conv / cs,
     rr4_vs_cs = cs / rr4)
end

"""
    mxfp4_softmax_costs(N::Integer; w = 16, redundant = false) -> NamedTuple

**Gate levels on the critical path** for a length-`N` soft-max, by phase, with each
phase marked for whether a redundant representation can help it.

As with [`mxfp4_dot_costs`](@ref) these are depths, not counts: the operation count is
`N` exponentials and `N-1` additions either way.

```
  max      N-1 comparisons, tree      ← RR4 HOSTILE: needs sign detection
  exp      per-element LUT + poly     ← unaffected; not an addition
  sum      N-1 adds, tree             ← RR4 addressable
  divide   reciprocal + N multiplies  ← needs a canonical divisor anyway
```

The max phase is the problem. A comparison is a subtract followed by a **sign test**,
and [`sign_detect_depth`](@ref) is 0 for two's complement and `⌈log₂⌈w/2⌉⌉` for a
redundant string. Carry-free addition makes the subtract cheaper and the test dearer,
and on narrow soft-max operands those cancel.

So soft-max's RR4-addressable fraction is **just the sum phase**, and the returned
`rr4_addressable` is the Amdahl ceiling on any win.
"""
function mxfp4_softmax_costs(N::Integer; w::Integer = 16, redundant::Bool = false)
    Nn = Int(N); ww = Int(w)
    lv = max(1, ceil(Int, log2(max(2, Nn))))
    pfx = max(1, ceil(Int, log2(max(2, ww))))

    # a comparison is subtract + sign test
    sub_conv, sub_rr4 = pfx, 3
    sgn_conv = sign_detect_depth(ww; redundant = false)
    sgn_rr4  = sign_detect_depth(ww; redundant = true)
    d_max_conv = lv * (sub_conv + sgn_conv)
    d_max_rr4  = lv * (sub_rr4 + sgn_rr4)

    d_exp = 6                                  # LUT + one poly step; identical either way
    d_sum_conv = _csa_levels(Nn) + pfx         # carry-save tree, the conventional best
    d_sum_rr4  = _rr4_tree_depth(Nn) + pfx
    d_div = 8 + pfx                            # reciprocal then a multiply

    conv = d_max_conv + d_exp + d_sum_conv + d_div
    rr4  = d_max_rr4 + d_exp + d_sum_rr4 + d_div
    (N = Nn, w = ww,
     max_conv = d_max_conv, max_rr4 = d_max_rr4,
     exp = d_exp, sum_conv = d_sum_conv, sum_rr4 = d_sum_rr4, divide = d_div,
     total_conv = conv, total_rr4 = rr4,
     rr4_addressable = d_sum_conv / conv,
     speedup = conv / rr4,
     sign_penalty = sgn_rr4 - sgn_conv)
end

"""
    rr4_opportunity(; Ns = (256, 1024, 4096, 16384), bf = MXFP4, acc_bits = 32,
                    io = stdout) -> NamedTuple

The whole verdict in one call: where RR4 pays on MXFP4 data, where it does not, and the
Amdahl ceiling on the best case.

Prints five sections — operand widths, element multiply, block reduction, dot product
scaling in `N`, soft-max scaling in `N` — and closes with the one configuration in
which RR4 genuinely wins.
"""
function rr4_opportunity(; Ns = (256, 1024, 4096, 16384), bf::BlockFormat = MXFP4,
                         acc_bits::Integer = 32, io::IO = stdout)
    w = mxfp4_widths(bf)
    println(io, "="^84)
    println(io, "  UNITS: every `depth` figure below is GATE LEVELS ON THE CRITICAL PATH")
    println(io, "  — logic depth, i.e. latency.  Not a count of adders, multipliers, gates")
    println(io, "  or operations.  `cells` and `rows` are the counts; `state` is flip-flops.")
    println(io, "="^84)
    println(io, "  1. THE WIDTHS — everything below follows from these being small")
    println(io, "="^84)
    @printf(io, "  %s element ×2 → integers %s\n", bf.elem.name, w.elem_ints)
    @printf(io, "  element      : %d magnitude bits + sign = %d\n", w.elem_mag_bits, w.elem_bits)
    @printf(io, "  one product  : |m_a·m_b| ≤ %d → %d bits   (%d distinct products exist)\n",
            w.product_max, w.product_bits, w.distinct_products)
    @printf(io, "  K=%d core sum : |Σ| ≤ %d → %d bits\n", w.K, w.core_sum_max, w.core_bits)
    @printf(io, "\n  At %d bits a prefix adder is %d levels deep.  Carry-free is 3.\n",
            w.core_bits, max(1, ceil(Int, log2(w.core_bits))))
    println(io, "  That 1.3× is the entire margin RR4 has to work with inside a block.")

    println(io, "\n", "="^84)
    println(io, "  2. ELEMENT MULTIPLY — nothing to sell at 4 bits (depth = levels, cells = count)")
    println(io, "="^84)
    @printf(io, "  %-14s %6s %8s %8s  %s\n", "method", "rows", "depth", "cells", "note")
    println(io, "  " * "─"^80)
    for o in mxfp4_multiply_options(bf)
        @printf(io, "  %-14s %6d %8d %8d  %s\n", o.method, o.rows, o.depth, o.cells, o.note)
    end

    println(io, "\n", "="^84)
    println(io, "  3. BLOCK REDUCTION — carry-save beats RR4, and by a lot (gate levels)")
    println(io, "="^84)
    @printf(io, "  %-20s %8s %8s  %s\n", "method", "levels", "depth", "note")
    println(io, "  " * "─"^80)
    ro = mxfp4_reduction_options(bf.K)
    for o in ro
        @printf(io, "  %-20s %8d %8d  %s\n", o.method, o.levels, o.depth, o.note)
    end
    cs = ro[1]; rr = ro[2]
    @printf(io, "\n  carry-save is %.2f× shallower than RR4 for the same reduction.\n",
            rr.depth / cs.depth)
    println(io, "  A 3:2 compressor is one gate level; an RR4 add is three.  A reduction")
    println(io, "  tree never needs digits back in range, so RR4's bounded digits are")
    println(io, "  a feature it pays for and cannot use.")

    println(io, "\n", "="^84)
    println(io, "  4. DOT PRODUCT vs N — three datapaths, tree schedule (gate levels)")
    println(io, "="^84)
    @printf(io, "  %7s %7s | %6s %8s | %9s %8s %10s | %9s %9s\n",
            "N", "blocks", "prod", "blk red", "canonical", "RR4", "carry-save",
            "RR4 v can", "RR4 v CS")
    println(io, "  " * "─"^92)
    dots = []
    for N in Ns
        d = mxfp4_dot_costs(N; bf, acc_bits, schedule = :tree)
        push!(dots, d)
        @printf(io, "  %7d %7d | %6d %8d | %9d %8d %10d | %8.2f× %8.2f×\n",
                d.N, d.blocks, d.products, d.block_reduce,
                d.total_conv, d.total_rr4, d.total_cs, d.speedup, d.rr4_vs_cs)
    end
    println(io, "\n  RR4 beats a canonical prefix tree — carry-free addition works.  It loses")
    println(io, "  to carry-save, which is what the hardware already does, because a 3:2")
    println(io, "  compressor is one gate level per combine and an RR4 add is three.")

    println(io, "\n", "="^84)
    println(io, "  5. SOFT-MAX vs N — the max phase blocks redundancy (gate levels)")
    println(io, "="^84)
    @printf(io, "  %7s | %8s %8s | %6s %8s %8s %8s | %9s %8s\n",
            "N", "max cv", "max rr4", "exp", "sum cv", "sum rr4", "divide", "address.", "speedup")
    println(io, "  " * "─"^92)
    for N in Ns
        s = mxfp4_softmax_costs(N)
        @printf(io, "  %7d | %8d %8d | %6d %8d %8d %8d | %8.1f%% %8.2f×\n",
                s.N, s.max_conv, s.max_rr4, s.exp, s.sum_conv, s.sum_rr4, s.divide,
                100s.rr4_addressable, s.speedup)
    end
    s0 = mxfp4_softmax_costs(first(Ns))
    @printf(io, "\n  Sign detection costs %+d levels per comparison in redundant form:\n",
            s0.sign_penalty)
    println(io, "  two's complement HAS the sign as its top bit; a signed-digit string")
    println(io, "  must priority-encode to its leading non-zero digit.  Carry-free makes")
    println(io, "  the subtract cheaper and the test dearer, and on soft-max they cancel.")

    println(io, "\n", "="^84)
    println(io, "  6. THE BEST CASE FOR RR4 — and the incumbent it still loses to")
    println(io, "="^84)
    println(io, "  Streaming accumulation of block results: no tree available, loop-carried,")
    println(io, "  wide accumulator.  This is the shape carry-free was invented for.")
    @printf(io, "\n  %9s | %10s %10s %10s | %11s %11s\n",
            "acc bits", "canonical", "RR4", "carry-save", "RR4 vs canon", "RR4 vs CS")
    println(io, "  " * "─"^70)
    best = nothing
    for aw in (16, 32, 48, 64)
        d = mxfp4_dot_costs(4096; bf, acc_bits = aw, schedule = :sequential)
        @printf(io, "  %9d | %10d %10d %10d | %10.2f× %10.2f×\n",
                aw, d.total_conv, d.total_rr4, d.total_cs, d.speedup, d.rr4_vs_cs)
        best = d
    end
    @printf(io, "\n  Against a CANONICAL adder RR4 wins %.2f× at %d bits — a real result.\n",
            best.speedup, best.acc_bits)
    @printf(io, "  Against CARRY-SAVE it loses %.2f×, because a 3:2 accumulator absorbs one\n",
            1 / best.rr4_vs_cs)
    println(io, "  term per gate level, (S,C)+x = csa(S,C,x), where RR4 needs three.")
    println(io, "\n  Carry-save is itself a redundant scheme, and it is the incumbent: it is")
    println(io, "  what Wallace trees and MAC accumulators have used since 1964.  So the")
    println(io, "  honest question is never RR4 vs binary, it is RR4 vs carry-save, and on")
    println(io, "  DEPTH carry-save leads in every schedule measured here.")
    println(io, "\n  What RR4 still has over carry-save, and it is not nothing:")
    @printf(io, "    state     %d bits vs %d — 1.5w against 2w, a %.0f%% saving\n",
            (3 * cld(best.acc_bits, 2)), 2 * best.acc_bits,
            100 * (1 - 3 * cld(best.acc_bits, 2) / (2 * best.acc_bits)))
    println(io, "    position  bounded digits form a real positional number: shiftable,")
    println(io, "              truncatable, indexable.  A (sum, carry) pair is none of these")
    println(io, "              until the exit CPA has run.")
    (widths = w, multiply = mxfp4_multiply_options(bf), reduction = ro,
     dots = dots, best_sequential = best)
end
