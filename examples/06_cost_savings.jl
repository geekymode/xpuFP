#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Example 6 — what Booth, CSD and RR4 cost and save, in three currencies.
#
#   julia --project=. examples/06_cost_savings.jl
#
# Depth is the currency the rest of the package reports, because depth is what
# redundancy buys.  This script adds the two that decide whether to take the
# trade: STATE (flip-flops) and CELLS (combinational area).
# ---------------------------------------------------------------------------
using xpuFP, Printf, Random

cost_report(w = 64, n = 1024, mulbits = 24)

println("\n", "="^80)
println("  BOOTH RECODING vs OPERAND WIDTH  (measured, not modelled)")
println("="^80)
@printf("  %6s  %6s %6s   %7s %7s   %7s %7s   %8s\n",
        "n bits", "plain", "booth", "levels", "levels", "cells", "cells", "cell")
@printf("  %6s  %6s %6s   %7s %7s   %7s %7s   %8s\n",
        "", "rows", "rows", "plain", "booth", "plain", "booth", "ratio")
println("  " * "─"^70)
for n in (8, 16, 24, 32, 53, 64)
    s = booth_tree_saving(n)
    @printf("  %6d  %6d %6d   %7d %7d   %7d %7d   %7.2f×\n",
            n, s.plain_rows, s.booth_rows, s.plain_levels, s.booth_levels,
            s.plain_cells, s.booth_cells, s.cell_ratio)
end
println("\n  Rows halve exactly; CSA levels fall by 1-2; cells fall with the rows.")
println("  The recoded digits are consumed immediately, never stored — which is why")
println("  Booth escapes the 3-bits-per-digit penalty that the accumulator pays.")

println("\n", "="^80)
println("  CSD CONSTANT MULTIPLIERS  (adders saved, per constant)")
println("="^80)
@printf("  %10s  %8s %6s   %8s %6s   %8s\n",
        "constant", "binary", "adds", "CSD", "adds", "saved")
println("  " * "─"^60)
tot_b = tot_c = 0
for c in (231, 255, 1023, 4095, 12345, 65535, 1_000_003)
    r = constant_multiply_costs(c)
    global tot_b += r.binary_adders; global tot_c += r.csd_adders
    @printf("  %10d  %8d %6d   %8d %6d   %8d\n",
            c, r.binary_weight, r.binary_adders, r.csd_weight, r.csd_adders,
            r.adders_saved)
end
@printf("\n  total adders: binary %d → CSD %d   (%.2f×, %d saved)\n",
        tot_b, tot_c, tot_c / tot_b, tot_b - tot_c)
println("  Runs of 1s are where CSD pays: 2^k−1 needs k adders in binary, 1 in CSD.")

println("\n", "="^80)
println("  MEMORY — the savings are in the FORMAT, not in the redundancy")
println("="^80)
format_memory_costs(4096 * 4096)
println("\n  One 4096×4096 weight matrix.  Narrowing the format saves 7.5×; a")
println("  carry-free accumulator downstream SPENDS 1.5× more state.  Both are real,")
println("  they act on different memories, and only the first is measured in MB.")

println("\n", "="^80)
println("  PUTTING IT TOGETHER — one MXFP4 dot product of length 4096")
println("="^80)
K, N = 32, 4096
elem_bits = bits_per_element(MXFP4)
@printf("  operand storage : %d values × %.2f b = %.1f KB   (FP32: %.1f KB, %.2f× less)\n",
        N, elem_bits, N * elem_bits / 8 / 1024, N * 32 / 8 / 1024, 32 / elem_bits)
mul4 = multiply_costs(4)
mul24 = multiply_costs(24)
b4 = mul4[findfirst(x -> x.method === :booth_wallace, mul4)]
b24 = mul24[findfirst(x -> x.method === :booth_wallace, mul24)]
@printf("  multiplier cells: E2M1 4×4 %d vs FP32 24×24 %d   → %.0f× smaller\n",
        b4.cells, b24.cells, b24.cells / b4.cells)
@printf("  core sum bound  : %d, so the adder tree is exact fixed point — no rounding\n",
        Int(core_sum_bound(MXFP4)))
acc = accumulate_costs(N ÷ K, 16)
rip, rr4 = acc[1], acc[end]
@printf("  scale fixups    : %d (one per block of %d), each a small multiply\n", N ÷ K, K)
@printf("  fixup accumulate: ripple %d levels vs carry-free %d   → %.1f× shallower\n",
        rip.depth, rr4.depth, rip.depth / rr4.depth)
println("\n  Three independent savings, in three different currencies, none of which")
println("  is a substitute for either of the others.")
