#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Example 7 — with weights already in MXFP4, what is RR4 worth?
#
#   julia --project=. examples/07_mxfp4_rr4.jl
#   julia --project=. examples/07_mxfp4_rr4.jl 8192
#
# The redundant-arithmetic chapters are written at general word widths.  MXFP4
# changes the widths — 4-bit elements, 9-bit products, 14-bit block sums — and
# that changes the answer.  This script measures the answer rather than assuming
# it, and the answer is mostly "no".
# ---------------------------------------------------------------------------
using xpuFP, Printf

N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4096

rr4_opportunity(Ns = (256, 1024, N, 4N))

println("\n", "="^84)
println("  7. SCALING LAWS — how each stage grows with N")
println("="^84)
println("""
  ALL NUMBERS BELOW ARE GATE LEVELS ON THE CRITICAL PATH — logic depth, i.e.
  latency.  They are NOT counts of adders, multipliers, gates or operations.
  A dot product of N=65536 does 65536 multiplies however you schedule it; what
  the 38 measures is how many gate delays separate the inputs from the answer.
  For counts, see cost_report()'s `cells` column and mxfp4_multiply_options().
""")
println("""
  dot product, K=32 blocks, tree schedule
    products      O(1)        depth 2, all N in parallel
    block reduce  O(1)        depth 12, all N/K blocks in parallel
    cross-block   O(log N)    the only stage that grows
    ⇒ total depth O(log N); RR4 changes the CONSTANT, not the order

  soft-max
    max           O(log N)    comparisons — RR4 hostile (sign detection)
    exp           O(1)        per element, LUT
    sum           O(log N)    RR4 addressable
    divide        O(1)        one reciprocal + N parallel multiplies
    ⇒ total depth O(log N); RR4 addresses about a quarter of it
""")
@printf("  %8s | %10s %10s | %10s %10s     (gate levels)\n",
        "N", "dot conv", "dot RR4", "smax conv", "smax RR4")
println("  " * "─"^58)
for n in (64, 256, 1024, 4096, 16384, 65536)
    d = mxfp4_dot_costs(n)
    s = mxfp4_softmax_costs(n)
    @printf("  %8d | %10d %10d | %10d %10d\n",
            n, d.total_conv, d.total_rr4, s.total_conv, s.total_rr4)
end
println("\n  Both columns grow like log N.  RR4 is above the conventional column in")
println("  every row, for both kernels.  It is not a scaling win; it is a constant-")
println("  factor loss that log N slowly dilutes.")

println("\n", "="^84)
println("  8. THE AMDAHL CEILING — best case even if RR4 addition were FREE")
println("="^84)
println("  (ceilings are depth ratios, not throughput)")
@printf("  %8s | %12s %12s %12s\n", "N", "dot addr.", "dot ceiling", "smax ceiling")
println("  " * "─"^54)
for n in (256, 1024, 4096, 16384)
    d = mxfp4_dot_costs(n)
    s = mxfp4_softmax_costs(n)
    dceil = d.total_conv / (d.total_conv - d.cross_conv)
    sceil = s.total_conv / (s.total_conv - s.sum_conv)
    @printf("  %8d | %11.1f%% %11.2f× %11.2f×\n",
            n, 100d.rr4_addressable, dceil, sceil)
end
println("\n  Set RR4's addition cost to ZERO and these are the speedups you would get.")
println("  They are the ceiling on any redundant scheme, RR4 or otherwise, because")
println("  the rest of the depth is products, exp, comparisons and the exit CPA.")

println("\n", "="^84)
println("  9. WHAT TO DO INSTEAD")
println("="^84)
println("""
  The MXFP4 saving is already banked, and it is memory:""")
format_memory_costs(4096 * 4096)
println("""
  7.5× off the weights, before any arithmetic is considered.  Token generation
  is memory-bound, so that compression is close to a proportional speedup.

  On the arithmetic side the wins available are, in order of size:
    1. the element multiply is a 64-entry ROM        — no multiplier at all
    2. the block reduction is a carry-save tree      — already standard
    3. the scale is an E8M0 exponent ADD             — no multiply per block
    4. the core sum is exact fixed point             — no rounding, no normaliser

  None of those is RR4.  RR4's contribution to an MXFP4 inference datapath, on
  the evidence above, is a 25% state saving in a streaming cross-block
  accumulator, against a carry-save accumulator that is 2.7× shallower.
""")
