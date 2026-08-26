#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Example 5 — an accumulator, and whether carry-free addition actually scales.
#
#   julia --project=. examples/05_accumulator.jl
#   julia --project=. examples/05_accumulator.jl 4096
#   julia --project=. examples/05_accumulator.jl 1024 positive
#
# An accumulator is the hardest case for carry propagation: `acc += x` N times,
# strictly sequential, with the register widening as the sum grows.  Conventional
# adders pay a carry chain every step and the chain gets longer.  RR4 pays none.
# ---------------------------------------------------------------------------
using xpuFP, Printf, Random

N    = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1024
KIND = length(ARGS) >= 2 ? Symbol(ARGS[2])     : :random
rng  = MersenneTwister(20260824)

println("="^88)
println("  1. THE CARRY-FREE CONVERSION  (binary → RR4 {-2..2}, depth 1)")
println("="^88)
v = 0b1011_0111_0011          # 2931
println("  v = $v = 0b$(string(v, base=2))\n")
println("  radix-4 digits of v are the bit-pairs; each carry looks at ITS OWN digit only:")
println("      c_{i+1} = [ d_i ≥ 2 ]        z_i = d_i − 4·c_{i+1} + c_i\n")
@printf("  %3s %6s %8s %7s %7s\n", "i", "d_i", "c_out", "c_in", "z_i")
println("  " * "─"^36)
for r in rr4_recode_table(v)
    @printf("  %3d %6d %8d %7d %7d\n", r.i, r.d, r.c_out, r.c_in, r.z)
end
z = rr4_recode(v)
println("\n  RR4 digits : ", digit_string(z))
println("  value back : ", Int(value(z)), "   exact: ", Int(value(z)) == v)
println("\n  Every row above is computable without waiting for any other row — that is")
println("  what makes it carry-free.  to_rr4 reaches the same VALUE by a sequential")
println("  rewrite whose carry depends on the running digit:")
println("      to_rr4      : carry out  iff  d_i + c_in > 2     (needs c_in — sequential)")
println("      rr4_recode  : carry out  iff  d_i     >= 2       (own digit — parallel)")
println("\n  The rules agree except at d_i = 2 with no incoming carry, so on many values")
println("  they coincide; on others they pick different — equally valid — spellings:")
@printf("      %6s  %-22s %-22s %s\n", "v", "to_rr4", "rr4_recode", "same spelling?")
for w in (v, 38, 2666, 10)
    @printf("      %6d  %-22s %-22s %s\n", w,
            digit_string(to_rr4(w)), digit_string(rr4_recode(w)),
            same_spelling(to_rr4(w), rr4_recode(w)))
end
println("\n  Both always decode to the same number — that redundancy IS the point.")

println("\n", "="^88)
println("  2. ONE ACCUMULATOR, $N TERMS, kind=:$KIND")
println("="^88)
terms = accumulator_inputs(N; kind = KIND, rng, magnitude = 1000)
println("  first 8 terms: ", terms[1:min(8, end)], " …\n")
compare_accumulators(terms)

println("\n", "="^88)
println("  3. YOUR OWN VECTOR")
println("="^88)
compare_accumulators([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3])

println("\n", "="^88)
println("  4. DOES IT SCALE?  sequential accumulator, depth vs N")
println("="^88)
accumulator_scaling(Ns = (16, 64, 256, 1024, 4096, 16384); rng = MersenneTwister(1))
println("\n  RR4 is flat at 3 levels per add whatever the accumulator width; ripple pays")
println("  the full width every single step, so its advantage erodes as the sum grows.")

println("\n", "="^88)
println("  5. THE SAME SWEEP AS A TREE  (both methods get to parallelise)")
println("="^88)
accumulator_scaling(Ns = (16, 64, 256, 1024, 4096, 16384);
                    schedule = :tree, rng = MersenneTwister(1))
println("\n  A tree helps everyone — log N levels instead of N — so the ratios shrink.")
println("  Redundancy is worth most exactly where you CANNOT tree: a sequential")
println("  accumulator with a loop-carried dependence, which is case 4 above.")

println("\n", "="^88)
println("  6. WIDTH IS THE VARIABLE THAT MATTERS")
println("="^88)
let
    @printf("  %12s %6s  %9s %9s %9s   %10s %10s\n",
            "magnitude", "bits", "ripple", "prefix", "RR4", "rip/RR4", "pre/RR4")
    println("  " * "─"^74)
    for mag in (10, 10^3, 10^6, 10^9, 10^15)
        tv = accumulator_inputs(512; kind = :positive,
                                rng = MersenneTwister(2), magnitude = mag)
        a = accumulate_carry(tv; adder = :ripple)
        b = accumulate_carry(tv; adder = :prefix)
        c = accumulate_rr4(tv)
        @printf("  %12d %6d  %9d %9d %9d   %9.1f× %9.1f×\n",
                mag, c.acc_bits, a.depth, b.depth, c.depth,
                a.depth / c.depth, b.depth / c.depth)
    end
end
println("\n  The RR4 column does not move.  That is the whole claim: its per-add depth is")
println("  a constant of the number system, not a function of the word length.")

println("\n", "="^88)
println("  7. WHERE IT DOES *NOT* WIN — read this before believing the other tables")
println("="^88)
println("""
  Against ripple carry the win is large and grows with width: 4-5x on the sweeps
  above, 19x at 58-bit accumulators.  Against a Kogge-Stone prefix network it is
  far more modest — 1.2x to 2.0x — because a prefix adder is already logarithmic
  in the width.  Carry-free replaces log(w) with a constant; it does not replace
  something linear.

  And at small widths RR4 can LOSE.  Section 3 above accumulates 16 terms into a
  7-bit register: prefix needs ceil(log2 7) = 3 levels per add, exactly what RR4
  needs, and RR4 still owes the exit conversion — so it comes out 0.8x, i.e.
  slightly behind.  The crossover is roughly where ceil(log2 w) exceeds 3, near
  w = 8 bits.

  The honest summary: redundancy buys width-independence, which is worth a great
  deal in a wide sequential accumulator and almost nothing in a narrow one.""")
