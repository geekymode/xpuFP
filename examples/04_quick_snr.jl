#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Example 4 — the quick SNR sim: sweep the FP4 design space in a few seconds.
#
#   julia --project=. examples/04_quick_snr.jl
#
# No figures, no quadrature, no confidence intervals — just the number, fast
# enough that asking a new question is cheaper than looking the old answer up.
# ---------------------------------------------------------------------------
using xpuFP, Printf, Random

println("="^82)
println("  1. THE SHIPPED FORMATS, plus what modifying them buys")
println("="^82)
quick_compare([MXFP4,                                   # OCP: K=32, E8M0, floor
               NVFP4,                                   # NVIDIA: K=16, E4M3, max/6
               fp4_variant(rule = OPT_SHIFT),           # +1 comparison in the encoder
               fp4_variant(rule = BEST_POW2),           # +1 block MSE evaluation
               fp4_variant(K = 16),                     # MX rule on NVFP4's block
               fp4_variant(K = 16, scale = E4M3, rule = MSE_OPTIMAL),
               MXINT4])                                 # the uniform-grid rival

println("\n", "="^82)
println("  2. BLOCK LENGTH — note the SNR column runs the OPPOSITE way to intuition")
println("="^82)
quick_sweep(Ks = (4, 8, 16, 32, 64, 128, 256))

println("\n", "="^82)
println("  3. THE SAME SWEEP UNDER THE OPTIMIZED SHIFT RULE — the trend inverts back")
println("="^82)
quick_sweep(Ks = (4, 8, 16, 32, 64, 128, 256), rule = OPT_SHIFT)
println("\nSo sweep 2's inversion was never a fact about block length — it was the floor")
println("rule wasting most of a binade, an overhead a longer block dilutes. Give the")
println("encoder one extra comparison and short blocks win again, as intuition said all")
println("along. Two sweeps, four seconds, and the confound is separated from the effect.")

println("\n", "="^82)
println("  4. DISTRIBUTION SENSITIVITY — where each scheme stops working")
println("="^82)
@printf("%-14s", "distribution")
schemes = [MXFP4, NVFP4, fp4_variant(rule = OPT_SHIFT, name = "MX+shift"), XPFP4_32]
for s in schemes; @printf("%14s", s.name); end
println("\n", "─"^(14 + 14 * length(schemes)))
for d in (:gaussian, :uniform, :laplace, :student_t3, :lognormal, :sparse, :outlier)
    x = quick_data(d, 300_000; rng = MersenneTwister(1))
    @printf("%-14s", d)
    for s in schemes
        @printf("%14.2f", quick_snr(s, x).snr)
    end
    println()
end
println("\n(dB; the heavy-tailed rows are where a single block maximum drags every")
println(" other element in its block down into the dead zone. Note that the shift rule,")
println(" worth +0.25 dB on Gaussians, is worth +3.3 dB on uniforms and almost nothing")
println(" on the heavy-tailed rows — a scale rule is only ever as good as its data.)")

println("\n", "="^82)
println("  5. WHAT SNR HIDES — the zeroed fraction on the same data")
println("="^82)
for s in schemes
    q = quick_snr(s; n = 300_000, dist = :student_t3)
    @printf("  %-24s SNR %7.2f dB   zeroed %6.2f %%   clipped %5.2f %%\n",
            q.scheme, q.snr, 100 * q.zeroed, 100 * q.clipped)
end
