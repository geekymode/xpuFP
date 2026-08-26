#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Example 2 — what an outlier does to a block, and what rotation does about it.
#
#   julia --project=. examples/02_outlier_study.jl
# ---------------------------------------------------------------------------
using xpuFP, CairoMakie, Random, Printf, Statistics

const OUT = joinpath(@__DIR__, "figures"); mkpath(OUT)
rng = MersenneTwister(42)

# one block: 32 weights at the typical scale of trained weights, plus one outlier
x = 0.02 .* randn(rng, 32); x[7] = 0.35

println("="^78)
println("  ONE BLOCK, ONE OUTLIER")
println("="^78)
@printf("  max/median magnitude ratio: %.1f×\n\n", maximum(abs, x) / median(abs.(x)))

@printf("  %-16s %8s %8s %9s %9s\n", "scheme", "SNR dB", "zeroed", "cosine", "worst")
for f in (MXFP4, MXFP4_OPT32, NVFP4, XPFP4_32, rotated(MXFP4), rotated(XPFP4_32))
    m = measure_scheme(f, x; K = 32)
    @printf("  %-16s %8.2f %7.1f%% %9.5f %8.1f%%\n",
            m.scheme, m.snr, 100m.zeroed, m.cosine, 100m.worst_rel)
end

# how the damage scales with outlier magnitude
println("\n--- rest-of-block SNR as the outlier grows (the 6 dB/octave law) ---")
@printf("  %6s %12s %12s\n", "R", "MXFP4", "H·MXFP4")
for R in (1, 2, 4, 8, 16, 32, 64)
    a = randn(MersenneTwister(9), 32); a[1] = R * sign(a[1])
    s1 = snr_db(a[2:end], quantize_all(MXFP4, a)[2:end])
    s2 = snr_db(a[2:end], quantize_all(rotated(MXFP4), a)[2:end])
    @printf("  %6d %12.2f %12.2f\n", R, s1, s2)
end

savefig(plot_rotation_effect(x), joinpath(OUT, "02_rotation.png"))
savefig(plot_scale_placement(x), joinpath(OUT, "02_scale_placement.png"))
savefig(plot_block_reconstruction((MXFP4, NVFP4, XPFP4_32, rotated(XPFP4_32)), x),
        joinpath(OUT, "02_reconstruction.png"))
savefig(plot_outlier_law(; trials = 120), joinpath(OUT, "02_outlier_law.png"))
println("\nfigures → ", OUT)
