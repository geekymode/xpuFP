#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Example 3 — dot-product accuracy vs length, and the per-element error profile.
#
#   julia --project=. examples/03_dot_and_profile.jl
# ---------------------------------------------------------------------------
using xpuFP, CairoMakie, Random, Printf

const OUT = joinpath(@__DIR__, "figures"); mkpath(OUT)
rng = MersenneTwister(3)

println("="^78)
println("  DOT-PRODUCT SNR vs LENGTH  (should be flat: the block tree is exact)")
println("="^78)
lengths = [32, 128, 512, 2048, 8192]
@printf("  %-14s", "scheme")
for N in lengths; @printf(" %9d", N); end
println()
for f in (MXFP4, MXFP4_OPT32, NVFP4, XPFP4_32)
    c = dot_snr_curve(f, lengths; trials = 300, rng = MersenneTwister(1))
    @printf("  %-14s", _fmtname(f))
    for r in c; @printf(" %9.2f", r.snr); end
    println()
end

println("\n--- every element product is exact; all error is representational ---")
a = randn(rng, 256); b = randn(rng, 256)
for f in (MXFP4, MXFP4_OPT32, NVFP4, XPFP4_32)
    r = block_dot(f, a, b)
    @printf("  %-14s products exact: %-5s   core bound %d\n",
            f.name, r.products_exact, Int(core_sum_bound(f)))
end

savefig(plot_dot_snr_schemes((MXFP4, MXFP4_OPT32, NVFP4, XPFP4_32);
                             trials = 300, rng = MersenneTwister(1)),
        joinpath(OUT, "03_dot_snr.png"))
savefig(plot_error_profiles((MXFP4, NVFP4, XPFP4_32)), joinpath(OUT, "03_profiles.png"))
savefig(plot_block_snr_distribution((MXFP4, MXFP4_OPT32, XPFP4_32, rotated(XPFP4_32)),
                                    randn(MersenneTwister(5), 60_000)),
        joinpath(OUT, "03_block_snr.png"))
println("\nfigures → ", OUT)
