#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Example 1 — benchmark every block-scaled FP4 scheme across seven distributions.
#
#   julia --project=. examples/01_compare_schemes.jl
#
# Writes figures to examples/figures/.
# ---------------------------------------------------------------------------
using xpuFP, CairoMakie, Random, Printf

const OUT = joinpath(@__DIR__, "figures"); mkpath(OUT)
rng = MersenneTwister(20260824)

schemes = (MXFP4, MXFP4_OPT32, MXFP4_OPT16, MXFP4_BEST32,
           NVFP4, NVFP4_BEST16, XPFP4_32)
data = test_distributions(60_000; rng)

println("="^78)
println("  BLOCK-SCALED FP4 SCHEMES — ", length(schemes), " schemes × ", length(data), " distributions")
println("="^78, "\n")

rows = benchmark_schemes(schemes, data)
print_benchmark(rows; metric = :snr)
println()
print_benchmark(rows; metric = :zeroed)
println()
print_benchmark(rows; metric = :block_snr_p10)

println("\n--- the Pareto frontier on Gaussian data ---")
for r in pareto_frontier(rows)
    @printf("  %-14s %5.3f bits  %7.3f dB\n", r.scheme, r.bits, r.snr)
end

println("\n--- with Hadamard rotation ---")
rrows = benchmark_schemes(schemes, data; rotate = true)
print_benchmark(rrows; metric = :snr)
println()
print_benchmark(rrows; metric = :zeroed)

savefig(plot_benchmark_grid(rows), joinpath(OUT, "01_grid.png"))
savefig(plot_benchmark_grid(rrows), joinpath(OUT, "01_grid_rotated.png"))
savefig(plot_scheme_pareto(rows), joinpath(OUT, "01_pareto.png"))
savefig(plot_block_schemes(data[[1, 3, 5]]), joinpath(OUT, "01_schemes.png"))
println("\nfigures → ", OUT)
