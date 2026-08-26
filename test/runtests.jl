using xpuFP
using Test
using Random
using LinearAlgebra
using Statistics

# Every numeric target below is drawn from the source report; the point of this suite
# is that the package reproduces them rather than merely runs.

@testset "xpuFP" begin

@testset "FloatFormat geometry" begin
    @test maxfinite(FP32) == 3.4028234663852886e38
    @test minnormal(FP32) == 1.1754943508222875e-38
    @test minsubnormal(FP32) == 1.401298464324817e-45
    @test machine_eps(FP32) == 2.0^-23
    @test maxfinite(E4M3) == 448.0          # one NaN code, not a whole exponent
    @test maxfinite(E5M2) == 57344.0
    @test maxfinite(FP16) == 65504.0
    @test maxfinite(E2M1) == 6.0
    @test nbits(FP32) == 32 && nbits(E2M1) == 4 && nbits(E8M0) == 8
    @test emin(FP32) == -126 && emax(FP32) == 127
end

@testset "encode / decode round trips" begin
    @test encode(FP32, -6.375) == 0xc0cc0000
    @test decode(FP32, 0x41C80000) == 25.0
    @test encode(FP32, 0.1) == 0x3dcccccd
    @test encode(FP32, 1440) == 0x44b40000
    @test quantize(FP32, 0.1) == 13421773 / 2.0^27
    for f in (FP16, BF16, E5M2, E4M3, E2M1, E1M2, E3M0)
        for c in 0:((1 << nbits(f)) - 1)
            v = decode(f, c)
            isfinite(v) || continue
            @test decode(f, encode(f, v)) == v
        end
    end
end

@testset "E2M1 is the whole number system" begin
    @test grid(E2M1) == [-6.0,-4.0,-3.0,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0]
    @test length(grid(E2M1)) == 15
    @test midpoints(E2M1) == [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    # every tie resolves to an M = 0 (even) code
    @test [quantize(E2M1, m) for m in midpoints(E2M1)] == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0]
    @test posgrid(E1M2) == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    @test posgrid(E3M0) == [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
end

@testset "FP4 arithmetic" begin
    @test fadd(E2M1, 1.5, 2.0) == 4.0        # tie → even
    @test fadd(E2M1, 4.0, 0.5) == 4.0        # absorbed
    @test fadd(E2M1, 6.0, 0.5) == 6.0        # saturates
    @test fmul(E2M1, 2.0, 4.0) == 6.0
    @test fmul(E2M1, 3.0, 4.0) == 6.0
    @test fmul(E2M1, 1.5, 1.5) == 2.0
    @test stagnation_trace(E2M1, 0.5, 8) == [0.5, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0]
    @test stagnation_threshold(E2M1, 0.5) == 1.0
    # non-associativity, at full strength
    @test fadd(E2M1, fadd(E2M1, 4.0, -3.0), -0.5) == 0.5
    @test fadd(E2M1, 4.0, fadd(E2M1, -3.0, -0.5)) == 0.0
    # only 141 of the 225 products land on the grid
    g = grid(E2M1)
    @test count(p -> abs(p) <= 6 && quantize(E2M1, p) == p, [a*b for a in g, b in g]) == 141
end

@testset "datapath traces" begin
    t = fpadd(FP32, 12.0, 6.0)
    @test t.result == 18.0 && t.info.align_shift == 1 && t.info.norm_shift == 1
    t = fpmul(FP32, 6.0, 6.0)
    @test t.result == 36.0 && t.info.stored_E == 131 && t.info.sig_product == 2.25
    t = fpdiv(FP32, 9.0, 6.0)
    @test t.result == 1.5 && t.info.norm_shift == -1
    @test fpadd(FP32, 1.0e8, 1.0).result == 1.0e8      # absorption
    f4 = FloatFormat("toy", 4, 4)
    @test fpfma(f4, 1.125, 1.375, -1.5).result == 0.046875
    @test fpmul_then_add(f4, 1.125, 1.375, -1.5).result == 0.0625
    @test relerror(fpmul_then_add(f4, 1.125, 1.375, -1.5)) ≈ 1/3 atol=1e-12
    for t in (fpadd(FP32, 3.7, -1.9), fpmul(FP32, 3.7, -1.9), fpdiv(FP32, 3.7, -1.9))
        @test abs(rounding_ulps(t)) <= 0.5 + 1e-12     # correctly rounded
    end
end

@testset "fixed point" begin
    f = FixedFormat(7, 8)
    @test nbits(f) == 16 && resolution(f) == 2.0^-8
    @test encode(f, -3.65) == -934
    @test quantize(f, -3.65) == -3.6484375
    @test fxadd(FixedFormat(3,4), 2.5, 1.25) == 3.75
    @test fxmul(FixedFormat(3,4), 1.5, 2.25) == 3.375
    @test fxadd(f, 100.0, 100.0) == fxmax(f)                     # saturates
    @test fxadd(f, 100.0, 100.0; mode = WRAP) < 0                 # wraps
end

@testset "summation schedules" begin
    x = [1.0; fill(2.0^-25, 3)]
    @test seq_sum(FP32, x) == 1.0                    # all three grains absorbed
    @test kahan_sum(FP32, x) > 1.0                   # compensation recovers them
    u = [1.5, 3.0, 0.5]; v = [1.5, -2.0, 6.0]
    @test fp_dot(E2M1, u, v; accumulate = FP32) == -0.75
    @test fp_dot(E2M1, u, v) == -1.0
end

@testset "block formats" begin
    bf = BlockFormat("mx4", 4, E2M1, E8M0, MX_FLOOR_POW2)
    qb = quantize_block(bf, [0.11, -0.35, 0.02, 0.24])
    @test qb.scale == 2.0^-4
    @test qb.scale_code == 123
    @test qb.elements == [2.0, -6.0, 0.5, 4.0]
    @test qb.values == [0.125, -0.375, 0.03125, 0.25]
    r = block_dot(bf, [0.11,-0.35,0.02,0.24], [1.3,0.7,-2.1,0.44])
    @test r.core_sum == -1.0
    @test r.scale_product == 2.0^-5
    @test r.value == -0.03125
    @test r.products_exact
    @test relerror(r) ≈ 0.186 atol = 0.002
    @test bits_per_element(MXFP4) == 4.25
    @test bits_per_element(NVFP4) == 4.5
    @test core_sum_bound(MXFP4) == 1152.0
    @test compression_ratio(MXFP4, FP32) ≈ 7.53 atol = 0.01
    @test scale_fixup_overhead(MXFP4) ≈ 0.031 atol = 0.001
    @test elem_emax(MXFP4) == 2
    # the alignment theorem: S is the unique power of two landing M in [4,8)
    rng = MersenneTwister(9)
    for _ in 1:200
        x = exp(4 * randn(rng)) .* randn(rng, 32)
        S = block_scale(MXFP4, x)
        @test 4 <= maximum(abs, x) / S < 8
    end
end

@testset "improved block schemes" begin
    rng = MersenneTwister(20260823)
    g = randn(rng, 60_000)

    # every improved scheme keeps MXFP4's four structural properties
    for bf in IMPROVED_BLOCK_FORMATS
        @test bf.elem === E2M1                       # FP4-native MAC still possible
        @test core_sum_bound(bf) == bf.K * 36        # adder tree still exact
        a = randn(rng, bf.K); b = randn(rng, bf.K)
        r = block_dot(bf, a, b)
        @test r.products_exact                       # products still exact in FP32
    end
    @test bits_per_element(MXFP4_BEST32) == 4.25
    @test bits_per_element(XPFP4_32) == 4.25
    @test bits_per_element(NVFP4_BEST16) == 4.5
    # the power-of-two schemes keep the exponent-add scale
    @test MXFP4_BEST32.scale === E8M0 && MXFP4_BEST16.scale === E8M0

    # XPFP4_16 is an alias, not a second format that would measure identically
    @test XPFP4_16 === NVFP4_BEST16

    # each improvement actually improves, at equal block size
    @test measure_snr(MXFP4_BEST32, g) > measure_snr(MXFP4, g)
    @test measure_snr(NVFP4_BEST16, g) > measure_snr(NVFP4, g)
    # and the headline claim: better SNR than NVFP4 at FEWER bits
    @test measure_snr(XPFP4_32, g) > measure_snr(NVFP4, g)
    @test bits_per_element(XPFP4_32) < bits_per_element(NVFP4)
    # report targets
    @test measure_snr(MXFP4, g) ≈ 18.8 atol = 0.25
    @test measure_snr(MXFP4_BEST32, g) ≈ 19.05 atol = 0.25
    @test measure_snr(NVFP4, g) ≈ 20.4 atol = 0.25

    # the scale must always be usable, whatever the data magnitude
    for bf in IMPROVED_BLOCK_FORMATS, _ in 1:40
        x = exp(3 * randn(rng)) .* randn(rng, bf.K)
        S = block_scale(bf, x)
        @test S > 0 && isfinite(S)
        r = maximum(abs, x) / S
        if bf.rule == MSE_OPTIMAL
            @test r <= 16                # deliberate overload, bounded by the search window
        else
            @test r <= 8 + 1e-9          # alignment rules never strand the max
        end
    end
    # extreme magnitudes must not break the scale, in either direction
    for bf in IMPROVED_BLOCK_FORMATS, mag in (1e-300, 1e-30, 1.0, 1e30, 1e300)
        x = mag .* randn(MersenneTwister(2), bf.K)
        S = block_scale(bf, x)
        @test S > 0 && isfinite(S)
        @test maximum(abs, x) / S <= 16
    end
    # regression: E4M3 scales silently returned 0 for small blocks (÷0 downstream)
    @test needs_tensor_scale(NVFP4) && !needs_tensor_scale(MXFP4)
    @test tensor_scale(MXFP4, randn(MersenneTwister(1), 64)) == 1.0
    for mag in (1e-30, 1e-6, 1.0, 1e6, 1e30)
        v = mag .* randn(MersenneTwister(4), 256)
        for bf in (NVFP4, XPFP4_32, NVFP4_BEST16)
            xh = reconstruct(bf, v)
            @test all(isfinite, xh)
            @test measure_snr(bf, v) > 15      # and still actually works
        end
    end
end

@testset "optimized shift rule" begin
    rng = MersenneTwister(20260824)
    x = randn(rng, 120_000)

    @test PHISTAR[32] == 0.86 && PHISTAR[16] == 0.82
    @test opt_shift_threshold(32) == 0.86 && opt_shift_threshold(16) == 0.82
    @test 0.82 <= opt_shift_threshold(24) <= 0.86        # interpolates between them

    # the rule itself, from the block maximum alone
    @test mx_scale_opt(0.35) == 0.0625                   # φ ≈ 0.49, no shift
    @test mx_scale_opt(0.95) == 0.25                     # φ ≈ 0.93 > 0.86, shift fires
    # disabling the threshold must reproduce plain MXFP4 exactly
    for M in (0.35, 0.95, 1.9, 7.7, 1e-8, 1e12)
        @test mx_scale_opt(M; phistar = 1.0) == block_scale(MXFP4, [M])
    end
    # the shift is exactly one binade, never more
    for M in exp.(range(-8, 8; length = 200))
        r = mx_scale_opt(M) / mx_scale_opt(M; phistar = 1.0)
        @test r == 1.0 || r == 2.0
    end

    # the measured claims
    for (K, bf, floorbf, bestbf) in
        ((32, MXFP4_OPT32, BlockFormat("f", 32, E2M1, E8M0, MX_FLOOR_POW2), MXFP4_BEST32),
         (16, MXFP4_OPT16, BlockFormat("f", 16, E2M1, E8M0, MX_FLOOR_POW2), MXFP4_BEST16))
        sf = measure_snr(floorbf, x); so = measure_snr(bf, x); sc = measure_snr(bestbf, x)
        @test so > sf + 0.2                        # it genuinely improves
        @test so <= sc + 1e-9                      # and cannot beat the class ceiling
        @test sc - so < 0.02                       # but lands within 0.02 dB of it
        r = opt_shift_rate(x, K)
        @test 0.10 < r < 0.25                      # fires on a minority of blocks
    end
    @test measure_snr(MXFP4_OPT32, x) ≈ 19.06 atol = 0.05
    @test measure_snr(MXFP4_OPT16, x) ≈ 19.19 atol = 0.05

    # THE point of the scheme: the decoder cannot tell which encoder ran
    for _ in 1:50
        a = randn(rng, 32)
        qo = quantize_block(MXFP4_OPT32, a)
        qm = quantize_block(MXFP4, a)
        @test bits_per_block(MXFP4_OPT32) == bits_per_block(MXFP4)
        @test length(qo.codes) == length(qm.codes)
        # both decode by the identical E8M0 rule, x̂ = e · 2^(X-127)
        @test qo.values ≈ qo.elements .* exp2(Int(qo.scale_code) - 127)
        @test qm.values ≈ qm.elements .* exp2(Int(qm.scale_code) - 127)
        # and the scale is always a legal E8M0 code
        @test 0 <= qo.scale_code <= 254
    end
    @test MXFP4_OPT32.scale === E8M0                # still an exponent add
    @test bits_per_element(MXFP4_OPT32) == 4.25     # and still 4.25 bits/value

    # opt_shift_fires agrees with the scale it produces
    for _ in 1:200
        a = randn(rng, 32) .* exp(3 * randn(rng))
        fired = mx_scale_opt(maximum(abs, a)) != mx_scale_opt(maximum(abs, a); phistar = 1.0)
        @test opt_shift_fires(a; K = 32) == fired
    end
end

@testset "Hadamard rotation" begin
    for K in (2, 4, 16, 32, 64)
        H = hadamard(K)
        @test size(H) == (K, K)
        @test H' * H ≈ Matrix(1.0I, K, K) atol = 1e-12
        x = randn(MersenneTwister(K), K)
        @test H' * (H * x) ≈ x atol = 1e-12
        @test sum(abs2, H * x) ≈ sum(abs2, x) atol = 1e-10    # orthogonal
    end
    @test_throws ArgumentError hadamard(24)
    @test_throws ArgumentError rotated(BlockFormat("odd", 24, E2M1, E8M0, MX_FLOOR_POW2))

    rng = MersenneTwister(5)
    N = 40_000
    g = randn(rng, N)
    o = copy(g); for i in 1:40:N; o[i] *= 20; end

    R = rotated(MXFP4)
    @test bits_per_element(R) == bits_per_element(MXFP4)     # rotation is free in storage
    # rotation is ~neutral on isotropic data and a large win on outliers
    @test measure_snr(R, g) ≈ measure_snr(MXFP4, g) atol = 0.3
    @test measure_snr(R, o) > measure_snr(MXFP4, o) + 3.0
    # and it collapses the annihilation rate, which is the metric that matters
    zr(f, v) = (xh = quantize_all(f, v); count(i -> xh[i] == 0 && v[i] != 0, eachindex(v)) / length(v))
    @test zr(R, o) < zr(MXFP4, o) / 5
end

@testset "H·XPFP4-32 worked example" begin
    rng = MersenneTwister(42)
    x = 0.02 .* randn(rng, 32); x[7] = 0.35
    H = hadamard(32)
    y = H * x

    # the rotation preserves every quantity a GEMM cares about
    @test norm(y) ≈ norm(x) atol = 1e-12
    b = randn(MersenneTwister(3), 32)
    @test dot(H * x, H * b) ≈ dot(x, b) atol = 1e-12       # dot products exactly preserved
    @test H' * y ≈ x atol = 1e-12
    # and it removes the concentration, which is the thing block scaling cannot tolerate
    @test maximum(abs, x) / median(abs.(x)) > 20
    @test maximum(abs, y) / median(abs.(y)) < 3

    zc(f) = (xh = quantize_all(f, x); count(i -> xh[i] == 0 && x[i] != 0, eachindex(x)))
    @test zc(MXFP4) >= 15
    @test zc(rotated(XPFP4_32)) == 0                        # the headline of the example
    @test zc(rotated(XPFP4_32)) < zc(XPFP4_32)
    @test cosine_similarity(x, quantize_all(rotated(XPFP4_32), x)) >
          cosine_similarity(x, quantize_all(MXFP4, x))

    # the SNR trap: NVFP4 scores well here while destroying 11 coordinates
    @test snr_db(x, quantize_all(NVFP4, x)) > snr_db(x, quantize_all(MXFP4, x))
    @test zc(NVFP4) > 5

    # the arithmetic stays exact for every scheme in play
    a2 = 0.02 .* randn(rng, 32); b2 = 0.02 .* randn(rng, 32)
    for bf in (MXFP4, NVFP4, XPFP4_32)
        @test block_dot(bf, a2, b2).products_exact
    end
end

@testset "block scheme comparison" begin
    rng = MersenneTwister(3)
    ds = ["g" => randn(rng, 20_000)]
    rows = compare_block_schemes(ds)
    @test length(rows) == length(IMPROVED_BLOCK_FORMATS)
    @test all(r -> length(r.snr) == 1 && length(r.zeroed) == 1, rows)
    @test all(r -> 0 <= r.zeroed[1] <= 1, rows)
    @test any(r -> r.exponent_only, rows) && any(r -> !r.exponent_only, rows)
    rrows = compare_block_schemes(ds; rotate = true)
    @test all(r -> startswith(r.name, "H·"), rrows)
    io = IOBuffer(); print_scheme_table(rows; io)
    @test occursin("scheme", String(take!(io)))
end

@testset "benchmark harness" begin
    rng = MersenneTwister(4)
    ds = test_distributions(8_000; rng)
    @test length(ds) == 7
    @test all(p -> length(last(p)) == 8_000, ds)
    @test all(p -> all(isfinite, last(p)), ds)

    m = measure_scheme(MXFP4, randn(rng, 20_000); dataset = "g")
    @test m.scheme == "MXFP4" && m.dataset == "g" && m.bits == 4.25
    @test m.snr ≈ 18.8 atol = 0.4
    @test m.eff_bits ≈ m.snr / DB_PER_BIT
    @test m.db_per_bit ≈ m.snr / m.bits
    @test 0 <= m.zeroed <= 1 && 0 <= m.cosine <= 1
    @test m.block_snr_p10 <= m.block_snr_median

    rows = benchmark_schemes((MXFP4, MXFP4_OPT32, NVFP4), ds[1:3])
    @test length(rows) == 9
    @test length(unique(r.scheme for r in rows)) == 3
    io = IOBuffer(); print_benchmark(rows; io); @test occursin("scheme", String(take!(io)))
    for met in (:snr, :zeroed, :db_per_bit, :block_snr_p10)
        io = IOBuffer(); print_benchmark(rows; io, metric = met)
        @test !isempty(String(take!(io)))
    end

    pf = pareto_frontier(rows)
    @test !isempty(pf)
    @test issorted([r.bits for r in pf])
    @test issorted([r.snr for r in pf])            # a frontier is monotone by construction

    ep = error_profile(MXFP4)
    @test length(ep) == 400 && all(>=(0), ep)
    @test ep[1] == 1.0                             # t = 1e-3 is inside the dead zone
end

@testset "SNR simulation and estimates" begin
    rng = MersenneTwister(11)
    # the derived element SNR must reproduce the report's figures
    @test estimate_element_snr(MXFP4) ≈ 18.79 atol = 0.06
    @test estimate_element_snr(NVFP4) ≈ 20.44 atol = 0.10
    @test estimate_element_snr(MXFP4_OPT32) > estimate_element_snr(MXFP4)
    # and agree with the measurement
    for f in (MXFP4, NVFP4, MXFP4_OPT32)
        s = simulate_snr(f; n = 20_000, trials = 6, rng = MersenneTwister(2), dotlength = 1024)
        @test s.estimate !== nothing
        @test abs(s.measured - s.estimate) < 0.15          # derivation matches simulation
        @test s.ci[1] <= s.measured <= s.ci[2]
        # the dot product must obey the −3.01 dB law within a decibel or so
        @test abs(s.dot_measured - s.dot_predicted) < 2.0
    end
    # the closed form for floats is distribution-free, so it holds on non-Gaussian data
    for f in (FP16, BF16, FP32)
        e, _ = analytic_snr(f)
        @test e ≈ predicted_snr(f)
        s = simulate_snr(f; n = 20_000, trials = 4, rng = MersenneTwister(3),
                         dist = (r, m) -> exp.(randn(r, m)), distname = "lognormal",
                         dotlength = 512)
        @test abs(s.measured - e) < 1.0
    end
    @test analytic_snr(INT4)[1] === nothing            # no closed form invented
    @test dot_snr_law(18.79) ≈ 15.78 atol = 0.01
    @test grid_distance(E2M1, 2.25) == 0.25
    @test grid_distance(E2M1, 10.0) == 4.0             # saturation grows without bound

    # the dot-product law is length-invariant: that is the headline claim
    c = dot_snr_curve(MXFP4, [64, 512, 4096]; trials = 250, rng = MersenneTwister(5))
    v = [r.snr for r in c]
    @test maximum(v) - minimum(v) < 2.0
    @test all(r -> r.lo <= r.snr <= r.hi, c)
end

@testset "SNR measurement" begin
    rng = MersenneTwister(20240821)
    x = randn(rng, 200_000)
    @test measure_snr(E2M1, x) ≈ 16.3 atol = 0.3
    @test measure_snr(MXFP4, x) ≈ 18.8 atol = 0.2
    @test measure_snr(NVFP4, x) ≈ 20.4 atol = 0.2
    @test measure_snr(FP32, x) ≈ 151.9 atol = 0.5     # the RMS figure
    @test predicted_snr(FP32) ≈ 151.93 atol = 0.05    # its closed form
    @test effective_bits(measure_snr(BF16, x)) ≈ 9.2 atol = 0.3
    # bare FP4 fails totally when the data scale is unknown; MXFP4 does not
    xs = 0.02 .* randn(rng, 50_000)
    @test measure_snr(E2M1, xs) == 0.0
    @test measure_snr(MXFP4, xs) ≈ 18.8 atol = 0.3
    @test DB_PER_BIT ≈ 6.0206 atol = 1e-4
end

@testset "conversions" begin
    @test convert_format(E2M1, FP32, 2.25) == 2.0
    @test convert_format(FP32, E2M1, 6.0) == 6.0
    @test narrow(BF16, FP32, 3.14159).value == 3.140625
    @test narrow(FP16, FP32, 1.0e30).overflowed
    r = int_to_float(FP32, 1440)
    @test r.clz == 21 && r.exponent == 10 && r.fraction == 3407872 && r.exact
    @test !int_to_float(FP32, 16_777_217).exact
    @test nextafter(FP32, 1.0) - 1.0 == machine_eps(FP32)
    @test nextafter(E2M1, 2.0) == 3.0
    @test prevafter(FP32, nextafter(FP32, 1.0)) == 1.0
    for f in (E2M1, E4M3, E5M2, FP16)
        @test encodings_monotone(f)
    end
    ts = top_seam(FP32)
    @test ts.phantom == 2.0^128
    @test ts.overflow_threshold == 2.0^128 - 2.0^103
    bs = bottom_seam(FP32)
    @test bs.gap == minsubnormal(FP32) && bs.codes_differ_by == 1 && bs.seamless
end

@testset "excursions" begin
    e = multiplier_excursion(FP32, 0.1, 10.0)
    @test e.result == 1.0 && e.bits_identical && e.class == INVISIBLE
    e = redundant_dot_excursion(FP32, [1.0, 2.0^-25, 2.0^-25, 2.0^-25], ones(4))
    @test e.reference == 1.0
    @test e.result == 1.0 + 2.0^-23
    @test e.class == VISIBLE_BETTER
end

@testset "CSD" begin
    @test digit_string(csd(231)) == "1001̄01001̄"
    @test weight(csd(231)) == 4 && adders(csd(231)) == 3
    @test binary_weight(231) == 6
    @test digit_string(csd(27)) == "1001̄01̄"
    @test value(csd(231)) == 231
    @test csd_multiply(231, 1234) == 231 * 1234
    @test csd_multiplier_taps(231) == [(0,-1), (3,1), (5,-1), (8,1)]
    @test !has_adjacent_nonzeros(csd(231))
    sp = all_signed_spellings(231, 9)
    @test length(sp) == 31
    @test count(s -> weight(s) == 4, sp) == 2
    @test [count(s -> weight(s) == w, sp) for w in 4:9] == [2, 6, 10, 8, 4, 1]
    @test verify_csd_minimal(3000)
    for x in 1:500
        @test weight(csd(x)) == min_signed_weight(x)
    end
end

@testset "SignedDigits alphabet is enforced" begin
    # regression: the compiler-generated inner constructor used to shadow the
    # validating one, silently admitting out-of-alphabet digits
    @test_throws ArgumentError SignedDigits([4, 0], 4, 3)
    @test_throws ArgumentError SignedDigits([3, 0], 4, 2)
    @test_throws ArgumentError SignedDigits([-4], 4, 3)
    @test_throws ArgumentError SignedDigits([2], 1, 2)
    @test value(SignedDigits([3, -3], 4, 3)) == -9
    @test redundancy(SignedDigits([1], 4, 3)) == 3      # maximal
    @test redundancy(SignedDigits([1], 4, 2)) == 1      # minimal
    @test !is_redundant(SignedDigits([1], 4, 1))
end

@testset "RR4" begin
    @test all(r -> abs(r.u) <= 2, rr4_split_table())
    @test length(rr4_split_table()) == 13
    @test verify_absorption()
    @test all(w -> abs(w) <= 3, absorption_table())
    @test verify_sign_coupling()
    @test verify_minimal_closure()
    @test verify_rr4_random(2000)
    @test verify_rr4_random(2000; alphabet = MIN_REDUNDANT)

    tr = rr4_add(6, 43; alphabet = MAX_REDUNDANT)
    @test value(tr.result) == 49 && tr.exact && tr.depth == 3
    @test value(rr4_add(6, 43).result) == 49          # MIN_REDUNDANT is the default
    @test all(c -> abs(c.t_out) <= 1, tr.columns)
    @test all(c -> abs(c.w) <= 2, tr.columns)

    # the report's non-canonical spellings, reproduced digit for digit
    x = SignedDigits([2, -3, 1], 4, 3); y = SignedDigits([-1, 3, 2], 4, 3)
    @test value(x) == 6 && value(y) == 43
    t2 = rr4_add(x, y; alphabet = MAX_REDUNDANT)
    @test [c.s for c in t2.columns] == [1, 0, 3]
    @test t2.result.digits == [1, 0, -1, 1]        # (1,1̄,0,1)₄ = 49
    @test value(t2.result) == 49

    # the minimal-set worked example
    xm = SignedDigits([1, 2, -1, 2], 4, 2); ym = SignedDigits([1, 1, 2, 1], 4, 2)
    @test value(xm) == 121 && value(ym) == 101
    t3 = rr4_add(xm, ym; alphabet = MIN_REDUNDANT)
    @test [c.s for c in t3.columns] == [2, 3, 1, 3]
    @test t3.result.digits == [-2, 0, 2, -1, 1]
    @test value(t3.result) == 222

    # the peek earns its keep: 7 + (-1)
    t4 = rr4_add(SignedDigits([-1, 2], 4, 2), SignedDigits([-1, 0], 4, 2); alphabet = MIN_REDUNDANT)
    @test value(t4.result) == 6
    @test all(c -> abs(c.z) <= 2, t4.columns)

    @test spelling_census(10, 4, 2) == 3 && spelling_census(10, 4, 3) == 6
    @test spelling_census(25, 4, 2) == 3 && spelling_census(25, 4, 3) == 8
    @test spelling_census(-7, 4, 2) == 2 && spelling_census(-7, 4, 3) == 6
    @test rr4_to_canonical(to_rr4(49)) == [1, 0, 3]
    m = maximal_to_minimal(SignedDigits([3, 3, 1], 4, 3))
    @test value(m) == 31 && m.maxdigit == 2
end

@testset "to_rr4 across input types" begin
    # the default alphabet is the one hardware uses
    @test to_rr4(49).maxdigit == 2
    @test to_rr4_minimal(49).maxdigit == 2
    @test to_rr4_maximal(49).maxdigit == 3

    @test value(to_rr4(49)) == 49
    @test value(to_rr4(0)) == 0
    @test value(to_rr4(-1234)) == -1234
    # floats are dyadic, so radix 4 holds them EXACTLY — no rounding invented
    for x in (49.2, 49.25, 0.1, -3.75, 2.0^-30, 1.0e6)
        sd = to_rr4(x)
        @test value(sd) == Rational{BigInt}(x)
        @test float_value(sd) == x
    end
    @test value(to_rr4(-7//8)) == -7//8
    @test value(to_rr4(3//16)) == 3//16
    # non-dyadic rationals have no finite expansion and must say so
    @test_throws ArgumentError to_rr4(1//3)
    @test value(to_rr4(1//3; fracdigits = 6)) == 1365//4096
    @test to_rr4(1//3; fracdigits = 6).exponent == -6
    # SignedDigits in, SignedDigits out: a re-spelling across alphabets
    m = to_rr4_maximal(49)
    @test value(to_rr4(m)) == 49 && to_rr4(m).maxdigit == 2
    @test value(to_rr4_maximal(to_rr4(49))) == 49

    @test dyadic_parts(49) == (49, 0)
    @test dyadic_parts(49.25) == (197, -1)
    @test_throws ArgumentError to_rr4(Inf)
end

@testset "RR4 enumeration and lengths" begin
    @test max_representable(3, MIN_REDUNDANT) == 42
    @test max_representable(3, MAX_REDUNDANT) == 63
    @test min_ndigits(49) == 4                       # {-2..2}
    @test min_ndigits(49; alphabet = MAX_REDUNDANT) == 3
    @test min_ndigits(0) == 1

    # every spelling, cross-checked against an independent DP count
    for v in (10, 25, -7, 49, 0), n in (3, 4, 5), alp in (MIN_REDUNDANT, MAX_REDUNDANT)
        reps = all_rr4_representations(v, n; alphabet = alp)
        @test length(reps) == count_rr4_representations(v, n; alphabet = alp)
        @test all(r -> value(r) == v, reps)
        @test all(r -> length(r) == n, reps)
        @test all(r -> all(d -> abs(d) <= alphabet_halfwidth(alp), r.digits), reps)
    end
    # ndigits is optional: it defaults to the shortest width the alphabet allows
    @test length(all_rr4_representations(12)) == length(all_rr4_representations(12, min_ndigits(12)))
    @test length(all_rr4_representations(12.5)) == 2
    @test all(r -> value(r) == 25//2, all_rr4_representations(12.5))
    @test all(r -> length(r) == min_ndigits(12.5), all_rr4_representations(12.5))
    @test count_rr4_representations(12) == length(all_rr4_representations(12))
    for x in (12, 12.5, 49, 231, -7)
        @test count_rr4_representations(x) == length(all_rr4_representations(x))
    end

    # the scalar form: "how many ways can this be spelled?", default method and width
    @test rr4_representation_counts(6) == 2
    @test rr4_representation_counts(10) == 1
    @test rr4_representation_counts(10; method = :maximally_redundant) == 2
    @test rr4_representation_counts(12) == 1
    @test rr4_representation_counts(25; method = :maximally_redundant) == 4
    @test rr4_representation_counts(12.5) == 2
    for v in (6, 10, 12, 25, 49, 231, 12.5), m in (:minimally_redundant, :maximally_redundant)
        @test rr4_representation_counts(v; method = m) ==
              count_rr4_representations(v; method = m) ==
              length(all_rr4_representations(v; method = m))
    end
    @test rr4_representation_counts(49; ndigits = 6) == count_rr4_representations(49, 6)

    # `method` is accepted wherever `alphabet` was
    @test count_rr4_representations(6; method = :maximally_redundant) ==
          count_rr4_representations(6; alphabet = MAX_REDUNDANT)
    @test min_ndigits(49; method = :maximally_redundant) == 3
    @test min_ndigits(49) == 4
    @test length(all_rr4_representations(6; method = :maximally_redundant)) == 2

    # spelling freedom as a function of width, measured not assumed
    @test rr4_representation_counts(25, 3:6) == [3 => 2, 4 => 3, 5 => 3, 6 => 3]
    @test rr4_representation_counts(25, 3:6; method = :maximally_redundant) ==
          [3 => 4, 4 => 8, 5 => 12, 6 => 16]
    @test rr4_representation_counts(49, 3:5) == [3 => 0, 4 => 1, 5 => 1]
    # a width below the minimum yields nothing, rather than erroring
    @test isempty(all_rr4_representations(49, 3))
    # and more room never means fewer spellings
    for v in (10, 12, 25, 49), alp in (MIN_REDUNDANT, MAX_REDUNDANT)
        c = last.(rr4_representation_counts(v, 3:7; method = alp))
        @test issorted(c)
    end

    # the report's censuses
    @test length(all_rr4_representations(10, 4)) == 3
    @test length(all_rr4_representations(10, 4; alphabet = MAX_REDUNDANT)) == 6
    @test length(all_rr4_representations(25, 4)) == 3
    @test length(all_rr4_representations(25, 4; alphabet = MAX_REDUNDANT)) == 8
    @test length(all_rr4_representations(-7, 4)) == 2
    @test length(all_rr4_representations(-7, 4; alphabet = MAX_REDUNDANT)) == 6

    # shortest spelling, and the tie broken by weight
    mn = minimal_rr4(49)
    @test length(mn) == min_ndigits(49) && value(mn) == 49
    @test weight(minimal_rr4(1000)) <= weight(to_rr4(1000))
    @test value(minimal_rr4(1000)) == 1000
    @test length(minimal_rr4(49; by = :canonical)) == min_ndigits(49)

    # arbitrary length and exponent — padding is free
    for n in 4:9
        sd = rr4_with_length(49, n)
        @test length(sd) == n && value(sd) == 49
    end
    for e in (-3, -2, -1, 0)
        sd = rr4_with_length(49, 8; exponent = e)
        @test length(sd) == 8 && value(sd) == 49 && sd.exponent == e
        @test nfracdigits(sd) == max(0, -e)
    end
    @test_throws ArgumentError rr4_with_length(1000, 2)
    @test_throws ArgumentError to_rr4(49.25; exponent = 0)   # would discard the .25
end

@testset "exponents and mixed-scale addition" begin
    sd = SignedDigits([1, 1], 4, 2, -1)
    @test value(sd) == 5//4 && nfracdigits(sd) == 1 && !isintegral(sd)
    @test scale(SignedDigits([1], 4, 2, -3)) == 1//64
    @test digit_string(to_rr4(49.25)) == "301.1" || value(to_rr4(49.25)) == 197//4
    @test value(rr4_rescale(to_rr4(49), -3)) == 49
    @test length(rr4_rescale(to_rr4(49), -3)) == length(to_rr4(49)) + 3

    # the adder aligns operands of different exponents, as an FP adder aligns significands
    for (a, b) in ((49.25, 6.5), (0.75, 0.125), (3, 0.5), (-7//8, 1//4))
        t = rr4_add(a, b)
        @test value(t.result) == Rational{BigInt}(a) + Rational{BigInt}(b)
        @test t.exact
        @test all(c -> abs(c.z) <= 2, t.columns)
    end
end

@testset "DigitFormat and universal conversion" begin
    @test alphabet(RR4) == -2:2
    @test alphabet(RADIX4) == 0:3
    @test alphabet(BINARY) == 0:1
    @test is_redundant(RR4) && is_redundant(RR4_MAX) && is_redundant(CSD)
    @test !is_redundant(BINARY) && !is_redundant(RADIX4)
    @test redundancy(RR4) == 1 && redundancy(RR4_MAX) == 3
    # both radix-4 alphabets fit the same 3-bit cell — redundancy free at storage level
    @test nbits(RR4) == nbits(RR4_MAX) == 3

    @test digit_string(to_digits(RADIX4, 49)) == "301"
    @test digit_string(to_digits(CSD, 231)) == "1001̄01001̄"
    @test value(to_digits(RR4, 231)) == 231
    @test conforms(CSD, to_digits(CSD, 231))
    @test !conforms(BINARY, to_digits(RR4, 231))
    @test_throws ArgumentError to_digits(BINARY, -5)      # unsigned
    @test_throws ArgumentError to_digits(RADIX4, -5)

    # the requested call, and exactness of floats in a power-of-two radix
    for x in (16.1656375, 16.15625, 49.2, -3.75, 0.1)
        for f in (RR4, RR4_MAX, CSD, SIGNED_BINARY)
            @test value(to_digits(f, x)) == Rational{BigInt}(x)
        end
    end
    @test_throws ArgumentError to_digits(RR4, 1//3)
    @test value(to_digits(RR4, 1//3; fracdigits = 5)) == round(BigInt, (1//3) * 4^5)//4^5

    @test radix_parts(49, 4) == (49, 0)
    @test radix_parts(0.25, 2) == (1, -2)

    # every ordered pair of families
    @test convert_format(E2M1, FP32, 2.25) == 2.0
    @test convert_format(BF16, FP32, 3.14159) == 3.140625
    @test value(convert_format(RR4, FP32, 6.5)) == 6.5
    @test convert_format(FP32, RR4, to_digits(RR4, 6.5)) == 6.5
    @test convert_format(FixedFormat(7, 8), FP32, -3.65) == -3.6484375
    @test convert_format(INT4, FP32, 2.6) == 3.0
    @test value(convert_format(RADIX4, BINARY, 231)) == 231
    @test convert_format(FP32, 2.25) == 2.25                    # two-arg form
    @test realvalue(to_digits(RR4, 6.5)) == 13//2

    # digit systems are re-spellings: lossless in both directions
    for x in (231, 49, 1024, 7)
        for f in (BINARY, CSD, RADIX4, RR4, RR4_MAX)
            @test conversion_report(f, BINARY, x).exact
        end
    end

    r = conversion_report(E2M1, FP32, 2.25)
    @test r.output == 2.0 && !r.exact && r.relerror ≈ 1/9
    @test conversion_report(FP16, FP32, 1.0e30).overflowed
    @test conversion_report(FP32, FP32, 2.0^-140).subnormal
    @test size(conversion_matrix((FP32, BF16, E2M1), 2.25)) == (3, 3)
end

@testset "addition algorithms" begin
    for (x, y) in ((255, 1), (93, 118), (0, 0), (4095, 4096), (-7, 3))
        rs = compare_adders(x, y)
        @test all(r -> r.value == x + y, rs)          # exact: redundancy trades cost only
    end
    @test serial_add(255, 1).parallel == false
    @test parallel_prefix_add(255, 1).parallel
    @test carry_free_add(255, 1).depth == 3
    @test carry_save_add(93, 118, 45).depth == 1
    @test carry_save_add(93, 118, 45).value == 93 + 118 + 45
    # depth ordering is the whole point
    d = Dict(r.algorithm => r.depth for r in compare_adders(255, 1))
    @test d[:carry_save] < d[:carry_free] < d[:parallel_prefix] < d[:serial]
    # carry-free depth is constant in the word length; serial is not
    @test carry_free_add(2^40 - 1, 1).depth == carry_free_add(3, 1).depth
    @test serial_add(2^40 - 1, 1).depth > serial_add(3, 1).depth
end

@testset "multiplication algorithms" begin
    for (a, b, n) in ((93, 118, 8), (11, 13, 6), (93, -74, 8), (0, 5, 6), (255, 255, 10))
        rs = compare_multipliers(a, b, n)
        @test all(r -> r.value == a * b, rs)
    end
    w = wallace_multiply(93, 118, 8; recode = false)
    bw = wallace_multiply(93, 118, 8; recode = true)
    @test w.value == bw.value == 93 * 118
    @test bw.nrows < w.nrows                       # recoding halves the rows
    @test bw.tree_levels < w.tree_levels           # which shortens the tree
    @test !w.recoded && bw.recoded

    # the report's FP32 significand figures
    @test wallace_multiply(12345, 9999, 24; recode = false).nrows == 24
    @test wallace_multiply(12345, 9999, 24; recode = false).tree_levels == 7
    u = wallace_multiply(12345, 9999, 24; recode = true, unsigned = true)
    @test u.nrows == 13 && u.tree_levels == 5 && u.value == 12345 * 9999

    # the unsigned guard digit, at every width
    for n in (4, 7, 8, 11, 16, 23, 24, 53)
        @test length(booth_radix4_unsigned(1, n).digits) == booth_rows(n)[2]
    end
    for x in (0, 1, 12345, 2^23 - 1)
        @test value(booth_radix4_unsigned(x, 24)) == x
    end

    c = csd_constant_multiply(231, 1234)
    @test c.value == 231 * 1234 && c.cells == adders(csd(231))
    @test rr4_multiply(93, 118).value == 93 * 118
    @test rr4_multiply(93, -74).value == 93 * -74
    @test shift_add_multiply(93, 118, 8).value == 93 * 118
end

@testset "digit array and integer scale" begin
    p = to_rr4_parts(49)
    @test p.digits == [1, 0, -1, 1]
    @test p.digits_msb == reverse(p.digits)
    @test p.scale == 0
    @test p.scaled_integer == 49
    @test p.value == 49
    @test p.radix == 4 && p.maxdigit == 2 && p.alphabet == -2:2
    @test p.ndigits == 4 && p.nonzeros == 3

    q = to_rr4_parts(49.25)
    @test q.digits == [1, 1, 0, -1, 1]
    @test q.scale == -1
    @test q.scaled_integer == 197
    @test q.value == 197//4

    # the defining identity: value == scaled_integer × radix^scale
    for x in (49, 49.25, 16.1656375, -3.75, 231, 0, 1//8)
        for alp in (MIN_REDUNDANT, MAX_REDUNDANT)
            sd = to_rr4(x; alphabet = alp)
            d = digit_parts(sd)
            @test d.value == d.scaled_integer * Rational{BigInt}(4)^d.scale
            @test d.value == value(sd)
            @test d.digits == sd.digits && d.scale == sd.exponent
            @test all(v -> v in d.alphabet, d.digits)
            @test d.nonzeros == count(!=(0), d.digits)
        end
    end
    @test scaled_integer(to_rr4(49.25)) == 197

    # the printed string and the digit array must never disagree: rebuild the string
    # from the array independently and demand equality
    mac = "\u0304"
    for x in (49, 49.25, 712.3, 16.1656375, -3.75, 0.1, 231, 1//8, 0)
        for alp in (MIN_REDUNDANT, MAX_REDUNDANT)
            sd = to_rr4(x; alphabet = alp)
            d, e = sd.digits, sd.exponent
            nf = max(0, -e)
            ch = [v == 0 ? "0" : (v < 0 ? string(-v) * mac : string(v)) for v in d]
            rebuilt = if nf == 0
                join(reverse(ch)) * (e > 0 ? repeat("0", e) : "")
            else
                ip = length(ch) > nf ? join(reverse(ch[nf+1:end])) : "0"
                fp = join(reverse(ch[1:min(nf, length(ch))]))
                length(ch) < nf && (fp = repeat("0", nf - length(ch)) * fp)
                ip * "." * fp
            end
            @test digit_string(sd) == rebuilt
            # and the array itself must reconstruct the value
            @test sum(BigInt(d[i]) * Rational{BigInt}(4)^(e + i - 1) for i in eachindex(d);
                      init = zero(Rational{BigInt})) == value(sd)
        end
    end
    # 712.3 specifically, since it prompted the check
    let sd = to_rr4(712.3)
        @test value(sd) == Rational{BigInt}(712.3)
        @test Float64(value(sd)) == 712.3
        @test length(sd.digits) == 27          # 53 significand bits ÷ 2 bits/digit
        @test sd.exponent == -21
        @test scaled_integer(sd) == 3132728529859379
    end
    @test to_digits_parts(CSD, 231).digits == [-1, 0, 0, 1, 0, -1, 0, 0, 1]
    @test to_digits_parts(RADIX4, 49).digits == [1, 0, 3]
    @test to_digits_parts(BINARY, 231).scale == 0
end

@testset "digit-string arithmetic" begin
    a, b = to_rr4(712.3), to_rr4(12.3)
    @test value(a + b) == Rational{BigInt}(712.3) + Rational{BigInt}(12.3)
    @test value(a - b) == Rational{BigInt}(712.3) - Rational{BigInt}(12.3)
    @test value(a + 12.3) == value(a + b)          # mixed with a plain number
    @test value(12.3 + a) == value(a + b)
    @test value(a * b) == Rational{BigInt}(712.3) * Rational{BigInt}(12.3)

    # exact where Float64 addition is not
    @test value(a + b) != Rational{BigInt}(712.3 + 12.3)

    # negation is free: flip the digits, no borrow
    @test (-to_rr4(49)).digits == .-to_rr4(49).digits
    @test value(-to_rr4(49)) == -49
    @test value(to_rr4(49) - to_rr4(49)) == 0

    for (x, y) in ((49, 43), (712.3, 12.3), (-7//8, 1//4), (0, 5), (1024, -1023))
        for alp in (MIN_REDUNDANT, MAX_REDUNDANT)
            u = to_rr4(x; alphabet = alp); v = to_rr4(y; alphabet = alp)
            @test value(u + v) == Rational{BigInt}(x) + Rational{BigInt}(y)
            @test value(u - v) == Rational{BigInt}(x) - Rational{BigInt}(y)
            @test value(u * v) == Rational{BigInt}(x) * Rational{BigInt}(y)
            @test all(d -> abs(d) <= alphabet_halfwidth(alp), (u + v).digits)
            @test all(d -> abs(d) <= alphabet_halfwidth(alp), (u * v).digits)
        end
    end

    # == is VALUE equality, because the system is redundant
    @test to_rr4(49) == to_rr4_maximal(49)
    @test to_rr4(49) == 49
    @test !same_spelling(to_rr4(49), to_rr4_maximal(49))
    @test same_spelling(to_rr4(49), to_rr4(49))
    @test hash(to_rr4(49)) == hash(to_rr4_maximal(49))

    @test iszero(to_rr4(0)) && !iszero(to_rr4(1))
    @test value(abs(to_rr4(-49))) == 49
    @test to_rr4(5) < to_rr4(99)
    @test value(maximum([to_rr4(5), to_rr4(99), to_rr4(2)])) == 99
    @test Float64(to_rr4(49.25)) == 49.25

    @test value(sum_rr4([1, 2, 3, 4, 5])) == 15
    @test value(sum_rr4(Float64[])) == 0
    @test value(dot_rr4([1, 2, 3], [4, 5, 6])) == 32
    @test value(sum_rr4([712.3, 12.3, 0.4])) ==
          Rational{BigInt}(712.3) + Rational{BigInt}(12.3) + Rational{BigInt}(0.4)
    @test_throws DimensionMismatch dot_rr4([1, 2], [1, 2, 3])

    # mixing radices must be refused, not silently coerced
    @test_throws ArgumentError to_rr4(5) + to_digits(CSD, 5)
    # CSD (radix 2) arithmetic still works within its own system
    @test value(to_digits(CSD, 231) + to_digits(CSD, 17)) == 248
    @test value(to_digits(CSD, 231) * to_digits(CSD, 3)) == 693
end

@testset "rr4_representations table" begin
    r = rr4_representations(12.5)
    @test r.method == :minimally_redundant
    @test r.alphabet == MIN_REDUNDANT && r.maxdigit == 2
    @test r.ndigits == 4 && r.scale == -1
    @test Base.size(r.digits) == (2, 4)
    @test r.value == 25//2
    @test length(r) == 2 && !isempty(r)
    # rows are MSB-first, i.e. the reverse of the .digits field
    for (i, rep) in enumerate(r.reps)
        @test r.digits[i, :] == reverse(rep.digits)
        @test value(rep) == r.value
        @test rep.exponent == r.scale
    end
    # the scale column really is the scale
    ws = digit_matrix_with_scale(r)
    @test Base.size(ws) == (2, 5)
    @test all(ws[:, end] .== r.scale)
    @test ws[:, 1:end-1] == digit_matrix(r)

    # method selects the alphabet, by either spelling
    @test rr4_representations(25; method = :maximally_redundant).maxdigit == 3
    @test rr4_representations(25; method = :max).maxdigit == 3
    @test rr4_representations(25; method = MAX_REDUNDANT).maxdigit == 3
    @test rr4_representations(25; method = :minimally_redundant).maxdigit == 2
    @test_throws ArgumentError rr4_representations(25; method = :sideways)

    @test length(rr4_representations(25; method = :maximally_redundant)) == 4
    @test length(rr4_representations(10; method = :maximally_redundant, ndigits = 4)) == 6
    @test length(rr4_representations(10; ndigits = 4)) == 3

    # rows sorted cheapest-first, and every row genuinely equals the value
    for v in (10, 25, 49, 12.5, -7), m in (:minimally_redundant, :maximally_redundant)
        t = rr4_representations(v; method = m, ndigits = 5)
        @test issorted([weight(x) for x in t.reps])
        @test all(x -> value(x) == t.value, t.reps)
        @test all(d -> abs(d) <= t.maxdigit, t.digits)
        # the matrix must reconstruct the value, read back LSB-first
        for i in axes(t.digits, 1)
            row = reverse(t.digits[i, :])
            @test sum(BigInt(row[k]) * Rational{BigInt}(4)^(t.scale + k - 1)
                      for k in eachindex(row)) == t.value
        end
    end

    # regression: enumeration used to try every digit at every position (5^n), so any
    # Float64 — which needs ~27 radix-4 digits — never returned.  Divisibility pruning
    # means only digits d ≡ rem (mod 4) are explored, at most two per position.
    @test count_rr4_representations(6.3) == 2
    @test length(rr4_representations(6.3)) == 2
    @test all(x -> value(x) == Rational{BigInt}(6.3), rr4_representations(6.3).reps)
    for v in (6.3, 712.3, 49.2, 16.1656375)
        r = rr4_representations(v)
        @test r.total == count_rr4_representations(v)
        @test all(x -> value(x) == Rational{BigInt}(v), r.reps)
    end
    # enumeration and the independent DP count must agree everywhere
    for v in (0, 1, 6, 10, 12, 25, 49, 231, -7, 12.5, 6.3, 1//8), n in (3, 4, 5, 6),
        alp in (MIN_REDUNDANT, MAX_REDUNDANT)
        @test length(all_rr4_representations(v, n; alphabet = alp)) ==
              count_rr4_representations(v, n; alphabet = alp)
    end
    # the limit guard reports the true total while listing only part of it
    @test count_rr4_representations(0.1) == 317811
    let r = rr4_representations(0.1)
        @test r.total == 317811
        @test length(r.reps) == 10_000 && r.truncated
        @test all(x -> value(x) == Rational{BigInt}(0.1), r.reps)
    end
    let r = rr4_representations(16.1656375; limit = 10)
        @test r.total == 336 && length(r.reps) == 10 && r.truncated
    end
    @test !rr4_representations(6.3).truncated
    @test length(all_rr4_representations(0.1, min_ndigits(0.1); limit = 5)) == 5

    # the cost preview: exact count, no enumeration
    let c = rr4_complexity(6.3)
        @test c.ndigits == 27 && c.total == 2 && c.listable
        @test c.naive_search == big(5)^27
        @test c.method == :minimally_redundant && c.maxdigit == 2
    end
    let c = rr4_complexity(0.1)
        @test c.total == 317811 && !c.listable
        @test c.bytes > 0
    end
    @test rr4_complexity(25; method = :maximally_redundant).maxdigit == 3
    @test rr4_complexity(0.1; limit = 10^6).listable
    for v in (6.3, 12.5, 49, 231, 0.1)
        @test rr4_complexity(v).total == count_rr4_representations(v)
    end
    # counting stays cheap at widths no enumerator could touch
    @test count_rr4_representations(0.1, 600) == 514229
    @test count_rr4_representations(0.1, 120; alphabet = MAX_REDUNDANT) > 6e9
    # verbose is suppressible
    @test length(rr4_representations(0.1; verbose = false).reps) == 10_000

    # why the counts behave as they do: only rem ≡ 2 (mod 4) offers a choice
    @test rr4_branch_residues() == [0 => [0], 1 => [1], 2 => [-2, 2], 3 => [-1]]
    @test count(p -> length(p.second) > 1, rr4_branch_residues()) == 1
    @test count(p -> length(p.second) > 1,
                rr4_branch_residues(:maximally_redundant)) == 3

    # closed form for the maximum count, checked against exhaustive search
    @test [Int(rr4_max_count(n)) for n in 1:8] == [1, 2, 3, 5, 8, 13, 21, 34]
    @test [Int(rr4_max_count(n; method = :maximally_redundant)) for n in 1:8] ==
          [1, 2, 4, 8, 16, 32, 64, 128]
    for n in 1:9, m in (:minimally_redundant, :maximally_redundant)
        a = alphabet_halfwidth(xpuFP._method_alphabet(m))
        best = 0
        for N in 0:(a * 4^n)
            c = count_rr4_representations(N, n; method = m)
            c > best && (best = c)
        end
        @test best == rr4_max_count(n; method = m)
    end
    # no value can exceed the bound
    for v in (0.1, 26.35, 26.4, 712.3, 231), n in (min_ndigits(v), min_ndigits(v) + 1)
        @test count_rr4_representations(v, n) <= rr4_max_count(n)
    end
    # 0.1 actually attains it at width 27 and 28
    @test count_rr4_representations(0.1, 27) == rr4_max_count(27) == 317811
    @test count_rr4_representations(0.1, 28) == rr4_max_count(28) == 514229

    # a width that cannot hold the value yields an empty table, not an error
    e = rr4_representations(49; ndigits = 3)
    @test isempty(e) && length(e) == 0
end

@testset "minimal weight and its statistics" begin
    # the DP must agree with brute-force enumeration wherever enumeration is feasible
    for v in (0, 1, 6, 10, 12, 25, 49, 231, 1000), n in 3:6,
        m in (:minimally_redundant, :maximally_redundant)
        reps = all_rr4_representations(v, n; method = m)
        dp = minimal_weight(v; method = m, ndigits = n)
        if isempty(reps)
            @test dp == -1
        else
            @test dp == minimum(weight, reps)
            # the distribution must agree with the count and locate the minimum
            d = weight_distribution(v; method = m, ndigits = n)
            @test sum(d; init = big(0)) == length(reps)
            @test findfirst(!iszero, d) - 1 == dp
            for k in 0:length(d)-1
                @test d[k+1] == count(r -> weight(r) == k, reps)
            end
        end
    end

    @test minimal_weight(231) == 4 && weight(to_rr4(231)) == 5
    @test minimal_weight(1000) == 3 && weight(to_rr4(1000)) == 4
    # the minimum can never exceed the one-pass conversion
    for v in (6, 26, 231, 1000, 12345, 26.35, 712.3)
        n = min_ndigits(v)
        @test minimal_weight(v) <= weight(to_rr4(v; ndigits = n))
    end
    # nested alphabets: more redundancy cannot need more nonzeros at a fixed width
    for v in (231, 1000, 12345, 99999)
        n = max(min_ndigits(v), min_ndigits(v; method = :maximally_redundant))
        @test minimal_weight(v; method = :maximally_redundant, ndigits = n) <=
              minimal_weight(v; method = :minimally_redundant, ndigits = n)
    end

    st = weight_stats(collect(1:200))
    @test st.nsamples == 200
    @test st.min_weight <= st.median_weight <= st.max_weight
    @test 0 <= st.density <= 1
    @test st.canonical_mean >= st.mean_weight        # searching cannot do worse
    @test 0 <= st.saving < 1

    rows = weight_scaling([8, 16]; samples = 60, rng = MersenneTwister(3))
    @test length(rows) == 2
    @test all(r -> 0.55 < r.density < 0.80, rows)
    @test all(r -> r.canonical_density >= r.density, rows)
    # the measured limits, loosely bounded so the test is about the law not the noise
    big2 = weight_scaling([48]; samples = 120, rng = MersenneTwister(7))[1]
    big3 = weight_scaling([48]; method = :maximally_redundant, samples = 120,
                          rng = MersenneTwister(7))[1]
    @test abs(big2.density - 2/3) < 0.02
    @test abs(big3.density - 3/5) < 0.02
    @test big3.density < big2.density                # redundancy lowers the density
end

@testset "canonical forms" begin
    # :minweight is exact, minimum weight, and deterministic
    for v in 0:400
        c = canonical_rr4(v)
        @test value(c) == v
        @test weight(c) == minimal_weight(v)
        @test same_spelling(c, canonical_rr4(v))
        @test all(d -> abs(d) <= 2, c.digits)
    end
    for v in (231, 1000, 12345, 99999, 26.5, -7)
        @test value(canonical_rr4(v)) == (v isa Integer ? v : Rational{BigInt}(v))
        @test weight(canonical_rr4(v)) == minimal_weight(v)
    end
    @test weight(canonical_rr4(231)) == 4 && weight(to_rr4(231)) == 5
    @test value(canonical_rr4(231; method = :maximally_redundant)) == 231
    @test weight(canonical_rr4(231; rule = :onepass)) == weight(to_rr4(231))
    @test_throws ArgumentError canonical_rr4(231; rule = :sideways)

    # the non-adjacent form: unique and minimum weight WHERE IT EXISTS, but it usually
    # does not — which is exactly why CSD's construction fails to carry over to radix 4
    for v in 0:300, m in (:minimally_redundant, :maximally_redundant)
        n = min_ndigits(v; method = m) + 2
        reps = all_rr4_representations(v, n; method = m)
        isempty(reps) && continue
        naf = filter(r -> !has_adjacent_nonzeros(r), reps)
        @test length(naf) <= 1                       # never ambiguous
        f = nonadjacent_rr4(v; method = m, ndigits = n)
        if isempty(naf)
            @test f === nothing
            @test !has_nonadjacent_form(v; method = m, ndigits = n)
        else
            @test f !== nothing && value(f) == v
            @test weight(f) == minimum(weight, reps)  # and always minimum weight
            @test has_nonadjacent_form(v; method = m, ndigits = n)
        end
    end
    # most values have none at all
    @test count(v -> has_nonadjacent_form(v), 0:600) / 601 < 0.25
    @test nonadjacent_rr4(231) === nothing
    @test_throws ArgumentError canonical_rr4(231; rule = :nonadjacent)

    # the one-pass form is genuinely not minimal, for a substantial fraction of values
    @test count(v -> weight(to_rr4(v)) > minimal_weight(v), 0:2000) > 400
end

@testset "csd on wide integers" begin
    # regression: csd() narrowed its argument with Int(x) and overflowed on BigInt
    for x in (big(15655175250120600661), big(2)^200 + 12345, big(10)^40)
        d = csd(x)
        @test value(d) == x
        @test !has_adjacent_nonzeros(d)
        @test value(csd(-x)) == -x
    end
    @test binary_weight(big(2)^200 + 7) == 4
    @test digit_string(csd(231)) == "1001̄01001̄"     # small values unchanged
    @test [r.digit for r in csd_trace(27)] == [-1, 0, -1, 0, 0, 1]
    # NAF density: the classical 1/3, measured on wide operands
    rng = MersenneTwister(2)
    dens = Statistics.mean(weight(csd(rand(rng, big(2)^127:big(2)^128))) / 128 for _ in 1:100)
    @test abs(dens - 1/3) < 0.03
end

@testset "Booth" begin
    @test value(booth_radix4(13, 6)) == 13
    @test booth_radix4(13, 6).digits == [1, -1, 1]
    @test value(booth_radix4(-74, 8)) == -74
    @test booth_radix4(-74, 8).digits == [-2, 2, -1, -1]
    @test verify_booth_exhaustive(8)
    @test verify_booth_exhaustive(10)
    bp = booth_multiply(11, 13, 6)
    @test bp.rows == [11, -44, 176] && bp.product == 143
    bp = booth_multiply(93, -74, 8)
    @test bp.rows == [-186, 744, -1488, -5952] && bp.product == -6882
    @test booth_rows(24) == (24, 13)
    p = booth_pairing(-74, 8)
    @test p.packed == p.radix4.digits
    @test value(booth_radix4_unsigned(12345, 16)) == 12345
end

@testset "carry-save" begin
    @test verify_csa_identity(3000)
    u, t = csa(93, 118, 45)
    @test u + t == 93 + 118 + 45
    @test reduction_schedule(24) == [24, 16, 11, 8, 6, 4, 3, 2]
    @test reduction_schedule(13) == [13, 9, 6, 4, 3, 2]
    A, B = 93, 118
    plain = [((B >> i) & 1) * (A << i) for i in 0:7]
    w = wallace_reduce(plain)
    @test w.pair == (5342, 5632) && w.total == A * B && w.exact
    bp = booth_multiply(A, B, 8)
    w2 = wallace_reduce(bp.rows)
    @test w2.pair == (-8738, 19712) && w2.total == A * B
    @test verify_tree_invariant(plain)
    @test verify_tree_invariant(collect(1:24))
    @test verify_tree_invariant(collect(1:24); schedule = :dadda)
    sv = booth_tree_saving(24)
    @test sv.plain_levels == 7 && sv.booth_levels == 5 && sv.cell_ratio == 0.5
    # u* is NOT the xor of everything — the invariant is the whole proof
    @test reduce(xor, plain) != w.pair[1]
    @test wallace_reduce(collect(1:24)).nlevels == dadda_reduce(collect(1:24)).nlevels
end

@testset "systolic array" begin
    r = systolic_run([1.0 2.0; 3.0 4.0], [5.0 6.0; 7.0 8.0])
    @test r.C == [19.0 22.0; 43.0 50.0]
    @test r.cycles == 4
    @test r.activity == [1, 3, 3, 1]   # wavefront; sums to M·P·K = 8
    @test r.exact
    @test systolic_cycles(4, 3, 5) == 10
    r2 = systolic_run(randn(MersenneTwister(2), 4, 3), randn(MersenneTwister(3), 3, 5))
    @test r2.exact
    @test sum(r2.activity) == 4 * 5 * 3
    @test systolic_utilization(4, 3, 5) ≈ 0.3 atol = 1e-9
end


@testset "quick SNR simulation" begin
    # The LUT must reproduce the reference quantizer exactly, or the sweep path is
    # measuring a different format from the one the rest of the package implements.
    for f in (E2M1, E1M2, E3M0, E4M3, E5M2, INT4, INT8)
        L = gridlut(f)
        g = grid(f)
        probes = Float64[]
        append!(probes, g)
        append!(probes, [(g[i] + g[i+1]) / 2 for i in 1:length(g)-1])   # exact ties
        append!(probes, g .* 1.0001, g .* 0.9999)
        append!(probes, (2 .* rand(MersenneTwister(7), 5000) .- 1) .* (2 * maxfinite(f)))
        @test all(u -> lut_quantize(L, u) == quantize(f, u), filter(isfinite, probes))
    end
    LE = gridlut(E8M0)                                  # unsigned, and has no zero code
    ge = filter(>(0), grid(E8M0))
    @test all(u -> lut_quantize(LE, u) == quantize(E8M0, u),
              vcat(ge, 1.3 .* ge, 0.7 .* ge))

    # quick_snr is an optimisation of measure_snr, not a different estimator: same
    # data in, same decibels out, for every scale rule and both scale formats.
    for dist in (:gaussian, :outlier, :sparse, :lognormal)
        x = quick_data(dist, 20_000; rng = MersenneTwister(11))
        for bf in (MXFP4, NVFP4, MXINT4, MXFP4_BEST32, XPFP4_32,
                   fp4_variant(rule = OPT_SHIFT), fp4_variant(K = 64),
                   fp4_variant(K = 16, scale = E4M3, rule = NV_MAXDIV))
            @test quick_snr(bf, x).snr ≈ measure_snr(bf, x) atol = 1e-9
        end
    end

    # a trailing partial block is encoded at its own length, as reconstruct does
    xp = randn(MersenneTwister(4), 32 * 7 + 5)
    @test quick_snr(MXFP4, xp).snr ≈ measure_snr(MXFP4, xp) atol = 1e-9
    @test quick_snr(MXFP4, xp).blocks == 8

    q = quick_snr(MXFP4; n = 200_000)
    @test q.eff_bits ≈ q.snr / DB_PER_BIT
    @test q.dot_snr ≈ q.snr - 3.01
    @test q.bits ≈ 4.25
    @test q.db_per_bit ≈ q.snr / 4.25
    # The report's 18.8 dB, which `estimate_element_snr`'s order-statistic quadrature
    # independently derives — checked here against the constant rather than by calling
    # it, because that quadrature takes minutes and this whole file takes seconds.
    @test q.snr ≈ 18.8 atol = 0.1
    @test 0.05 < q.zeroed < 0.15                    # SNR cannot see these
    @test q.blocks == 200_000 ÷ 32

    # the zeroed fraction is the same one quantize_block reports
    xb = randn(MersenneTwister(9), 32 * 500)
    zq = sum(zeroed_count, quantize_blocked(MXFP4, xb)) / count(!=(0), xb)
    @test quick_snr(MXFP4, xb).zeroed ≈ zq atol = 1e-12

    # Every rule clips a little, which is the point of reporting the fraction. The MX
    # floor rule leaves M/S ∈ [4,8) against a grid that stops at 6, so the block maximum
    # saturates whenever it lands high in its binade; NVFP4 aims the maximum *at* the top
    # code but has to round M/6 onto E4M3, and rounding down overloads. Neither is a bug,
    # and neither is zero — read `clipped` as a dial, not an alarm.
    @test quick_snr(MXFP4; n = 100_000).clipped > 0
    @test quick_snr(NVFP4; n = 100_000).clipped > 0
    @test quick_snr(XPFP4_32; n = 100_000).snr > quick_snr(MXFP4; n = 100_000).snr

    # the design-space facts the sweep exists to show
    g = randn(MersenneTwister(13), 400_000)
    @test quick_snr(NVFP4, g).snr > quick_snr(MXFP4, g).snr                 # finer scale
    @test quick_snr(fp4_variant(rule = OPT_SHIFT), g).snr >
          quick_snr(MXFP4, g).snr                                            # one compare
    # Block length runs the *other* way from the obvious guess: under the floor rule a
    # longer block measures slightly better, because the rule only ever sees the block
    # maximum and a larger sample of Gaussians lands its maximum higher in the binade,
    # where less of the element grid is wasted. So NVFP4's advantage over MXFP4 is bought
    # by its E4M3 scale format, not by its shorter block — the sweep exists to catch
    # exactly this kind of inverted intuition.
    @test quick_snr(fp4_variant(K = 64), g).snr > quick_snr(fp4_variant(K = 16), g).snr
    @test bits_per_element(fp4_variant(K = 16)) >
          bits_per_element(fp4_variant(K = 64))                               # and costs more
    @test quick_snr(MXINT4, g).snr < quick_snr(MXFP4, g).snr                 # uniform grid

    v = fp4_variant(K = 16, scale = E4M3, rule = NV_MAXDIV)
    @test v.K == 16 && v.scale === E4M3 && v.rule == NV_MAXDIV && v.elem === E2M1
    @test bits_per_element(v) ≈ bits_per_element(NVFP4)
    @test occursin("K16", v.name)
    @test fp4_variant(name = "mine").name == "mine"

    @test_throws ArgumentError quick_data(:nonesuch, 10)
    @test_throws ArgumentError quick_snr(MXFP4, Float64[])
end


@testset "FP4-K16-E4M3-MSE_OPTIMAL (docs/src/mseopt16.md)" begin
    # Every number quoted on the page is pinned here, so the prose cannot drift away
    # from the code without a test going red.
    F = fp4_variant(K = 16, scale = E4M3, rule = MSE_OPTIMAL)

    # the tuple is NVFP4_BEST16 / XPFP4_16 under a descriptive name, not a new format
    @test (F.K, F.elem, F.scale, F.rule) ===
          (NVFP4_BEST16.K, NVFP4_BEST16.elem, NVFP4_BEST16.scale, NVFP4_BEST16.rule)
    @test XPFP4_16 === NVFP4_BEST16
    @test bits_per_block(F) == 72
    @test bits_per_element(F) ≈ 4.5
    @test maxfinite(E2M1) == 6 && elem_emax(F) == 2

    # the E2M1 code table on the page, all sixteen rows
    @test [decode(E2M1, UInt64(c)) for c in 0:15] ==
          [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
           -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

    # E4M3's span, quoted as 2^-9 … 448
    @test minsubnormal(E4M3) == 2.0^-9
    @test maxfinite(E4M3) == 448
    @test E4M3.bias == 7 && E4M3.mbits == 3

    # the tensor scale formula G = 2^(floor(log2(M/6)) - 7)
    xt = vcat(round.(0.02 .* randn(MersenneTwister(116), 16); digits = 5),
              0.02 .* randn(MersenneTwister(8), 16 * 40))
    @test tensor_scale(F, xt) ==
          exp2(binade_exponent(maximum(abs, xt) / 6) - 7)
    @test tensor_scale(F, xt) == 2.0^-14

    # ---- the worked block ----------------------------------------------------
    x = round.(0.02 .* randn(MersenneTwister(116), 16); digits = 5)
    M = maximum(abs, x)
    @test M ≈ 0.05310 atol = 1e-9
    anchor = M / 6
    @test anchor ≈ 0.008850 atol = 1e-6

    # the search window: one octave below the anchor, half an octave above
    S = block_scale(F, x)
    S0 = quantize(E4M3, anchor)                     # what NV_MAXDIV would pick
    @test S0 == 0.009765625 && encode(E4M3, S0) == 0x05
    @test S  == 0.0078125   && encode(E4M3, S)  == 0x04
    @test S < S0                                    # the rule deliberately overloads
    @test M / S ≈ 6.7968 atol = 1e-4                # past the top code of 6
    @test 6 / sqrt(2) <= M / S <= 12                # inside the window's bounds

    # only a handful of E4M3 scales exist in a 1.5-octave window
    cands = unique(quantize(E4M3, exp2(t))
                   for t in range(log2(anchor) - 1, log2(anchor) + 0.5; length = 200))
    @test length(cands) == 5
    @test S in cands

    # and S is genuinely the SSE minimiser over them
    sse(c) = sum(v -> (quantize(E2M1, v / c) * c - v)^2, x)
    @test all(sse(S) <= sse(c) for c in cands)

    qb = quantize_block(F, x)
    @test qb.scale == S
    @test qb.scale_code == 0x04
    @test length(qb.codes) == 16
    @test qb.values ≈ [quantize(E2M1, v / S) * S for v in x]

    # row 3 of the table: the clamped maximum, traced by hand on the page
    @test x[3] ≈ -0.05310 atol = 1e-9
    @test x[3] / S ≈ -6.7968 atol = 1e-4
    @test qb.elements[3] == -6.0                    # saturates, does not wrap
    @test qb.codes[3] == 0b1111
    @test qb.values[3] ≈ -0.046875 atol = 1e-9
    @test qb.values[3] - x[3] ≈ 0.006225 atol = 1e-6

    # row 4: annihilated, and SNR cannot see it
    @test abs(x[4] / S) < 0.25                      # below the round-to-zero threshold
    @test qb.values[4] == 0 && x[4] != 0
    @test zeroed_count(qb) == 1
    @test count(v -> abs(v) > 6S, x) == 2           # two clamped

    # the +4.35 dB the search bought on this block
    v0 = [quantize(E2M1, v / S0) * S0 for v in x]
    @test snr_db(x, v0) ≈ 16.939 atol = 0.001
    @test snr_db(x, qb.values) ≈ 21.289 atol = 0.001
    @test snr_db(x, qb.values) - snr_db(x, v0) ≈ 4.350 atol = 0.002
    @test count(i -> x[i] != 0 && v0[i] == 0, 1:16) == 2   # anchoring zeroes one more

    # ---- the population claim ------------------------------------------------
    a = quick_snr(NVFP4;  n = 1_000_000)
    b = quick_snr(F;      n = 1_000_000)
    @test a.snr ≈ 20.440 atol = 0.01
    @test b.snr ≈ 21.588 atol = 0.01
    @test b.snr - a.snr ≈ 1.15 atol = 0.02          # "+1.15 dB for free"
    @test a.bits == b.bits == 4.5                   # same wire cost
    @test b.zeroed > a.zeroed                       # the honest caveat on the page

    # the dot-product core bound quoted as 16 x 6 x 6
    @test core_sum_bound(F) == 16 * 6 * 6 == 576
end

@testset "carry-free recoding and accumulators" begin
    # ---- rr4_recode: the parallel conversion -------------------------------
    # exactness over a wide range, both signs, including the awkward small cases
    for v in vcat(0:200, [255, 256, 1023, 2931, 65535, 10^6, 2^40 + 7],
                  -1 .* [1, 2, 3, 38, 255, 2931, 10^6])
        z = rr4_recode(v)
        @test value(z) == v
        @test z.radix == 4 && z.maxdigit == 2
        @test all(d -> -2 <= d <= 2, z.digits)      # inside the alphabet
    end
    for v in rand(MersenneTwister(5), -10^9:10^9, 400)
        @test value(rr4_recode(v)) == v
    end

    # the carry of position i is a function of digit i ALONE — that is what makes it
    # parallel, and it is the one property the sequential rewrite does not have.
    for v in rand(MersenneTwister(6), 0:10^7, 200)
        tbl = rr4_recode_table(v)
        @test all(r -> r.c_out == (r.d >= 2 ? 1 : 0), tbl)
        @test all(r -> r.z == r.d - 4r.c_out + r.c_in, tbl)
        @test all(r -> -2 <= r.z <= 2, tbl)
    end

    # same value as to_rr4, and the spellings diverge exactly where the rules do:
    # at digit 2 with no incoming carry.
    for v in 0:3000
        @test value(rr4_recode(v)) == value(to_rr4(v))
    end
    @test same_spelling(to_rr4(2931), rr4_recode(2931))       # rules happen to agree
    @test !same_spelling(to_rr4(38), rr4_recode(38))          # d=2, c_in=0 — diverge
    @test !same_spelling(to_rr4(2666), rr4_recode(2666))
    @test digit_string(to_rr4(38)) != digit_string(rr4_recode(38))

    # ---- inputs -------------------------------------------------------------
    @test length(accumulator_inputs(64)) == 64
    @test all(>(0), accumulator_inputs(50; kind = :positive))
    @test accumulator_inputs(5; kind = :ramp) == [1, 2, 3, 4, 5]
    @test accumulator_inputs(5; kind = :ones) == ones(Int, 5)
    @test sum(accumulator_inputs(10; kind = :alternating, magnitude = 7)) == 0
    @test_throws ArgumentError accumulator_inputs(4; kind = :nonesuch)

    # ---- every accumulator computes the SAME value --------------------------
    # redundancy trades cost, never accuracy; this is the assertion that says so.
    for kind in (:random, :positive, :ramp, :ones, :alternating)
        for n in (1, 2, 7, 64, 300)
            xs = accumulator_inputs(n; kind, rng = MersenneTwister(n))
            ref = sum(BigInt, xs)
            for sched in (:sequential, :tree)
                r1 = accumulate_rr4(xs; schedule = sched)
                r2 = accumulate_carry(xs; adder = :ripple, schedule = sched)
                r3 = accumulate_carry(xs; adder = :prefix, schedule = sched)
                @test r1.value == r2.value == r3.value == ref
                @test r1.exact && r2.exact && r3.exact
                @test r1.n == r2.n == n
            end
        end
    end
    # an explicit vector, not generated
    @test accumulate_rr4([3, 1, 4, 1, 5, 9, 2, 6]).value == 31

    # ---- the cost model's defining property ---------------------------------
    # RR4 add depth is 3 per term REGARDLESS of accumulator width; that is the claim
    # the whole example exists to test.
    for mag in (10, 10^3, 10^9, 10^15)
        xs = accumulator_inputs(256; kind = :positive, magnitude = mag,
                                rng = MersenneTwister(2))
        r = accumulate_rr4(xs)
        @test r.add_depth == 3 * (256 - 1)          # independent of mag
        @test r.exit_cpa <= 8                        # one small amortised conversion
    end
    # ...whereas ripple's depth grows with the width it is asked to carry
    narrow = accumulate_carry(accumulator_inputs(256; kind = :positive, magnitude = 10,
                                                 rng = MersenneTwister(2)); adder = :ripple)
    wide   = accumulate_carry(accumulator_inputs(256; kind = :positive, magnitude = 10^15,
                                                 rng = MersenneTwister(2)); adder = :ripple)
    @test wide.depth > 2 * narrow.depth
    @test wide.acc_bits > narrow.acc_bits

    # per-term depth is flat in N for RR4, and rising for ripple
    rr4_pt = [accumulate_rr4(accumulator_inputs(n; rng = MersenneTwister(1))).per_term
              for n in (64, 1024, 4096)]
    @test all(p -> 3.0 <= p <= 3.2, rr4_pt)
    rip_pt = [accumulate_carry(accumulator_inputs(n; rng = MersenneTwister(1));
                               adder = :ripple).per_term for n in (64, 1024, 4096)]
    @test issorted(rip_pt)                           # gets worse as the sum widens

    # a tree beats a sequential chain for every method
    xs = accumulator_inputs(1024; rng = MersenneTwister(3))
    for m in (() -> accumulate_rr4(xs; schedule = :tree),
              () -> accumulate_carry(xs; adder = :ripple, schedule = :tree))
        @test m().depth < 200                        # log-depth, not linear
    end
    @test accumulate_rr4(xs; schedule = :tree).depth <
          accumulate_rr4(xs; schedule = :sequential).depth

    # exit model is a choice, and is paid once either way
    rp = accumulate_rr4(xs; exit = :prefix)
    rr = accumulate_rr4(xs; exit = :ripple)
    @test rp.add_depth == rr.add_depth               # only the exit differs
    @test rr.exit_cpa > rp.exit_cpa
    @test rr.exit_cpa == rr.acc_bits + 1

    @test_throws ArgumentError accumulate_rr4(Int[])
    @test_throws ArgumentError accumulate_carry([1, 2]; adder = :nonesuch)
    @test_throws ArgumentError accumulate_rr4([1, 2]; schedule = :nonesuch)
end


@testset "cost model — memory, cells, depth" begin
    # The model's job is to reproduce the structural facts; these pin them so the
    # numbers quoted in examples/06 and the docs cannot drift.

    # ---- addition ----------------------------------------------------------
    a = add_costs(64)
    m(v, k) = v[findfirst(x -> x.method === k, v)]
    @test m(a, :ripple).depth == 65
    @test m(a, :prefix).depth == 6                    # ceil(log2 64)
    @test m(a, :carry_save).depth == 1
    @test m(a, :rr4_carry_free).depth == 3            # the whole point
    # carry-free depth does NOT move with width; every other scheme's does
    for w in (8, 16, 64, 256, 1024)
        @test m(add_costs(w), :rr4_carry_free).depth == 3
        @test m(add_costs(w), :ripple).depth == w + 1
    end
    # ...and it is paid for in state, always, by exactly 1.5x
    for w in (8, 16, 64, 256)
        r = m(add_costs(w), :rr4_carry_free)
        @test r.state_bits == cld(w, 2) * RR4_BITS_PER_DIGIT
        @test r.state_bits / m(add_costs(w), :ripple).state_bits ≈ 1.5 atol = 0.1
        @test m(add_costs(w), :carry_save).state_bits == 2w      # two words
    end
    @test RR4_BITS_PER_DIGIT == 3 && BINARY_BITS_PER_RADIX4_DIGIT == 2

    # ---- accumulation ------------------------------------------------------
    b = accumulate_costs(1024, 64)
    @test m(b, :rr4_carry_free).depth == 3 * 1024 + 6      # 3 per term + one exit
    @test m(b, :ripple).depth == 1024 * 65
    @test m(b, :ripple).depth / m(b, :rr4_carry_free).depth > 20
    # the exit is paid once, so its share shrinks as n grows
    share(n) = (accumulate_costs(n, 64)[end].depth - 3n) / accumulate_costs(n, 64)[end].depth
    @test share(16) > share(1024) > share(65536)
    # the model agrees with the executed arithmetic on the RR4 add depth
    xs = accumulator_inputs(256; rng = MersenneTwister(4))
    @test accumulate_rr4(xs).add_depth == 3 * 255
    @test accumulate_costs(256, 64)[end].depth - accumulate_costs(256, 64)[end].depth % 1 ==
          3 * 256 + 6

    # ---- multiplication ----------------------------------------------------
    c = multiply_costs(24)
    @test m(c, :booth_wallace).cells < m(c, :wallace_plain).cells
    @test m(c, :booth_wallace).state_bits < m(c, :wallace_plain).state_bits
    @test m(c, :booth_wallace).depth <= m(c, :wallace_plain).depth
    # Booth is the only scheme here that wins cells AND state AND depth at once
    bw, wp = m(c, :booth_wallace), m(c, :wallace_plain)
    @test bw.cells / wp.cells < 0.75
    @test wp.state_bits / bw.state_bits > 1.5
    # rows really do halve, at every width, and that drives the PP storage
    for n in (8, 16, 24, 32, 53, 64)
        plain, booth = booth_rows(n)
        @test booth == cld(n + 1, 2)
        mc = multiply_costs(n)
        @test m(mc, :wallace_plain).state_bits == plain * 2n
        @test m(mc, :booth_wallace).state_bits == booth * 2n
        s = booth_tree_saving(n)
        @test s.booth_rows == booth && s.cell_ratio < 0.6
        @test s.booth_levels <= s.plain_levels
    end
    # a tree buys depth by spending storage — the honest counterpart
    @test m(c, :wallace_plain).state_bits > m(c, :shift_add).state_bits
    @test m(c, :wallace_plain).depth < m(c, :shift_add).depth

    # ---- CSD constants -----------------------------------------------------
    r = constant_multiply_costs(231)
    @test r.binary_weight == binary_weight(231)
    @test r.csd_weight == length(filter(!iszero, csd(231).digits))
    @test r.csd_adders <= r.binary_adders
    @test r.adders_saved == r.binary_adders - r.csd_adders
    # runs of ones are exactly where CSD pays: 2^k - 1 needs k adders binary, 1 in CSD
    for k in (4, 8, 10, 12, 16)
        rr = constant_multiply_costs(2^k - 1)
        @test rr.binary_weight == k
        @test rr.csd_weight == 2                      # 2^k - 1 = 2^k - 1
        @test rr.csd_adders == 1
        @test rr.adders_saved == k - 2
    end
    # never worse than binary, over a wide sweep
    @test all(constant_multiply_costs(c).csd_adders <=
              constant_multiply_costs(c).binary_adders for c in 1:2000)

    # ---- format memory: the savings that ARE memory ------------------------
    @test compression_ratio(MXFP4, FP32) ≈ 32 / 4.25
    @test storage_bytes(MXFP4, 4096 * 4096) < storage_bytes(FP32, 4096 * 4096) / 7
    @test bits_per_element(MXFP4) == 4.25

    @test_throws ArgumentError add_costs(0)
    @test_throws ArgumentError multiply_costs(1)
    @test_throws ArgumentError accumulate_costs(0, 8)
end


@testset "MXFP4 vs RR4 opportunity" begin
    # ---- widths derived from the format, not assumed -----------------------
    w = mxfp4_widths(MXFP4)
    @test w.elem_ints == [-12, -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 12]
    @test w.elem_mag_bits == 4 && w.elem_bits == 5
    @test w.product_max == 144 && w.product_bits == 9
    @test w.core_sum_max == 32 * 144 == 4608 && w.core_bits == 14
    @test w.distinct_products == 37
    @test w.lut_entries == 64
    # the core sum bound agrees with the package's own, in E2M1 units
    @test w.core_sum_max == 4 * Int(core_sum_bound(MXFP4))
    # doubling E2M1 really does give integers — the premise of the whole analysis
    @test all(g -> g * 2 == round(g * 2), filter(isfinite, grid(E2M1)))
    # K=16 needs one bit less
    @test mxfp4_widths(NVFP4).core_bits == 13

    # ---- sign detection: the asymmetry that blocks soft-max ----------------
    @test sign_detect_depth(32; redundant = false) == 0     # it IS the top bit
    @test sign_detect_depth(32; redundant = true) == 4      # priority encode
    for ww in (8, 16, 32, 64)
        @test sign_detect_depth(ww; redundant = true) > sign_detect_depth(ww; redundant = false)
    end

    # ---- element multiply: the LUT wins at 4 bits --------------------------
    mo = mxfp4_multiply_options()
    g(k) = mo[findfirst(x -> x.method === k, mo)]
    @test g(:lut).depth < g(:array).depth
    @test g(:lut).depth < g(:booth_r4).depth
    @test g(:rr4_digits).depth > g(:array).depth       # recoding 4 bits is pure overhead
    @test g(:lut).cells == 64

    # ---- block reduction: carry-save beats RR4 -----------------------------
    ro = mxfp4_reduction_options(32)
    r(k) = ro[findfirst(x -> x.method === k, ro)]
    @test r(:carry_save_tree).depth < r(:rr4_tree).depth
    @test r(:carry_save_tree).depth < r(:prefix_tree).depth
    @test r(:sequential_prefix).depth > 5 * r(:carry_save_tree).depth
    # and it holds at every block size
    for K in (8, 16, 32, 64, 128)
        rr = mxfp4_reduction_options(K)
        cs = rr[findfirst(x -> x.method === :carry_save_tree, rr)]
        r4 = rr[findfirst(x -> x.method === :rr4_tree, rr)]
        @test cs.depth < r4.depth
    end

    # ---- dot product: RR4 is slower on a tree, at every N ------------------
    for n in (256, 1024, 4096, 16384)
        d = mxfp4_dot_costs(n)
        # three genuinely distinct datapaths, and the ordering is the whole finding:
        # carry-save < RR4 < canonical.  Carry-free beats a prefix tree and loses to
        # the compressor tree the hardware already uses.
        @test d.total_cs < d.total_rr4 < d.total_conv
        @test d.speedup > 1                        # RR4 does beat canonical
        @test d.rr4_vs_cs < 1                      # ...and does not beat carry-save
        @test d.products == 2 && d.block_reduce == 12
        @test 0 < d.rr4_addressable < 1
    end
    # depth grows like log N, not N: 1024× the work for ~2.25× the depth
    d1, d2 = mxfp4_dot_costs(1024), mxfp4_dot_costs(1024 * 1024)
    @test d2.N == 1024 * d1.N
    @test d2.total_conv < 3 * d1.total_conv               # logarithmic...
    @test d2.total_conv < d1.total_conv * (d2.N ÷ d1.N) / 100   # ...emphatically not linear
    for m in (:total_conv, :total_rr4, :total_cs)
        @test getproperty(d2, m) < 3 * getproperty(d1, m)
    end

    # ---- soft-max: the max phase is RR4-hostile ----------------------------
    for n in (256, 1024, 4096)
        sm = mxfp4_softmax_costs(n)
        @test sm.max_rr4 > sm.max_conv             # sign detection costs more
        @test sm.sum_rr4 > sm.sum_conv             # carry-save tree wins the sum too
        @test sm.total_rr4 > sm.total_conv
        @test sm.sign_penalty > 0
        @test sm.rr4_addressable < 0.35            # only the sum phase is addressable
    end

    # ---- the sequential niche: RR4 beats canonical, loses to carry-save ----
    for aw in (32, 48, 64)
        d = mxfp4_dot_costs(4096; acc_bits = aw, schedule = :sequential)
        @test d.speedup > 1                        # genuinely beats a canonical adder
        @test d.rr4_vs_cs < 1                      # but not carry-save
        @test d.total_cs < d.total_rr4 < d.total_conv
    end
    @test mxfp4_dot_costs(4096; acc_bits = 64, schedule = :sequential).speedup > 1.5

    # RR4's real edge over carry-save is state, not depth: 1.5w vs 2w
    for ww in (16, 32, 64)
        a = add_costs(ww)
        m(k) = a[findfirst(x -> x.method === k, a)]
        @test m(:rr4_carry_free).state_bits < m(:carry_save).state_bits
        @test m(:rr4_carry_free).depth > m(:carry_save).depth
    end

    @test_throws ArgumentError mxfp4_dot_costs(64; schedule = :nonesuch)
end


@testset "cost model documentation (docs/src/costmodel.md)" begin
    # Every number quoted on the cost-model page, pinned. The page explains how
    # computations map to gate levels; if the mapping changes, the prose must too.
    m(v, k) = v[findfirst(x -> x.method === k, v)]

    # §3 addition — the four schedules at w=64
    a = add_costs(64)
    @test m(a, :ripple).depth == 65                    # w + 1
    @test m(a, :prefix).depth == 6                     # ceil(log2 64)
    @test m(a, :carry_save).depth == 1                 # one compressor
    @test m(a, :rr4_carry_free).depth == 3             # sum, transfer, digit
    # the derivations hold at every width, not just 64
    for w in (4, 9, 14, 16, 32, 64, 128, 1024)
        b = add_costs(w)
        @test m(b, :ripple).depth == w + 1
        @test m(b, :prefix).depth == max(1, ceil(Int, log2(max(2, w))))
        @test m(b, :carry_save).depth == 1
        @test m(b, :rr4_carry_free).depth == 3         # constant in w — the whole claim
    end

    # §4 reduction trees at K=32
    r = mxfp4_reduction_options(32)
    @test (m(r, :carry_save_tree).levels, m(r, :carry_save_tree).depth) == (8, 12)
    @test (m(r, :rr4_tree).levels, m(r, :rr4_tree).depth) == (5, 19)
    @test (m(r, :prefix_tree).levels, m(r, :prefix_tree).depth) == (5, 20)
    @test (m(r, :sequential_prefix).levels, m(r, :sequential_prefix).depth) == (31, 124)
    # CSA levels come from reduction_schedule's rows → ceil(2·rows/3)
    @test length(reduction_schedule(32)) - 1 == 8
    @test reduction_schedule(32) == [32, 22, 15, 10, 7, 5, 4, 3, 2]

    # §6 sign detection — 0 vs a priority-encode tree
    @test sign_detect_depth(16; redundant = false) == 0
    @test sign_detect_depth(16; redundant = true) == 3       # ceil(log2 ceil(16/2))
    for w in (8, 16, 32, 64, 128)
        @test sign_detect_depth(w; redundant = true) ==
              max(1, ceil(Int, log2(max(2, cld(w, 2)))))
    end

    # §8 the worked dot product, stage by stage
    d = mxfp4_dot_costs(64)
    @test (d.blocks, d.products, d.block_reduce, d.scale) == (2, 2, 12, 1)
    @test (d.cross_conv, d.cross_rr4, d.cross_cs) == (5, 8, 5)
    @test (d.total_conv, d.total_rr4, d.total_cs) == (20, 23, 20)
    # the stages really do sum — depth is a sequential critical path
    @test d.total_conv == d.products + d.block_reduce + d.scale + d.cross_conv
    @test d.total_rr4  == d.products + d.block_reduce + d.scale + d.cross_rr4
    @test d.total_cs   == d.products + d.block_reduce + d.scale + d.cross_cs
    # block_reduce = CSA levels over K + one exit CPA on the core width
    @test d.block_reduce == (length(reduction_schedule(32)) - 1) +
                            max(1, ceil(Int, log2(mxfp4_widths().core_bits)))

    # §8 the worked soft-max, phase by phase
    sm = mxfp4_softmax_costs(64)
    @test (sm.max_conv, sm.max_rr4) == (24, 36)
    @test (sm.exp, sm.sum_conv, sm.sum_rr4, sm.divide) == (6, 14, 22, 12)
    @test (sm.total_conv, sm.total_rr4) == (56, 76)
    @test sm.total_conv == sm.max_conv + sm.exp + sm.sum_conv + sm.divide
    @test sm.total_rr4  == sm.max_rr4  + sm.exp + sm.sum_rr4  + sm.divide
    # max = tree levels × (subtract + sign test)
    @test sm.max_conv == 6 * (4 + sign_detect_depth(16; redundant = false))
    @test sm.max_rr4  == 6 * (3 + sign_detect_depth(16; redundant = true))

    # §9 the three currencies, and that they genuinely trade
    for (k, dp, ce, st) in ((:ripple, 65, 64, 64), (:prefix, 6, 448, 64),
                            (:carry_save, 1, 64, 128), (:rr4_carry_free, 3, 96, 96))
        x = m(a, k)
        @test (x.depth, x.cells, x.state_bits) == (dp, ce, st)
    end
    @test m(a, :prefix).cells == 64 * 6 + 64            # w·ceil(log2 w) + w
    @test m(a, :rr4_carry_free).cells == 3 * cld(64, 2) # 3 per digit
    @test m(a, :rr4_carry_free).state_bits == RR4_BITS_PER_DIGIT * cld(64, 2)
    @test m(a, :carry_save).state_bits == 2 * 64        # sum word + carry word
    # prefix buys depth with area, on the identical sum
    @test m(a, :prefix).cells > 6 * m(a, :ripple).cells
    @test m(a, :ripple).depth > 10 * m(a, :prefix).depth

    # §7 constants the stage tables rest on
    @test mxfp4_widths().core_bits == 14
    @test mxfp4_widths().lut_entries == 64
    @test booth_rows(24) == (24, 13)

    # "Known simplifications": the sensitivity table, same basis for both columns
    let pfx1(w) = max(1, ceil(Int, log2(max(2, w)))), pfx2(w) = pfx1(w) + 2,
        csa(n) = length(reduction_schedule(Int(n))) - 1, K = 32, B = 128, aw = 64,
        core = 14
        for (pf, want) in ((pfx1, (777, 402, 148)), (pfx2, (1033, 406, 152)))
            fixed = 2 + (csa(K) + pf(core)) + 1
            canon = fixed + (B - 1) * pf(aw)
            rr4   = fixed + 3 * (B - 1) + pf(aw)
            cs    = fixed + (B - 1) + pf(aw)
            @test (canon, rr4, cs) == want
            @test cs < rr4 < canon                       # ordering survives either way
        end
        # block reduction ordering survives too
        for pf in (pfx1, pfx2)
            @test csa(K) + pf(core) < 3 * ceil(Int, log2(K)) + pf(core)
        end
        # ...and the soft-max max phase weakens from a loss to a tie, as documented
        @test 6 * pfx1(16) < 6 * (3 + 3)                 # as modelled: RR4 loses
        @test 6 * pfx2(16) == 6 * (3 + 3)                # corrected: a tie
    end
end


@testset "cycle model — depth ÷ clock budget" begin
    m(v, k) = v[findfirst(x -> x.method === k, v)]

    # ---- the conversion itself ---------------------------------------------
    @test cycles_for(65; levels_per_cycle = 16) == 5
    @test cycles_for(3;  levels_per_cycle = 16) == 1
    @test cycles_for(0;  levels_per_cycle = 16) == 1      # never less than one
    @test cycles_for(32; levels_per_cycle = 16) == 2      # exact multiple
    @test cycles_for(33; levels_per_cycle = 16) == 3      # one over
    for d in (1, 3, 6, 65, 1000), L in (4, 8, 16, 64)
        @test cycles_for(d; levels_per_cycle = L) == max(1, cld(d, L))
    end
    @test DEFAULT_LEVELS_PER_CYCLE == 16

    # ---- addition: the central caveat --------------------------------------
    # at a normal clock budget every scheme but ripple collapses to ONE cycle,
    # so carry-free's depth win over a prefix adder is invisible in cycles.
    a = add_cycles(64; levels_per_cycle = 16)
    @test m(a, :ripple).latency == 5
    @test m(a, :prefix).latency == 1
    @test m(a, :carry_save).latency == 1
    @test m(a, :rr4_carry_free).latency == 1
    @test m(a, :prefix).latency == m(a, :rr4_carry_free).latency   # a tie in cycles
    # ...but at an aggressive clock they separate again
    a4 = add_cycles(64; levels_per_cycle = 4)
    @test m(a4, :ripple).latency == 17
    @test m(a4, :prefix).latency == 2
    @test m(a4, :rr4_carry_free).latency == 1
    @test m(a4, :rr4_carry_free).latency < m(a4, :prefix).latency  # now it shows
    # and at a very coarse clock even ripple catches up
    a64 = add_cycles(64; levels_per_cycle = 64)
    @test m(a64, :ripple).latency == 2
    @test all(x -> x.latency <= 2, a64)

    # ---- multiplication: iterative vs pipelined ----------------------------
    mc = multiply_cycles(24; levels_per_cycle = 16)
    @test m(mc, :shift_add).latency == 24            # one row per cycle
    @test m(mc, :shift_add).ii == 24                 # occupies the unit
    @test !m(mc, :shift_add).pipelined
    @test m(mc, :wallace_plain).latency == 1
    @test m(mc, :booth_wallace).latency == 1
    @test m(mc, :wallace_plain).ii == 1              # one product per cycle
    # Booth's 1-level depth saving is zero cycles at any sane clock
    @test m(mc, :booth_wallace).latency == m(mc, :wallace_plain).latency
    @test m(mc, :shift_add).throughput < m(mc, :booth_wallace).throughput / 20

    # ---- accumulation: where depth DOES become cycles ----------------------
    ac = accumulate_cycles(1024, 64; levels_per_cycle = 16)
    @test m(ac, :ripple).latency == 5120             # 5 cycles/term
    @test m(ac, :prefix).latency == 1024             # 1 cycle/term
    @test m(ac, :rr4_carry_free).latency == 1025     # 1/term + one exit
    @test m(ac, :carry_save).latency == 1025
    # the honest result: carry-free costs ONE CYCLE MORE than a prefix adder here,
    # because the exit conversion is real and the depth win is not cashable
    @test m(ac, :rr4_carry_free).latency > m(ac, :prefix).latency
    @test m(ac, :ripple).latency == 5 * m(ac, :prefix).latency
    # a loop-carried accumulator cannot pipeline
    @test all(x -> !x.pipelined, ac)
    # a tree can
    at = accumulate_cycles(1024, 64; levels_per_cycle = 16, schedule = :tree)
    @test all(x -> x.ii == 1, at)
    @test m(at, :carry_save).latency <= m(at, :rr4_carry_free).latency
    @test_throws ArgumentError accumulate_cycles(8, 16; schedule = :nonesuch)

    # ---- dot product: latency vs issue-bound throughput --------------------
    d = dot_cycles(4096; levels_per_cycle = 16)
    @test d.latency_cs <= d.latency_rr4 <= d.latency_conv
    @test (d.latency_conv, d.latency_rr4, d.latency_cs) == (4, 3, 2)
    # with finite MACs the issue stream dominates and the datapaths converge
    for macs in (64, 256, 1024)
        dm = dot_cycles(4096; levels_per_cycle = 16, macs)
        @test dm.issue_cycles == cld(4096, macs)
        @test dm.total_conv - dm.total_rr4 <= 1        # a one-cycle difference at most
        @test dm.total_conv > dm.issue_cycles          # fill is on top of issue
    end
    @test dot_cycles(4096; macs = 64).issue_cycles == 64

    # ---- soft-max: pass-bound, not gate-bound ------------------------------
    for lanes in (1, 8, 32, 128)
        s = softmax_cycles(4096; levels_per_cycle = 16, lanes)
        @test s.stream_cycles == cld(4096, lanes)
        @test s.passes == 3
        @test s.pass_bound == 3 * s.stream_cycles
        @test s.arithmetic_share < 0.10                # arithmetic is a rounding error
        @test s.total_conv == s.pass_bound + s.fill_conv
    end
    # even RR4's worse fill barely moves the total
    s1 = softmax_cycles(4096; lanes = 1)
    @test (s1.total_rr4 - s1.total_conv) / s1.total_conv < 0.01
end


@testset "level → physical grounding, and the FP32 multiply" begin
    # ---- FO4 conversion ----------------------------------------------------
    @test FO4_PER_LEVEL == (2, 3)
    @test fo4_range(1) == (2, 3)
    @test fo4_range(22) == (44, 66)            # one FP32 multiply
    for L in (4, 8, 16, 32)
        lo, hi = fo4_range(L)
        @test lo == 2L && hi == 3L
    end
    # the default budget is the relaxed one, and the docs say so
    @test fo4_range(DEFAULT_LEVELS_PER_CYCLE) == (32, 48)

    # ---- FP32 multiply, stage by stage -------------------------------------
    st = float_multiply_stages(FP32)
    byname(n) = st[findfirst(x -> x.stage == n, st)]
    @test byname("unpack").levels == 1
    @test byname("sign").levels == 1
    @test byname("exponent add").levels == 3           # ceil(log2 8)
    @test byname("significand multiply").levels == 12  # 24×24 Booth-Wallace
    @test byname("normalise").levels == 1
    @test byname("round").levels == 5                  # ceil(log2 24)
    @test byname("post-normalise").levels == 1
    @test byname("pack / specials").levels == 2

    # sign and exponent are OFF the critical path — the shape of an FP multiplier
    @test !byname("sign").critical
    @test !byname("exponent add").critical
    @test byname("significand multiply").critical
    @test float_multiply_depth(FP32) == 22
    @test float_multiply_depth(FP32) == sum(x.levels for x in st if x.critical)

    # the significand multiply agrees with the standalone multiplier model
    mc = multiply_costs(24)
    @test byname("significand multiply").levels ==
          mc[findfirst(x -> x.method === :booth_wallace, mc)].depth

    # rounding is nearly half the multiply — why exact block sums win
    @test byname("round").levels > 0.3 * byname("significand multiply").levels

    # ---- across formats: depth is logarithmic in significand width ---------
    depths = Dict(f => float_multiply_depth(f) for f in (E2M1, E4M3, BF16, FP16, FP32, FP64))
    @test depths[E2M1] == 9 && depths[E4M3] == 12
    @test depths[BF16] == 16 && depths[FP16] == 18
    @test depths[FP32] == 22 && depths[FP64] == 26
    # FP64 has 2.2× FP32's significand but only ~1.2× the depth
    @test (FP64.mbits + 1) / (FP32.mbits + 1) > 2
    @test depths[FP64] / depths[FP32] < 1.3
    # monotone in significand width
    ordered = [E2M1, E4M3, BF16, FP16, FP32, FP64]
    @test issorted([depths[f] for f in ordered])

    # ---- cycles for the FP32 multiply at each budget ------------------------
    d32 = float_multiply_depth(FP32)
    @test cycles_for(d32; levels_per_cycle = 4) == 6
    @test cycles_for(d32; levels_per_cycle = 8) == 3
    @test cycles_for(d32; levels_per_cycle = 16) == 2
    @test cycles_for(d32; levels_per_cycle = 32) == 1
    # real FP32 multipliers are 3-5 stages, bracketing L = 4 and L = 8
    @test 3 <= cycles_for(d32; levels_per_cycle = 8) <= 5
    @test 3 <= cycles_for(d32; levels_per_cycle = 4) <= 6

    # ---- the crossover the docs warn about ---------------------------------
    # "redundancy buys no cycles" holds at L=8 and L=16, and fails at L=4
    m(v, k) = v[findfirst(x -> x.method === k, v)]
    for L in (8, 16)
        a = add_cycles(64; levels_per_cycle = L)
        @test m(a, :rr4_carry_free).latency == m(a, :prefix).latency   # a tie
    end
    a4 = add_cycles(64; levels_per_cycle = 4)
    @test m(a4, :rr4_carry_free).latency < m(a4, :prefix).latency      # RR4 wins here
end


@testset "pipelining, and the one-cycle multiply" begin
    # ---- pipeline_plan: a homogeneous block, balanced -----------------------
    p = pipeline_plan(22; levels_per_cycle = 8, width = 58)
    @test p.latency == cycles_for(22; levels_per_cycle = 8) == 3
    @test p.depth == 22                       # registers do not remove logic
    @test sum(s.levels for s in p.stages) == 22
    @test maximum(s.levels for s in p.stages) <= 8
    @test maximum(s.levels for s in p.stages) - minimum(s.levels for s in p.stages) <= 1
    @test p.ii == 1
    @test p.registers == 2 * 58               # (latency - 1) cuts
    @test all(s.slack >= 0 for s in p.stages)
    @test_throws ArgumentError pipeline_plan(0)
    @test_throws ArgumentError pipeline_plan(8; levels_per_cycle = 0)

    # a single-cycle block needs no registers at all
    p1 = pipeline_plan(6; levels_per_cycle = 8, width = 32)
    @test p1.latency == 1 && p1.registers == 0

    # ---- the FP32 multiplier, cut ------------------------------------------
    for L in (4, 8, 16, 32)
        q = float_multiply_pipeline(FP32; levels_per_cycle = L)
        @test q.depth == float_multiply_depth(FP32) == 22
        # no stage can exceed the budget: deep stages are split internally
        @test maximum(s.levels for s in q.stages) <= L
        @test all(s.slack >= 0 for s in q.stages)
        # greedy packing of named stages can never beat the ideal free cut
        @test q.latency >= cycles_for(22; levels_per_cycle = L)
        @test q.registers == (q.latency - 1) * q.width
        @test q.slack == L * q.latency - 22
    end

    # ... and at L = 4 it genuinely loses a cycle: the 1-level unpack cannot share
    # a cycle with a 4-level slice of the tree, so it burns a stage on its own
    @test float_multiply_pipeline(FP32; levels_per_cycle = 4).latency == 7
    @test cycles_for(22; levels_per_cycle = 4) == 6
    for L in (8, 16, 32)
        @test float_multiply_pipeline(FP32; levels_per_cycle = L).latency ==
              cycles_for(22; levels_per_cycle = L)
    end

    # L = 8 gives the industrial 3-stage shape, cut through the Wallace tree
    q8 = float_multiply_pipeline(FP32; levels_per_cycle = 8)
    @test q8.latency == 3
    @test any(occursin("significand multiply (1/2)", s.name) for s in q8.stages)
    @test any(occursin("significand multiply (2/2)", s.name) for s in q8.stages)
    # real FP32 multipliers are 3-5 stages
    @test 3 <= q8.latency <= 5

    # non-critical stages consume no budget (note "significand" contains "sign")
    @test !any(occursin("exponent", s.name) for s in q8.stages)
    @test !any("sign" in split(s.name, " + ") for s in q8.stages)
    @test q8.depth == 22 < sum(x.levels for x in float_multiply_stages(FP32))

    # deeper pipelines waste less of what they buy
    q16 = float_multiply_pipeline(FP32; levels_per_cycle = 16)
    @test q16.slack > q8.slack
    @test q8.registers > q16.registers          # ... and cost more flip-flops

    # ---- throughput arithmetic ---------------------------------------------
    @test pipeline_time(q8, 1) == q8.latency
    @test pipeline_time(q8, 1024) == 3 + 1023
    @test pipeline_speedup(q8, 1) ≈ 1.0         # never helps a single op
    @test pipeline_speedup(q8, 1024) > 2.99
    @test pipeline_speedup(q8, 10^7) < q8.latency   # approaches, never reaches
    @test pipeline_utilization(q8, 1) ≈ 1 / 3
    @test pipeline_utilization(q8, 10^6) > 0.999
    for m in (1, 4, 16, 1024)
        @test pipeline_utilization(q8, m) ≈ m / pipeline_time(q8, m)
        @test pipeline_speedup(q8, m) ≈ q8.latency * pipeline_utilization(q8, m)
    end

    # break-even scales with depth
    q4 = float_multiply_pipeline(FP32; levels_per_cycle = 4)
    @test pipeline_breakeven(q8) == 19
    @test pipeline_breakeven(q4) > pipeline_breakeven(q8)
    @test pipeline_speedup(q8, pipeline_breakeven(q8)) >= 0.9 * q8.latency
    @test pipeline_breakeven(q8; fraction = 0.99) > pipeline_breakeven(q8)
    @test_throws ArgumentError pipeline_breakeven(q8; fraction = 1.0)

    # ---- the one-cycle ladder for FP32 -------------------------------------
    L = DEFAULT_LEVELS_PER_CYCLE
    r = one_cycle_multiply(FP32)
    @test length(r) == 5
    @test r[1].step == "IEEE baseline" && r[1].depth == 22 && r[1].saved == 0
    @test !r[1].fits && r[1].cycles == 2       # IEEE FP32 is not single-cycle at L=16
    @test r[2].saved == 1                      # Booth radix-8 buys ~nothing
    @test r[3].saved == 6 && r[3].depth == 15  # dropping the exit CPA is the big one
    @test r[3].fits                            # ... and it is what gets us to one cycle
    @test r[4].saved == 7 && r[4].depth == 8   # deferred rounding
    @test occursin("N/A", r[5].step)           # FP32's table would need 2^62 entries
    # the ladder is monotone and cumulative
    @test issorted([x.depth for x in r]; rev = true)
    for i in 2:length(r)
        @test r[i].depth == r[i-1].depth - r[i].saved
        @test r[i].cycles == cycles_for(r[i].depth; levels_per_cycle = L)
        @test r[i].fits == (r[i].depth <= L)
    end
    # rung 4 is single-cycle even at an aggressive clock
    @test cycles_for(r[4].depth; levels_per_cycle = 8) == 1

    # ---- E2M1: single-cycle before you start -------------------------------
    e = one_cycle_multiply(E2M1)
    @test e[1].depth == 9 && e[1].fits         # plain IEEE E2M1 is already one cycle
    @test e[end].step == "+ table lookup"
    @test e[end].depth == 2                    # 64-entry ROM
    @test occursin("64-entry", e[end].cost)
    @test cycles_for(e[end].depth; levels_per_cycle = 4) == 1   # one cycle at any clock
    # agrees with the MXFP4-specific model
    @test e[end].depth ==
          mxfp4_multiply_options(MXFP4)[findfirst(x -> x.method === :lut,
                                                  mxfp4_multiply_options(MXFP4))].depth

    # ---- format is worth more than the whole toolbox -----------------------
    rows = one_cycle_formats(; io = devnull)
    byfmt(n) = rows[findfirst(x -> x.format == n, rows)]
    @test byfmt("E2M1").ieee_cycles == 1 && byfmt("FP32").ieee_cycles == 2
    @test byfmt("E2M1").lut && !byfmt("FP32").lut && !byfmt("FP64").lut
    @test byfmt("FP64").best_levels == 10      # no Int overflow on the LUT test
    @test all(x.one_cycle for x in rows)       # every format lands at one cycle eventually
    @test issorted([x.ieee_levels for x in rows[[1, 2, 4, 3, 5, 6]]])
    # the headline ratio: 2.75x from every technique, 11x from the format
    @test byfmt("FP32").ieee_levels / byfmt("FP32").best_levels ≈ 22 / 8
    @test byfmt("FP32").ieee_levels / byfmt("E2M1").best_levels ≈ 11.0
    @test byfmt("FP32").ieee_levels / byfmt("E2M1").best_levels >
          byfmt("FP32").ieee_levels / byfmt("FP32").best_levels

    # ---- the clock a single-cycle multiply demands --------------------------
    c = one_cycle_clock(FP32)
    @test c.baseline_levels == 22
    @test c.baseline_fo4 == fo4_range(22) == (44, 66)
    @test c.baseline_cycles_at_core_clock == (2, 5)   # against a 15-25 FO4 core clock
    @test one_cycle_clock(E2M1).baseline_cycles_at_core_clock[1] == 1
    @test length(c.rungs) == 5
    @test c.rungs[1].levels_needed == 22

    # ---- printers run ------------------------------------------------------
    @test pipeline_report(q8; io = devnull) === q8
    @test pipeline_timeline(q8, 5; io = devnull) === nothing
    @test one_cycle_report(FP32; io = devnull) === nothing
end


@testset "float grid as a sampled log axis" begin
    # ---- the analogy: exponent field == octave index -----------------------
    for f in (E4M3, E5M2, FP16, BF16, FP32), e in max(emin(f), -6):min(emax(f), 6)
        x = exp2(e)
        @test log2(x) == e                       # powers of two are exact
        @test quantize(f, x) == x
        @test floor(Int, log2(x)) == e           # ... and land on the tick
    end

    # ---- the difference: 2^mbits values per binade, spaced LINEARLY --------
    for f in (E2M1, E4M3, E5M2, FP16)
        g = filter(x -> x > 0 && isfinite(x), grid(f))
        for e in (emin(f) + 1):(emax(f) - 1)
            b = filter(x -> exp2(e) <= x < exp2(e + 1), g)
            isempty(b) && continue
            @test length(b) == 1 << f.mbits      # exactly 2^m per octave
            steps = diff(b)
            @test all(≈(exp2(e - f.mbits)), steps)   # linear inside
            @test allequal(steps)
            # ... and therefore NOT logarithmic: log2 steps shrink across the binade
            if length(b) > 2
                ls = diff(log2.(b))
                @test issorted(ls; rev = true)   # denser toward the top
                @test ls[1] > ls[end]
            end
        end
    end

    # ---- the sag: max of m - log2(1+m), independent of mantissa width ------
    mpk = 1 / log(2) - 1
    gap(m) = log2(1 + m) - m
    @test mpk ≈ 0.4426950408889634
    @test gap(mpk) ≈ 0.08607133205593431
    @test round(gap(mpk); digits = 4) == 0.0861
    for m in range(0, 1; length = 2001)          # it really is the maximum
        @test gap(m) <= gap(mpk) + 1e-12
    end
    @test gap(0.0) == 0.0 && gap(1.0) == 0.0     # pinned at both octave ends
    # independent of the format: same curve, sampled more finely
    for f in (E2M1, E4M3, FP32)
        sampled = maximum(gap(k * exp2(-f.mbits)) for k in 0:(1 << f.mbits) - 1)
        @test sampled <= gap(mpk) + 1e-12
    end
    @test maximum(gap(k * exp2(-3)) for k in 0:7) ≈ gap(0.5)  # E4M3 worst is at 1.5

    # ---- the bit pattern IS a piecewise-linear log2 -------------------------
    bits2log(x) = Float64(reinterpret(Int32, Float32(x))) / 2^23 - 127
    worst = 0.0
    for i in 0:4000
        x = 2.0^(i / 4000)
        worst = max(worst, abs(bits2log(x) - log2(x)))
    end
    @test worst ≈ gap(mpk) atol = 1e-4           # the same 0.0861
    @test bits2log(1.0) == 0.0 && bits2log(2.0) == 1.0

    # ---- the wobble: relative step is a sawtooth of ratio exactly 2 ---------
    for f in (E2M1, E4M3, FP32)
        eps_f = machine_eps(f)
        @test eps_f == exp2(-f.mbits)
        for e in (-2, 0, 3)
            top = exp2(e - f.mbits) / exp2(e)                  # start of binade
            bot = exp2(e - f.mbits) / prevfloat(exp2(e + 1.0))  # top of binade
            @test top ≈ eps_f
            @test bot ≈ eps_f / 2 rtol = 1e-6
            @test top / bot ≈ 2 rtol = 1e-6      # exactly the radix
        end
    end

    # ---- where the analogy stops: subnormals are LINEAR, so they fly apart --
    for f in (E4M3, FP16)
        g = filter(x -> x > 0 && isfinite(x), grid(f))
        sub = filter(x -> x < minnormal(f), g)
        @test !isempty(sub)
        @test allequal(diff(sub))                 # a fixed linear grid
        @test all(≈(minsubnormal(f)), diff(sub))
        ls = diff(log2.(sub))
        @test issorted(ls; rev = true)            # gaps SHRINK going up...
        @test ls[1] > ls[end]                     # ...i.e. they fly apart going down
        # a normal binade fits 2^m values into one octave; the subnormals do not
        span = log2(sub[end]) - log2(sub[1])
        @test span > 1.0
    end
    # E4M3: 7 subnormals spanning nearly three octaves
    let g = filter(x -> x > 0 && isfinite(x), grid(E4M3)),
        sub = filter(x -> x < minnormal(E4M3), g)
        @test length(sub) == 7
        @test log2(sub[1]) ≈ -9.0
        @test log2(sub[end]) ≈ -6.19 atol = 0.01
        @test 2.7 < log2(sub[end]) - log2(sub[1]) < 3.0
    end
    # and zero has no coordinate on the axis at all
    @test log2(0.0) == -Inf
    @test 0.0 in grid(E4M3)

    # ---- E2M1 samples each octave exactly twice ----------------------------
    @test E2M1.mbits == 1
    @test filter(x -> x > 0 && isfinite(x), grid(E2M1)) == [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    @test log2(1.5) ≈ 0.5849625007211562
    @test round(log2(1.5); digits = 3) == 0.585

    # (plot_log_axis_analogy is exercised by the docs build, which loads CairoMakie)
end


@testset "HLS: derived widths, accumulator sizing, energy split" begin
    # ---- declarations are DERIVED from the format, not rounded to a word ----
    d = hls_types(MXFP4)
    @test length(d) == 5
    byrole(v, r) = v[findfirst(x -> x.role == r, v)]
    w = mxfp4_widths(MXFP4)
    @test byrole(d, "element").bits == w.elem_bits == 5
    @test byrole(d, "product").bits == w.product_bits == 9
    @test byrole(d, "block accumulator").bits == w.core_bits == 14
    @test byrole(d, "block scale").bits == 8              # E8M0, unsigned
    @test byrole(d, "element").decl == "ap_int<5>"
    @test byrole(d, "block scale").decl == "ap_uint<8>"
    # every width is strictly below the machine word it would otherwise take
    for r in ("element", "product", "block accumulator", "block scale")
        @test byrole(d, r).bits < 32
    end
    # the block sum really is exact: K * max|product| fits
    @test MXFP4.K * w.product_max == w.core_sum_max == 4608
    @test 2^(w.core_bits - 1) > w.core_sum_max            # signed, fits
    @test 2^(w.core_bits - 2) <= w.core_sum_max           # ...and is not oversized

    # more blocks widen only the dot accumulator
    for nb in (1, 8, 128, 2048)
        dn = hls_types(MXFP4; blocks = nb)
        @test byrole(dn, "block accumulator").bits == w.core_bits
        @test byrole(dn, "dot accumulator").bits ==
              w.core_bits + (nb > 1 ? ceil(Int, log2(nb)) : 0)
    end

    # ---- accumulator sizing ------------------------------------------------
    for N in (32, 256, 4096, 65536)
        a = hls_accumulator_bits(MXFP4; blocks = cld(N, 32))
        @test a.blocks == cld(N, 32)
        @test a.per_block == 14                    # exact within a block, always
        @test a.exact_within_block
        @test a.per_block_roundings == a.blocks    # N/K roundings, not N
        @test a.per_block_roundings < N            # ...which is the point
        @test a.wide_fixed > a.per_block
        @test a.wide_fixed < a.full_kulisch
    end
    @test hls_accumulator_bits(MXFP4; blocks = 2048).wide_fixed == 57
    @test hls_accumulator_bits(MXFP4; blocks = 128).wide_fixed == 53
    # the span is a knob, and it moves the width one-for-one
    let a1 = hls_accumulator_bits(MXFP4; blocks = 128, scale_span = 32),
        a2 = hls_accumulator_bits(MXFP4; blocks = 128, scale_span = 64)
        @test a2.wide_fixed - a1.wide_fixed == 32
    end
    # full Kulisch over all of E8M0 is impractical, which is why it is shown
    @test hls_accumulator_bits(MXFP4).full_kulisch == 14 + 254
    @test hls_accumulator_bits(MXFP4).full_kulisch > 4 * 57

    # ---- the control table -------------------------------------------------
    @test length(HLS_CONTROL) == 6
    @test all(!isempty(r.op) && !isempty(r.lever) for r in HLS_CONTROL)
    @test any(occursin("recurrence", r.tool_picks) for r in HLS_CONTROL)

    # ---- energy: memory dominates, and DRAM dominates harder ---------------
    for bf in (MXFP4, NVFP4, MXINT4)
        es = hls_energy(bf; from_dram = false)
        ed = hls_energy(bf; from_dram = true)
        @test es.memory_fraction > 0.5            # even from local SRAM
        @test ed.memory_fraction > 0.95           # from DRAM it is everything
        @test ed.memory_fraction > es.memory_fraction
        @test es.arithmetic_fraction + es.memory_fraction ≈ 1.0
        @test es.e_total ≈ es.e_multiply + es.e_reduce + es.e_registers + es.e_memory
        @test ed.e_total > es.e_total
    end
    # bits/element agrees with the memory model elsewhere in the package
    @test hls_energy(MXFP4).bits_per_element ≈ 4.25
    @test hls_energy(NVFP4).bits_per_element ≈ 4.5
    @test hls_energy(MXINT4).bits_per_element ≈ 4.25
    # and the element may be a float format or an integer one
    @test MXFP4.elem isa FloatFormat && MXINT4.elem isa IntFormat

    # ---- the 4-bit multiply: the table wins, Booth loses -------------------
    mo = mxfp4_multiply_options(MXFP4)
    m(k) = mo[findfirst(x -> x.method === k, mo)]
    @test m(:lut).depth == 2
    @test m(:lut).depth < m(:booth_r4).depth < m(:array).depth
    @test m(:booth_r4).cells > m(:array).cells      # recoding costs MORE cells here
    @test m(:rr4_digits).depth > m(:array).depth    # RR4 at 4 bits is pure overhead
    @test m(:lut).cells == mxfp4_widths(MXFP4).lut_entries == 64

    # ---- the reduction gap that survives into HLS --------------------------
    red = mxfp4_reduction_options(32)
    r(k) = red[findfirst(x -> x.method === k, red)]
    @test r(:sequential_prefix).depth > 10 * r(:carry_save_tree).depth
    @test r(:carry_save_tree).depth < r(:rr4_tree).depth < r(:prefix_tree).depth

    # ---- printers run ------------------------------------------------------
    @test hls_operator_table(; io = devnull) === nothing
    @test hls_pragmas(MXFP4; blocks = 128, io = devnull) === nothing
    @test hls_report(MXFP4; blocks = 128, io = devnull) === nothing
    @test hls_energy_compare(; io = devnull) === nothing
end

end
