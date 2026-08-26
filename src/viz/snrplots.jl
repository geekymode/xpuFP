# ---------------------------------------------------------------------------
# SNR and cost figures.
# ---------------------------------------------------------------------------

"""
    plot_format_snr(x = randn(MersenneTwister(1), 200_000); formats=..., size=(900,400)) -> Figure

Measured SNR and effective bits for a set of formats on the same data.

The effective-bits axis is the calibration proof of the whole metric: dividing by 6.02
recovers *exactly* the significand widths of the IEEE formats, and the block formats'
fractional readings are honest measurements of how much resolution their four stored
bits actually deliver.
"""
function plot_format_snr(x::AbstractVector = randn(MersenneTwister(1), 200_000);
                         formats = (MXINT4, E2M1, MXFP4, NVFP4, E4M3, E5M2, BF16, FP16, FP32),
                         size = (900, 400))
    set_theme!(xpufp_theme())
    names = [_fmtname(f) for f in formats]
    snrs = [measure_snr(f, x) for f in formats]
    bits = [_storage_bits(f) for f in formats]
    fig = Figure(; size = size)
    ax = Axis(fig[1, 1]; xticks = (1:length(formats), names), ylabel = "measured SNR (dB)",
        title = "Format SNR on the same data  (n = $(length(x)))",
        xticklabelrotation = π/6)
    barplot!(ax, 1:length(formats), snrs; color = [format_color(f) for f in formats])
    for i in eachindex(snrs)
        text!(ax, i, snrs[i]; text = @sprintf("%.1f\n%.1f bits", snrs[i], effective_bits(snrs[i])),
              align = (:center, :bottom), fontsize = 8.5, offset = (0, 2))
    end
    ylims!(ax, 0, maximum(snrs) * 1.20)
    ax2 = Axis(fig[2, 1]; xticks = (1:length(formats), names), ylabel = "dB per bit",
               xticklabelrotation = π/6)
    dpb = snrs ./ bits
    barplot!(ax2, 1:length(formats), dpb; color = [format_color(f) for f in formats])
    hlines!(ax2, [DB_PER_BIT]; color = ACCENT, linestyle = :dash)
    text!(ax2, 0.6, DB_PER_BIT; text = " Shannon ceiling 6.02 dB/bit",
          align = (:left, :bottom), fontsize = 9, color = ACCENT)
    ylims!(ax2, 0, DB_PER_BIT * 1.18)
    rowsize!(fig.layout, 2, Relative(0.36))
    fig
end

"""
    plot_dot_snr_vs_length(; ...) -> Figure

The central duality, measured: dot-product SNR against vector length.

**MXFP4 is length-invariant** — all its noise enters at the encoder and the block
arithmetic is exact, so the curve is flat at `SNRₑ − 3.01 dB` forever.  **FP32's
sequential chain decays** as `152.3 − 10log₁₀N`, the random walk of per-operation
roundings.  **FP32's tree schedule is nearly flat**, `≈152 − 10log₁₀log₂N` — the decay
belongs to the *schedule*, not the format.

The two laws would meet only near `N ~ 10^13.7`: never in practice.
"""
function plot_dot_snr_vs_length(; lengths = [32, 128, 512, 2048, 8192, 32768],
                                trials = 24, rng = MersenneTwister(11), size = (900, 420))
    set_theme!(xpufp_theme())
    mx = Float64[]; fp32seq = Float64[]; fp32tree = Float64[]
    for N in lengths
        m = Float64[]; s = Float64[]; t = Float64[]
        for _ in 1:trials
            a = randn(rng, N); b = randn(rng, N)
            truth = dot(a, b)
            r = block_dot(MXFP4, a, b)
            push!(m, 20 * log10(abs(truth) / max(abs(r.value - truth), 1e-300)))
            ps = a .* b
            push!(s, 20 * log10(abs(truth) / max(abs(seq_sum(FP32, ps) - truth), 1e-300)))
            push!(t, 20 * log10(abs(truth) / max(abs(tree_sum(FP32, ps) - truth), 1e-300)))
        end
        push!(mx, median(m)); push!(fp32seq, median(s)); push!(fp32tree, median(t))
    end
    fig = Figure(; size = size)
    ax = Axis(fig[1, 1]; xscale = log10, xlabel = "vector length N", ylabel = "dot-product SNR (dB)",
        title = "Where the noise is born: encoder (flat) versus arithmetic (decaying)")
    scatterlines!(ax, lengths, fp32tree; color = ACCENT2, markersize = 9, label = "FP32, tree schedule")
    scatterlines!(ax, lengths, fp32seq; color = FRAC_COLOR, markersize = 9, label = "FP32, sequential chain")
    scatterlines!(ax, lengths, mx; color = ACCENT, markersize = 9, label = "MXFP4 (exact block tree)")
    pred = measure_snr(MXFP4, randn(rng, 100_000)) - 3.01
    hlines!(ax, [pred]; color = (ACCENT, 0.6), linestyle = :dash)
    text!(ax, lengths[1], pred; text = " predicted SNRₑ − 3.01 dB", align = (:left, :bottom),
          fontsize = 9, color = ACCENT)
    axislegend(ax; position = :lb)
    fig
end

"""
    plot_block_size_sweep(; ...) -> Figure

SNR against block size `K` for thin- and heavy-tailed data, with the storage overhead
`8/K` on the second axis.

Three forces pull on `K`, and 32 is where they balance: overhead falls as `8/K`;
accuracy pulls the other way but **only for realistic data** — on Gaussian samples the
SNR is nearly flat (the power-of-two scale, not the block size, is the bottleneck),
while on heavy-tailed data it degrades steadily, because each block's largest value
sets the scale; and hardware wants `32 × 4 = 128` bits, exactly a 16-byte payload.
"""
function plot_block_size_sweep(; Ks = (8, 16, 32, 64, 128), n = 200_000,
                               rng = MersenneTwister(3), size = (900, 400))
    set_theme!(xpufp_theme())
    g = randn(rng, n)
    t3 = _student_t(rng, n, 3)
    gs = Float64[]; ts = Float64[]; ov = Float64[]
    for K in Ks
        bf = BlockFormat("mx-K$(K)", K, E2M1, E8M0, MX_FLOOR_POW2)
        push!(gs, measure_snr(bf, g)); push!(ts, measure_snr(bf, t3))
        push!(ov, nbits(E8M0) / K)
    end
    fig = Figure(; size = size)
    ax = Axis(fig[1, 1]; xscale = log2, xticks = (collect(Ks), [string(k) for k in Ks]),
        xlabel = "block size K", ylabel = "SNR (dB)",
        title = "Why K = 32: overhead falls as 8/K, heavy-tail accuracy falls with K")
    scatterlines!(ax, collect(Ks), gs; color = FRAC_COLOR, markersize = 9, label = "Gaussian (thin tails)")
    scatterlines!(ax, collect(Ks), ts; color = ACCENT, markersize = 9, label = "Student-t₃ (heavy tails)")
    vlines!(ax, [32]; color = (GREY, 0.6), linestyle = :dash)
    text!(ax, 32, minimum(ts); text = " K = 32:\n 0.25 b/elem,\n 16-byte payload",
          align = (:left, :bottom), fontsize = 9, color = GREY)
    axislegend(ax; position = :rc)
    ax2 = Axis(fig[2, 1]; xscale = log2, xticks = (collect(Ks), [string(k) for k in Ks]),
               xlabel = "block size K", ylabel = "scale overhead\n(bits/element)")
    scatterlines!(ax2, collect(Ks), ov; color = EXP_COLOR, markersize = 9)
    rowsize!(fig.layout, 2, Relative(0.3))
    fig
end

function _student_t(rng, n, ν)
    [randn(rng) / sqrt(sum(abs2, randn(rng, ν)) / ν) for _ in 1:n]
end

"""
    plot_outlier_law(; ...) -> Figure

**Spread is MXFP4's true adversary**, in both proved and measured form.

Plant one outlier of magnitude `R` among 31 unit-normal elements: it captures the
scale, so the *rest* of the block is quantized on a grid whose step grows `∝ R` while
their signal stays fixed — their SNR must fall by `20log₁₀2 ≈ 6 dB per doubling of R`,
until they cross `M/32` and flatline at annihilation.

The same experiment exposes a trap: the **whole-block SNR is non-monotone**, dipping and
then *rising*, because the well-quantized outlier owns all the energy while most
coordinates are literally zero.  Energy metrics cannot be trusted under spread.
"""
function plot_outlier_law(; Rs = 2.0 .^ (0:12), trials = 400,
                          rng = MersenneTwister(5), size = (900, 420))
    set_theme!(xpufp_theme())
    rest = Float64[]; whole = Float64[]; zeroed = Float64[]
    for R in Rs
        rs = Float64[]; ws = Float64[]; zs = Float64[]
        for _ in 1:trials
            x = randn(rng, 32); x[1] = R * sign(randn(rng))
            qb = quantize_block(MXFP4, x)
            push!(ws, snr_db(x, qb.values))
            push!(rs, snr_db(x[2:end], qb.values[2:end]))
            push!(zs, zeroed_count(qb))
        end
        push!(rest, median(rs)); push!(whole, median(ws)); push!(zeroed, median(zs))
    end
    fig = Figure(; size = size)
    ax = Axis(fig[1, 1]; xscale = log2, xlabel = "outlier magnitude R", ylabel = "SNR (dB)",
        title = "The outlier law: 6 dB lost per doubling — and why whole-block SNR lies")
    scatterlines!(ax, collect(Rs), rest; color = ACCENT, markersize = 8,
                  label = "the other 31 elements")
    scatterlines!(ax, collect(Rs), whole; color = FRAC_COLOR, markersize = 8,
                  linestyle = :dash, label = "whole block (misleading!)")
    ref = rest[1] .- 6.02 .* log2.(collect(Rs) ./ Rs[1])
    lines!(ax, collect(Rs), max.(ref, 0); color = (GREY, 0.8), linestyle = :dot,
           label = "predicted −6.02 dB/octave")
    ylims!(ax, -2, max(maximum(whole), maximum(rest)) * 1.15)
    axislegend(ax; position = :lb)
    ax2 = Axis(fig[2, 1]; xscale = log2, xlabel = "outlier magnitude R",
               ylabel = "elements\nzeroed (of 32)")
    scatterlines!(ax2, collect(Rs), zeroed; color = ACCENT2, markersize = 8)
    rowsize!(fig.layout, 2, Relative(0.28))
    fig
end

"""
    plot_energy_bars(; size=(880,380)) -> Figure

Energy per 32-bit operation on a 45 nm process (Horowitz, ISSCC 2014; representative
values, log scale).

Compute is cheap and flat — add, multiply and FMA span half a decade — while a DRAM
access costs ~700× an FP32 add.  The bar chart is the roofline model in miniature: any
kernel that cannot keep its operands on-chip pays for **data motion**, not arithmetic.
"""
function plot_energy_bars(; size = (880, 380))
    set_theme!(xpufp_theme())
    labels = ["INT32 add", "FP32 add", "FP32 mult", "FP32 FMA", "SRAM 32 KB read", "DRAM read"]
    energy = [0.1, 0.9, 3.7, 4.6, 20.0, 640.0]
    cols = [EXP_COLOR, FRAC_COLOR, FRAC_COLOR, FRAC_COLOR, INT_COLOR, ACCENT]
    fig = Figure(; size = size)
    ax = Axis(fig[1, 1]; yscale = log10, xticks = (1:length(labels), labels),
        ylabel = "energy (pJ, 45 nm)", xticklabelrotation = π/7,
        title = "Compute is cheap; moving the operands is not")
    barplot!(ax, 1:length(labels), energy; color = cols)
    for i in eachindex(energy)
        text!(ax, i, energy[i]; text = string(energy[i]), align = (:center, :bottom),
              fontsize = 9.5, offset = (0, 2))
    end
    text!(ax, 6, 640; text = " ≈700× an FP32 add", align = (:left, :top),
          fontsize = 9.5, color = ACCENT, offset = (4, -6))
    ylims!(ax, 0.05, 2000)
    fig
end

"""
    plot_systolic_activity(M, K, P; size=(900,380)) -> Figure

The wavefront: each PE's start cycle is its taxicab distance `i+j` from the corner, so
equal start times form diagonals sweeping the array; the activity histogram over
`T = K+M+P−2` cycles ramps up and back down.

Its area is exactly the `MPK` required MACs — but a short reduction depth keeps the
array under-filled, which is the quantitative case for the long, chained `K` that
MX-block tiling provides.
"""
function plot_systolic_activity(M::Integer = 4, K::Integer = 3, P::Integer = 5;
                                size = (900, 380))
    set_theme!(xpufp_theme())
    T = systolic_cycles(M, K, P)
    act = [count(((i, j),) -> 0 <= t - i - j < K,
                 [(i, j) for i in 0:M-1, j in 0:P-1]) for t in 0:T-1]
    fig = Figure(; size = size)
    ax1 = Axis(fig[1, 1]; xlabel = "column j", ylabel = "row i", yreversed = true,
        xticks = 0:P-1, yticks = 0:M-1,
        title = "start cycle = i + j  (diagonal wavefronts)")
    st = [Float64(i + j) for i in 0:M-1, j in 0:P-1]
    hm = heatmap!(ax1, 0:P-1, 0:M-1, permutedims(st); colormap = :viridis)
    for i in 0:M-1, j in 0:P-1
        text!(ax1, j, i; text = string(i + j), align = (:center, :center),
              fontsize = 10, color = :white)
    end
    Colorbar(fig[1, 2], hm; label = "start cycle")

    ax2 = Axis(fig[1, 3]; xlabel = "cycle t", ylabel = "PEs computing",
        xticks = 0:T-1, title = "activity: Σ = $(sum(act)) = M·P·K")
    barplot!(ax2, 0:T-1, act; color = ACCENT)
    for (t, a) in enumerate(act)
        text!(ax2, t - 1, a; text = string(a), align = (:center, :bottom), fontsize = 9, offset = (0, 2))
    end
    ylims!(ax2, 0, maximum(act) * 1.2)
    Label(fig[2, 1:3],
          "T = K+M+P−2 = $(T) cycles,  utilisation = K/T = " *
          "$(round(100 * systolic_utilization(M, K, P), digits=0))% — utilisation → 1 as K grows, " *
          "which is why MX-block tiling chains many K=32 blocks",
          fontsize = 10, color = GREY, tellwidth = false)
    fig
end

"""
    plot_block_schemes(datasets; formats=IMPROVED_BLOCK_FORMATS, rotate=false,
                       size=(980, 520)) -> Figure

Compare block-scaled FP4 schemes on the two axes that decide deployability: measured
SNR (top) and the share of nonzero inputs annihilated to exact zero (bottom).

The second panel exists because SNR is **energy-weighted** and therefore blind to
destroyed small coordinates — a block whose outlier is well represented can post a
flattering SNR while most of its neighbours have been zeroed. Read the panels together.

```julia
using Random
rng = MersenneTwister(1)
plot_block_schemes(["gaussian" => randn(rng, 40_000)]; rotate = true)
```
"""
function plot_block_schemes(datasets; formats = IMPROVED_BLOCK_FORMATS,
                            rotate::Bool = false, size = (980, 520))
    set_theme!(xpufp_theme())
    rows = compare_block_schemes(datasets; formats, rotate)
    sort!(rows; by = r -> r.snr[1])
    names = [r.name for r in rows]
    labs = rows[1].labels
    nd = length(labs)
    fig = Figure(; size = size)

    ax1 = Axis(fig[1, 1]; xticks = (1:length(rows), names), ylabel = "SNR (dB)",
        xticklabelrotation = π/7,
        title = "Block-scaled FP4 schemes" * (rotate ? " (with Hadamard rotation)" : ""))
    w = 0.8 / nd
    for j in 1:nd
        off = (j - (nd + 1) / 2) * w
        barplot!(ax1, (1:length(rows)) .+ off, [r.snr[j] for r in rows];
                 width = w * 0.92, color = cgrad(:viridis)[(j - 0.5) / nd],
                 label = labs[j])
    end
    for (i, r) in enumerate(rows)
        text!(ax1, i, maximum(r.snr) + 0.25; text = "$(round(r.bits, digits=2))b" *
              (r.exponent_only ? " · exp" : " · ×"),
              align = (:center, :bottom), fontsize = 8, color = GREY)
    end
    ylims!(ax1, 0, maximum(maximum(r.snr) for r in rows) * 1.18)
    axislegend(ax1; position = :lt, labelsize = 9, orientation = :horizontal)

    ax2 = Axis(fig[2, 1]; xticks = (1:length(rows), names),
        ylabel = "inputs zeroed (%)", xticklabelrotation = π/7)
    for j in 1:nd
        off = (j - (nd + 1) / 2) * w
        barplot!(ax2, (1:length(rows)) .+ off, [100 * r.zeroed[j] for r in rows];
                 width = w * 0.92, color = cgrad(:viridis)[(j - 0.5) / nd])
    end
    rowsize!(fig.layout, 2, Relative(0.34))
    Label(fig[3, 1],
          "top: SNR is energy-weighted.  bottom: the share of nonzero inputs stored as " *
          "exact zero — the acceptance metric SNR cannot see.  " *
          "annotations give bits/value and whether the scale is an exponent add (exp) or a multiply (×).",
          fontsize = 9, color = GREY, halign = :left, tellwidth = false)
    fig
end
