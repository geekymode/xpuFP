# ---------------------------------------------------------------------------
# Plots for the benchmark harness.
# ---------------------------------------------------------------------------

"""
    plot_scheme_pareto(rows; dataset="gaussian", size=(880,440)) -> Figure

SNR against storage cost, with the Pareto frontier joined.

A scheme below the frontier is dominated: something else gets more SNR for fewer bits.
The frontier is the honest shortlist for a given storage budget.
"""
function plot_scheme_pareto(rows::Vector{SchemeMetrics}; dataset::AbstractString = "gaussian",
                            size = (880, 440))
    set_theme!(xpufp_theme())
    sel = filter(r -> r.dataset == dataset, rows)
    isempty(sel) && throw(ArgumentError("no rows for dataset $(dataset)"))
    front = pareto_frontier(rows; dataset)
    fig = Figure(; size = size)
    ax = Axis(fig[1, 1]; xlabel = "storage (bits per value)", ylabel = "SNR (dB)",
        title = "Accuracy against cost — $(dataset)")
    lines!(ax, [r.bits for r in front], [r.snr for r in front];
           color = (ACCENT, 0.5), linewidth = 2.0, linestyle = :dash, label = "Pareto frontier")
    for r in sel
        on = any(f -> f.scheme == r.scheme, front)
        scatter!(ax, [r.bits], [r.snr]; color = on ? ACCENT : GREY,
                 markersize = on ? 13 : 9)
        text!(ax, r.bits, r.snr; text = " " * r.scheme, align = (:left, :center),
              fontsize = 9, color = on ? ACCENT : GREY)
    end
    xlims!(ax, minimum(r.bits for r in sel) - 0.35, maximum(r.bits for r in sel) + 1.4)
    axislegend(ax; position = :rb, labelsize = 9)
    Label(fig[2, 1], "orange = on the frontier (nothing achieves both fewer bits and higher SNR)",
          fontsize = 9, color = GREY, halign = :left, tellwidth = false)
    fig
end

"""
    plot_benchmark_grid(rows; metrics=(:snr, :zeroed), size=nothing) -> Figure

A scheme × dataset heat map per metric — the whole benchmark on one page.

Reading `:snr` and `:zeroed` together is the point: a scheme can look strong on the first
while destroying a fifth of its coordinates, which only the second reveals.
"""
function plot_benchmark_grid(rows::Vector{SchemeMetrics};
                             metrics = (:snr, :zeroed), size = nothing)
    set_theme!(xpufp_theme())
    schemes = unique(r.scheme for r in rows)
    dsets = unique(r.dataset for r in rows)
    sc = sort(schemes; by = s -> -mean(r.snr for r in rows if r.scheme == s))
    sz = size === nothing ?
         (240 + 105 * length(dsets) * length(metrics), 150 + 30 * length(sc)) : size
    fig = Figure(; size = sz)
    for (mi, m) in enumerate(metrics)
        M = [begin
                 i = findfirst(r -> r.scheme == s && r.dataset == d, rows)
                 i === nothing ? NaN : getfield(rows[i], m)
             end for s in sc, d in dsets]
        pct = m === :zeroed
        disp = pct ? 100 .* M : M
        ax = Axis(fig[1, mi];
            xticks = (1:length(dsets), dsets), yticks = (1:length(sc), sc),
            xticklabelrotation = π/5, yreversed = true,
            ylabelvisible = mi == 1,
            title = pct ? "annihilated (%) — lower is better" : "$(m) — higher is better")
        mi > 1 && hideydecorations!(ax; grid = false)
        # low-is-good metrics get a reversed ramp so "good" is always the light end
        cmap = pct ? cgrad(:viridis; rev = true) : cgrad(:viridis)
        hm = heatmap!(ax, 1:length(dsets), 1:length(sc), permutedims(disp); colormap = cmap)
        for i in eachindex(sc), j in eachindex(dsets)
            isnan(disp[i, j]) && continue
            text!(ax, j, i; text = pct ? string(round(disp[i, j], digits = 1)) :
                                        string(round(disp[i, j], digits = 1)),
                  align = (:center, :center), fontsize = 8.5,
                  color = text_on(cmap[(disp[i, j] - minimum(filter(!isnan, disp))) /
                      max(maximum(filter(!isnan, disp)) - minimum(filter(!isnan, disp)), eps())]))
        end
    end
    Label(fig[2, 1:length(metrics)],
          "SNR is energy-weighted and blind to annihilated coordinates; read both panels together.",
          fontsize = 9, color = GREY, halign = :left, tellwidth = false)
    fig
end

"""
    plot_dot_snr_schemes(formats, lengths=[32,128,512,2048,8192]; trials=16, rng, size)
        -> Figure

Dot-product SNR against vector length, per scheme, with the 10–90 % band.

The expected shape is **flat**: block arithmetic is exact, so all noise enters at the
encoder and the ratio of signal to noise power is independent of length. A sagging curve
would mean the arithmetic is leaking error.
"""
function plot_dot_snr_schemes(formats, lengths = [32, 128, 512, 2048, 8192];
                              trials::Integer = 16,
                              rng::AbstractRNG = Random.default_rng(), size = (900, 430))
    set_theme!(xpufp_theme())
    fig = Figure(; size = size)
    ax = Axis(fig[1, 1]; xscale = log10, xlabel = "vector length N",
        ylabel = "dot-product SNR (dB)",
        title = "Dot-product accuracy is length-invariant when the block tree is exact")
    cols = cgrad(:viridis, max(length(formats), 2); categorical = true)
    for (i, f) in enumerate(formats)
        c = dot_snr_curve(f, lengths; trials, rng)
        band!(ax, [r.N for r in c], [r.lo for r in c], [r.hi for r in c];
              color = (cols[i], 0.15))
        scatterlines!(ax, [r.N for r in c], [r.snr for r in c];
                      color = cols[i], markersize = 8, label = _fmtname(f))
    end
    axislegend(ax; position = :rb, labelsize = 9)
    Label(fig[2, 1], "points: median over trials.  band: 10–90 %.",
          fontsize = 9, color = GREY, halign = :left, tellwidth = false)
    fig
end

"""
    plot_error_profiles(formats; ref=FP32, size=(900,430)) -> Figure

Per-element relative error against `t = |xᵢ|/M`, the element's size relative to its block
maximum — the like-for-like comparison across schemes.

`FP32` is a flat line: the same guarantee to every element at every `t`. A block-scaled
format is a ripple inside its window, degrading through the twilight, and 100 % beyond
the wall at `t = 1/32`.
"""
function plot_error_profiles(formats; ref = FP32, size = (900, 430))
    set_theme!(xpufp_theme())
    ts = exp10.(range(-3, 0; length = 800))
    fig = Figure(; size = size)
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10,
        xlabel = "t = |xᵢ| / M   (element size relative to its block maximum)",
        ylabel = "per-element relative error",
        title = "The like-for-like profile every spread experiment samples")
    cols = cgrad(:viridis, max(length(formats), 2); categorical = true)
    for (i, f) in enumerate(formats)
        e = max.(error_profile(f, ts), 1e-9)
        lines!(ax, ts, e; color = cols[i], linewidth = 1.6, label = _fmtname(f))
    end
    if ref !== nothing
        lines!(ax, ts, max.(error_profile(ref, ts), 1e-9);
               color = GREY, linewidth = 1.6, linestyle = :dash, label = _fmtname(ref))
    end
    vlines!(ax, [1/32]; color = ACCENT, linestyle = :dash)
    text!(ax, 1/32, 3e-2; text = "the wall\nt = 1/32", align = (:right, :top),
          fontsize = 9, color = ACCENT)
    ylims!(ax, 1e-9, 3)
    axislegend(ax; position = :lb, labelsize = 9)
    fig
end

"""
    plot_block_snr_distribution(formats, x; K=32, size=(900,430)) -> Figure

The distribution of **per-block** SNR, not just its mean.

Tails threaten guarantees, not medians: it is the unlucky, outlier-bearing blocks in the
lower percentiles that break a network, and they are exactly what a single aggregate SNR
conceals.
"""
function plot_block_snr_distribution(formats, x::AbstractVector; K::Integer = 32,
                                     size = (900, 430))
    set_theme!(xpufp_theme())
    xs = collect(Float64, x)
    fig = Figure(; size = size)
    ax = Axis(fig[1, 1]; xlabel = "per-block SNR (dB)", ylabel = "density",
        title = "Per-block SNR — the lower tail is what decides deployability")
    cols = cgrad(:viridis, max(length(formats), 2); categorical = true)
    for (i, f) in enumerate(formats)
        xh = quantize_all(f, xs)
        bs = Float64[]
        for j in 1:K:length(xs)
            k = min(j + K - 1, length(xs))
            s = snr_db(@view(xs[j:k]), @view(xh[j:k]))
            isfinite(s) && push!(bs, s)
        end
        isempty(bs) && continue
        density!(ax, bs; color = (cols[i], 0.20), strokecolor = cols[i],
                 strokewidth = 1.8, label = _fmtname(f))
        vlines!(ax, [quantile(bs, 0.10)]; color = (cols[i], 0.8), linestyle = :dot)
    end
    axislegend(ax; position = :lt, labelsize = 9)
    Label(fig[2, 1], "dotted verticals mark each scheme's 10th percentile.",
          fontsize = 9, color = GREY, halign = :left, tellwidth = false)
    fig
end

"""
    plot_snr_simulation(sims; size=(920,430)) -> Figure

Measured SNR against its analytical estimate, for a set of [`SNRSimulation`](@ref)s.

**Left** — the per-trial spread with the 95 % confidence interval and, where one exists,
the derived estimate as a marker. Agreement within the interval validates both the
measurement and the derivation; a gap localises the error to one of them.

**Right** — measured dot-product SNR against the `SNRₑ − 3.01 dB` law.

```julia
using Random
sims = [simulate_snr(f; n = 30_000, trials = 12, rng = MersenneTwister(1))
        for f in (MXFP4, MXFP4_OPT32, NVFP4, XPFP4_32)]
plot_snr_simulation(sims)
```
"""
function plot_snr_simulation(sims::AbstractVector{SNRSimulation}; size = (920, 430))
    set_theme!(xpufp_theme())
    n = length(sims)
    names = [s.scheme for s in sims]
    fig = Figure(; size = size)

    ax1 = Axis(fig[1, 1]; xticks = (1:n, names), ylabel = "element SNR (dB)",
        xticklabelrotation = π/7, title = "measured vs derived")
    for (i, s) in enumerate(sims)
        scatter!(ax1, fill(Float64(i), length(s.samples)) .+ 0.06 .* randn(length(s.samples)),
                 s.samples; color = (FRAC_COLOR, 0.35), markersize = 5)
        lines!(ax1, [i - 0.22, i + 0.22], [s.measured, s.measured];
               color = FRAC_COLOR, linewidth = 2.5)
        lines!(ax1, [i, i], [s.ci[1], s.ci[2]]; color = FRAC_COLOR, linewidth = 1.2)
        s.estimate === nothing && continue
        scatter!(ax1, [Float64(i)], [s.estimate]; color = ACCENT, markersize = 13,
                 marker = :diamond)
    end
    scatter!(ax1, [NaN], [NaN]; color = ACCENT, marker = :diamond, markersize = 11,
             label = "analytical estimate")
    scatter!(ax1, [NaN], [NaN]; color = FRAC_COLOR, markersize = 6, label = "per-trial measurement")
    axislegend(ax1; position = :lt, labelsize = 9)

    ax2 = Axis(fig[1, 2]; xticks = (1:n, names), ylabel = "dot-product SNR (dB)",
        xticklabelrotation = π/7, title = "dot product vs the −3.01 dB law")
    barplot!(ax2, 1:n, [s.dot_measured for s in sims]; color = (FRAC_COLOR, 0.75),
             width = 0.55, label = "measured")
    scatter!(ax2, 1:n, [s.dot_predicted for s in sims]; color = ACCENT, markersize = 13,
             marker = :hline, label = "SNRₑ − 3.01")
    axislegend(ax2; position = :lb, labelsize = 9)
    Label(fig[2, 1:2],
          "dot SNR is the ensemble ratio 10log₁₀(Σy²/Σe²); the median of per-trial ratios " *
          "is a different, far noisier statistic.",
          fontsize = 9, color = GREY, halign = :left, tellwidth = false)
    fig
end
