# ---------------------------------------------------------------------------
# Error-curve figures.
# ---------------------------------------------------------------------------

"""
    plot_relative_error(f::FloatFormat = E2M1; xmax=nothing, npts=6000, size=(920,400)) -> Figure

The complete relative-error function of a format, computed pointwise from the rounding
rule, saturation included.

For E2M1 it has three regimes, each a quantified failure mode:

- **The dead zone** — every `|x| < 0.25` rounds to 0: a 100% relative error.  The
  format cannot distinguish small from zero.
- **The sawtooth** — on `[0.25, 6]` the error oscillates between 0 (on grid points) and
  peaks at the midpoints.  The familiar `ε/2 = 25%` bound holds only on `[1,6]`; the
  true worst case is **33%**, at `x = 0.75` in the wide subnormal-adjacent gap.
- **Saturation** — beyond 6 the error grows without bound.

The dashed line marks the textbook `ε/2` bound; note where the true curve exceeds it.
"""
function plot_relative_error(f::FloatFormat = E2M1; xmax = nothing, npts = 6000,
                             size = (920, 400))
    set_theme!(xpufp_theme())
    hi = xmax === nothing ? maxfinite(f) * 2.2 : xmax
    lo = minsubnormal(f) / 8
    xs = exp10.(range(log10(lo), log10(hi); length = npts))
    err = [abs(quantize(f, x) - x) / x for x in xs]
    g = filter(>(0), posgrid(f))
    fig = Figure(; size = size)
    ax = Axis(fig[1, 1]; xscale = log10,
        xlabel = "x", ylabel = "relative error  |x − Q(x)| / |x|",
        title = "$(f.name): the complete relative-error function")
    dz = minsubnormal(f) / 2
    poly!(ax, Rect2f(lo, 0, dz - lo, 1.15); color = (ACCENT, 0.10), strokewidth = 0)
    poly!(ax, Rect2f(maxfinite(f), 0, hi - maxfinite(f), 1.15); color = (ACCENT, 0.10), strokewidth = 0)
    lines!(ax, xs, err; color = FRAC_COLOR, linewidth = 1.4)
    hlines!(ax, [machine_eps(f) / 2]; color = GREY, linestyle = :dash, linewidth = 1.1)
    scatter!(ax, g, zeros(length(g)); color = ACCENT, markersize = 8)
    peak = maximum(err[xs .<= maxfinite(f)])
    text!(ax, lo * 1.3, 1.0; text = "dead zone\n(→ 0, 100 % error)",
          align = (:left, :top), fontsize = 9, color = ACCENT)
    text!(ax, maxfinite(f) * 1.05, 0.9; text = "saturation\n(unbounded)",
          align = (:left, :top), fontsize = 9, color = ACCENT)
    text!(ax, sqrt(dz * maxfinite(f)), peak + 0.05;
          text = "sawtooth — worst case $(round(100*peak, digits=1)) %",
          align = (:center, :bottom), fontsize = 9.5, color = FRAC_COLOR)
    text!(ax, hi, machine_eps(f)/2 + 0.012; text = "textbook ε/2 bound",
          align = (:right, :bottom), fontsize = 9, color = GREY)
    ylims!(ax, -0.03, 1.15); xlims!(ax, lo, hi)
    fig
end

"""
    plot_verdict_curve(; bf=MXFP4, ref=FP32, npts=4000, size=(900,420)) -> Figure

The direct, like-for-like comparison: per-element relative representation error against
the element's size **relative to its block maximum**, `t = |xᵢ|/M`.

FP32 promises `6e-8` to every element at every `t`, forever.  MXFP4 promises a 5–33 %
ripple inside its ~20 dB window, degrades through the twilight, and stores everything
below `t = 1/32` as exact zero.

Every spread experiment is this single profile, sampled by different data — and the
acceptance test for MXFP4 is simply how much of your block lives left of the wall.
"""
function plot_verdict_curve(; bf::BlockFormat = MXFP4, ref::FloatFormat = FP32,
                            npts = 4000, size = (900, 420))
    set_theme!(xpufp_theme())
    M = 1.0
    S = exp2(binade_exponent(M) - elem_emax(bf))     # the canonical scale, S = M/4
    ts = exp10.(range(-3, 0; length = npts))
    mxerr = [begin
                 x = t * M
                 abs(quantize(bf.elem, x / S) * S - x) / x
             end for t in ts]
    referr = [begin
                  x = t * M
                  abs(quantize(ref, x) - x) / x
              end for t in ts]
    fig = Figure(; size = size)
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10,
        xlabel = "t = |xᵢ| / M   (element size relative to its block maximum)",
        ylabel = "per-element relative error",
        title = "The like-for-like verdict: $(bf.name) against $(ref.name)")
    dead = 1 / 32
    poly!(ax, Rect2f(1e-3, 1e-9, dead - 1e-3, 10); color = (ACCENT, 0.10), strokewidth = 0)
    lines!(ax, ts, max.(mxerr, 1e-9); color = format_color(bf), linewidth = 1.6,
           label = "$(bf.name)  ($(round(bits_per_element(bf), digits=2)) b/value)")
    lines!(ax, ts, max.(referr, 1e-9); color = format_color(ref), linewidth = 1.6,
           label = "$(ref.name)  ($(nbits(ref)) b/value)")
    vlines!(ax, [dead]; color = ACCENT, linestyle = :dash, linewidth = 1.2)
    text!(ax, dead * 0.9, 3e-2; text = "the wall:\nt = 1/32\n→ exact zero",
          align = (:right, :top), fontsize = 9, color = ACCENT)
    axislegend(ax; position = :lb)
    ylims!(ax, 1e-9, 3); xlims!(ax, 1e-3, 1.05)
    fig
end

"""
    plot_stagnation(f::FloatFormat = E2M1, addend = 0.5, n = 8; size=(880,330)) -> Figure

Accumulator stagnation, drawn on the format's actual grid.

Four exact hops, then permanent stagnation: `2 + 0.5` lands on the midpoint 2.5 and
ties-to-even returns it to 2 on every subsequent addition.  The computed sum is 2; the
true sum is 4.
"""
function plot_stagnation(f::FloatFormat = E2M1, addend::Real = 0.5, n::Integer = 8;
                         size = (880, 330))
    set_theme!(xpufp_theme())
    tr = stagnation_trace(f, addend, n)
    truth = [addend * k for k in 1:n]
    fig = Figure(; size = size)
    ax = Axis(fig[1, 1]; xlabel = "additions", ylabel = "accumulator",
        title = "Summing $(n) copies of $(addend) with an $(f.name) accumulator")
    g = [v for v in posgrid(f) if v <= maximum(truth) * 1.15]
    hlines!(ax, g; color = (GREY, 0.25), linewidth = 0.8)
    lines!(ax, 1:n, truth; color = GREY, linestyle = :dash, linewidth = 1.4, label = "true sum")
    scatterlines!(ax, 1:n, tr; color = ACCENT, markersize = 11, linewidth = 1.8,
                  label = "$(f.name) accumulator")
    thr = stagnation_threshold(f, addend)
    hlines!(ax, [thr]; color = ACCENT2, linestyle = :dot, linewidth = 1.2)
    text!(ax, n, thr; text = " stall threshold = addend/ε = $(_fmtnum(thr))",
          align = (:right, :bottom), fontsize = 9, color = ACCENT2)
    stall = findfirst(k -> k < n && tr[k] == tr[k+1], 1:n-1)
    if stall !== nothing
        vlines!(ax, [stall]; color = (ACCENT, 0.4), linestyle = :dash)
        text!(ax, stall + 0.1, maximum(truth) * 0.35;
              text = "stagnates here:\n$(_fmtnum(tr[stall])) + $(addend) = " *
                     "$(_fmtnum(tr[stall]+addend)) is a midpoint\n→ ties-to-even sends it back",
              align = (:left, :center), fontsize = 9, color = ACCENT)
    end
    axislegend(ax; position = :lt)
    fig
end

"""
    plot_fp4_error_regimes(; size=(900,340)) -> Figure

The three FP4 regimes as a bar chart of *where the probability mass goes* for Gaussian
data at several scales — the "window problem" made visual.

Bare FP4 does not degrade gracefully as data shrinks; it fails **totally**, because a
12:1 window cannot meet data whose scale it does not know.
"""
function plot_fp4_error_regimes(; sigmas = (1.0, 0.3, 0.1, 0.02), n = 100_000,
                                rng = MersenneTwister(7), size = (900, 340))
    set_theme!(xpufp_theme())
    dz = minsubnormal(E2M1) / 2
    sat = maxfinite(E2M1)
    labels = String[]; deadfrac = Float64[]; livefrac = Float64[]; satfrac = Float64[]
    snrs = Float64[]
    for s in sigmas
        x = s .* randn(rng, n)
        a = abs.(x)
        push!(labels, "σ = $(s)")
        push!(deadfrac, count(<(dz), a) / n)
        push!(satfrac, count(>(sat), a) / n)
        push!(livefrac, 1 - deadfrac[end] - satfrac[end])
        push!(snrs, measure_snr(E2M1, x))
    end
    fig = Figure(; size = size)
    ax = Axis(fig[1, 1]; xticks = (1:length(sigmas), labels), ylabel = "fraction of samples",
        title = "Bare FP4 against Gaussian data: the window problem")
    barplot!(ax, 1:length(sigmas), deadfrac; color = (ACCENT, 0.75), label = "dead zone (→ 0)")
    barplot!(ax, 1:length(sigmas), livefrac; offset = deadfrac, color = (FRAC_COLOR, 0.7),
             label = "live range")
    barplot!(ax, 1:length(sigmas), satfrac; offset = deadfrac .+ livefrac,
             color = (EXP_COLOR, 0.7), label = "saturated")
    for (i, s) in enumerate(snrs)
        text!(ax, i, 1.02; text = @sprintf("%.1f dB", s), align = (:center, :bottom),
              fontsize = 10, font = :bold)
    end
    ylims!(ax, 0, 1.16)
    axislegend(ax; position = :lb, orientation = :horizontal)
    fig
end
