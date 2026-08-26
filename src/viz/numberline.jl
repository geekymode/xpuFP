# ---------------------------------------------------------------------------
# Number-line figures: where the representable points actually sit.
# ---------------------------------------------------------------------------

"""
    plot_number_line(f::FloatFormat = E2M1; size=(950,240)) -> Figure

The **entire** number line of a narrow format, drawn to scale — no schematic needed.

For E2M1 all fifteen distinct values fit on one axis: spacing doubles at each power of
two exactly as in FP32, the subnormal ±0.5 extends the innermost step uniformly to zero
(gradual underflow in miniature), and the line simply stops at ±6, where arithmetic
saturates instead of producing ±∞.
"""
function plot_number_line(f::FloatFormat = E2M1; size = (950, 240))
    set_theme!(xpufp_theme())
    g = grid(f)
    lo, hi = minimum(g), maximum(g)
    fig = Figure(; size = size)
    ax = Axis(fig[1, 1];
        title = "The complete $(f.name) number line — all $(length(g)) distinct values, to scale",
        xlabel = "value", yticksvisible = false, yticklabelsvisible = false,
        ygridvisible = false)
    hidespines!(ax, :l, :r, :t)
    lines!(ax, [lo * 1.08, hi * 1.08], [0, 0]; color = GREY, linewidth = 1.0)
    sub = [v for v in g if v != 0 && abs(v) < minnormal(f)]
    nor = [v for v in g if abs(v) >= minnormal(f)]
    scatter!(ax, nor, zeros(length(nor)); color = FRAC_COLOR, markersize = 11)
    !isempty(sub) && scatter!(ax, sub, zeros(length(sub)); color = ACCENT, markersize = 11,
                              marker = :diamond)
    scatter!(ax, [0.0], [0.0]; color = :black, markersize = 9)
    for v in g
        text!(ax, v, 0.16; text = _fmtnum(v), align = (:center, :bottom),
              fontsize = 8.5, rotation = π/2.4, color = GREY)
    end
    # binade spacing annotations
    for e in emin(f):emax(f)
        a, b = exp2(e), exp2(e + 1)
        b > hi * 1.02 && continue
        step = exp2(e - f.mbits)
        lines!(ax, [a, min(b, hi)], [-0.30, -0.30]; color = (GREY, 0.6), linewidth = 1.0)
        text!(ax, (a + min(b, hi)) / 2, -0.36; text = "step $(_fmtnum(step))",
              align = (:center, :top), fontsize = 8, color = GREY)
    end
    ylims!(ax, -0.75, 0.95); xlims!(ax, lo * 1.12, hi * 1.12)
    if !isempty(sub)
        text!(ax, 0, 0.72;
              text = "◆ subnormal (hidden bit 0, scale pinned at 2^$(emin(f)))   " *
                     "● normal   •  saturates at ±$(maxfinite(f))",
              align = (:center, :bottom), fontsize = 9, color = GREY)
    end
    fig
end

"""
    plot_binade_spacing(f::FloatFormat = FP32; size=(880,380)) -> Figure

The defining property of floating point, plotted: absolute spacing (ulp) against
magnitude on log–log axes is a staircase that **doubles at every power of two**, while
*relative* spacing stays bounded — within a factor of 2, forever.

That factor of 2 is not slack in the plot; it is real. Relative spacing is a **sawtooth**
between ``2^{-mbits}`` and ``2^{-mbits-1}``, flat only by comparison with an absolute
staircase spanning eighty octaves. [`plot_log_axis_analogy`](@ref) is the same quantity
drawn over six binades instead of all of them, where the wobble is the point.

Within one binade `[2ᵉ, 2ᵉ⁺¹)` there are exactly `2^mbits` equally spaced values, so the
gap between neighbours is `2^(e−mbits)`.  Around 1.0 the FP32 spacing is `ε ≈ 1.2e-7`;
around `2²⁴` it reaches 2, so FP32 cannot even represent the odd integer 16 777 217.
"""
function plot_binade_spacing(f::FloatFormat = FP32; size = (880, 380))
    set_theme!(xpufp_theme())
    es = max(emin(f), -40):min(emax(f), 40)
    xs = Float64[]; ulps = Float64[]; rel = Float64[]
    for e in es
        a = exp2(e)
        push!(xs, a); push!(ulps, exp2(e - f.mbits)); push!(rel, exp2(-f.mbits))
        b = prevfloat(exp2(e + 1))
        push!(xs, b); push!(ulps, exp2(e - f.mbits)); push!(rel, exp2(e - f.mbits) / b)
    end
    fig = Figure(; size = size)
    ax1 = Axis(fig[1, 1]; xscale = log10, yscale = log10,
               xlabel = "magnitude", ylabel = "ulp (absolute spacing)",
               title = "$(f.name): absolute spacing doubles every binade")
    lines!(ax1, xs, ulps; color = FRAC_COLOR, linewidth = 1.6)
    ax2 = Axis(fig[1, 2]; xscale = log10, yscale = log10,
               xlabel = "magnitude", ylabel = "ulp / value (relative spacing)",
               title = "…while relative spacing stays within a factor of 2")
    lines!(ax2, xs, rel; color = ACCENT, linewidth = 1.6)
    hlines!(ax2, [machine_eps(f)]; color = (GREY, 0.7), linestyle = :dash)
    text!(ax2, xs[1] * 4, machine_eps(f) * 1.3; text = "ε = 2^-$(f.mbits)",
          align = (:left, :bottom), fontsize = 10, color = GREY)
    fig
end

"""
    plot_edges_map(f::FloatFormat = FP32; size=(950,250)) -> Figure

The positive half of a format's universe on a logarithmic axis: the subnormal ramp, the
normal range, and the overflow cliff.

Zero itself sits off the logarithmic axis, which is exactly the point — the subnormals
exist to bridge that gap in even steps.
"""
function plot_edges_map(f::FloatFormat = FP32; size = (950, 250))
    set_theme!(xpufp_theme())
    fig = Figure(; size = size)
    ax = Axis(fig[1, 1]; xscale = log10, yticksvisible = false, yticklabelsvisible = false,
              ygridvisible = false, xlabel = "magnitude (log scale)",
              title = "$(f.name): the positive half of the universe")
    hidespines!(ax, :l, :r, :t)
    ms, mn, mx = minsubnormal(f), minnormal(f), maxfinite(f)
    poly!(ax, Rect2f(ms, -0.16, mn - ms, 0.32); color = (ACCENT, 0.25), strokewidth = 0)
    poly!(ax, Rect2f(mn, -0.16, mx - mn, 0.32); color = (FRAC_COLOR, 0.22), strokewidth = 0)
    vlines!(ax, [ms, mn, mx]; color = GREY, linewidth = 1.0)
    for (v, lab) in ((ms, "smallest\nsubnormal\n2^$(emin(f)-f.mbits)"),
                     (mn, "smallest\nnormal\n2^$(emin(f))"),
                     (mx, "largest\nnormal"))
        text!(ax, v, 0.24; text = lab, align = (:center, :bottom), fontsize = 8.5, color = GREY)
        text!(ax, v, -0.22; text = @sprintf("%.4g", v), align = (:center, :top),
              fontsize = 8.5, color = :black)
    end
    text!(ax, sqrt(ms * mn), 0.0; text = "subnormal ramp", align = (:center, :center), fontsize = 10)
    text!(ax, sqrt(mn * mx), 0.0; text = "normals", align = (:center, :center), fontsize = 10)
    text!(ax, mx * 8, 0.0; text = f.saturate ? "saturate" : "→ +∞",
          align = (:left, :center), fontsize = 10, color = ACCENT)
    xlims!(ax, ms / 8, mx * 400); ylims!(ax, -0.62, 0.72)
    fig
end

"""
    plot_seams(f::FloatFormat = FP32; size=(940,380)) -> Figure

Both boundaries under a microscope, on **linear** axes so the grid geometry is honest.

*Top seam*: the open circle is the phantom grid point `2^(emax+1)` — exactly one ulp
above `x_max`, but its encoding needs the reserved exponent, so the slot is occupied by
`+∞`, one increment of the bit pattern away.  The dashed line is the round-to-nearest
overflow threshold, the midpoint of that final gap.

*Bottom seam*: unlike the top, the spacing here is **identical on both sides** —
pinning the subnormal scale makes the last subnormal and the first normal exactly one
subnormal step apart.  The ladder crosses without a seam you could feel.
"""
function plot_seams(f::FloatFormat = FP32; size = (940, 380))
    set_theme!(xpufp_theme())
    ts = top_seam(f); bs = bottom_seam(f)
    fig = Figure(; size = size)

    ax1 = Axis(fig[1, 1]; title = "Top seam: one encoding step, $(_fmtnum(ts.ulp)) apart",
               xlabel = "value", yticksvisible = false, yticklabelsvisible = false, ygridvisible = false)
    hidespines!(ax1, :l, :r, :t)
    pts = [ts.xmax - k * ts.ulp for k in 3:-1:0]
    scatter!(ax1, pts, zeros(length(pts)); color = FRAC_COLOR, markersize = 11)
    scatter!(ax1, [ts.phantom], [0.0]; color = :white, strokecolor = ACCENT,
             strokewidth = 1.8, markersize = 13)
    vlines!(ax1, [ts.overflow_threshold]; color = ACCENT, linestyle = :dash, linewidth = 1.2)
    text!(ax1, ts.phantom, 0.14; text = "phantom 2^$(emax(f)+1)\n= +∞ slot",
          align = (:center, :bottom), fontsize = 9, color = ACCENT)
    text!(ax1, ts.xmax, -0.14; text = "x_max", align = (:center, :top), fontsize = 9)
    text!(ax1, ts.overflow_threshold, -0.32; text = "RN overflow\nthreshold",
          align = (:center, :top), fontsize = 8.5, color = ACCENT)
    ylims!(ax1, -0.75, 0.75)

    ax2 = Axis(fig[1, 2]; title = "Bottom seam: spacing identical on both sides",
               xlabel = "value", yticksvisible = false, yticklabelsvisible = false, ygridvisible = false)
    hidespines!(ax2, :l, :r, :t)
    st = bs.subnormal_step
    p2 = [bs.smallest_normal + k * st for k in -3:3]
    cols = [v < bs.smallest_normal ? ACCENT : FRAC_COLOR for v in p2]
    scatter!(ax2, p2, zeros(length(p2)); color = cols, markersize = 11)
    vlines!(ax2, [bs.smallest_normal]; color = GREY, linestyle = :dot)
    text!(ax2, bs.largest_subnormal, 0.14; text = "largest\nsubnormal",
          align = (:center, :bottom), fontsize = 8.5, color = ACCENT)
    text!(ax2, bs.smallest_normal, -0.16; text = "smallest\nnormal",
          align = (:center, :top), fontsize = 8.5, color = FRAC_COLOR)
    text!(ax2, bs.smallest_normal, 0.42;
          text = "gap = $(_fmtnum(bs.gap)) = the subnormal step — seamless",
          align = (:center, :bottom), fontsize = 9, color = GREY)
    ylims!(ax2, -0.75, 0.75)
    fig
end

"""
    plot_log_axis_analogy(f::FloatFormat = E4M3; size=(950,400)) -> Figure

**A float grid is a piecewise-linear approximation of a log axis** — this is the picture
that says why, and where the approximation fails.

Left: inside one binade the exponent is fixed, so a format places its `2^mbits` values by
their mantissa, **linearly**. A logarithmic axis would place them along `log₂(1+m)`. The
two curves are pinned together at both ends of the octave — floats are exact on powers of
two — and pull apart in between, worst at `m = 1/ln2 − 1 ≈ 0.4427`, where the gap reaches
`0.0861` of an octave. That is the same constant that makes the integer reading of a
float's bit pattern an approximate `log₂`.

Right: the consequence, as relative step size. A true log axis has **constant** relative
resolution; a float's `ulp(x)/x` is a **sawtooth** that halves across each binade and
doubles at every boundary. The ratio is exactly the radix, `2`, at every mantissa width —
narrowing the format does not reduce the wobble, it only raises the whole band.

See [`plot_binade_spacing`](@ref) for the absolute-spacing staircase behind this, and
[`plot_edges_map`](@ref) for the two places the analogy fails outright: the subnormal ramp
below, where a linear grid on a log axis flies apart, and the zero that a log axis has no
coordinate for.
"""
function plot_log_axis_analogy(f::FloatFormat = E4M3; size = (950, 400))
    set_theme!(xpufp_theme())
    fig = Figure(; size = size)

    # ---- left: inside one octave, linear mantissa vs true log ----
    ax1 = Axis(fig[1, 1];
        xlabel = "mantissa fraction m,  value = 1 + m",
        ylabel = "position within the octave",
        title = "$(f.name): the mantissa is linear, the axis is logarithmic")
    ms = range(0, 1; length = 400)
    lines!(ax1, ms, collect(ms); color = ACCENT, linewidth = 2,
           label = "float placement (linear in m)")
    lines!(ax1, ms, log2.(1 .+ ms); color = FRAC_COLOR, linewidth = 2,
           label = "true log₂(1 + m)")
    # the format's own grid points
    step = exp2(-f.mbits)
    gm = 0:step:(1 - step)
    scatter!(ax1, collect(gm), collect(gm); color = ACCENT, markersize = 9)
    scatter!(ax1, collect(gm), log2.(1 .+ gm); color = FRAC_COLOR, markersize = 9)
    for m in gm
        lines!(ax1, [m, m], [m, log2(1 + m)]; color = (GREY, 0.55), linewidth = 0.9)
    end
    # the peak of the gap, at m = 1/ln2 - 1
    mpk = 1 / log(2) - 1
    gap = log2(1 + mpk) - mpk
    lines!(ax1, [mpk, mpk], [mpk, log2(1 + mpk)]; color = :black, linewidth = 1.8)
    text!(ax1, mpk + 0.03, (mpk + log2(1 + mpk)) / 2;
          text = "max gap $(round(gap; digits = 4)) octaves\nat m = $(round(mpk; digits = 4))",
          align = (:left, :center), fontsize = 9)
    axislegend(ax1; position = :lt, framevisible = false, labelsize = 9)

    # ---- right: relative step size, the wobble ----
    ax2 = Axis(fig[1, 2]; xscale = log2, yscale = log2,
        xlabel = "value", ylabel = "ulp(x) / x  (relative step)",
        title = "…so relative resolution wobbles by exactly 2")
    e0 = max(emin(f), -4)
    es = e0:(e0 + 5)
    eps_f = machine_eps(f)
    # the *continuous* relative step, sampled densely: it reaches ε/2 at the top of
    # every binade whether or not the format has a grid point that far up
    xs = Float64[]; ys = Float64[]
    for e in es
        a = exp2(e)
        for t in range(0, 1; length = 120)
            x = a * (1 + t * (1 - 1e-9))
            push!(xs, x); push!(ys, exp2(e - f.mbits) / x)
        end
        push!(xs, NaN); push!(ys, NaN)          # break the line at the boundary
    end
    hlines!(ax2, [eps_f]; color = (GREY, 0.8), linestyle = :dash)
    hlines!(ax2, [eps_f / 2]; color = (GREY, 0.8), linestyle = :dash)
    lines!(ax2, xs, ys; color = ACCENT, linewidth = 2)
    # where this format actually samples that curve
    gx = Float64[]; gy = Float64[]
    for e in es, k in 0:(1 << f.mbits) - 1
        x = exp2(e) * (1 + k * step)
        push!(gx, x); push!(gy, exp2(e - f.mbits) / x)
    end
    scatter!(ax2, gx, gy; color = ACCENT, markersize = 6)
    hlines!(ax2, [eps_f / sqrt(2)]; color = FRAC_COLOR, linewidth = 2)
    text!(ax2, exp2(e0) * 1.04, eps_f * 0.97; text = "ε",
          align = (:left, :top), fontsize = 10, color = GREY)
    text!(ax2, exp2(e0) * 1.04, eps_f / 2 * 1.04; text = "ε / 2",
          align = (:left, :bottom), fontsize = 10, color = GREY)
    text!(ax2, exp2(e0) * 1.04, eps_f / sqrt(2) * 1.16;
          text = "a log axis would be flat", align = (:left, :bottom),
          fontsize = 9, color = FRAC_COLOR)
    fig
end
