# ---------------------------------------------------------------------------
# Illustrations for the composed scheme H·XPFP4_32.
# ---------------------------------------------------------------------------

"""
    plot_xpfp4_pipeline(; size=(1000,430)) -> Figure

The `H·XPFP4_32` encoder and decoder as a dataflow, with each stage annotated by what it
costs and what it buys.

Three ingredients, each doing exactly one job: the rotation destroys outlier
concentration, the E4M3 scale over 32 elements places the block accurately at MXFP4's
storage cost, and the MSE-optimal search trades a small clip at the top for resolution
in the middle.
"""
function plot_xpfp4_pipeline(; size = (1000, 430))
    set_theme!(xpufp_theme())
    fig = Figure(; size = size)
    ax = blank_axis(fig[1, 1]; aspect = nothing, title = "H·XPFP4-32 — encode")
    stages = [
        ("32 real values\nxᵢ", LIGHT_GREY, "one block along\nthe reduction axis"),
        ("Hadamard\ny = Hx/√32", (EXP_COLOR, 0.30), "adds & subtracts only\nno multiplies"),
        ("scale search\nS = argmin ‖y − Ŝê‖²", (ACCENT, 0.28), "encoder-side only\nwire format unchanged"),
        ("round to E2M1\neᵢ = R(yᵢ/S)", (FRAC_COLOR, 0.28), "the ONLY lossy step"),
        ("store\n32 nibbles + E4M3", (INT_COLOR, 0.35), "136 bits\n4.25 b/value"),
    ]
    w, h, gap = 2.5, 1.15, 0.75
    for (i, (lab, col, note)) in enumerate(stages)
        x = (i - 1) * (w + gap)
        labelbox!(ax, x, 0, w, h, lab; fill = col, fontsize = 9.5)
        text!(ax, x + w/2, -0.16; text = note, align = (:center, :top),
              fontsize = 8, color = GREY)
        i < length(stages) && arrow!(ax, (x + w, h/2), (x + w + gap, h/2);
                                     color = GREY, headsize = 0.24)
    end
    xlims!(ax, -0.4, 5 * (w + gap)); ylims!(ax, -1.05, h + 0.35)

    ax2 = blank_axis(fig[2, 1]; aspect = nothing, title = "decode and MAC — the scale never enters the inner loop")
    d = [
        ("read nibbles\neᵃᵢ, eᵇᵢ", (FRAC_COLOR, 0.28), "4-bit operands"),
        ("32 tiny multiplies\neᵃᵢ·eᵇᵢ", (ACCENT, 0.28), "EXACT — 2×2 significands"),
        ("adder tree\nΣ, |Σ| ≤ 1152", (EXP_COLOR, 0.30), "EXACT — narrow fixed point"),
        ("one scale fixup\n× SᵃSᵇ", (INT_COLOR, 0.35), "1 multiply per 32 MACs"),
        ("Hᵀ on the output\ntile", LIGHT_GREY, "adds & subtracts only"),
    ]
    for (i, (lab, col, note)) in enumerate(d)
        x = (i - 1) * (w + gap)
        labelbox!(ax2, x, 0, w, h, lab; fill = col, fontsize = 9.5)
        text!(ax2, x + w/2, -0.16; text = note, align = (:center, :top),
              fontsize = 8, color = GREY)
        i < length(d) && arrow!(ax2, (x + w, h/2), (x + w + gap, h/2);
                                color = GREY, headsize = 0.24)
    end
    xlims!(ax2, -0.4, 5 * (w + gap)); ylims!(ax2, -1.05, h + 0.35)
    Label(fig[3, 1],
          "the four structural properties of MXFP4 survive intact: exact products, exact tree, " *
          "scale factored out of the loop, E2M1 elements for FP4-native tensor cores",
          fontsize = 9, color = GREY, halign = :left, tellwidth = false)
    fig
end

"""
    plot_rotation_effect(x; K=32, size=(960,400)) -> Figure

What the Hadamard rotation does to a block: before and after, on a log magnitude axis,
with each scheme's dead-zone threshold drawn in.

An outlier captures the shared scale, which pushes the threshold up and annihilates its
neighbours. Rotating spreads that one large value across all `K` coordinates, so no
single element dominates and almost nothing falls into the dead zone.
"""
function plot_rotation_effect(x::AbstractVector; K::Integer = 32, size = (960, 400))
    set_theme!(xpufp_theme())
    xs = collect(Float64, x)[1:K]
    H = hadamard(K)
    y = H * xs
    fig = Figure(; size = size)
    rmax(v) = maximum(abs, v) / median(abs.(v))
    # one shared y range: different axes would make the two panels look alike when the
    # entire point is that their spread differs
    lo = min(minimum(abs.(xs)), minimum(abs.(y))) / 3
    hi = max(maximum(abs, xs), maximum(abs, y)) * 4

    for (col, (v, ttl)) in enumerate((
            (xs, "before: one element dominates"),
            (y,  "after H: energy spread evenly")))
        ax = Axis(fig[1, col]; yscale = log10, xlabel = "element",
                  ylabel = col == 1 ? "|value|" : "", title = ttl)
        thr = maximum(abs, v) / 32          # the MX dead-zone threshold, M/32
        dead = findall(a -> a < thr, abs.(v))
        barplot!(ax, 1:K, abs.(v); color = [i in dead ? ACCENT : FRAC_COLOR for i in 1:K])
        hlines!(ax, [thr]; color = ACCENT, linestyle = :dash, linewidth = 1.3)
        text!(ax, K + 0.4, thr; text = "M/32", align = (:left, :center),
              fontsize = 8.5, color = ACCENT)
        text!(ax, 0.5, hi / 1.3;
              text = "max/median $(round(rmax(v), digits=1))×,  $(length(dead)) annihilated",
              align = (:left, :top), fontsize = 9.5, color = length(dead) > 0 ? ACCENT : GREY)
        ylims!(ax, lo, hi); xlims!(ax, 0.2, K + 2.6)
    end
    Label(fig[2, 1:2],
          "orange bars fall below the block's dead-zone threshold and are annihilated. " *
          "The rotation is orthogonal, so ‖x‖ and every dot product are unchanged — " *
          "only the concentration is.",
          fontsize = 9, color = GREY, halign = :left, tellwidth = false)
    fig
end

"""
    plot_scale_placement(x; size=(960,340)) -> Figure

Where each scale rule lands the block on the E2M1 grid.

`MXFP4`'s floor rule can only put the maximum *somewhere* in the top binade; `NVFP4`
anchors it at the top code; `XPFP4_32`'s MSE-optimal search deliberately overloads a
little, accepting a small clip on the largest element in exchange for finer resolution
on all the others — which is where its extra decibel comes from.
"""
function plot_scale_placement(x::AbstractVector; size = (960, 340))
    set_theme!(xpufp_theme())
    xs = collect(Float64, x)
    M = maximum(abs, xs)
    rules = [("MXFP4  floor 2^e", MXFP4), ("NVFP4  anchor M/6", NVFP4),
             ("XPFP4-32  MSE-optimal", XPFP4_32)]
    fig = Figure(; size = size)
    ax = Axis(fig[1, 1]; xscale = log10, xlabel = "|value| / S   (position on the E2M1 grid)",
              yticks = (1:length(rules), [r[1] for r in rules]),
              title = "Where each rule places the block on the element grid")
    g = filter(>(0), posgrid(E2M1))
    for (i, (nm, bf)) in enumerate(rules)
        S = block_scale(bf, xs)
        u = sort(abs.(xs) ./ S)
        vlines!(ax, g; color = (GREY, 0.20), linewidth = 0.8)
        scatter!(ax, u, fill(Float64(i), length(u)); color = FRAC_COLOR, markersize = 6)
        scatter!(ax, [M / S], [Float64(i)]; color = ACCENT, markersize = 11, marker = :diamond)
        clipped = M / S > 6
        text!(ax, maximum(u) * 1.15, i;
              text = "M/S = $(round(M/S, digits=2))" * (clipped ? "  ← clipped to 6" : ""),
              align = (:left, :center), fontsize = 9, color = clipped ? ACCENT : GREY)
    end
    for v in g
        text!(ax, v, length(rules) + 0.55; text = _fmtnum(v), align = (:center, :bottom),
              fontsize = 8, color = GREY)
    end
    vlines!(ax, [0.25]; color = ACCENT, linestyle = :dash)
    text!(ax, 0.25, 0.35; text = "dead zone\nbelow 0.25", align = (:center, :top),
          fontsize = 8.5, color = ACCENT)
    ylims!(ax, 0.1, length(rules) + 1.1)
    Label(fig[2, 1],
          "grey rules are the E2M1 codes; the orange diamond is the block maximum. " *
          "Anything left of 0.25 is stored as exact zero.",
          fontsize = 9, color = GREY, halign = :left, tellwidth = false)
    fig
end

"""
    plot_block_reconstruction(fmts, x; size=(980,420)) -> Figure

One block reconstructed by several schemes, element by element, with the error drawn as
a whisker and annihilated elements marked.

This is the picture SNR summarises and, on spread data, misleads about: a scheme can
post a good SNR while a third of its coordinates have been replaced by exact zero.
"""
function plot_block_reconstruction(fmts, x::AbstractVector; size = (980, 420))
    set_theme!(xpufp_theme())
    xs = collect(Float64, x)
    n = length(xs)
    fig = Figure(; size = size)
    for (i, f) in enumerate(fmts)
        xh = quantize_all(f, xs)
        z = count(k -> xh[k] == 0 && xs[k] != 0, 1:n)
        # symlog: a linear window around zero, logarithmic outside, so the outlier and
        # the small coordinates are both legible on one axis
        M = maximum(abs, xs)
        lin = max(M / 200, eps())
        # explicit ticks: Symlog10's automatic ones crowd into an illegible stack
        tk = [-M, -M/10, 0.0, M/10, M]
        tl = [_fmtnum(round(t, sigdigits = 2)) for t in tk]
        ax = Axis(fig[i, 1]; ylabel = _fmtname(f), yscale = Makie.Symlog10(lin),
                  yticks = (tk, tl), ylabelsize = 10, yticklabelsize = 9,
                  xlabelvisible = i == length(fmts), xlabel = i == length(fmts) ? "element" : "",
                  title = i == 1 ? "One block, reconstructed  (symlog axis: linear near 0, log outside)" : "")
        ylims!(ax, -M * 2.2, M * 2.2)
        for k in 1:n
            lines!(ax, [k, k], [xs[k], xh[k]]; color = (GREY, 0.6), linewidth = 1.0)
        end
        scatter!(ax, 1:n, xs; color = :black, markersize = 5)
        dead = [k for k in 1:n if xh[k] == 0 && xs[k] != 0]
        live = setdiff(1:n, dead)
        scatter!(ax, live, xh[live]; color = FRAC_COLOR, markersize = 8,
                 marker = :circle, strokewidth = 0)
        !isempty(dead) && scatter!(ax, dead, xh[dead]; color = ACCENT, markersize = 9,
                                   marker = :xcross)
        text!(ax, 1, maximum(xs);
              text = " SNR $(round(snr_db(xs, xh), digits=2)) dB   •   zeroed $(z)/$(n)",
              align = (:left, :top), fontsize = 9.5,
              color = z > 0 ? ACCENT : EXP_COLOR)
    end
    Label(fig[length(fmts) + 1, 1],
          "black dots: the original values.  blue circles: the reconstruction.  " *
          "orange ×: elements annihilated to exact zero.",
          fontsize = 9, color = GREY, halign = :left, tellwidth = false)
    fig
end

"""
    plot_opt_shift(; K=32, x=randn(MersenneTwister(1), 200_000), size=(980,430)) -> Figure

The optimized shift rule, in two panels.

**Left** — the decision. The floor rule leaves the scaled maximum at `u = 4·2^φ ∈ [4,8)`;
the shaded sliver above `u*` is the insurance zone, where the clamp toward code 6 costs
the most. The rule fires there and only there, shifting one binade so the maximum lands
at `u/2 ∈ [2,4)` instead.

**Right** — the SNR as a function of the threshold, showing the optimum is broad: the
curve is flat to within 0.01 dB over a wide band, so the constant is not delicate.
`φ* = 1` disables the rule and recovers plain MXFP4.
"""
function plot_opt_shift(; K::Integer = 32,
                        x::AbstractVector = randn(MersenneTwister(1), 200_000),
                        size = (980, 430))
    set_theme!(xpufp_theme())
    phistar = opt_shift_threshold(K)
    ustar = 4 * 2^phistar
    fig = Figure(; size = size)

    ax1 = Axis(fig[1, 1]; xlabel = "u = M/S, the scaled block maximum",
        ylabel = "relative error on the maximum",
        title = "the decision: where the floor rule leaves the max")
    us = range(4, 8; length = 500)
    err = [u <= 6 ? abs(quantize(E2M1, u) - u) / u : (u - 6) / u for u in us]
    poly!(ax1, Rect2f(ustar, 0, 8 - ustar, 0.30); color = (ACCENT, 0.13), strokewidth = 0)
    lines!(ax1, us, err; color = FRAC_COLOR, linewidth = 1.6, label = "floor rule (no shift)")
    lines!(ax1, us, [abs(quantize(E2M1, u/2) - u/2) / (u/2) for u in us];
           color = ACCENT, linewidth = 1.6, label = "after the one-binade shift")
    vlines!(ax1, [ustar]; color = ACCENT, linestyle = :dash, linewidth = 1.3)
    text!(ax1, ustar, 0.28; text = " u* = $(round(ustar, digits=2))\n φ* = $(phistar)",
          align = (:left, :top), fontsize = 9, color = ACCENT)
    text!(ax1, (ustar + 8)/2, 0.015; text = "insurance zone",
          align = (:center, :bottom), fontsize = 9, color = ACCENT)
    ylims!(ax1, 0, 0.30); xlims!(ax1, 4, 8)
    axislegend(ax1; position = :lt, labelsize = 9)

    ax2 = Axis(fig[1, 2]; xlabel = "threshold φ*", ylabel = "SNR (dB)",
        title = "the optimum is broad")
    ps = 0.70:0.01:1.00
    snrs = Float64[]
    for p in ps
        n = length(x); e = 0.0; sg = 0.0
        for i in 1:K:n
            j = min(i + K - 1, n); seg = @view x[i:j]
            M = maximum(abs, seg); M == 0 && continue
            S = mx_scale_opt(M; phistar = p)
            for v in seg
                d = quantize(E2M1, v / S) * S - v
                e += d * d; sg += v * v
            end
        end
        push!(snrs, 10log10(sg / e))
    end
    lines!(ax2, collect(ps), snrs; color = ACCENT, linewidth = 1.8)
    scatter!(ax2, [phistar], [snrs[argmin(abs.(collect(ps) .- phistar))]];
             color = ACCENT, markersize = 11)
    hlines!(ax2, [snrs[end]]; color = (GREY, 0.7), linestyle = :dash)
    text!(ax2, 0.70, snrs[end]; text = " plain MXFP4 (rule disabled)",
          align = (:left, :bottom), fontsize = 9, color = GREY)
    text!(ax2, phistar, maximum(snrs);
          text = "φ* = $(phistar)\n$(round(maximum(snrs), digits=3)) dB ",
          align = (:right, :top), fontsize = 9, color = ACCENT)
    Label(fig[2, 1:2],
          "K = $(K):  the rule fires on $(round(100 * opt_shift_rate(x, K), digits=1))% of blocks, " *
          "costs one comparison, and leaves the decoder bit-identical to standard MXFP4.",
          fontsize = 9, color = GREY, halign = :left, tellwidth = false)
    fig
end
