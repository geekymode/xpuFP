# ---------------------------------------------------------------------------
# The structural diagrams: dependency graphs, Forney factor graphs, prefix
# butterflies, and reduction trees.
#
# These are the package's answers to the source report's TikZ figures.  The claim
# they all make is one claim: carry propagation is *memory*, and a redundant
# alphabet buys the model down from "chain with state" to "sliding window, order two".
# ---------------------------------------------------------------------------

"""
    plot_dependency_graph(n::Integer = 5; size=(980,470)) -> Figure

The impossibility of carry-forward in RR4, as a statement about **graphs**.

Both panels draw the full dependency graph of an addition; an arrow into a node means
"argument of".

*Left, standard binary*: `c_{k+1} = maj(x_k, y_k, c_k)` has the previous carry among
its arguments, so the graph contains the horizontal path `c₁ → c₂ → ⋯` by construction.
That path **is** the ripple, and its length is the word width.

*Right, minimal RR4*: every transfer `t_{k+1} = f(s_k, s_{k−1})` has two parents, both
in the `s` row.  The transfer row has **no internal edges at all**, so the dashed,
crossed arrows mark edges that would have to exist for a carry to travel two steps —
and the rule gives them nothing to be.  The longest dependency anywhere is
`x → s → t → z`: three hops at any width.
"""
function plot_dependency_graph(n::Integer = 5; size = (980, 470))
    set_theme!(xpufp_theme())
    fig = Figure(; size = size)
    dx = 1.9

    # ---- left: binary ripple ----
    ax1 = blank_axis(fig[1, 1]; title = "standard binary:  cₖ₊₁ = maj(xₖ, yₖ, cₖ)")
    for k in 0:n-1
        x = k * dx
        text!(ax1, x, 3.5; text = "x$(_sub(k))y$(_sub(k))", align = (:center, :center),
              fontsize = 9, color = GREY)
        _gnode!(ax1, x, 2.2, "c$(_sub(k+1))"; fill = (ACCENT, 0.30))
        _gnode!(ax1, x, 0.6, "z$(_sub(k))"; fill = (FRAC_COLOR, 0.25))
        arrow!(ax1, (x, 3.3), (x, 2.5); color = GREY, headsize = 0.16)
        arrow!(ax1, (x, 1.95), (x, 0.9); color = GREY, headsize = 0.16)
        if k < n - 1
            arrow!(ax1, (x + 0.32, 2.2), (x + dx - 0.32, 2.2); color = ACCENT,
                   linewidth = 2.0, headsize = 0.20)
        end
    end
    text!(ax1, (n - 1) * dx / 2, -0.5;
          text = "the carry row is a CHAIN: length n, one step per cell",
          align = (:center, :top), fontsize = 10, color = ACCENT)
    xlims!(ax1, -1.0, (n - 1) * dx + 1.0); ylims!(ax1, -1.4, 4.4)

    # ---- right: minimal RR4 ----
    ax2 = blank_axis(fig[1, 2]; title = "minimal RR4:  tₖ₊₁ = f(sₖ, sₖ₋₁)")
    for k in 0:n-1
        x = k * dx
        text!(ax2, x, 4.2; text = "x$(_sub(k))y$(_sub(k))", align = (:center, :center),
              fontsize = 9, color = GREY)
        _gnode!(ax2, x, 3.1, "s$(_sub(k))"; fill = (EXP_COLOR, 0.28))
        _gnode!(ax2, x, 1.8, "t$(_sub(k+1))"; fill = (ACCENT, 0.30))
        _gnode!(ax2, x, 0.5, "z$(_sub(k))"; fill = (FRAC_COLOR, 0.25))
        arrow!(ax2, (x, 4.0), (x, 3.4); color = GREY, headsize = 0.15)
        arrow!(ax2, (x, 2.85), (x, 2.1); color = GREY, headsize = 0.15)      # own sum
        if k > 0                                                             # the peek
            arrow!(ax2, (x - dx + 0.28, 3.0), (x - 0.26, 1.95); color = ACCENT2,
                   linewidth = 1.1, headsize = 0.15)
        end
        arrow!(ax2, (x + 0.22, 2.9), (x + 0.22, 0.78); color = GREY, headsize = 0.15)  # bypass s→z
        arrow!(ax2, (x, 1.55), (x, 0.8); color = GREY, headsize = 0.15)      # payment
        if k < n - 1                                                         # receipt
            arrow!(ax2, (x + 0.3, 1.72), (x + dx - 0.3, 0.62); color = GREY, headsize = 0.15)
        end
        # the forbidden t → t edge
        if k < n - 1
            lines!(ax2, [x + 0.32, x + dx - 0.32], [1.8, 1.8];
                   color = (ACCENT, 0.5), linewidth = 1.2, linestyle = :dash)
            xm = x + dx/2
            lines!(ax2, [xm - 0.13, xm + 0.13], [1.67, 1.93]; color = ACCENT, linewidth = 1.8)
            lines!(ax2, [xm - 0.13, xm + 0.13], [1.93, 1.67]; color = ACCENT, linewidth = 1.8)
        end
    end
    text!(ax2, (n - 1) * dx / 2, -0.5;
          text = "the transfer row has NO internal edges — longest path x→s→t→z: 3 hops, any width",
          align = (:center, :top), fontsize = 10, color = ACCENT)
    xlims!(ax2, -1.0, (n - 1) * dx + 1.0); ylims!(ax2, -1.4, 5.0)
    fig
end

function _gnode!(ax, x, y, label; fill = LIGHT_GREY, r = 0.30)
    poly!(ax, Circle(Point2f(x, y), r); color = fill, strokecolor = (:black, 0.55), strokewidth = 0.9)
    text!(ax, x, y; text = label, align = (:center, :center), fontsize = 9)
end

"""
    plot_factor_graph(n::Integer = 4; size=(980,500)) -> Figure

The carry-forward contrast in **Forney (normal) factor-graph** form, where variables
live on edges, constraints are boxes, external variables are half-edges, and dots
duplicate a shared variable.

*Left*: binary addition is a **state-space model** — one full-adder constraint per
column, and the carry variables form a rail threading every box, because each `FA` has
both a `c`-in and a `c`-out port.  This is the trellis of a system with memory, and any
exact evaluation must sweep it end to end.

*Right*: minimal RR4.  Half-edges `x_k, y_k` enter sum factors `σ_k`, whose `s_k` edges
branch at dots to the transfer factories `f_{k+1}` and the combiners `g_k`.  Each `t`
edge fans to two combiners — into `g_k` as the payment `−4t_{k+1}` and into `g_{k+1}` as
the receipt — the borrow scheme's conservation pair as wiring.

**No factory has a `t` port on its input side**, so no edge can thread two factories in
sequence: feedforward, window two, no state variable at all.
"""
function plot_factor_graph(n::Integer = 4; size = (980, 500))
    set_theme!(xpufp_theme())
    fig = Figure(; size = size)
    dx = 2.3

    ax1 = blank_axis(fig[1, 1]; title = "binary: a state-space model — the carry rail")
    for k in 0:n-1
        x = k * dx
        labelbox!(ax1, x - 0.42, 1.4, 0.84, 0.7, "FA$(_sub(k))"; fill = (FRAC_COLOR, 0.22), fontsize = 9)
        lines!(ax1, [x - 0.18, x - 0.18], [2.1, 2.8]; color = GREY, linewidth = 1.0)
        lines!(ax1, [x + 0.18, x + 0.18], [2.1, 2.8]; color = GREY, linewidth = 1.0)
        text!(ax1, x - 0.18, 2.9; text = "x$(_sub(k))", align = (:center, :bottom), fontsize = 8, color = GREY)
        text!(ax1, x + 0.18, 2.9; text = "y$(_sub(k))", align = (:center, :bottom), fontsize = 8, color = GREY)
        lines!(ax1, [x, x], [1.4, 0.7]; color = GREY, linewidth = 1.0)
        text!(ax1, x, 0.6; text = "z$(_sub(k))", align = (:center, :top), fontsize = 8, color = GREY)
        if k < n - 1
            lines!(ax1, [x + 0.42, x + dx - 0.42], [1.75, 1.75]; color = ACCENT, linewidth = 2.2)
            text!(ax1, x + dx/2, 1.85; text = "c$(_sub(k+1))", align = (:center, :bottom),
                  fontsize = 8, color = ACCENT)
        end
    end
    text!(ax1, (n-1)*dx/2, -0.3;
          text = "every cut severing the word is crossed by a carry edge ⇒ n boundaries fire in series",
          align = (:center, :top), fontsize = 9.5, color = ACCENT)
    xlims!(ax1, -1.2, (n-1)*dx + 1.2); ylims!(ax1, -1.3, 3.5)

    ax2 = blank_axis(fig[1, 2]; title = "minimal RR4: feedforward, window two, no state")
    yσ, yf, yg = 3.4, 2.1, 0.7
    for k in 0:n-1
        x = k * dx
        labelbox!(ax2, x - 0.36, yσ - 0.28, 0.72, 0.56, "σ$(_sub(k))"; fill = (EXP_COLOR, 0.25), fontsize = 9)
        labelbox!(ax2, x - 0.36, yf - 0.28, 0.72, 0.56, "f$(_sub(k+1))"; fill = (ACCENT, 0.28), fontsize = 9)
        labelbox!(ax2, x - 0.36, yg - 0.28, 0.72, 0.56, "g$(_sub(k))"; fill = (FRAC_COLOR, 0.25), fontsize = 9)
        lines!(ax2, [x - 0.16, x - 0.16], [yσ + 0.28, yσ + 0.9]; color = GREY, linewidth = 1.0)
        lines!(ax2, [x + 0.16, x + 0.16], [yσ + 0.28, yσ + 0.9]; color = GREY, linewidth = 1.0)
        text!(ax2, x, yσ + 1.0; text = "x$(_sub(k)) y$(_sub(k))", align = (:center, :bottom),
              fontsize = 8, color = GREY)
        # s_k drops to its factory, with a duplication dot
        lines!(ax2, [x, x], [yσ - 0.28, yf + 0.28]; color = GREY, linewidth = 1.0)
        scatter!(ax2, [x], [(yσ - 0.28 + yf + 0.28)/2]; color = :black, markersize = 5)
        # bypass: s_k continues past its factory to the combiner, down the east lane
        lines!(ax2, [x, x + 0.72, x + 0.72, x + 0.36],
                    [(yσ-0.28+yf+0.28)/2, (yσ-0.28+yf+0.28)/2, yg, yg];
               color = GREY, linewidth = 1.0)
        # payment: t_{k+1} straight down into g_k
        lines!(ax2, [x, x], [yf - 0.28, yg + 0.28]; color = ACCENT, linewidth = 1.4)
        # peek: s_{k-1} into f_{k+1}
        if k > 0
            # one clean diagonal into the factory's top-left corner; deliberately NOT a
            # shared horizontal run, which would read as a rail — the very thing absent here
            lines!(ax2, [x - dx + 0.10, x - 0.30],
                        [(yσ - 0.28 + yf + 0.28)/2, yf + 0.26];
                   color = ACCENT2, linewidth = 1.1)
        end
        # receipt: t_{k+1} across into g_{k+1}
        if k < n - 1
            lines!(ax2, [x + 0.30, x + dx - 0.36], [yf - 0.20, yg + 0.10];
                   color = ACCENT, linewidth = 1.0)
        end
        lines!(ax2, [x, x], [yg - 0.28, yg - 0.75]; color = GREY, linewidth = 1.0)
        text!(ax2, x, yg - 0.85; text = "z$(_sub(k))", align = (:center, :top), fontsize = 8, color = GREY)
    end
    # the two horizontal cuts
    hlines!(ax2, [(yσ + yf)/2]; color = (GREY, 0.7), linestyle = :dash)
    hlines!(ax2, [(yf + yg)/2]; color = (GREY, 0.7), linestyle = :dash)
    text!(ax2, -1.25, (yσ+yf)/2 + 0.10; text = "cut 1", align = (:left, :bottom),
          fontsize = 8.5, color = GREY)
    text!(ax2, -1.25, (yf+yg)/2 + 0.10; text = "cut 2", align = (:left, :bottom),
          fontsize = 8.5, color = GREY)
    text!(ax2, (n-1)*dx/2, -0.9;
          text = "two horizontal full-width cuts ⇒ the whole graph clocks in THREE stages, any width",
          align = (:center, :top), fontsize = 9.5, color = ACCENT)
    xlims!(ax2, -1.4, (n-1)*dx + 1.0); ylims!(ax2, -1.9, yσ + 1.7)
    fig
end

"""
    plot_butterfly_vs_planes(n::Integer = 16; size=(980,430)) -> Figure

The two escapes from the carry chain, side by side.

*Left*: keep standard binary and compute all carries as a **parallel prefix** of the
associative generate-and-propagate composition.  The Kogge–Stone network does this in
`⌈log₂n⌉` levels, and its diagonals — strides 1, 2, 4, … — are exactly the wings of a
radix-2 FFT butterfly, because prefix networks and butterflies belong to the same graph
family.  The price is wiring: `Θ(n log n)` combine edges and long spans.  The network
*organises* global dependence; it does not remove it.

*Right*: change the number system instead.  The RR4 window graph has constant depth
three at any width, linear wiring, and a bounded light cone of at most three columns per
digit.  There is no dependence left to organise.
"""
function plot_butterfly_vs_planes(n::Integer = 16; size = (980, 430))
    set_theme!(xpufp_theme())
    fig = Figure(; size = size)
    L = ceil(Int, log2(n))

    ax1 = blank_axis(fig[1, 1];
        title = "Kogge–Stone prefix: $(L) levels, Θ(n log n) wires, spans up to n/2")
    for lev in 0:L
        y = -lev * 1.0
        for k in 0:n-1
            lines!(ax1, [k, k], [y, y - 1.0]; color = (GREY, 0.35), linewidth = 0.7)
        end
        lev == L && continue
        stride = 1 << lev
        for k in stride:n-1
            lines!(ax1, [k - stride, k], [y, y - 1.0]; color = (ACCENT, 0.75), linewidth = 1.0)
        end
        text!(ax1, -1.4, y - 0.5; text = "stride $(stride)", align = (:right, :center),
              fontsize = 8.5, color = GREY)
    end
    xlims!(ax1, -5.0, n); ylims!(ax1, -L - 1.4, 0.9)

    ax2 = blank_axis(fig[1, 2];
        title = "RR4 planes: 3 levels at ANY width, Θ(n) wires, span ≤ 2 columns")
    for (lev, lab) in enumerate(("x,y → s", "s → t", "s,t → z"))
        y = -(lev - 1) * 1.0
        for k in 0:n-1
            lines!(ax2, [k, k], [y, y - 1.0]; color = (GREY, 0.35), linewidth = 0.7)
        end
        if lev == 2
            for k in 1:n-1
                lines!(ax2, [k - 1, k], [y, y - 1.0]; color = (ACCENT, 0.75), linewidth = 1.0)
            end
        elseif lev == 3
            for k in 0:n-2
                lines!(ax2, [k + 1, k], [y, y - 1.0]; color = (ACCENT, 0.75), linewidth = 1.0)
            end
        end
        text!(ax2, -1.4, y - 0.5; text = lab, align = (:right, :center), fontsize = 8.5, color = GREY)
    end
    xlims!(ax2, -5.0, n); ylims!(ax2, -L - 1.4, 0.9)
    Label(fig[2, 1:2],
          "left: the wire mass grows like n log n and the height like log n.   " *
          "right: both height and per-column wiring are constants — only the ribbon widens.",
          fontsize = 10, color = GREY, tellwidth = false)
    fig
end

"""
    plot_wallace_tree(rows::AbstractVector{<:Integer}; size=(980,430)) -> Figure

The Wallace tree in both of its native pictures.

*Left*: the rows enter as vertical wires and the bundle visibly narrows `9→6→4→3→2` —
each 3:2 box consumes three input lanes and continues only two.  The narrowing **is**
the tree: trace any final wire upward and its ancestry is a ternary tree of boxes with
the rows as leaves.  The exit CPA at the bottom is the sole carry chain in the whole
structure.

*Right*: the same discipline per weight column as the classic **dot diagram** — triples
of equal-weight dots compress to a sum dot in place plus a carry dot exactly one column
leftward, and never further.  That is the whole carry-dodging discipline in one sentence.
"""
function plot_wallace_tree(rows::AbstractVector{<:Integer}; size = (980, 430))
    set_theme!(xpufp_theme())
    tr = wallace_reduce(rows)
    counts = [length(l) for l in tr.levels]
    fig = Figure(; size = size)

    ax1 = blank_axis(fig[1, 1];
        title = "the narrowing bundle: " * join(string.(counts), "→") *
                "   ($(tr.nlevels) levels, $(tr.ncells) cells)")
    for (lev, c) in enumerate(counts)
        y = -(lev - 1) * 1.2
        for k in 1:c
            x = k - c/2
            lines!(ax1, [x, x], [y, y - 1.2]; color = (GREY, 0.4), linewidth = 0.8)
        end
        lev == length(counts) && continue
        nboxes = (c) ÷ 3
        for b in 1:nboxes
            x0 = (3b - 2) - c/2; x2 = (3b) - c/2
            poly!(ax1, Rect2f(x0 - 0.22, y - 0.72, (x2 - x0) + 0.44, 0.42);
                  color = (ACCENT, 0.30), strokecolor = (:black, 0.5), strokewidth = 0.7)
            b == 1 && text!(ax1, (x0 + x2)/2, y - 0.51; text = "3:2",
                            align = (:center, :center), fontsize = 7.5)
        end
        text!(ax1, -c/2 - 1.6, y - 0.6; text = "$(c)→$(counts[lev+1])",
              align = (:right, :center), fontsize = 8.5, color = GREY)
    end
    ylast = -(length(counts) - 1) * 1.2 - 1.2
    labelbox!(ax1, -1.6, ylast - 0.85, 3.2, 0.6, "exit CPA — the ONLY carry chain";
              fill = (FRAC_COLOR, 0.25), fontsize = 8.5)
    xlims!(ax1, -maximum(counts)/2 - 4.0, maximum(counts)/2 + 1.5)
    ylims!(ax1, ylast - 1.5, 0.7)

    ax2 = blank_axis(fig[1, 2]; title = "dot diagram: carry dots move exactly one column left")
    ncol = 7
    for lev in 0:2
        y = -lev * 2.2
        for col in 1:ncol
            ndots = max(0, 5 - lev * 2 - abs(col - 4))
            for d in 1:ndots
                scatter!(ax2, [Float64(col)], [y - 0.25 * d];
                         color = lev == 0 ? FRAC_COLOR : ACCENT, markersize = 7)
            end
        end
        if lev < 2
            for col in 2:ncol
                arrow!(ax2, (col - 0.1, y - 1.45), (col - 0.85, y - 1.9);
                       color = ACCENT, linewidth = 0.9, headsize = 0.16)
            end
            text!(ax2, ncol + 0.4, y - 0.7; text = "sum stays,\ncarry ← 1 col",
                  align = (:left, :center), fontsize = 8, color = GREY)
        end
    end
    xlims!(ax2, 0.2, ncol + 3.0); ylims!(ax2, -6.0, 0.6)
    fig
end

"""
    plot_reduction_pyramids(n::Integer = 24; size=(900,380)) -> Figure

What Booth/RR4 buys, drawn as the two reduction pyramids: `n` rows for plain binary
(one per bit) against `⌈n/2⌉+1` for Booth (one per bit-pair, plus the unsigned guard).

Operand recoding starts the right-hand funnel lower, which propagates down the whole
schedule — fewer constant-time compression levels and about half the full-adder cells —
while the carry-save form of every intermediate row is what makes each level `O(1)` in
the first place.
"""
function plot_reduction_pyramids(n::Integer = 24; size = (900, 380))
    set_theme!(xpufp_theme())
    pr, br = booth_rows(n)
    ps, bs = reduction_schedule(pr), reduction_schedule(br)
    fig = Figure(; size = size)
    ax = Axis(fig[1, 1]; xlabel = "live partial-product rows", ylabel = "compression level",
        title = "Reduction pyramids for an $(n)-bit multiplier", yreversed = true)
    for (sched, col, off, lab) in ((ps, FRAC_COLOR, -0.18, "plain binary"),
                                   (bs, ACCENT, 0.18, "Booth radix-4"))
        for (i, c) in enumerate(sched)
            barplot!(ax, [i + off], [c]; direction = :x, color = (col, 0.75), width = 0.34)
        end
        lines!(ax, sched, (1:length(sched)) .+ off; color = col, linewidth = 1.6, label = lab)
    end
    for (i, c) in enumerate(ps)
        text!(ax, c + 0.4, i - 0.18; text = string(c), align = (:left, :center), fontsize = 9, color = FRAC_COLOR)
    end
    for (i, c) in enumerate(bs)
        text!(ax, c + 0.4, i + 0.18; text = string(c), align = (:left, :center), fontsize = 9, color = ACCENT)
    end
    axislegend(ax; position = :rb)
    sv = booth_tree_saving(n)
    Label(fig[2, 1],
          "levels $(sv.plain_levels) → $(sv.booth_levels);   " *
          "cells $(sv.plain_cells) → $(sv.booth_cells) (ratio $(round(sv.cell_ratio, digits=2)));   " *
          "both funnels end in the same single carry-propagate adder",
          fontsize = 10, color = GREY, tellwidth = false)
    fig
end
