# ---------------------------------------------------------------------------
# Figures for the redundant number systems.
#
# Several of these are the package's answers to the source report's TikZ diagrams:
# the dependency graph, the Forney factor graph, the butterfly-versus-planes
# comparison, and the Wallace tree in both of its native pictures.
# ---------------------------------------------------------------------------

"""
    plot_csd_digits(x::Integer; size=(900,300)) -> Figure

A constant in ordinary binary and in canonical signed digit form, weights in grey and
nonzero digits emphasised.

The run-collapsing identity `2ᵏ − 1` turns each block of consecutive ones into two
digits; the canonical no-adjacent-nonzeros form is unique and provably minimal in
nonzero count, pricing the constant multiplier at fewer adders.
"""
function plot_csd_digits(x::Integer; size = (900, 300))
    set_theme!(xpufp_theme())
    b = to_binary(abs(Int(x)))
    sd = csd(x)
    n = max(length(b), length(sd.digits))
    fig = Figure(; size = size)
    ax = blank_axis(fig[1, 1];
        title = "$(x) in binary and canonical signed digit  —  " *
                "$(binary_weight(x)) vs $(weight(sd)) nonzero digits " *
                "($(binary_weight(x)-1) vs $(adders(sd)) adders)")
    cw = 1.0
    for (row, (digs, lab, col)) in enumerate(((b, "binary", FRAC_COLOR),
                                              (sd.digits, "CSD", ACCENT)))
        y = (2 - row) * 1.6
        text!(ax, -0.4, y + 0.45; text = lab, align = (:right, :center), fontsize = 12, font = :bold)
        for i in n:-1:1
            xx = (n - i) * cw
            d = i <= length(digs) ? digs[i] : nothing
            if d === nothing
                poly!(ax, Rect2f(xx, y, cw, 0.9); color = (:white, 0.0),
                      strokecolor = (GREY, 0.3), strokewidth = 0.6, linestyle = :dot)
            else
                fill = d == 0 ? LIGHT_GREY : (col, 0.45)
                bitcell!(ax, xx, y, d == -1 ? "1̄" : string(d); fill = fill, w = cw, h = 0.9,
                         fontsize = 12)
            end
        end
    end
    for i in n:-1:1
        text!(ax, (n - i) * cw + cw/2, 3.35; text = "2$(_sup(i-1))",
              align = (:center, :bottom), fontsize = 8.5, color = GREY)
    end
    terms = join([(d > 0 ? "+" : "−") * "2$(_sup(i-1))"
                  for (i, d) in enumerate(sd.digits) if d != 0] |> reverse, " ")
    text!(ax, 0, -0.7; text = "CSD:  $(x) = $(lstrip(terms, ['+',' ']))",
          align = (:left, :top), fontsize = 11, color = ACCENT)
    xlims!(ax, -3.5, n * cw + 0.4); ylims!(ax, -1.6, 4.0)
    fig
end

const _SUPCHARS = ['⁰','¹','²','³','⁴','⁵','⁶','⁷','⁸','⁹']
_sup(n::Integer) = join([c == '-' ? '⁻' : _SUPCHARS[parse(Int, c)+1] for c in string(n)])

"""
    plot_csd_census(x::Integer, ndig::Integer; size=(880,340)) -> Figure

Every signed-binary spelling of `x` that fits in `ndig` digits, grouped by adder cost.

The landscape shows why the canonical form matters twice over: cost varies wildly for
spellings of the very same number, and even the *minimum* need not be unique — the
non-adjacency rule is what breaks the tie canonically.
"""
function plot_csd_census(x::Integer, ndig::Integer; size = (880, 340))
    set_theme!(xpufp_theme())
    sp = all_signed_spellings(x, ndig)
    ws = weight.(sp)
    lo, hi = minimum(ws), maximum(ws)
    counts = [count(==(w), ws) for w in lo:hi]
    canon = csd(x)
    fig = Figure(; size = size)
    ax = Axis(fig[1, 1]; xticks = (lo:hi, ["$(w)\n($(w-1) adders)" for w in lo:hi]),
        xlabel = "nonzero digits", ylabel = "number of spellings",
        title = "All $(length(sp)) signed-binary spellings of $(x) in $(ndig) digits")
    cols = [w == weight(canon) ? ACCENT : FRAC_COLOR for w in lo:hi]
    barplot!(ax, collect(lo:hi), counts; color = cols)
    for (i, w) in enumerate(lo:hi)
        text!(ax, w, counts[i]; text = string(counts[i]), align = (:center, :bottom),
              fontsize = 10, offset = (0, 2))
    end
    bw = binary_weight(x)
    vlines!(ax, [bw]; color = GREY, linestyle = :dash)
    text!(ax, bw, maximum(counts) * 0.92; text = " ordinary binary sits here",
          align = (:left, :center), fontsize = 9.5, color = GREY)
    text!(ax, weight(canon), maximum(counts) * 0.6;
          text = "canonical (NAF)\n$(digit_string(canon))",
          align = (:center, :center), fontsize = 9.5, color = ACCENT)
    fig
end

"""
    plot_booth_windows(B::Integer, n::Integer; size=(900,330)) -> Figure

Radix-4 Booth recoding drawn as overlapping 3-bit windows sliding across the multiplier.

Each brace shares one bit with its neighbour — **the overlap is what makes the recoding
correct** — and each window emits one digit `dⱼ = b₂ⱼ₋₁ + b₂ⱼ − 2b₂ⱼ₊₁`.  The digits
reassemble the two's-complement value exactly, negatives included, with no correction.
"""
function plot_booth_windows(B::Integer, n::Integer; size = (900, 330))
    set_theme!(xpufp_theme())
    iseven(n) || (n += 1)
    bits = twos_complement_bits(B, n)
    d = booth_radix4(B, n)
    fig = Figure(; size = size)
    ax = blank_axis(fig[1, 1];
        title = "Radix-4 Booth recoding of $(B) ($(n)-bit two's complement)")
    cw = 1.0
    # bits, LSB on the right
    for i in n:-1:1
        xx = (n - i) * cw
        bitcell!(ax, xx, 1.4, bits[i]; fill = (FRAC_COLOR, 0.28), w = cw, h = 0.85, fontsize = 11)
        text!(ax, xx + cw/2, 2.4; text = "b$(_sub(i-1))", align = (:center, :bottom),
              fontsize = 8, color = GREY)
    end
    # the padded b_{-1}
    bitcell!(ax, n * cw, 1.4, "0"; fill = LIGHT_GREY, w = cw, h = 0.85, fontsize = 11)
    text!(ax, n * cw + cw/2, 2.4; text = "b₋₁", align = (:center, :bottom), fontsize = 8, color = GREY)

    for j in 0:(n ÷ 2)-1
        x1 = (n - (2j + 2)) * cw           # left edge (b_{2j+1})
        x0 = (n - (2j - 1)) * cw           # right edge (b_{2j-1})
        yb = 1.25 - (j % 2) * 0.28
        shade = j % 2 == 0 ? (EXP_COLOR, 0.14) : (ACCENT, 0.12)
        poly!(ax, Rect2f(x1, 1.38, x0 - x1, 0.89); color = shade, strokewidth = 0)
        lines!(ax, [x1, x1, x0, x0], [yb, yb - 0.12, yb - 0.12, yb]; color = GREY, linewidth = 1.0)
        xm = (x0 + x1) / 2
        text!(ax, xm, yb - 0.20; text = "d$(_sub(j)) = $(d.digits[j+1] >= 0 ? "+" : "")$(d.digits[j+1])",
              align = (:center, :top), fontsize = 10, font = :bold,
              color = d.digits[j+1] == 0 ? GREY : ACCENT)
    end
    text!(ax, 0, -0.35;
          text = "Σⱼ dⱼ·4ʲ = " * join(["$(d.digits[j])·4$(_sup(j-1))" for j in length(d.digits):-1:1], " + ") *
                 " = $(value(d))  ✓",
          align = (:left, :top), fontsize = 10.5)
    text!(ax, 0, -0.95;
          text = "each digit selects a partial product from {0, ±A, ±2A} — all free shifts and negations",
          align = (:left, :top), fontsize = 9.5, color = GREY)
    xlims!(ax, -0.4, (n + 1) * cw + 0.4); ylims!(ax, -1.7, 2.9)
    fig
end

"""
    plot_rr4_addition(tr::RR4AddTrace; size=(900,420)) -> Figure

One carry-free addition, column by column, with every transfer drawn as an arrow that
hops exactly one position and **stops**.

The picture makes the timing claim visible as geometry: no wire spans more than one
position, every column computes without waiting for any other, and the critical path is
three column-local stages whether the adder is four digits wide or four thousand.
"""
function plot_rr4_addition(tr::RR4AddTrace; size = (960, 470))
    set_theme!(xpufp_theme())
    cols = tr.columns
    n = length(cols)
    hascarry = tr.carry_out != 0
    fig = Figure(; size = size)
    alab = tr.alphabet == MAX_REDUNDANT ? "maximal {-3..3}" : "minimal {-2..2}, one-column peek"
    ax = blank_axis(fig[1, 1]; aspect = nothing,
        title = "RR4 carry-free addition ($(alab)):  " *
                "$(value(tr.x)) + $(value(tr.y)) = $(value(tr.result))")
    cw = 1.0
    # rows, with a dedicated lane for the transfer arrows so they never sit in a cell
    rows = ("xᵢ", "yᵢ", "sᵢ = xᵢ + yᵢ", "t out", "", "wᵢ = sᵢ − 4·t out", "zᵢ = wᵢ + t in")
    ys   = (6.4,   5.5,   4.4,            3.3,     2.55, 1.5,               0.4)
    rowh = 0.72
    # the carry-out digit occupies one extra cell to the LEFT of the most significant
    xoff = hascarry ? cw : 0.0
    for (r, lab) in enumerate(rows)
        isempty(lab) && continue
        text!(ax, -0.35, ys[r] + rowh/2; text = lab, align = (:right, :center),
              fontsize = 10, color = GREY)
    end
    cellx(k) = xoff + (n - k) * cw          # k = 1 is position 0, drawn rightmost
    for (k, c) in enumerate(cols)
        xx = cellx(k)
        text!(ax, xx + cw/2, ys[1] + rowh + 0.18; text = "pos $(c.i)",
              align = (:center, :bottom), fontsize = 9, color = GREY)
        bitcell!(ax, xx, ys[1], c.x; fill = LIGHT_GREY, w = cw, h = rowh, fontsize = 10)
        bitcell!(ax, xx, ys[2], c.y; fill = LIGHT_GREY, w = cw, h = rowh, fontsize = 10)
        bitcell!(ax, xx, ys[3], c.s; fill = (FRAC_COLOR, 0.25), w = cw, h = rowh, fontsize = 10)
        bitcell!(ax, xx, ys[4], c.t_out == 0 ? "0" : (c.t_out > 0 ? "+1" : "−1");
                 fill = c.t_out == 0 ? LIGHT_GREY : (ACCENT, 0.35), w = cw, h = rowh, fontsize = 10)
        bitcell!(ax, xx, ys[6], c.w; fill = (EXP_COLOR, 0.22), w = cw, h = rowh, fontsize = 10)
        bitcell!(ax, xx, ys[7], c.z; fill = (ACCENT, 0.28), w = cw, h = rowh, fontsize = 11)
    end
    # transfer arrows live in their own lane: source centre → destination centre
    ylane = ys[5] + 0.12
    for (k, c) in enumerate(cols)
        c.t_out == 0 && continue
        xs = cellx(k) + cw/2
        xd = cellx(k) - cw/2                 # position i+1 sits one cell to the left
        lines!(ax, [xs, xs], [ys[4] + 0.02, ylane]; color = ACCENT, linewidth = 1.2)
        arrow!(ax, (xs, ylane), (xd + 0.10, ylane); color = ACCENT, linewidth = 1.5,
               headsize = 0.20)
        lines!(ax, [xd, xd], [ylane - 0.16, ylane + 0.16]; color = ACCENT, linewidth = 2.6)
        lines!(ax, [xd, xd], [ylane, ys[7] + rowh]; color = (ACCENT, 0.45),
               linewidth = 1.0, linestyle = :dot)
    end
    # the one-column peek, drawn under the s row
    for (k, c) in enumerate(cols)
        (c.peeked && k < n) || continue
        xs = cellx(k) + cw/2
        xr = cellx(k + 1) + cw/2
        lines!(ax, [xs, xr], [ys[3] - 0.14, ys[3] - 0.14]; color = ACCENT2,
               linewidth = 1.0, linestyle = :dot)
        text!(ax, (xs + xr)/2, ys[3] - 0.18; text = "peek", align = (:center, :top),
              fontsize = 7.5, color = ACCENT2)
    end
    if hascarry
        bitcell!(ax, 0.0, ys[7], tr.carry_out; fill = (ACCENT, 0.28), w = cw, h = rowh,
                 fontsize = 11)
        text!(ax, cw/2, ys[7] - 0.10; text = "carry-out", align = (:center, :top),
              fontsize = 8, color = GREY)
    end
    xlims!(ax, -3.4, n * cw + xoff + 0.4)
    ylims!(ax, -0.7, ys[1] + rowh + 0.9)
    Label(fig[2, 1],
          "every transfer travels exactly one position and stops (▌) — depth 3 at ANY word " *
          "length   •   value conserved: $(tr.exact ? "yes ✓" : "NO ✗")",
          fontsize = 10, color = ACCENT, tellwidth = false)
    fig
end

"""
    plot_absorption_table(; size=(560,420)) -> Figure

The absorption lemma by **total enumeration**: the final digit `w = (s − 4t(s)) + t_in`
for every one of the 39 possible (column sum, incoming transfer) pairs.

All entries lie in `[-3,3]` — legal digits, no case needing a second transfer — and the
extremes `|w| = 3` occur precisely at the rounding ties `s ∈ {-6,-2,2,6}` combined with
an aligned incoming transfer, confirming the half-step bound is tight.
"""
function plot_absorption_table(; size = (560, 420))
    set_theme!(xpufp_theme())
    A = absorption_table()
    fig = Figure(; size = size)
    ax = Axis(fig[1, 1]; xlabel = "incoming transfer t_in", ylabel = "column sum s",
        xticks = (1:3, ["−1", "0", "+1"]), yticks = (1:13, string.(-6:6)),
        title = "Absorption lemma: w = (s − 4t(s)) + t_in", yreversed = false)
    hm = heatmap!(ax, 1:3, 1:13, permutedims(A); colormap = :viridis)
    for i in 1:13, j in 1:3
        v = A[i, j]
        text!(ax, j, i; text = string(v), align = (:center, :center), fontsize = 11,
              color = abs(v) == 3 ? ACCENT : :white, font = abs(v) == 3 ? :bold : :regular)
    end
    Colorbar(fig[1, 2], hm; label = "final digit w")
    Label(fig[2, 1:2], "all |w| ≤ 3 ⇒ legal digits, no second transfer ever needed;\n" *
          "the four |w| = 3 extremes (orange) are exactly the rounding ties s ∈ {−6,−2,2,6}",
          fontsize = 10, color = GREY, tellwidth = false)
    fig
end

"""
    plot_minimal_maps(; size=(900,760)) -> Figure

The complete minimal-set addition rule as data, on four lookup maps.

(a) the column sum, a 5×5 addition table over the alphabet;
(b) the transfer `t_{i+1} = f(sᵢ, s_{i-1})`, taking only `{-1,0,1}` — **seven of the
nine rows are constant** (they read only `sᵢ`), and the two split rows at `sᵢ = ±2` are
the peek, flipping exactly where the neighbour's sign flips;
(c) the interim `wᵢ = sᵢ − 4t_{i+1}`, everywhere in `[-2,2]` — claim (i) as a picture;
(d) the landing `zᵢ = wᵢ + tᵢ`, everywhere in `[-2,2]`, with the two combinations the
sign-coupling lemma forbids left blank — **claim (ii) visible as two missing cells**.
"""
function plot_minimal_maps(; size = (900, 760))
    set_theme!(xpufp_theme())
    fig = Figure(; size = size)
    ds = -2:2; ss = -4:4

    ax1 = Axis(fig[1, 1]; title = "(a) column sum  sᵢ = xᵢ + yᵢ",
               xlabel = "yᵢ", ylabel = "xᵢ", xticks = (1:5, string.(ds)), yticks = (1:5, string.(ds)))
    S = [x + y for x in ds, y in ds]
    heatmap!(ax1, 1:5, 1:5, permutedims(S); colormap = :viridis)
    for i in 1:5, j in 1:5
        text!(ax1, j, i; text = string(S[i, j]), align = (:center, :center), fontsize = 10, color = :white)
    end

    ax2 = Axis(fig[1, 2]; title = "(b) transfer  t = f(sᵢ, s_{i−1})   ← the peek",
               xlabel = "s_{i−1}  (right neighbour)", ylabel = "sᵢ",
               xticks = (1:9, string.(ss)), yticks = (1:9, string.(ss)))
    T = [rr4_transfer_minimal(s, sr) for s in ss, sr in ss]
    heatmap!(ax2, 1:9, 1:9, permutedims(T); colormap = :viridis)
    for i in 1:9, j in 1:9
        borderline = abs(ss[i]) == 2
        text!(ax2, j, i; text = string(T[i, j]), align = (:center, :center), fontsize = 9,
              color = borderline ? ACCENT : :white, font = borderline ? :bold : :regular)
    end

    ax3 = Axis(fig[2, 1]; title = "(c) interim  wᵢ = sᵢ − 4t   — always in [−2,2]",
               xlabel = "s_{i−1}", ylabel = "sᵢ", xticks = (1:9, string.(ss)), yticks = (1:9, string.(ss)))
    W = [s - 4 * rr4_transfer_minimal(s, sr) for s in ss, sr in ss]
    heatmap!(ax3, 1:9, 1:9, permutedims(W); colormap = :viridis)
    for i in 1:9, j in 1:9
        text!(ax3, j, i; text = string(W[i, j]), align = (:center, :center), fontsize = 9, color = :white)
    end

    ax4 = Axis(fig[2, 2]; title = "(d) landing  zᵢ = wᵢ + t_in   — the ∅ cells are forbidden",
               xlabel = "incoming transfer t_in", ylabel = "interim wᵢ",
               xticks = (1:3, ["−1", "0", "+1"]), yticks = (1:5, string.(-2:2)))
    Z = fill(NaN, 5, 3)
    for (i, w) in enumerate(-2:2), (j, t) in enumerate(-1:1)
        (w == 2 && t == 1) && continue          # excluded by sign coupling
        (w == -2 && t == -1) && continue
        Z[i, j] = w + t
    end
    heatmap!(ax4, 1:3, 1:5, permutedims(Z); colormap = :viridis, nan_color = (GREY, 0.25))
    for i in 1:5, j in 1:3
        txt = isnan(Z[i, j]) ? "∅" : string(Int(Z[i, j]))
        text!(ax4, j, i; text = txt, align = (:center, :center), fontsize = 11,
              color = isnan(Z[i, j]) ? ACCENT : :white, font = isnan(Z[i, j]) ? :bold : :regular)
    end
    fig
end
