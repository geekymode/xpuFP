# ---------------------------------------------------------------------------
# One layout figure for every format family.
#
# Whatever the container — packed float fields, a fixed-point word, an integer, or a
# redundant digit string — the picture answers the same three questions: what is
# stored, what weight does each cell carry, and how does that reconstruct the value.
# ---------------------------------------------------------------------------

"""
    plot_digit_layout(fmt; value, size=(980,300), maxcells=48) -> Figure

Draw how `fmt` stores `value`, with every cell's positional weight and the decoded
reconstruction underneath.

Works across **all** format families:

| Format | What is drawn |
|:---|:---|
| [`DigitFormat`](@ref) (`RR4`, `CSD`, `BINARY`, …) | signed digit cells, radix point, weights `rᵏ` |
| [`FloatFormat`](@ref) | sign / exponent / fraction fields, with the decode |
| [`FixedFormat`](@ref) | sign / integer / fraction cells with power-of-two weights |
| [`IntFormat`](@ref) | two's-complement bit cells |

```julia
plot_digit_layout(RR4;  value = 16.1656375)
plot_digit_layout(CSD;  value = 231)
plot_digit_layout(FP32; value = -6.375)
plot_digit_layout(FixedFormat(7, 8); value = -3.65)
```

Long strings are truncated to `maxcells` cells from the most significant end, with the
omission marked — `16.1656375` in RR4 is exact but runs to ~27 digits, because the
`Float64` it came from is a dyadic rational and *that* is the number being represented.
"""
function plot_digit_layout end

function plot_digit_layout(f::DigitFormat; value, size = (980, 300),
                           maxcells::Integer = 48, kwargs...)
    set_theme!(xpufp_theme())
    sd = to_digits(f, value; kwargs...)
    v = xpuFP.value(sd)
    fig = Figure(; size = size)
    exact = v == (value isa Integer ? value : Rational{BigInt}(value))
    ax = blank_axis(fig[1, 1]; aspect = nothing,
        title = "$(f.name):  $(value)   →   $(digit_string(sd))")
    n = length(sd.digits)
    shown = min(n, Int(maxcells))
    trunc = n > shown
    cw = 1.0
    nf = nfracdigits(sd)
    # digits are drawn most significant first, left to right
    for k in 1:shown
        idx = n - k + 1                      # index into sd.digits (LSB-first)
        d = sd.digits[idx]
        x = (k - 1) * cw
        col = d == 0 ? LIGHT_GREY : (d < 0 ? (SIGN_COLOR, 0.40) : (FRAC_COLOR, 0.30))
        bitcell!(ax, x, 0, d == 0 ? "0" : (d < 0 ? "$(-d)̄" : string(d));
                 fill = col, w = cw, h = 0.9, fontsize = 12)
        p = sd.exponent + idx - 1
        text!(ax, x + cw/2, 1.02; text = "$(f.radix)$(_sup(p))",
              align = (:center, :bottom), fontsize = 8, color = GREY)
        # the radix point sits just left of the first fractional digit
        if nf > 0 && idx == nf + 1 && k < shown
            lines!(ax, [x + cw, x + cw], [-0.10, 1.0]; color = :black, linewidth = 2.0)
        end
    end
    if trunc
        text!(ax, shown * cw + 0.15, 0.45; text = "…  ($(n - shown) more digits)",
              align = (:left, :center), fontsize = 10, color = ACCENT)
    end
    terms = String[]
    for k in 1:min(n, 6)
        idx = n - k + 1
        d = sd.digits[idx]
        d == 0 && continue
        p = sd.exponent + idx - 1
        push!(terms, "$(d > 0 ? "+" : "−")$(abs(d))·$(f.radix)$(_sup(p))")
    end
    expr = lstrip(join(terms, " "), ['+', ' '])
    count(!=(0), sd.digits) > 6 && (expr *= " …")
    lines = [
        "alphabet {$(join(alphabet(f), ","))}   " *
        (is_redundant(f) ? "redundant, ρ = $(redundancy(f)) ⇒ many spellings" :
                           "non-redundant ⇒ unique"),
        "value = $(expr)",
        "      = $(v)" * (exact ? "   (exact)" : "   (rounded)"),
    ]
    for (i, l) in enumerate(lines)
        text!(ax, 0, -0.35 - 0.42 * (i - 1); text = l, align = (:left, :top),
              fontsize = 10.5, color = i == 1 ? GREY : :black)
    end
    xlims!(ax, -0.4, max(shown * cw, 12) + 3.0); ylims!(ax, -2.0, 1.7)
    fig
end

function plot_digit_layout(f::FloatFormat; value, size = (980, 300), kwargs...)
    plot_bit_layout(f; value = value, size = size)
end

function plot_digit_layout(f::FixedFormat; value, size = (980, 300), kwargs...)
    plot_bit_layout(f; value = value, size = size)
end

function plot_digit_layout(f::IntFormat; value, size = (980, 300), kwargs...)
    plot_bit_layout(f; value = value, size = size)
end

"""
    plot_layout_comparison(formats, value; size=(980,220*length(formats))) -> Figure

Stack one value's layout in several formats, so the representations can be read against
each other.

```julia
plot_layout_comparison((BINARY, CSD, RADIX4, RR4, RR4_MAX), 231)
```
"""
function plot_layout_comparison(formats, value; size = nothing, maxcells::Integer = 26)
    set_theme!(xpufp_theme())
    n = length(formats)
    sz = size === nothing ? (980, 90 * n + 90) : size
    fig = Figure(; size = sz)
    ax = blank_axis(fig[1, 1]; aspect = nothing,
        title = "$(value) in $(n) representations")
    cw = 1.0
    rowh = 0.78
    maxw = 0.0
    for (r, f) in enumerate(formats)
        y = (n - r) * 1.15
        sd = try
            to_digits(f, value)
        catch
            continue
        end
        m = length(sd.digits)
        shown = min(m, Int(maxcells))
        nf = nfracdigits(sd)
        text!(ax, -0.35, y + rowh/2; text = f.name, align = (:right, :center),
              fontsize = 11, font = :bold)
        for k in 1:shown
            idx = m - k + 1
            d = sd.digits[idx]
            x = (k - 1) * cw
            col = d == 0 ? LIGHT_GREY : (d < 0 ? (SIGN_COLOR, 0.40) : (FRAC_COLOR, 0.30))
            bitcell!(ax, x, y, d == 0 ? "0" : (d < 0 ? "$(-d)̄" : string(d));
                     fill = col, w = cw, h = rowh, fontsize = 10)
            if nf > 0 && idx == nf + 1 && k < shown
                lines!(ax, [x + cw, x + cw], [y - 0.06, y + rowh + 0.06];
                       color = :black, linewidth = 1.8)
            end
        end
        maxw = max(maxw, shown * cw)
        w = count(!=(0), sd.digits)
        text!(ax, shown * cw + 0.4, y + rowh/2;
              text = "$(m) digits, $(w) nonzero" * (m > shown ? "  (truncated)" : ""),
              align = (:left, :center), fontsize = 9, color = GREY)
    end
    xlims!(ax, -6.0, maxw + 12.0); ylims!(ax, -0.6, n * 1.15 + 0.3)
    fig
end

"""
    plot_rr4_representations(r::RR4Representations; kwargs...) -> Figure
    plot_rr4_representations(x; method=:minimally_redundant, ndigits=nothing, kwargs...) -> Figure

Draw every spelling of a value as a labelled colour map: one **row per representation**,
one column per digit position (most significant on the left, so a row reads like the
number), and a final, visually separated cell holding the **scaling exponent**.

The digit cells use a diverging colormap centred on zero, because the digits are signed
and zero is the meaningful middle. The scale cell is deliberately *not* on that
colormap — it is an exponent, not a digit, and sharing a scale would imply a comparison
that does not exist.

# Keywords
- `method` — `:minimally_redundant` (default) or `:maximally_redundant`.
- `ndigits` — width; defaults to the shortest the alphabet allows.
- `highlight` — outline the minimum-weight row (default `true`).
- `colormap` — defaults to `:viridis`.
- `colorbar` — off by default; the digits are printed in their own cells, so a legend
  would spend width restating them. Pass `true` if you want one anyway.
- `size`, `maxrows`.

```julia
plot_rr4_representations(12.5)
plot_rr4_representations(25; method = :maximally_redundant)
plot_rr4_representations(49; ndigits = 5)
```
"""
function plot_rr4_representations end

function plot_rr4_representations(r::RR4Representations;
                                  highlight::Bool = true, colormap = :viridis,
                                  maxrows::Integer = 40, size = nothing,
                                  colorbar::Bool = false)
    set_theme!(xpufp_theme())
    nrep = length(r.reps)
    nrep == 0 && throw(ArgumentError(
        "no spelling of $(r.value) fits in $(r.ndigits) digits; widen with ndigits="))
    shown = min(nrep, Int(maxrows))
    n = r.ndigits
    # no colorbar by default: with at most seven digit values, all of them printed in
    # their own cells, a legend would spend width to restate what the labels already say
    # Keep the title short so it never outruns a narrow figure; the format details go
    # in the caption, which spans the full width and has room.
    ttl = "$(r.value) in RR4 — $(r.total) spelling$(r.total == 1 ? "" : "s")"
    sz = size === nothing ?
         (max(390, 130 + 50 * (n + 1), 10 * length(ttl)) + (colorbar ? 95 : 0),
          150 + 32 * shown) : size
    fig = Figure(; size = sz)
    ax = blank_axis(fig[1, 1]; aspect = nothing)
    ax.title = ttl
    ax.titlefont = :regular

    cw, ch = 1.0, 1.0
    gap = 0.5                                   # visual break before the scale column
    lim = max(r.maxdigit, 1)
    grad = cgrad(colormap)
    nf = max(0, -r.scale)
    pointat = n - nf
    digitcolor(d) = grad[clamp((d + lim) / (2lim), 0.0, 1.0)]

    for i in 1:shown
        y = (shown - i) * ch
        text!(ax, -0.30, y + ch/2; text = string(i), align = (:right, :center),
              fontsize = 9, color = GREY)
        for k in 1:n
            d = r.digits[i, k]
            x = (k - 1) * cw
            col = digitcolor(d)
            poly!(ax, Rect2f(x, y, cw, ch); color = col,
                  strokecolor = (:white, 0.85), strokewidth = 1.0)
            text!(ax, x + cw/2, y + ch/2; text = string(d), align = (:center, :center),
                  fontsize = 12, color = text_on(col))
        end
        xs = n * cw + gap
        poly!(ax, Rect2f(xs, y, cw, ch); color = LIGHTSKYBLUE,
              strokecolor = (:white, 0.85), strokewidth = 1.0)
        text!(ax, xs + cw/2, y + ch/2; text = string(r.scale), align = (:center, :center),
              fontsize = 12, color = :black)
        text!(ax, xs + cw + 0.22, y + ch/2; text = string(weight(r.reps[i])),
              align = (:left, :center), fontsize = 9, color = GREY)
    end

    if nf > 0 && 0 < pointat < n
        # the radix point: a rule between columns, extended past the grid so it is not
        # mistaken for a cell border
        xp = pointat * cw
        lines!(ax, [xp, xp], [-0.16, shown * ch + 0.16]; color = :black, linewidth = 2.6)
        text!(ax, xp, -0.20; text = "radix point", align = (:center, :top),
              fontsize = 8, color = GREY)
    end

    nbest = 0
    if highlight
        # outline EVERY minimum-weight row: ties are common, and singling out the first
        # would imply a distinction the arithmetic does not make
        wmin = minimum(weight, r.reps)
        for i in 1:shown
            weight(r.reps[i]) == wmin || continue
            nbest += 1
            y = (shown - i) * ch
            poly!(ax, Rect2f(-0.05, y - 0.05, n * cw + 0.10, ch + 0.10);
                  color = (:white, 0.0), strokecolor = ACCENT, strokewidth = 2.0)
        end
    end

    for k in 1:n
        text!(ax, (k - 1) * cw + cw/2, shown * ch + 0.14;
              text = "4$(_sup(r.scale + n - k))", align = (:center, :bottom),
              fontsize = 8.5, color = GREY)
    end
    text!(ax, n * cw + gap + cw/2, shown * ch + 0.14; text = "scale",
          align = (:center, :bottom), fontsize = 8.5, color = GREY)
    text!(ax, n * cw + gap + cw + 0.22, shown * ch + 0.14; text = "nz",
          align = (:left, :bottom), fontsize = 8.5, color = GREY)

    xlims!(ax, -1.15, n * cw + gap + cw + 1.05)
    ylims!(ax, nf > 0 && 0 < pointat < n ? -0.62 : -0.35, shown * ch + 0.9)

    if colorbar
        Colorbar(fig[1, 2]; colormap = colormap, limits = (-lim, lim),
                 ticks = (collect(-lim:lim), string.(collect(-lim:lim))),
                 label = "digit value", width = 11, height = Relative(0.72),
                 ticklabelsize = 9, labelsize = 10)
        colsize!(fig.layout, 2, Auto(false))
    end

    # captions go in the layout, not in axis coordinates, so they can never be clipped
    spec = "$(r.method),  alphabet {-$(r.maxdigit)…$(r.maxdigit)},  $(n) digits,  " *
           "scale exponent $(r.scale)"
    notes = String["each row is one spelling, all equal to $(r.value);  " *
                   "the final cell is the scaling exponent, not a digit"]
    r.truncated && push!(notes, "$(r.total) spellings exist — $(nrep) listed")
    nrep > shown && push!(notes, "$(nrep - shown) further rows not drawn")
    Label(fig[2, 1:(colorbar ? 2 : 1)], spec;
          fontsize = 9, color = GREY, halign = :left, tellwidth = false)
    Label(fig[3, 1:(colorbar ? 2 : 1)], join(notes, "   •   ");
          fontsize = 9, color = GREY, halign = :left, tellwidth = false)
    if highlight
        Label(fig[4, 1:(colorbar ? 2 : 1)],
              nbest == nrep ? "orange outline = fewest nonzero digits — all spellings tie here" :
                              "orange outline = fewest nonzero digits ($(nbest) of $(nrep))";
              fontsize = 9, color = ACCENT, halign = :left, tellwidth = false)
    end
    rowgap!(fig.layout, 4)
    fig
end

plot_rr4_representations(x; method = :minimally_redundant, ndigits = nothing,
                         fracdigits = nothing, exponent = nothing, kwargs...) =
    plot_rr4_representations(
        rr4_representations(x; method, ndigits, fracdigits, exponent); kwargs...)

"""
    plot_weight_scaling(widths = [8, 16, 32, 64]; samples=200, rng=Random.default_rng(),
                        size=(880, 400)) -> Figure

How the **minimal-weight density** — nonzero digits per digit position — scales with
word length, for both radix-4 alphabets, with the one-pass conversion for comparison.

The measured limits are marked: ``2/3`` for the minimally redundant alphabet and
``3/5`` for the maximally redundant one. More redundancy buys a lower density, as it
must — the alphabets are nested, so the minimum over the larger set cannot exceed the
minimum over the smaller.

Values are sampled from the range each alphabet can actually represent at that width;
sampling beyond it would silently mix in values that do not fit.
"""
function plot_weight_scaling(widths = [8, 16, 32, 64]; samples::Integer = 200,
                             rng::AbstractRNG = Random.default_rng(), size = (880, 400))
    set_theme!(xpufp_theme())
    fig = Figure(; size = size)
    ax = Axis(fig[1, 1]; xscale = log2, xticks = (collect(widths), string.(widths)),
        xlabel = "word length (radix-4 digits)", ylabel = "nonzero digits per position",
        title = "Minimal-weight density: how cheap can a constant multiplier get?")
    for (m, col, lab) in ((:minimally_redundant, ACCENT, "{-2..2} minimally redundant"),
                          (:maximally_redundant, ACCENT2, "{-3..3} maximally redundant"))
        rows = weight_scaling(widths; method = m, samples, rng)
        scatterlines!(ax, collect(widths), [r.density for r in rows];
                      color = col, markersize = 9, label = "$(lab) — minimal")
        scatterlines!(ax, collect(widths), [r.canonical_density for r in rows];
                      color = (col, 0.45), markersize = 7, linestyle = :dash,
                      label = "$(lab) — one-pass")
    end
    hlines!(ax, [2/3]; color = (ACCENT, 0.55), linestyle = :dot)
    hlines!(ax, [3/5]; color = (ACCENT2, 0.55), linestyle = :dot)
    text!(ax, last(widths), 2/3; text = " 2/3", align = (:left, :bottom),
          fontsize = 9, color = ACCENT)
    text!(ax, last(widths), 3/5; text = " 3/5", align = (:left, :top),
          fontsize = 9, color = ACCENT2)
    ylims!(ax, 0.5, 0.85)
    axislegend(ax; position = :rt, labelsize = 9)
    Label(fig[2, 1],
          "dotted lines are the measured limits; searching for the minimum saves " *
          "≈10% of the nonzero digits over the one-pass conversion",
          fontsize = 9, color = GREY, halign = :left, tellwidth = false)
    fig
end
