# ---------------------------------------------------------------------------
# A consistent visual language for every figure in the package.
#
# The field colours deliberately echo the source report's palette so that a bit
# always means the same thing: red for sign, green for exponent, blue for fraction,
# amber for the integer part of a fixed-point word.
# ---------------------------------------------------------------------------

"Colour of the sign bit in every bit-layout figure."
const SIGN_COLOR = RGBf(214/255, 84/255, 84/255)
"Colour of the exponent field."
const EXP_COLOR = RGBf(86/255, 156/255, 104/255)
"Colour of the fraction/mantissa field."
const FRAC_COLOR = RGBf(92/255, 128/255, 188/255)
"Colour of the integer part of a fixed-point word."
const INT_COLOR = RGBf(230/255, 168/255, 84/255)
"Neutral grey for structure, rules and de-emphasised elements."
const GREY = RGBf(0.45, 0.45, 0.45)
"Light grey fill."
const LIGHT_GREY = RGBf(0.94, 0.94, 0.94)
"Accent colour for the thing the figure is actually about."
const ACCENT = RGBf(0.85, 0.37, 0.15)
"Light sky blue — used for the scale/exponent cell, deliberately off the digit colormap."
const LIGHTSKYBLUE = RGBf(135/255, 206/255, 250/255)

"""
    text_on(c) -> Symbol

Pick `:white` or `:black` for text drawn on the background colour `c`, by relative
luminance.  Necessary on a perceptual colormap like viridis, where the dark and light
ends both occur inside one figure."""
function text_on(c)
    rc = convert(RGBf, c)
    r, g, b = Float64(rc.r), Float64(rc.g), Float64(rc.b)
    lum = 0.2126r + 0.7152g + 0.0722b
    lum < 0.55 ? :white : :black
end

"Secondary accent, for the comparison series."
const ACCENT2 = RGBf(0.20, 0.45, 0.70)

"""    FORMAT_COLORS

A stable colour per named format, so a format keeps its identity across figures."""
const FORMAT_COLORS = Dict(
    "FP32"   => RGBf(0.16, 0.32, 0.55),
    "FP16"   => RGBf(0.25, 0.55, 0.75),
    "BF16"   => RGBf(0.45, 0.70, 0.85),
    "E5M2"   => RGBf(0.55, 0.45, 0.75),
    "E4M3"   => RGBf(0.70, 0.45, 0.65),
    "E2M1"   => RGBf(0.85, 0.37, 0.15),
    "E1M2"   => RGBf(0.90, 0.60, 0.25),
    "E3M0"   => RGBf(0.70, 0.30, 0.20),
    "E8M0"   => RGBf(0.40, 0.60, 0.35),
    "MXFP4"  => RGBf(0.85, 0.37, 0.15),
    "NVFP4"  => RGBf(0.20, 0.55, 0.35),
    "MXINT4" => RGBf(0.55, 0.55, 0.55),
    "INT4"   => RGBf(0.55, 0.55, 0.55),
)

"""
    format_color(f) -> RGBf

The stable colour for a format, falling back to grey for anything unregistered."""
format_color(f) = get(FORMAT_COLORS, _fmtname(f), GREY)
_fmtname(f::FloatFormat) = f.name
_fmtname(f::IntFormat) = f.name
_fmtname(f::FixedFormat) = f.name
_fmtname(f::BlockFormat) = f.name
_fmtname(s::AbstractString) = String(s)

"""
    xpufp_theme() -> Theme

The package's Makie theme: restrained, print-oriented, with a light background and
thin rules.  Applied automatically by every `plot_*` function; call
`set_theme!(xpufp_theme())` to use it for your own figures too."""
function xpufp_theme()
    Theme(
        fontsize = 13,
        figure_padding = 14,
        backgroundcolor = :white,
        Axis = (
            backgroundcolor = :white,
            xgridvisible = true, ygridvisible = true,
            xgridcolor = (:black, 0.06), ygridcolor = (:black, 0.06),
            xgridwidth = 0.8, ygridwidth = 0.8,
            topspinevisible = false, rightspinevisible = false,
            leftspinecolor = GREY, bottomspinecolor = GREY,
            xtickcolor = GREY, ytickcolor = GREY,
            titlesize = 15, titlealign = :left, titlefont = :bold,
            xlabelsize = 12, ylabelsize = 12,
        ),
        Legend = (framevisible = false, patchsize = (18, 12), labelsize = 11),
        Colorbar = (ticklabelsize = 10, labelsize = 11),
    )
end

# ---- diagram primitives ----------------------------------------------------

"""
    blank_axis(fig_or_pos; title="", aspect=DataAspect()) -> Axis

An axis with every decoration removed — the canvas for schematic figures (bit
layouts, dataflow diagrams, factor graphs) where the coordinates are drawing
positions, not data."""
function blank_axis(pos; title::AbstractString = "", aspect = DataAspect())
    ax = aspect === nothing ? Axis(pos; title = title) : Axis(pos; title = title, aspect = aspect)
    hidedecorations!(ax)
    hidespines!(ax)
    ax
end

"""
    bitcell!(ax, x, y, label; fill=LIGHT_GREY, w=1.0, h=1.0, fontsize=11, textcolor=:black)

Draw one labelled bit cell with its origin at `(x, y)` — the building block of every
bit-layout figure."""
function bitcell!(ax, x::Real, y::Real, label;
                  fill = LIGHT_GREY, w::Real = 1.0, h::Real = 1.0,
                  fontsize::Real = 11, textcolor = :black, strokecolor = (:black, 0.55))
    poly!(ax, Rect2f(x, y, w, h); color = fill, strokecolor = strokecolor, strokewidth = 0.8)
    text!(ax, x + w/2, y + h/2; text = string(label), align = (:center, :center),
          fontsize = fontsize, font = :regular, color = textcolor)
    nothing
end

"""
    bitrun!(ax, x, y, bits, fill; kwargs...) -> Float64

Draw a run of bit cells left to right, returning the x-coordinate just past the run."""
function bitrun!(ax, x::Real, y::Real, bits, fill; w::Real = 1.0, kwargs...)
    cx = Float64(x)
    for b in bits
        bitcell!(ax, cx, y, b; fill = fill, w = w, kwargs...)
        cx += w
    end
    cx
end

"""
    brace!(ax, x0, x1, y, label; below=true, depth=0.28, fontsize=10)

A horizontal brace spanning `[x0, x1]` with a centred label — used to name the
sign / exponent / fraction fields under a bit layout."""
function brace!(ax, x0::Real, x1::Real, y::Real, label;
                below::Bool = true, depth::Real = 0.28, fontsize::Real = 10,
                color = GREY)
    s = below ? -1 : 1
    xm = (x0 + x1) / 2
    pts = Point2f[(x0, y), (x0, y + s*depth), (xm, y + s*depth),
                  (xm, y + s*depth*1.5), (xm, y + s*depth), (x1, y + s*depth), (x1, y)]
    lines!(ax, pts; color = color, linewidth = 1.0)
    text!(ax, xm, y + s*depth*1.5 + s*0.12; text = string(label),
          align = (:center, below ? :top : :bottom), fontsize = fontsize, color = color)
    nothing
end

"""
    labelbox!(ax, x, y, w, h, text; fill, strokecolor, fontsize, rounded)

A labelled rounded box — the node primitive for dataflow and pipeline diagrams."""
function labelbox!(ax, x::Real, y::Real, w::Real, h::Real, label;
                   fill = :white, strokecolor = GREY, fontsize::Real = 10,
                   textcolor = :black, strokewidth = 1.0)
    poly!(ax, Rect2f(x, y, w, h); color = fill, strokecolor = strokecolor,
          strokewidth = strokewidth)
    text!(ax, x + w/2, y + h/2; text = string(label), align = (:center, :center),
          fontsize = fontsize, color = textcolor, justification = :center)
    nothing
end

"""
    arrow!(ax, p0, p1; color=GREY, linewidth=1.0, headsize=0.10, dashed=false)

A line from `p0` to `p1` with a filled triangular head at `p1`.  Makie's `arrows!`
scales its heads in data units, which fights schematic layouts; this draws the head
as an explicit polygon so it stays the size you asked for."""
function arrow!(ax, p0, p1; color = GREY, linewidth::Real = 1.0,
                headsize::Real = 0.10, dashed::Bool = false)
    x0, y0 = Float64(p0[1]), Float64(p0[2])
    x1, y1 = Float64(p1[1]), Float64(p1[2])
    dx, dy = x1 - x0, y1 - y0
    L = hypot(dx, dy)
    L == 0 && return nothing
    ux, uy = dx / L, dy / L
    bx, by = x1 - ux * headsize, y1 - uy * headsize
    lines!(ax, [x0, bx], [y0, by]; color = color, linewidth = linewidth,
           linestyle = dashed ? :dash : :solid)
    px, py = -uy, ux
    hw = headsize * 0.42
    poly!(ax, Point2f[(x1, y1), (bx + px*hw, by + py*hw), (bx - px*hw, by - py*hw)];
          color = color, strokewidth = 0)
    nothing
end

"""
    savefig(fig, path; px_per_unit=2) -> String

Save a figure, creating the directory if needed.  Returns the path.  PNG gets a 2×
pixel density by default so figures stay crisp in the docs."""
function savefig(fig, path::AbstractString; px_per_unit::Real = 2)
    mkpath(dirname(abspath(path)))
    if endswith(lowercase(path), ".png")
        save(path, fig; px_per_unit = px_per_unit)
    else
        save(path, fig)
    end
    path
end

function _fmtnum(v::Real)
    x = Float64(v)
    isfinite(x) || return string(x)
    # `Int(x)` overflows for values like 2.03e31 that are integers mathematically
    (isinteger(x) && abs(x) < 1e15) && return string(Int(x))
    (abs(x) >= 1e5 || (x != 0 && abs(x) < 1e-4)) && return string(round(x, sigdigits = 4))
    abs(x) >= 0.01 ? string(round(x, digits = 4)) : string(round(x, sigdigits = 3))
end
