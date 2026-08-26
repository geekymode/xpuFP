# ---------------------------------------------------------------------------
# Bit-layout figures: the packed word, drawn.
# ---------------------------------------------------------------------------

"""
    plot_bit_layout(f::FloatFormat; value=nothing, size=(900,260)) -> Figure

Draw a format's bit layout, every field coloured and braced.  With `value` supplied,
the cells carry that value's actual bits and the caption decodes it.

Each field is read with ordinary binary place value: the exponent bits form a plain
unsigned integer, and the fraction bits continue binary place value past the point, so
together with the hidden leading 1 they form `1.f₁f₂…`.

```julia
fig = plot_bit_layout(FP32; value = -6.375)      # the report's 0xC0CC0000
savefig(fig, "fp32_layout.png")
```
"""
function plot_bit_layout(f::FloatFormat; value = nothing, size = (900, 260))
    set_theme!(xpufp_theme())
    n = nbits(f)
    w = min(0.9, 26 / n)
    fig = Figure(; size = size)
    ttl = value === nothing ? "$(f.name) bit layout" :
          "$(f.name):  $(value)  =  0x$(uppercase(string(encode(f, value), base=16, pad=cld(n,4))))"
    ax = blank_axis(fig[1, 1]; title = ttl)

    bits = if value === nothing
        nothing
    else
        c = encode(f, value)
        [Int((c >> (n - 1 - i)) & 1) for i in 0:n-1]
    end

    x = 0.0
    if f.signed
        lab = bits === nothing ? "s" : string(bits[1])
        bitcell!(ax, x, 0, lab; fill = (SIGN_COLOR, 0.35), w = w)
        brace!(ax, x, x + w, -0.06, "sign"; depth = 0.18)
        x += w
    end
    xe0 = x
    for i in 1:f.ebits
        lab = bits === nothing ? "e$(_sub(f.ebits - i))" : string(bits[(f.signed ? 1 : 0) + i])
        bitcell!(ax, x, 0, lab; fill = (EXP_COLOR, 0.30), w = w, fontsize = w > 0.6 ? 11 : 8)
        x += w
    end
    brace!(ax, xe0, x, -0.06, "biased exponent  E,  bias $(f.bias)"; depth = 0.18)
    xf0 = x
    for i in 1:f.mbits
        lab = bits === nothing ? "f$(_sub(i))" : string(bits[(f.signed ? 1 : 0) + f.ebits + i])
        bitcell!(ax, x, 0, lab; fill = (FRAC_COLOR, 0.28), w = w, fontsize = w > 0.6 ? 11 : 8)
        x += w
    end
    f.mbits > 0 && brace!(ax, xf0, x, -0.06,
        "fraction  F   →  significand 1.f₁f₂… = 1 + F/2^$(f.mbits)   (hidden 1 not stored)";
        depth = 0.18)

    # bit indices above
    text!(ax, w/2, 1.12; text = string(n - 1), align = (:center, :bottom), fontsize = 9, color = GREY)
    text!(ax, x - w/2, 1.12; text = "0", align = (:center, :bottom), fontsize = 9, color = GREY)

    cap = if value === nothing
        "x = (-1)ˢ · 2^(E−$(f.bias)) · (1 + F/2^$(f.mbits))      normals: E = " *
        "$(f.zero_exp == SUBNORMAL_ZERO ? 1 : 0)…$(max_normal_E(f))"
    else
        p = unpack(f, value)
        "s = $(p.sign),  E = $(p.E)  ⇒ e = $(p.e),  F = $(p.M)   " *
        "⇒  (-1)^$(p.sign) · 2^$(p.e) · $(round(p.significand, digits=8)) = $(p.value)"
    end
    text!(ax, 0, -0.72; text = cap, align = (:left, :top), fontsize = 10.5, color = :black)

    xlims!(ax, -0.25, x + 0.25); ylims!(ax, -1.15, 1.45)
    fig
end

_sub(n::Integer) = join(['₀'+d for d in reverse(digits(n))])

"""
    plot_format_zoo(; formats=(FP32,FP16,BF16,E5M2,E4M3,E2M1), size=(900,420)) -> Figure

The IEEE-style family side by side with field widths **drawn to scale**, so the
trade-offs are visible as geometry.

The headline comparison: bfloat16 is simply FP32 with the bottom 16 fraction bits
chopped off — same exponent, same range, far less precision.
"""
function plot_format_zoo(; formats = (FP32, FP16, BF16, E5M2, E4M3, E2M1),
                         size = (900, 420))
    set_theme!(xpufp_theme())
    fig = Figure(; size = size)
    ax = blank_axis(fig[1, 1]; title = "The IEEE-style format family, field widths to scale")
    maxn = maximum(nbits(f) for f in formats)
    unit = 1.0
    for (row, f) in enumerate(reverse(collect(formats)))
        y = (row - 1) * 1.6
        x = 0.0
        if f.signed
            poly!(ax, Rect2f(x, y, unit, 0.9); color = (SIGN_COLOR, 0.45),
                  strokecolor = (:black, 0.4), strokewidth = 0.6); x += unit
        end
        poly!(ax, Rect2f(x, y, f.ebits * unit, 0.9); color = (EXP_COLOR, 0.40),
              strokecolor = (:black, 0.4), strokewidth = 0.6)
        f.ebits >= 2 && text!(ax, x + f.ebits*unit/2, y + 0.45; text = "$(f.ebits)e",
              align = (:center, :center), fontsize = 10)
        x += f.ebits * unit
        if f.mbits > 0
            poly!(ax, Rect2f(x, y, f.mbits * unit, 0.9); color = (FRAC_COLOR, 0.38),
                  strokecolor = (:black, 0.4), strokewidth = 0.6)
            f.mbits >= 2 && text!(ax, x + f.mbits*unit/2, y + 0.45; text = "$(f.mbits)m",
                  align = (:center, :center), fontsize = 10)
            x += f.mbits * unit
        end
        text!(ax, -0.6, y + 0.45; text = f.name, align = (:right, :center),
              fontsize = 12, font = :bold)
        text!(ax, maxn * unit + 1.0, y + 0.45;
              text = @sprintf("%2d bits   max %.3g   ε = 2^-%d", nbits(f), maxfinite(f), f.mbits),
              align = (:left, :center), fontsize = 9.5, color = GREY)
    end
    xlims!(ax, -7, maxn * unit + 12)
    ylims!(ax, -0.8, length(formats) * 1.6)
    fig
end

"""
    plot_fp4_codes(f::FloatFormat = E2M1; size=(760,360)) -> Figure

Every code of a 4-bit format, laid out as a table: bit pattern, fields, and the value
it decodes to.

For E2M1 this is half the entire number system — setting the sign bit mirrors each row.
The reserved-exponent machinery of FP32 shrinks to a single rule (`E = 0` drops the
hidden bit) and the special values vanish entirely: the top code is an ordinary number,
6, and results beyond it clamp rather than overflowing to infinity.
"""
function plot_fp4_codes(f::FloatFormat = E2M1; size = (760, 360))
    set_theme!(xpufp_theme())
    fig = Figure(; size = size)
    ax = blank_axis(fig[1, 1];
        title = "Every positive $(f.name) code — half the entire number system")
    ncodes = 1 << (f.ebits + f.mbits)
    rowh = 1.0
    text!(ax, 0.6, ncodes * rowh + 0.55; text = "s", align = (:center, :bottom), fontsize = 10, color = GREY)
    text!(ax, 1.0 + f.ebits/2, ncodes * rowh + 0.55; text = "E", align = (:center, :bottom), fontsize = 10, color = GREY)
    text!(ax, 1.0 + f.ebits + f.mbits/2, ncodes * rowh + 0.55; text = "M", align = (:center, :bottom), fontsize = 10, color = GREY)
    text!(ax, 5.6, ncodes * rowh + 0.55; text = "kind", align = (:left, :bottom), fontsize = 10, color = GREY)
    text!(ax, 8.4, ncodes * rowh + 0.55; text = "value", align = (:right, :bottom), fontsize = 10, color = GREY)

    for c in 0:ncodes-1
        y = (ncodes - 1 - c) * rowh
        E = (c >> f.mbits) & ((1 << f.ebits) - 1)
        M = c & ((1 << f.mbits) - 1)
        v = decode(f, c)
        bitcell!(ax, 0.2, y, "0"; fill = (SIGN_COLOR, 0.30), w = 0.8, h = 0.85)
        x = 1.0
        for i in f.ebits-1:-1:0
            bitcell!(ax, x, y, (E >> i) & 1; fill = (EXP_COLOR, 0.28), w = 0.8, h = 0.85)
            x += 0.8
        end
        for i in f.mbits-1:-1:0
            bitcell!(ax, x, y, (M >> i) & 1; fill = (FRAC_COLOR, 0.26), w = 0.8, h = 0.85)
            x += 0.8
        end
        kind = E == 0 ? (M == 0 ? "zero" : "subnormal (hidden bit 0)") : "normal"
        text!(ax, 5.6, y + 0.42; text = kind, align = (:left, :center), fontsize = 9.5, color = GREY)
        text!(ax, 8.4, y + 0.42; text = _fmtnum(v), align = (:right, :center),
              fontsize = 11, font = :bold)
    end
    text!(ax, 0.2, -0.9;
          text = "no ±∞, no NaN — all $(1<<nbits(f)) codes are finite; overflow saturates at ±$(maxfinite(f))",
          align = (:left, :top), fontsize = 10, color = ACCENT)
    xlims!(ax, -0.2, 9.2); ylims!(ax, -1.6, ncodes * rowh + 1.3)
    fig
end

"""
    plot_mx_block_layout(qb::QuantizedBlock; size=(950,330)) -> Figure

An MX block as stored, with the quantization made explicit cell by cell: the real
value, divided by the shared scale, rounded onto the element grid, and only then the
stored code — alongside the one scale byte.

Multiplying or dividing by `S` is an exponent add, never a real multiplication.
"""
function plot_mx_block_layout(qb::QuantizedBlock; size = (950, 330))
    set_theme!(xpufp_theme())
    bf = qb.fmt
    K = length(qb.original)
    fig = Figure(; size = size)
    ax = blank_axis(fig[1, 1];
        title = "$(bf.name) block as stored — $(bits_per_block(bf)) bits " *
                "($(round(bits_per_element(bf), digits=2)) bits/value)")
    cw = 1.0
    # scale byte
    poly!(ax, Rect2f(-2.6, 0, 2.2, 1.0); color = (EXP_COLOR, 0.30),
          strokecolor = (:black, 0.5), strokewidth = 1.0)
    text!(ax, -1.5, 0.5; text = "S = $(_fmtnum(qb.scale))", align = (:center, :center), fontsize = 10, font = :bold)
    text!(ax, -1.5, -0.25; text = "$(bf.scale.name)\ncode 0x$(string(qb.scale_code, base=16))",
          align = (:center, :top), fontsize = 8.5, color = GREY)
    text!(ax, -1.5, 1.15; text = "shared scale", align = (:center, :bottom), fontsize = 9, color = GREY)

    rows = ("xᵢ", "xᵢ / S", "code eᵢ", "x̂ᵢ = S·eᵢ")
    for (r, lab) in enumerate(rows)
        text!(ax, -2.9, 1.0 - (r - 1) * 1.0 - 0.5 - (r > 1 ? 1.3 : 0);
              text = lab, align = (:right, :center), fontsize = 9.5, color = GREY)
    end
    for i in 1:K
        x = (i - 1) * cw
        zeroed = qb.values[i] == 0 && qb.original[i] != 0
        col = zeroed ? (ACCENT, 0.35) : (FRAC_COLOR, 0.24)
        text!(ax, x + cw/2, 0.5 + 1.3 + 1.0; text = _fmtnum(qb.original[i]),
              align = (:center, :center), fontsize = 8, color = GREY)
        text!(ax, x + cw/2, 0.5 + 1.3; text = _fmtnum(qb.original[i] / qb.scale),
              align = (:center, :center), fontsize = 8, color = GREY)
        bitcell!(ax, x, 0, _fmtnum(qb.elements[i]); fill = col, w = cw, h = 1.0, fontsize = 9)
        text!(ax, x + cw/2, -0.55; text = _fmtnum(qb.values[i]),
              align = (:center, :center), fontsize = 8, color = zeroed ? ACCENT : :black)
    end
    z = zeroed_count(qb)
    text!(ax, 0, -1.5;
          text = "SNR $(round(snr_db(qb.original, qb.values), digits=2)) dB   •   " *
                 "elements zeroed: $z" * (z > 0 ? "  (shown in orange — information annihilated)" : ""),
          align = (:left, :top), fontsize = 10, color = z > 0 ? ACCENT : GREY)
    xlims!(ax, -3.4, K * cw + 0.4); ylims!(ax, -2.3, 3.4)
    fig
end


# --- plot_bit_layout across the other binary containers ---------------------

"""
    plot_bit_layout(f::FixedFormat; value=nothing, size=(900,260)) -> Figure

The fixed-point counterpart: sign, integer and fraction cells, each with its
power-of-two weight, and the binary point that never moves.

The whole character of the format is visible as geometry — the weights step down
uniformly through the point, which is why the spacing is constant everywhere and the
*relative* error explodes for small values.
"""
function plot_bit_layout(f::FixedFormat; value = nothing, size = (900, 260))
    set_theme!(xpufp_theme())
    n = nbits(f)
    w = min(0.9, 26 / n)
    fig = Figure(; size = size)
    ttl = value === nothing ? "$(f.name) bit layout" :
          "$(f.name):  $(value)  →  stored integer $(encode(f, value))  =  $(quantize(f, value))"
    ax = blank_axis(fig[1, 1]; aspect = nothing, title = ttl)
    bits = value === nothing ? nothing : begin
        I = encode(f, value)
        u = I < 0 ? I + (1 << n) : I
        [Int((u >> (n - 1 - i)) & 1) for i in 0:n-1]
    end
    x = 0.0
    if f.signed
        bitcell!(ax, x, 0, bits === nothing ? "s" : string(bits[1]);
                 fill = (SIGN_COLOR, 0.35), w = w)
        brace!(ax, x, x + w, -0.06, "sign"; depth = 0.16)
        x += w
    end
    xi = x
    for i in 1:f.m
        lab = bits === nothing ? "" : string(bits[(f.signed ? 1 : 0) + i])
        bitcell!(ax, x, 0, lab; fill = (INT_COLOR, 0.40), w = w)
        text!(ax, x + w/2, 1.02; text = "2$(_supb(f.m - i))", align = (:center, :bottom),
              fontsize = 7.5, color = GREY)
        x += w
    end
    f.m > 0 && brace!(ax, xi, x, -0.06, "integer part ($(f.m) bits)"; depth = 0.16)
    # the binary point
    lines!(ax, [x, x], [-0.05, 0.95]; color = :black, linewidth = 2.2)
    xf = x
    for i in 1:f.n
        lab = bits === nothing ? "" : string(bits[(f.signed ? 1 : 0) + f.m + i])
        bitcell!(ax, x, 0, lab; fill = (FRAC_COLOR, 0.28), w = w)
        text!(ax, x + w/2, 1.02; text = "2$(_supb(-i))", align = (:center, :bottom),
              fontsize = 7.5, color = GREY)
        x += w
    end
    brace!(ax, xf, x, -0.06, "fraction part ($(f.n) bits)"; depth = 0.16)
    cap = value === nothing ?
        "x = I / 2^$(f.n)      step $(resolution(f)) everywhere      range [$(fxmin(f)), $(fxmax(f))]" :
        "I = $(encode(f, value)),   x = I / 2^$(f.n) = $(quantize(f, value))   " *
        "(error $(round(quantize(f, value) - Float64(value), sigdigits=3)))"
    text!(ax, 0, -0.72; text = cap, align = (:left, :top), fontsize = 10.5)
    xlims!(ax, -0.25, x + 0.25); ylims!(ax, -1.15, 1.55)
    fig
end

"""
    plot_bit_layout(f::IntFormat; value=nothing, size=(900,240)) -> Figure

The two's-complement integer layout: every cell an ordinary power of two, except the
top one, which carries the **negative** weight `−2^(b−1)` that makes the encoding
signed with no correction step anywhere.
"""
function plot_bit_layout(f::IntFormat; value = nothing, size = (900, 240))
    set_theme!(xpufp_theme())
    n = f.bits
    w = min(0.9, 26 / n)
    fig = Figure(; size = size)
    ttl = value === nothing ? "$(f.name) bit layout" : "$(f.name):  $(quantize(f, value))"
    ax = blank_axis(fig[1, 1]; aspect = nothing, title = ttl)
    bits = value === nothing ? nothing : begin
        I = Int(quantize(f, value))
        u = I < 0 ? I + (1 << n) : I
        [Int((u >> (n - 1 - i)) & 1) for i in 0:n-1]
    end
    for i in 1:n
        x = (i - 1) * w
        top = f.signed && i == 1
        bitcell!(ax, x, 0, bits === nothing ? "" : string(bits[i]);
                 fill = top ? (SIGN_COLOR, 0.35) : (INT_COLOR, 0.35), w = w)
        text!(ax, x + w/2, 1.02;
              text = (top ? "−2" : "2") * _supb(n - i),
              align = (:center, :bottom), fontsize = 8, color = top ? SIGN_COLOR : GREY)
    end
    text!(ax, 0, -0.32;
          text = value === nothing ?
                 "range [$(intmin(f)), $(maxfinite(f))],  step 1 — uniform absolute error" :
                 "value = $(quantize(f, value));  the top cell carries the negative weight −2^$(n-1)",
          align = (:left, :top), fontsize = 10.5)
    xlims!(ax, -0.25, n * w + 0.25); ylims!(ax, -0.95, 1.55)
    fig
end

const _SUPB = ['⁰','¹','²','³','⁴','⁵','⁶','⁷','⁸','⁹']
_supb(k::Integer) = (k < 0 ? "⁻" : "") * join([_SUPB[parse(Int, c)+1] for c in string(abs(k))])
