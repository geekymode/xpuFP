# Formats

## Fixed point

A fixed-point number is an ordinary two's-complement integer that both parties agree
to read with a binary point at a fixed position. In Q``m``.``n`` notation the stored
integer ``I`` denotes ``x = I / 2^n``.

```@docs
FixedFormat
resolution
fxmax
fxmin
```

The properties are rigid: the step is ``2^{-n}`` **everywhere**, so absolute error is
bounded by ``2^{-n-1}`` but *relative* error explodes for tiny values. Storing
``0.001`` in Q7.8 gives ``0`` or ``0.0039`` — up to 100 % relative error. That is
precisely the weakness floating point fixes.

## Floating point

```@docs
FloatFormat
NaNStyle
ZeroExponent
```

One struct covers the whole zoo because the formats differ only in field widths and
in what the reserved codes mean.

### A float grid is a log axis, sampled

The quickest way to see what a floating-point format *is* geometrically: take the log of
its defining equation.

```math
x = 1.m \times 2^{e}
\qquad\Longrightarrow\qquad
\log_2 x = \underbrace{e}_{\text{integer}} + \underbrace{\log_2(1.m)}_{\in\,[0,1)}
```

The exponent field drops out as the **integer part** — the octave index, or *binade* —
and the mantissa contributes only the fraction. A semilog axis places every power of two
at an equal spacing, and so does the exponent field. **That correspondence is exact and
free**: it is the whole reason floating point handles dynamic range at all, and it is why
`plot_edges_map` and `plot_binade_spacing` draw a format's universe on a log axis rather
than a linear one.

What is **not** logarithmic is the inside. A format with ``m`` mantissa bits puts
``2^m`` values in every binade and spaces them **linearly** — the mantissa is an integer
counter, and it steps by a constant ``2^{e-m}``. So a float grid is a
**piecewise-linear approximation of a log axis**: pinned exactly at every power of two,
sagging in between.

```@example fmt
using xpuFP, CairoMakie # hide
plot_log_axis_analogy(E4M3)
```

#### How big is the sag

The gap between where a float puts a value and where a logarithm would is
``m - \log_2(1+m)``, maximised where the derivative vanishes:

```math
\frac{d}{dm}\bigl[m - \log_2(1+m)\bigr] = 0
\;\Longrightarrow\;
m = \frac{1}{\ln 2} - 1 \approx 0.4427,
\qquad \text{gap} = 0.0861 \text{ octaves}
```

**Independent of the mantissa width.** More mantissa bits sample the same sagging curve
more finely; they do not straighten it.

That constant shows up somewhere unexpected. Read a positive float's *bit pattern* as an
integer and you get ``2^{23}\,(\log_2 x + 127)`` — to within exactly that error. The
hardware encoding **is** a piecewise-linear logarithm, which is why the fast
inverse-square-root trick works by subtracting an integer.

#### The difference that matters numerically

A true log axis has **constant relative resolution**. A float does not: ``\mathrm{ulp}(x)/x``
is a **sawtooth** that halves smoothly across each binade and doubles instantly at the
boundary, so the best and worst relative step differ by **exactly the radix, 2**, at every
mantissa width. Numerical analysts call this *wobbling precision*, and it is why error
bounds carry a stray factor of the base.

[`machine_eps`](@ref) reports the top of that band, ``2^{-m}``; the bottom is
``2^{-m-1}``. Averaged over a binade the relative error is better than ``\epsilon``
suggests, and at the boundary it is exactly ``\epsilon`` — which is the honest reason to
quote ``\epsilon`` as a *bound* rather than a typical case.

#### Where the analogy stops

At both ends the two objects part company, in opposite directions:

| | Semilog axis | Floating point |
|:---|:---|:---|
| Octave spacing | equal by construction | equal — the exponent field *is* the tick index |
| Inside an octave | continuous, logarithmic | ``2^m`` points, spaced **linearly** |
| Relative resolution | constant everywhere | sawtooth, wobbling by exactly the radix |
| Multiplication | becomes addition | adds exponents, **multiplies** mantissas |
| Zero | unreachable, at ``-\infty`` | an exact, special-cased value |
| Below the floor | continues forever | linear subnormal ramp, then stops |
| Above the ceiling | continues forever | overflows to ``\pm\infty`` or saturates |

The bottom end is the interesting one. Below [`minnormal`](@ref) a format abandons the
logarithmic scheme entirely and switches to a **fixed linear grid** — the subnormals —
before stopping at an exact zero. Viewed on a log axis those subnormals do the opposite of
crowding: they fly apart. E4M3's seven of them span ``2^{-9}`` to ``2^{-6.19}``, nearly
three octaves for seven values, where a normal binade fits eight into one. And zero sits
at ``\log_2 0 = -\infty``, which is not a coordinate the axis has. [`plot_edges_map`](@ref)
draws exactly that gap, and the subnormals exist to bridge it in even steps.

!!! note "The format that closes the gap"
    A **logarithmic number system** stores ``\log_2 x`` directly in fixed point. It *is*
    the semilog axis: perfectly constant relative resolution, no wobble, and
    multiplication really does collapse to addition. What it gives up is the other
    operation — addition needs a lookup table. Floating point is the engineering
    compromise between the two, and the wobble is the price of the compromise.

Shrink the mantissa and the compromise becomes visible. **E2M1 — the element format
inside MXFP4 — has one mantissa bit**, so it samples each octave exactly twice, at log₂
offsets ``0`` and ``0.585``:

```@example fmt
grid(E2M1)          # 0.5 1 1.5 2 3 4 6, and their negatives
```

That is a log axis sampled twice per octave with a 0.0861-octave sag at each step — and
it is still close enough to logarithmic that a shared block exponent can carry the dynamic
range the mantissa gave up. That trade is the subject of [Block formats](@ref).

### The registry

```@docs
FP32
FP16
BF16
E4M3
E5M2
E2M1
E1M2
E3M0
E8M0
```

### Geometry

```@docs
nbits
emin
emax
maxfinite
minnormal
minsubnormal
machine_eps
dynamic_range
ulp
```

### Encoding and decoding

```@docs
encode
decode
quantize
grid
posgrid
midpoints
```

## Integer grids

```@docs
IntFormat
INT4
```
