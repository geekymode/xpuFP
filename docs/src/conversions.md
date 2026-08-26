# Conversions

Every scheme is a round trip of three maps — an **encoder** into the representation, a
**compute** step inside it, and a **decoder** back — and should be judged on nothing but
the end-to-end comparison against the true real-number result.

This framing exposes one structural fact all the schemes share: **loss, if any, has a
location**.

| Scheme | Where value is destroyed |
|:---|:---|
| FP32 | a little at encode, then again at *every operation* |
| RR4 / carry-save | only at encode (the fixed-point grid); computes exactly forever after |
| MXFP4 | its entire, much larger loss at the encoder; exact thereafter |
| CSD | **nowhere** — it re-spells rather than approximates (constants only) |

Decoders are exact in every scheme: leaving a representation costs circuitry, never
accuracy.

## Format to format

```@docs
convert_format
narrow
int_to_float
float_to_fixed
fixed_to_float
```

`int → float` is a count-leading-zeros, a shift, and one rounding. The exponent is found
**without any logarithm** — the position of the top set bit *is* the exponent:

```jldoctest
julia> using xpuFP

julia> r = int_to_float(FP32, 1440);

julia> r.clz, r.exponent, r.fraction, r.exact
(21, 10, 3407872, true)
```

## The monotone-encoding gift

```@docs
nextafter
prevafter
encodings_monotone
```

FP encodings are *monotone*: for non-negative values a larger bit pattern always means a
larger value, and one increment always means one grid step. That invariant is a gift to
systems programmers — `nextafter` is an integer increment — and it explains both seams.

```@docs
top_seam
bottom_seam
```

```@example conv
using xpuFP, CairoMakie # hide
plot_seams(FP32)
```

At the top seam that step is a gargantuan ``2^{104}``; at the bottom a tiny
``2^{-149}``; but each is exactly one ulp of its own neighbourhood.

## Redundant-domain conversions

```@docs
to_carrysave
from_carrysave
rr4_to_canonical
minimal_to_maximal
maximal_to_minimal
```

The alphabets are **strictly nested**: every minimal spelling *is* a maximal spelling,
but not conversely. They are different coordinate systems on the same integers, and they
interconvert at bounded cost, with the value invariant throughout.

## The hub and its excursions

Let FP32 be the **hub**. Values enter it once — one half-ulp, paid at the door — and
every other representation becomes an **excursion**: words leave, are converted into
some internal format chosen on merit for one operation, and convert back before anything
else sees them.

The decisive question is not "which format is best" but, per excursion: *does the
returning word have the same bits it would have had without the trip?*

```@docs
ExcursionClass
Excursion
excursion
multiplier_excursion
redundant_dot_excursion
mx_excursion
```

### Invisible — take them all, unconditionally

```jldoctest
julia> using xpuFP

julia> e = multiplier_excursion(FP32, 0.1, 10.0);

julia> e.result, e.bits_identical, e.class
(1.0, true, INVISIBLE)
```

The hub holds `0x3DCCCCCD` (0.1 rounded *up*); the Booth/carry-save core forms
``1 + 2^{-26}`` exactly, an amount the hub cannot represent; the exit rounding snaps it
to `0x3F800000` exactly. The error `0.1` picked up entering the hub is precisely
cancelled by the rounding on the way back — and now you know the circuit-level reason
`0.1f * 10f == 1.0f`.

### Visible, better

```jldoctest
julia> using xpuFP

julia> e = redundant_dot_excursion(FP32, [1.0, 2.0^-25, 2.0^-25, 2.0^-25], ones(4));

julia> e.reference == 1.0, e.result == 1.0 + 2.0^-23, e.class
(true, true, VISIBLE_BETTER)
```

Three sub-half-ulp grains that a per-step chain absorbs to nothing are recovered by
staying redundant and rounding once. Deferring the rounding did not merely save carries;
it saved **information**.

Deviating from reference bits is a semantics decision, not a correctness failure — and
it has precedent: the FMA is exactly such an excursion, considered valuable enough that
IEEE 754-2008 standardised it.

### Visible, worse

The MXFP4 trip returns a word measurably far from truth in exchange for ``7.5\times``
fewer bits and ``144\times`` smaller multipliers. Because it begins and ends at hub
words, its damage is **measurable in isolation** — which is precisely how one decides
that GEMM operands may take this trip while the softmax may not.
