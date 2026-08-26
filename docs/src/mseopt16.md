# FP4-K16-E4M3-MSE_OPTIMAL, end to end

```julia
julia> fp4_variant(K = 16, scale = E4M3, rule = MSE_OPTIMAL)
BlockFormat(FP4-K16-E4M3-MSE_OPTIMAL, K=16, E2M1 × E4M3, 4.5 b/elem)
```

!!! note "This tuple already has a name"
    `(K=16, E2M1, E4M3, MSE_OPTIMAL)` is exactly [`NVFP4_BEST16`](@ref), aliased
    [`XPFP4_16`](@ref) — NVFP4's wire format with the scale chosen by search instead of
    by anchoring. [`fp4_variant`](@ref) builds a structurally identical object under a
    descriptive name, which is convenient in a sweep but is *not* a new format. Anything
    below applies to all three names.

Four numbers define the code, and only the last is unusual:

| choice | value | consequence |
|:---|:---|:---|
| block length `K` | 16 | 16 elements share one scale |
| element format | `E2M1` | 4 bits, grid `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` |
| scale format | `E4M3` | 8 bits, spans ``2^{-9} \approx 0.00195`` … 448 |
| scale rule | `MSE_OPTIMAL` | the scale minimising block MSE, **not** the one that avoids overflow |

Storage is ``16 \times 4 + 8 = 72`` bits per block, **4.5 bits per value**.

## The pipeline

```
 ENCODE                                                             per tensor
 ══════                                                             ──────────
                        ┌──────────────────────────────┐
   x  (K=16 reals) ────►│  G = tensor_scale(x_tensor)  │────► G = 2^g   (8b, exact)
                        │  power of two, whole tensor  │      shared by all blocks
                        └───────────────┬──────────────┘
                                        │  u = x / G                   per block
                                        ▼                              ─────────
                        ┌──────────────────────────────┐
                        │  M = max |u_i|               │
                        │  anchor a = M / 6            │   6 = maxfinite(E2M1)
                        └───────────────┬──────────────┘
                                        ▼
                        ┌──────────────────────────────────────────┐
                        │  MSE-OPTIMAL SEARCH                      │
                        │  for S in E4M3 ∩ [a/2, a√2] :            │
                        │      SSE(S) = Σ (Q(u_i/S)·S − u_i)²      │
                        │  S* = argmin SSE          (ties → lower)  │
                        └───────────────┬──────────────────────────┘
                                        │  S*  ──► 8-bit E4M3 code
                                        ▼
                        ┌──────────────────────────────┐
   16 nibbles ◄─────────│  e_i = Q_E2M1(u_i / S*)      │
                        │  c_i = encode(E2M1, e_i)     │
                        └──────────────────────────────┘

 stored block:   ┌────────┬────┬────┬────┬─────────────────────┬────┐
                 │ scale  │ c₁ │ c₂ │ c₃ │  …                  │ c₁₆│   72 bits
                 │ 8b E4M3│ 4b │ 4b │ 4b │                     │ 4b │
                 └────────┴────┴────┴────┴─────────────────────┴────┘

 DECODE                        x̂_i = G · S* · decode(E2M1, c_i)
 ══════                        two multiplies, no rounding, no search
```

The asymmetry is the point: **the encoder searches, the decoder does not.** Nothing in
the stored block, the decode path, or the MAC datapath differs from shipped NVFP4 — the
rule changes only which 8-bit scale code the encoder writes.

## Encoding, step by step

### 1. The tensor scale ``G``

`E4M3` spans only ``2^{-9}`` … ``448``. Raw tensor data need not land in that window, so
a per-tensor power of two is divided out first ([`tensor_scale`](@ref)):

```math
G = 2^{\,\lfloor \log_2 (M_{\text{tensor}}/6) \rfloor \;-\; \lfloor \log_2 448 \rfloor \;+\; 1}
  = 2^{\,\lfloor \log_2 (M_{\text{tensor}}/6) \rfloor - 7}
```

which parks the largest per-block ratio near the upper-middle of E4M3's range. Because
``G`` is a power of two it is **exact**, and because it multiplies signal and
reconstruction identically it cancels out of every SNR ratio — it changes what is
*representable*, never what the error is. For the worked tensor below, ``G = 2^{-14}``.

This is what makes the scheme genuinely **three-level**: tensor ``G``, block ``S``,
element ``e_i``.

### 2. The anchor

With ``u = x/G`` and ``M = \max_i |u_i|``, the *anchor* is the scale that would place the
block maximum exactly on E2M1's top code:

```math
a = \frac{M}{\max_{\text{finite}}(\text{E2M1})} = \frac{M}{6}
```

Rounding ``a`` onto E4M3 and stopping is precisely the `NV_MAXDIV` rule that shipped
NVFP4 uses. It guarantees no element overflows. **It does not minimise error.**

### 3. The search

`MSE_OPTIMAL` instead evaluates the true block SSE over a window around the anchor and
takes the minimiser:

```math
S^\star = \arg\min_{S \,\in\, \mathcal{S}} \; \sum_{i=1}^{16}
          \bigl( Q_{\text{E2M1}}(u_i / S)\cdot S - u_i \bigr)^2 ,
\qquad
\mathcal{S} = \Bigl\{ Q_{\text{E4M3}}(2^t) \;:\; t \in [\log_2 a - 1,\; \log_2 a + \tfrac12] \Bigr\}
```

The window spans one octave below the anchor and half an octave above, so

```math
\frac{M}{S^\star} \in \Bigl[\tfrac{6}{\sqrt 2},\; 12\Bigr] \approx [4.24,\; 12]
```

— the block maximum may be overloaded by at most ``2\times`` before clamping to 6. Ties
go to the lower ``S``, since the scan runs upward in ``t`` and accepts only a *strict*
improvement.

!!! tip "200 trial exponents, ~5 distinct scales"
    The window is sampled at `npts = 200` points, but E4M3 carries 3 mantissa bits — 8
    steps per binade — so a 1.5-octave window holds only about a dozen representable
    scales, and often fewer. In the worked block below, 200 trial exponents collapse to
    **5 distinct candidates**. [`quick_snr`](@ref) skips a candidate equal to its
    predecessor, which evaluates the same set in the same order for the same answer at a
    fraction of the cost.

### 4. Elements

```math
e_i = Q_{\text{E2M1}}(u_i / S^\star), \qquad c_i = \text{encode}_{\text{E2M1}}(e_i)
```

``Q_{\text{E2M1}}`` is round-nearest-ties-to-even onto `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4,
±6}`, **saturating** at ±6 — FP4 has no infinities to escape to, which is what makes the
deliberate overload safe rather than catastrophic.

## Decoding

```math
\hat{x}_i \;=\; G \cdot S^\star \cdot \text{decode}_{\text{E2M1}}(c_i)
```

with the E2M1 code read back by its field decomposition — sign ``s``, exponent ``E``
(2 bits, bias 1), mantissa ``M`` (1 bit):

```math
\text{decode}_{\text{E2M1}}(c) = (-1)^{s} \times
\begin{cases}
2^{0} \cdot (M/2) & E = 0 \quad \text{(subnormal: } 0 \text{ or } 0.5) \\[4pt]
2^{E-1} \cdot (1 + M/2) & E \ge 1
\end{cases}
```

| code | s E E M | value | | code | s E E M | value |
|:---|:---|---:|---|:---|:---|---:|
| `0000` | 0 00 0 | 0 | | `1000` | 1 00 0 | −0 |
| `0001` | 0 00 1 | 0.5 | | `1001` | 1 00 1 | −0.5 |
| `0010` | 0 01 0 | 1 | | `1010` | 1 01 0 | −1 |
| `0011` | 0 01 1 | 1.5 | | `1011` | 1 01 1 | −1.5 |
| `0100` | 0 10 0 | 2 | | `1100` | 1 10 0 | −2 |
| `0101` | 0 10 1 | 3 | | `1101` | 1 10 1 | −3 |
| `0110` | 0 11 0 | 4 | | `1110` | 1 11 0 | −4 |
| `0111` | 0 11 1 | 6 | | `1111` | 1 11 1 | −6 |

The E4M3 scale decodes the same way with bias 7 and 3 mantissa bits, the single code
`S.1111.111` reserved for NaN.

**The decoder never searches.** ``G \cdot S^\star`` folds into one constant per block, so
each element costs a 16-entry table lookup and a single multiply. In a dot product even
that multiply leaves the inner loop:

```math
y = \sum_i (G_a S_a e^a_i)(G_b S_b e^b_i) = G_a G_b S_a S_b \sum_i e^a_i e^b_i
```

so the scales leave the inner loop entirely and the core sum stays exact integer
arithmetic, bounded by ``16 \times 6 \times 6 = 576``.

## A block traced all the way through

Sixteen weights at ``\sigma = 0.02``, `MersenneTwister(116)`:

```
x = [-0.00673,  0.00467, -0.05310, -0.00059, -0.03351, -0.00590,  0.02008, -0.00381,
     -0.00416, -0.00214,  0.02605,  0.04681, -0.00793, -0.01697, -0.04734,  0.02135]

M = max|x| = 0.053100        anchor a = M/6 = 0.008850
tensor scale (656-element tensor)  G = 2^-14
```

The five distinct candidate scales in the window, with the block SSE each produces:

```
    S             code   M/S      SSE         clipped  zeroed
    0.00390625    0x02   13.594   2.125e-03      5        1
    0.005859375   0x03    9.062   6.329e-04      3        1
    0.0078125     0x04    6.797   7.731e-05      2        1   <-- minimum
    0.009765625   0x05    5.437   2.105e-04      0        2   <-- the anchor
    0.01171875    0x06    4.531   7.797e-05      0        2
```

The anchor is `0x05`. The minimiser is `0x04`, **one step below it** — a scale that
overloads the block maximum to ``M/S = 6.797`` and clamps two elements to ±6. Read the
last two columns: the overload trades 2 clipped elements for 1 fewer annihilated one,
and cuts SSE by 2.7×.

Note also that the candidate *above* the anchor (`0x06`, SSE `7.797e-05`) is nearly tied
with the winner (`7.731e-05`). The SSE curve is shallow near its floor, which is why
sampling the window coarsely costs nothing — and why deduplicating 200 trial exponents
down to these 5 distinct scales changes no answer at all.

The encoded block, `S* = 0.0078125` (code `0x04` = `00000100`):

```
  i |        x_i |   x_i/S* |  e_i | code |          x̂_i |      err
  1 |  -0.006730 |  -0.8614 |   -1 | 1010 |   -0.0078125 | -0.001083
  2 |   0.004670 |   0.5978 |  0.5 | 0001 |    0.0039062 | -0.000764
  3 |  -0.053100 |  -6.7968 |   -6 | 1111 |   -0.0468750 | +0.006225   ← clamped
  4 |  -0.000590 |  -0.0755 |   -0 | 1000 |   -0.0000000 | +0.000590   ← annihilated
  5 |  -0.033510 |  -4.2893 |   -4 | 1110 |   -0.0312500 | +0.002260
  6 |  -0.005900 |  -0.7552 |   -1 | 1010 |   -0.0078125 | -0.001913
  7 |   0.020080 |   2.5702 |    3 | 0101 |    0.0234375 | +0.003357
  8 |  -0.003810 |  -0.4877 | -0.5 | 1001 |   -0.0039062 | -0.000096
  9 |  -0.004160 |  -0.5325 | -0.5 | 1001 |   -0.0039062 | +0.000254
 10 |  -0.002140 |  -0.2739 | -0.5 | 1001 |   -0.0039062 | -0.001766
 11 |   0.026050 |   3.3344 |    3 | 0101 |    0.0234375 | -0.002613
 12 |   0.046810 |   5.9917 |    6 | 0111 |    0.0468750 | +0.000065
 13 |  -0.007930 |  -1.0150 |   -1 | 1010 |   -0.0078125 | +0.000117
 14 |  -0.016970 |  -2.1722 |   -2 | 1100 |   -0.0156250 | +0.001345
 15 |  -0.047340 |  -6.0595 |   -6 | 1111 |   -0.0468750 | +0.000465   ← clamped
 16 |   0.021350 |   2.7328 |    3 | 0101 |    0.0234375 | +0.002087

  zeroed 1    clipped 2    block SNR 21.289 dB
```

Trace row 3 by hand as a check: ``x_3 = -0.05310``, ``x_3/S^\star = -6.7968``, which
saturates to ``e_3 = -6`` (code `1111`), so
``\hat{x}_3 = 0.0078125 \times (-6) = -0.046875`` — an error of ``+0.006225``, the
largest in the block and the price of the overload.

Row 4 shows the other failure mode: ``-0.000590 / S^\star = -0.0755``, below E2M1's
round-to-zero threshold of 0.25, so it is stored as exact zero. **SNR cannot see this
row** — that is what the `zeroed` count is for.

Reproduce the whole trace with:

```julia
using xpuFP, Random
F = fp4_variant(K = 16, scale = E4M3, rule = MSE_OPTIMAL)
x = round.(0.02 .* randn(MersenneTwister(116), 16); digits = 5)
quantize_block(F, x)        # displays the table above
```

## What the search is worth

On that block, against the anchoring rule NVFP4 ships:

| rule | ``S`` | code | ``M/S`` | zeroed | clipped | block SNR |
|:---|---:|:---:|---:|---:|---:|---:|
| `NV_MAXDIV` (anchor) | 0.009765625 | `0x05` | 5.437 | 2 | 0 | 16.939 dB |
| `MSE_OPTIMAL` | 0.0078125 | `0x04` | 6.797 | 1 | 2 | **21.289 dB** |

**+4.35 dB on one block, for zero change to the stored bits.** That block was selected
to show the mechanism clearly; across a million Gaussians the average gain is smaller
but consistent:

| scheme | bits/value | SNR | zeroed | clipped |
|:---|---:|---:|---:|---:|
| `NVFP4` (`NV_MAXDIV`) | 4.500 | 20.440 dB | 6.78 % | 3.47 % |
| `FP4-K16-E4M3-MSE_OPTIMAL` | 4.500 | **21.588 dB** | 7.31 % | 3.35 % |

**+1.15 dB for free at the wire level.** The cost is encoder time — about five block-SSE
evaluations per block — paid once when weights are quantized, never at inference.

Note the honest part of that table: the MSE-optimal rule **zeroes more** (7.31 % against
6.78 %), because minimising a sum of squares is content to annihilate a small coordinate
if that buys resolution for larger ones. If your consumer cares about individual small
coordinates rather than block energy, read [`QuickSNR`](@ref)'s `zeroed` column before
adopting it.

## Why overloading wins

The shared scale sets one step size for the whole block. Halving ``S`` halves the step
everywhere — a 6 dB resolution gain across all 16 elements — and costs error only on the
elements that then exceed ±6. With Gaussian data the block maximum is a lone outlier by
construction (it is the max of 16 draws), so:

* **the gain** applies to ~15 elements, weighted by their energy;
* **the loss** applies to one or two, and is bounded, because E2M1 saturates rather than
  wrapping.

The anchoring rule is optimising the wrong objective — it minimises the *worst-case*
element error, which for a block code is not what block MSE measures. The search costs
nothing at inference, so the only reason to prefer anchoring is encoder simplicity.

The reference docstrings live with the formats themselves:
[`NVFP4_BEST16`](@ref) / [`XPFP4_16`](@ref) for this tuple, [`ScaleRule`](@ref) for
`MSE_OPTIMAL` beside the four rules it competes with, and [`block_scale`](@ref) for the
dispatch that implements them.

See [Improved FP4](@ref) for the other scale rules measured on the same axis,
[Block formats](@ref) for the two-level code in general, and [Examples](@ref) for this
format inside a full design sweep.
