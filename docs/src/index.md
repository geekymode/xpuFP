# xpuFP.jl

A laboratory for the number representations accelerator hardware actually uses:
fixed point, the IEEE-style floating-point zoo down to four bits, the block-scaled
microscaling formats (MXFP4, NVFP4), the redundant systems inside every multiplier,
and the error analysis that decides between them.

Every format is implemented **from the bit pattern up** — encode, decode, round,
saturate — so the arithmetic can be *watched* rather than trusted.

!!! note "Build status"
    This documentation is under active construction. The Formats, Arithmetic and
    Block sections reflect implemented and verified code. Redundant arithmetic
    (CSD, Booth, RR4, Wallace trees), the systolic-array model, and the figure
    suite are still being written.

## The core problem

Real numbers are infinite objects squeezed into finite containers. With ``n`` bits we
can distinguish at most ``2^n`` values, so any ``n``-bit number system is a choice of
*which* ``2^n`` points on the real line are exact — everything else rounds to a
neighbour.

Fixed point spends its bits on **uniformly spaced** points; floating point spends them
on points that are **dense near zero and sparse far away**. Everything else follows
from that one decision.

Geometrically, a float grid is a **log axis sampled at finite resolution**: the exponent
field is exactly the octave index, and the mantissa interpolates inside each octave —
*linearly*, where a logarithm would curve. [Formats](@ref) works that analogy out, along
with the two places it fails and the 0.0861-octave sag it leaves behind.

## Quick start

```julia
using xpuFP

FP32                       # a FloatFormat; show it for its vital statistics
grid(E2M1)                 # the *entire* FP4 number system, 15 distinct values
quantize(E2M1, 2.25)       # 2.0 — even the smallest interesting product rounds

fpadd(FP32, 12.0, 6.0)     # a full datapath trace, not just an answer

qb = quantize_block(MXFP4, randn(32))   # one MX block: scale + 32 nibbles
measure_snr(MXFP4, randn(200_000))      # ≈ 18.8 dB

quick_snr(MXFP4)                        # the same 18.8 dB, plus what SNR hides
quick_compare([MXFP4, NVFP4, fp4_variant(rule = OPT_SHIFT)])
```

Then read [Examples](@ref) for four runnable studies with their expected output.

## What is verified against the source report

Every quantitative claim the package can check, it does:

| Claim | Report | `xpuFP` |
|:---|---:|---:|
| FP32 largest normal | ``3.4028235\times10^{38}`` | matches to the last digit |
| ``-6.375`` as FP32 | `0xC0CC0000` | `0xC0CC0000` |
| ``0.1`` as FP32 | `0x3DCCCCCD` | `0x3DCCCCCD` |
| E2M1 distinct values | 15 | 15 |
| Exact FP4 products | 141 of 225 | 141 of 225 |
| MX worked block scale | ``2^{-4}``, code 123 | ``2^{-4}``, code 123 |
| MX worked dot-product error | 18.6 % | 18.6 % |
| Bare FP4 SNR (Gaussian) | 16.3 dB | 16.34 dB |
| MXFP4 SNR (Gaussian) | 18.8 dB | 18.77 dB |
| NVFP4 SNR (Gaussian) | 20.4 dB | 20.42 dB |
| MXFP4 compression vs FP32 | ``7.5\times`` | ``7.53\times`` |

Two independent paths reach the block-format numbers and are held to each other:
[`quick_snr`](@ref) measures by simulation, [`estimate_element_snr`](@ref) derives by
order-statistic quadrature, and on MXFP4 they agree to **0.002 dB** — 18.790 against
18.788. The simulation runs in 12 ms and the quadrature in about 400 s, so the
agreement is what licenses using the fast one everywhere else.

## Guide

```@contents
Pages = ["formats.md", "arithmetic.md", "blocks.md", "analysis.md",
         "examples.md", "api.md"]
Depth = 2
```
