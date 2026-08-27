# Error analysis

Throughout, a number format is treated as a **noisy channel**: a value ``x`` goes in,
its representable stand-in ``\hat x`` comes out, and the difference is bundled as
additive noise. The framing is borrowed from communications and is legitimate for a
reason (Bennett, 1948): over varied data, deterministic rounding *behaves
statistically* like independent additive noise.

```@docs
snr_db
effective_bits
rel_rms_error
DB_PER_BIT
```

The effective-bits column is the calibration proof: dividing by 6.02 recovers
*exactly* the significand widths 8, 11 and 24 of BF16, FP16 and FP32.

## Measurement

```@docs
measure_snr
quantize_all
per_element_relerror
cosine_similarity
cosine_from_snr
```

## QSNR: the same ratio, told block by block

The quantization literature writes **QSNR** for

```math
\mathrm{QSNR} = 10\log_{10}\frac{\sum_i x_i^2}{\sum_i (x_i - q(x_i))^2}
```

which is [`measure_snr`](@ref) exactly — same formula, same number, to the last decimal.
Worth saying plainly rather than shipping a second function that returns the first one's
answer.

What the published figure hides is the **aggregation**. That sum runs over the whole
tensor, so each block contributes in proportion to its energy: a high-energy block can
carry the score while a low-energy block is quietly ruined. It is the failure mode
[`snr_db`](@ref)'s own warning describes, and on block-scaled formats it is not
hypothetical, because each block gets its own scale and therefore its own fate.

[`measure_qsnr`](@ref) computes the ratio *within* each block of `K` and reports the
distribution, so every block counts equally:

```julia
julia> using Random

julia> x = quick_data(:sparse, 200_000; rng = MersenneTwister(0));

julia> q = measure_qsnr(MXFP4, x);

julia> round.((q.pooled, q.median, q.p10, q.min), digits = 2)
(18.16, 18.95, 14.81, 12.03)
```

Pooled says 18.16 dB. The tenth-percentile block says 14.81, and the worst says 12.03 —
a **`gap` of +3.34 dB** between the headline and the tail on the same data.

Two rules follow:

* **Pooled QSNR answers "how much signal energy survived".** It is the right number for
  comparing formats at a glance, and it is what to quote against published results.
* **Per-block QSNR answers "is there a block I ruined".** It is the right number for
  deciding whether a format is safe to deploy, because a downstream layer sees blocks,
  not tensors.

The two agree to a tenth of a dB on Gaussian data and diverge by whole decibels the
moment the data is sparse or heavy-tailed — precisely when the question matters. The
`gap` field makes that divergence a number you can sort on.

!!! note "An empty block is not a damaged one"
    [`snr_db`](@ref) returns `0.0` when the signal itself is zero. Scoring an all-zero
    block as "0 dB" would report absent data as total destruction, so
    [`qsnr_blocks`](@ref) skips zero-energy blocks rather than counting them. The
    `dead_blocks` field counts only blocks that had energy and lost it.

The result type is [`QSNR`](@ref) — documented on the API page, since
[API — Analysis](@ref) already autodocs this file and listing the type twice would
collide.

```@docs
qsnr
qsnr_blocks
measure_qsnr
```

## Coding efficiency

```@docs
db_per_bit
resolution_fraction
predicted_snr
```

The efficiency law is nothing but the **resolution fraction**:

```math
\frac{\text{dB}}{\text{bit}} \approx 6.02 \times \frac{p}{b_{\text{total}}}
```

Exponent bits buy *range*, not SNR. BF16's mediocre score is a theorem, not an
observation: half its bits are exponent.

!!! note "Two conventions, and they disagree instructively"
    *Guaranteed* SNR is ``6.02p`` (144.5 dB for FP32). *Typical* (RMS) SNR is
    ``6.02p + 7.44`` dB (151.9 dB measured). This package measures the RMS figure;
    [`predicted_snr`](@ref) gives the closed form. Both appear in the literature and
    they differ by a constant 7.44 dB.
