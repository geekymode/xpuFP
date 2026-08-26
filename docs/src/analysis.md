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
