# ---------------------------------------------------------------------------
# The format zoo, as named constants.
# ---------------------------------------------------------------------------

"""    FP32

IEEE 754 binary32: 1 sign, 8 exponent (bias 127), 23 fraction bits.  ~7.2 decimal
digits over a range of ``10^{\\pm 38}``; machine epsilon ``2^{-23}``."""
const FP32 = FloatFormat("FP32", 8, 23)

"""    FP64

IEEE 754 binary64.  Note that values of this format are *not* generally exactly
representable in the `Float64` working type used by [`quantize`](@ref) — it is
included for reference and for the significand-width `p = 53` in cost formulas."""
const FP64 = FloatFormat("FP64", 11, 52)

"""    FP16

IEEE 754 binary16 (half): 1 sign, 5 exponent (bias 15), 10 fraction bits."""
const FP16 = FloatFormat("FP16", 5, 10)

"""    BF16

bfloat16: FP32 with the bottom 16 fraction bits chopped off.  Same 8 exponent bits
and therefore the same range as FP32, with only 8 significand bits — a trade machine
learning makes happily, and the reason BF16 scores a poor 3.0 dB/bit."""
const BF16 = FloatFormat("BF16", 8, 7)

"""    E5M2

OCP FP8 E5M2: the range-favouring 8-bit float, with IEEE-style ±Inf and NaN."""
const E5M2 = FloatFormat("E5M2", 5, 2)

"""    E4M3

OCP FP8 E4M3: the precision-favouring 8-bit float.  Only the single code
`S.1111.111` is NaN, so the format reaches `448 = 2^8 × 1.75` instead of stopping at
240.  Used as the block-scale format of NVFP4."""
const E4M3 = FloatFormat("E4M3", 4, 3; nan_style = E4M3_NAN, saturate = true)

"""    E2M1

FP4 as standardised by OCP Microscaling: 1 sign, 2 exponent (bias 1), 1 mantissa bit.
All sixteen codes are finite — there is no ±Inf and no NaN, because at this size the
special-value apparatus would burn 12.5% of the code space.  The positive magnitudes
are exactly `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`, machine epsilon is `0.5`, and overflow
saturates at ±6."""
const E2M1 = FloatFormat("E2M1", 2, 1; nan_style = NO_SPECIAL)

"""    E1M2

The precision-favouring FP4 sibling: finer steps, but a range of only a few units.
Grid: `{0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5}`."""
const E1M2 = FloatFormat("E1M2", 1, 2; nan_style = NO_SPECIAL)

"""    E3M0

The range-favouring FP4 sibling.  With no mantissa bits every value is a pure power
of two, so multiplication degenerates into an exponent *addition* — no multiplier
array at all.  Grid: `{0, 0.25, 0.5, 1, 2, 4, 8, 16}`."""
const E3M0 = FloatFormat("E3M0", 3, 0; nan_style = NO_SPECIAL)

"""    E8M0

The MXFP4 block-scale format: eight exponent bits, no sign, no mantissa, bias 127 —
the FP32 exponent field promoted to a standalone number.  Decodes as `2^(X-127)`,
so applying it is an exponent add rather than a multiplication.  `X = 255` is NaN."""
const E8M0 = FloatFormat("E8M0", 8, 0; bias = 127, signed = false,
                         subnormals = false, nan_style = E8M0_NAN,
                         zero_exp = NORMAL_ZERO, saturate = true)

"""    ALL_FLOAT_FORMATS

Every registered [`FloatFormat`](@ref), for sweeps and comparison tables."""
const ALL_FLOAT_FORMATS = (FP32, FP16, BF16, E5M2, E4M3, E2M1, E1M2, E3M0, E8M0)

"""    FP4_FAMILY

The three 4-bit splits of the same four bits: range, balance, precision."""
const FP4_FAMILY = (E3M0, E2M1, E1M2)
