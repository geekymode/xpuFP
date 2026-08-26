# ---------------------------------------------------------------------------
# The optimized shift rule.
#
# Standard MXFP4 with one added comparison in the encoder and a decoder that cannot
# tell the difference.  The floor rule leaves the block maximum somewhere in the top
# binade, u = M/S ∈ [4,8); the interval just below 8 is where the clamp toward code 6
# bites hardest (relative error up to 1 − 6/8 = 25% on the block's most energetic
# element).  Shifting one binade up moves the maximum to u/2 ∈ [2,4) — clamp-free — at
# the cost of one bit of resolution for everyone else.  Whether that trade pays is a
# single threshold test on the maximum's mantissa.
# ---------------------------------------------------------------------------

"""    PHISTAR

The mantissa-fraction thresholds `φ*` of the optimized shift rule, by block size.

`φ* = 0.86` at `K = 32` and `0.82` at `K = 16`, i.e. shift when the scaled maximum would
land beyond `u* = 4·2^{φ*} ≈ 7.26` (resp. `7.06`).

Both were confirmed by sweeping `φ*` on 2×10⁵ Gaussian samples: the SNR curve is flat
to within 0.001 dB across `φ* ∈ [0.84, 0.86]` at `K = 32` and `[0.82, 0.84]` at
`K = 16`, so the exact constant is not delicate."""
const PHISTAR = Dict(32 => 0.86, 16 => 0.82)

"""
    opt_shift_threshold(K::Integer) -> Float64

The threshold `φ*` for block size `K`.  Known values are tabulated in [`PHISTAR`](@ref);
other sizes interpolate linearly in `log₂K` between them, which tracks the measured
optimum closely enough that the residual is under 0.01 dB.
"""
function opt_shift_threshold(K::Integer)
    k = Int(K)
    haskey(PHISTAR, k) && return PHISTAR[k]
    # the optimum drifts slowly with block size; a line in log₂K through the two known
    # points is well within the flat region of the SNR-vs-φ* curve
    lo, hi = 16, 32
    t = (log2(k) - log2(lo)) / (log2(hi) - log2(lo))
    clamp(PHISTAR[lo] + t * (PHISTAR[hi] - PHISTAR[lo]), 0.5, 1.0)
end

"""
    mx_scale_opt(M::Real; phistar=0.86, emax=2) -> Float64

The optimized shift rule's scale, from the block maximum `M` alone.

```math
E' = \\lfloor \\log_2 M \\rfloor, \\qquad
\\varphi = \\log_2 M - E', \\qquad
e_x = E' - e_{\\max} + \\mathbb{1}[\\varphi > \\varphi^*]
```

and `S = 2^{e_x}`.  With `emax = 2` (E2M1) this is MXFP4's floor rule plus a one-bit
insurance shift, fired only when the maximum would otherwise land in the top sliver of
the binade where the clamp toward code 6 is most severe.

**The exponent is still a plain E8M0 value**, so the decoder is bit-identical to
standard MXFP4 — it cannot tell which encoder ran.

```jldoctest
julia> mx_scale_opt(0.35)                  # φ ≈ 0.49 → no shift, the ordinary rule
0.0625

julia> mx_scale_opt(0.95)                  # φ ≈ 0.93 > 0.86 → shift one binade up
0.25

julia> mx_scale_opt(0.95; phistar = 1.0)   # threshold disabled ⇒ plain MXFP4
0.125
```
"""
function mx_scale_opt(M::Real; phistar::Real = 0.86, emax::Integer = 2)
    Mf = Float64(M)
    Mf > 0 || return 1.0
    l = log2(Mf)
    Ep = floor(Int, l)
    # guard the boundary: log2 can land a hair on the wrong side of a power of two
    exp2(Ep) > Mf && (Ep -= 1)
    exp2(Ep + 1) <= Mf && (Ep += 1)
    phi = l - Ep
    exp2(Ep - Int(emax) + (phi > phistar ? 1 : 0))
end

"""
    opt_shift_fires(x::AbstractVector; phistar=…, K=length(x)) -> Bool

Whether the insurance shift fires for this block — i.e. whether the maximum's mantissa
fraction exceeds `φ*`.

Measured firing rates on Gaussian data: **13.3%** of blocks at `K = 32` and **20.6%** at
`K = 16`."""
function opt_shift_fires(x::AbstractVector; K::Integer = length(x),
                         phistar::Real = opt_shift_threshold(K))
    M = maximum(abs, x)
    M == 0 && return false
    l = log2(Float64(M))
    (l - floor(l)) > phistar
end

"""
    opt_shift_rate(x::AbstractVector, K::Integer; phistar=…) -> Float64

The fraction of `K`-blocks of `x` for which the shift fires."""
function opt_shift_rate(x::AbstractVector, K::Integer;
                        phistar::Real = opt_shift_threshold(K))
    n = length(x); fired = 0; nb = 0
    for i in 1:K:n
        j = min(i + K - 1, n)
        seg = @view x[i:j]
        maximum(abs, seg) == 0 && continue
        nb += 1
        opt_shift_fires(seg; K, phistar) && (fired += 1)
    end
    nb == 0 ? 0.0 : fired / nb
end

"""    MXFP4_OPT32

MXFP4 with the **optimized shift rule** at `K = 32`.

Standard MXFP4 plus one comparison in the encoder; the decoder is bit-identical, because
`e_x` is still an ordinary E8M0 exponent.  Fires on 13.3% of Gaussian blocks and lifts
**18.80 → 19.058 dB**, against a best-of-two-powers ceiling of 19.063 dB — it captures
98% of the gain the entire power-of-two class can offer, and is within 0.005 dB of it.

Cheaper than [`MXFP4_BEST32`](@ref) for the same result: best-of-two must quantize the
block twice to compare MSE, while this needs one comparison on the maximum's mantissa."""
const MXFP4_OPT32 = BlockFormat("MXFP4opt32", 32, E2M1, E8M0, OPT_SHIFT)

"""    MXFP4_OPT16

The optimized shift rule at `K = 16`, threshold `φ* = 0.82`.  Fires on 20.6% of Gaussian
blocks and lifts **18.587 → 19.188 dB**, within 0.005 dB of the 19.193 dB class ceiling."""
const MXFP4_OPT16 = BlockFormat("MXFP4opt16", 16, E2M1, E8M0, OPT_SHIFT)
