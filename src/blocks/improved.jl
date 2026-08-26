# ---------------------------------------------------------------------------
# Improved block-scaled FP4 schemes.
#
# The constraint that makes these interesting is what they must NOT change.  MXFP4's
# computational structure rests on four facts, and every scheme here preserves all of
# them:
#
#   1. every E2M1 × E2M1 product is exact in the wide accumulator;
#   2. the per-block adder tree is exact (bounded by K·6²), so no rounding in flight;
#   3. the scale factors out of the inner loop entirely, y = S_a S_b Σ eᵢ e'ᵢ;
#   4. the element grid stays E2M1, so FP4-native tensor cores can multiply natively.
#
# What is left to improve is therefore only the *scale*: which value it takes, and how
# it is chosen.  That is an encoder-side decision, and — for the power-of-two rules —
# it costs no hardware at all.
# ---------------------------------------------------------------------------

"""    MXFP4_BEST32

MXFP4 with the **best-of-two-powers** scale rule at `K = 32`.

Identical wire format, identical decoder, identical MAC path, and the scale is still
E8M0 so applying it remains an **exponent add**.  The only change is at encode: try both
`2^e` and `2^(e+1)` and keep whichever gives lower block MSE — one comparison against a
precomputed threshold, no multiplies.

Measured **19.05 dB** on Gaussian data against MXFP4's 18.80: a free quarter-decibel."""
const MXFP4_BEST32 = BlockFormat("MXFP4best32", 32, E2M1, E8M0, BEST_POW2)

"""    MXFP4_BEST16

The same best-of-two-powers rule at `K = 16`.  Measured **19.18 dB** Gaussian, and it
gains far more on heavy tails (18.42 dB on Student-t₃ against MXFP4-32's 16.82), because
a shorter block confines each outlier to fewer neighbours."""
const MXFP4_BEST16 = BlockFormat("MXFP4best16", 16, E2M1, E8M0, BEST_POW2)

"""    NVFP4_BEST16

NVFP4 with an **MSE-optimal** scale instead of max-anchoring.

Anchoring the block maximum at the top code is a no-overflow heuristic, not the error
minimiser: deliberately overloading slightly — accepting a small clamp on the largest
element — buys finer resolution for all the others.  The search happens entirely inside
the encoder; wire format, decoder and MAC path are untouched.

Measured **21.60 dB** against shipped NVFP4's 20.44."""
const NVFP4_BEST16 = BlockFormat("NVFP4best16", 16, E2M1, E4M3, MSE_OPTIMAL)

"""    XPFP4_32

**The best operating point in this package**: `K = 32` blocks, E2M1 elements, an E4M3
scale placed by MSE-optimal search.

Measured **20.76 dB** on Gaussian data at **4.25 bits per value** — better SNR *and*
fewer bits than shipped NVFP4 (20.44 dB at 4.5 bits).  It reaches NVFP4's accuracy class
while paying MXFP4's storage overhead, by spending the finer scale format on twice as
many elements.

The cost, relative to MXFP4, is that the scale is a genuine mini-float, so the per-block
fixup is a small multiply rather than an exponent add.  All four structural properties
above survive, and the fixup is amortised over 32 MACs rather than 16 — halving NVFP4's
scale-application rate."""
const XPFP4_32 = BlockFormat("XPFP4-32", 32, E2M1, E4M3, MSE_OPTIMAL)

"""    XPFP4_16

Alias for [`NVFP4_BEST16`](@ref) — the `K = 16` sibling of [`XPFP4_32`](@ref) *is*
NVFP4 with an MSE-optimal scale; they are the same `(K, element, scale, rule)` tuple, so
the package exposes one object under both names rather than two formats that would
silently measure identically."""
const XPFP4_16 = NVFP4_BEST16

"""    IMPROVED_BLOCK_FORMATS

The improved schemes, paired with the two shipped baselines for comparison."""
const IMPROVED_BLOCK_FORMATS =
    (MXFP4, NVFP4, MXFP4_OPT32, MXFP4_OPT16, MXFP4_BEST32, MXFP4_BEST16,
     NVFP4_BEST16, XPFP4_32)

# ---------------------------------------------------------------------------
# Hadamard rotation — orthogonal to every rung of the ladder.
# ---------------------------------------------------------------------------

"""
    hadamard(K::Integer) -> Matrix{Float64}

The orthonormal Sylvester–Hadamard matrix of order `K` (a power of two), scaled by
`1/√K` so that `H' * H == I`.

Applying it costs only additions and subtractions — no multiplies — which is why it
composes with block quantization at negligible hardware cost."""
function hadamard(K::Integer)
    k = Int(K)
    (k > 0 && (k & (k - 1)) == 0) ||
        throw(ArgumentError("hadamard: order must be a power of two, got $(k)"))
    H = ones(Float64, 1, 1)
    while size(H, 1) < k
        H = [H H; H -H]
    end
    H ./ sqrt(k)
end

"""
    RotatedBlockFormat

A [`BlockFormat`](@ref) preceded by an orthonormal Hadamard rotation of each block.

The rotation is **orthogonal to the whole design ladder**: it composes with any scale
rule, costs only adds and subtracts, and is exactly invertible.

On i.i.d. Gaussian data it changes nothing — a rotation of an isotropic distribution is
the same distribution — so the SNR is unmoved. Its value is entirely on *real* tensors
with outliers: rotating smears a lone large value across all `K` coordinates, which is
precisely the failure mode block scaling has, since one outlier otherwise captures the
shared scale and degrades its 31 neighbours.
"""
struct RotatedBlockFormat
    base::BlockFormat
    name::String
end

"""
    rotated(bf::BlockFormat; name=…) -> RotatedBlockFormat

Wrap a block format in a Hadamard rotation.  Requires a power-of-two block size."""
function rotated(bf::BlockFormat; name::AbstractString = "H·" * bf.name)
    (bf.K > 0 && (bf.K & (bf.K - 1)) == 0) ||
        throw(ArgumentError("rotated: block size must be a power of two, got $(bf.K)"))
    RotatedBlockFormat(bf, String(name))
end

_fmtname(f::RotatedBlockFormat) = f.name
bits_per_element(f::RotatedBlockFormat) = bits_per_element(f.base)
_storage_bits(f::RotatedBlockFormat) = bits_per_element(f.base)

"""
    reconstruct(f::RotatedBlockFormat, x) -> Vector{Float64}

Rotate each block, quantize, decode, and rotate back.  Blocks shorter than `K` at the
tail are passed through unrotated, since the rotation is only defined at full length."""
function reconstruct(f::RotatedBlockFormat, x::AbstractVector)
    bf = f.base
    K = bf.K
    H = hadamard(K)
    xs = collect(Float64, x)
    out = similar(xs)
    n = length(xs)
    for i in 1:K:n
        j = min(i + K - 1, n)
        seg = @view xs[i:j]
        if length(seg) == K
            y = H * collect(seg)
            qb = quantize_block(bf, y)
            out[i:j] .= H' * qb.values
        else
            out[i:j] .= quantize_block(bf, seg).values
        end
    end
    out
end

quantize_all(f::RotatedBlockFormat, x::AbstractVector) = reconstruct(f, x)

Base.show(io::IO, f::RotatedBlockFormat) =
    print(io, "RotatedBlockFormat(", f.name, ", K=", f.base.K, ")")

# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------

"""
    BlockSchemeReport

One row of a scheme comparison: name, storage cost, how the scale is applied, and the
measured SNR on each supplied dataset."""
struct BlockSchemeReport
    name::String
    K::Int
    scale_format::String
    bits::Float64
    exponent_only::Bool
    snr::Vector{Float64}
    zeroed::Vector{Float64}
    labels::Vector{String}
end

"""
    compare_block_schemes(datasets; formats=IMPROVED_BLOCK_FORMATS, rotate=false)
        -> Vector{BlockSchemeReport}

Measure a set of block schemes on a set of datasets, reporting SNR alongside the two
costs that decide deployability: **bits per value**, and whether the scale can be
applied by an exponent add.

`datasets` is a collection of `label => vector` pairs.

```julia
using Random
rng = MersenneTwister(1)
compare_block_schemes(["gaussian" => randn(rng, 50_000)])
```
"""
function compare_block_schemes(datasets;
                               formats = IMPROVED_BLOCK_FORMATS,
                               rotate::Bool = false)
    labels = String[first(d) for d in datasets]
    out = BlockSchemeReport[]
    for bf in formats
        f = rotate ? rotated(bf) : bf
        snrs = [measure_snr(f, last(d)) for d in datasets]
        # SNR is energy-weighted and therefore blind to annihilated small coordinates,
        # so carry the acceptance metric alongside it
        zs = [begin
                  v = last(d)
                  xh = quantize_all(f, v)
                  count(i -> xh[i] == 0 && v[i] != 0, eachindex(v)) / length(v)
              end for d in datasets]
        base = bf isa RotatedBlockFormat ? bf.base : bf
        push!(out, BlockSchemeReport(_fmtname(f), base.K, base.scale.name,
                                     bits_per_element(base), base.scale === E8M0,
                                     snrs, zs, labels))
    end
    out
end

"""
    print_scheme_table(rows::Vector{BlockSchemeReport}; io=stdout)

Render [`compare_block_schemes`](@ref) output as a table, sorted by the first dataset's
SNR."""
function print_scheme_table(rows::Vector{BlockSchemeReport}; io = stdout)
    isempty(rows) && return
    labs = rows[1].labels
    @printf(io, "%-14s %4s %7s %7s %9s", "scheme", "K", "scale", "b/val", "scale via")
    for l in labs
        @printf(io, " %9s", first(l, 9))
    end
    for l in labs
        @printf(io, " %8s", "%0:" * first(l, 5))
    end
    println(io)
    println(io, "-"^(48 + 10 * length(labs) + 9 * length(labs)))
    for r in sort(rows; by = x -> -x.snr[1])
        @printf(io, "%-14s %4d %7s %7.3f %9s", r.name, r.K, r.scale_format, r.bits,
                r.exponent_only ? "exp add" : "multiply")
        for s in r.snr
            @printf(io, " %9.2f", s)
        end
        for z in r.zeroed
            @printf(io, " %7.2f%%", 100z)
        end
        println(io)
    end
    println(io, "  SNR is energy-weighted; \"%0\" is the share of nonzero inputs stored as")
    println(io, "  exact zero — the acceptance metric SNR cannot see.")
end
