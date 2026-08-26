# ---------------------------------------------------------------------------
# Arithmetic on block-scaled operands.  The whole design rests on one
# factorisation:  y = Sa·Sb · Σ eᵃᵢ eᵇᵢ  — the scales never enter the inner loop.
# ---------------------------------------------------------------------------

"""
    BlockDotResult

The outcome of one block dot product, with the evidence that the arithmetic was exact.

# Fields
- `value::Float64` — the computed dot product.
- `exact::Float64` — the true dot product of the *original* (unquantized) vectors.
- `core_sum::Float64` — `Σ eᵃᵢeᵇᵢ`, the integer-ish core the hardware actually adds.
- `scale_product::Float64` — `Sa·Sb`, applied once as a single exponent add.
- `products_exact::Bool` — whether every element product landed on the accumulator
  grid without rounding.  For E2M1 operands into an FP32 accumulator this is always
  true, and it is the reason the format's 25% pointwise crudeness coexists with
  usable end-to-end computation.
- `tree_bound::Float64` — the provable bound `K · max|e|²` on the core sum, which is
  what lets the adder tree be plain narrow fixed-point with no rounding anywhere.
"""
struct BlockDotResult
    value::Float64
    exact::Float64
    core_sum::Float64
    scale_product::Float64
    products_exact::Bool
    tree_bound::Float64
end

relerror(r::BlockDotResult) = r.exact == 0 ? abs(r.value) : abs(r.value - r.exact) / abs(r.exact)

"""
    block_dot(bf::BlockFormat, a, b; accumulate=FP32) -> BlockDotResult

Dot product of two block-quantized vectors, computed the way MX hardware does it.

Both operands are cut into matching blocks of `bf.K` along the shared reduction axis.
Within a block the scales factor straight out:

```math
y = \\sum_i (S_a e^a_i)(S_b e^b_i) = S_a S_b \\sum_i e^a_i e^b_i
```

so the inner loop runs on tiny exact multiplies feeding an adder tree that is
*provably* free of rounding, and the two scale bytes add **once** per block.

```jldoctest
julia> a = [0.11, -0.35, 0.02, 0.24]; b = [1.3, 0.7, -2.1, 0.44];

julia> r = block_dot(BlockFormat("mx4", 4, E2M1, E8M0, MX_FLOOR_POW2), a, b);

julia> r.core_sum, r.scale_product, r.value
(-1.0, 0.03125, -0.03125)

julia> r.products_exact
true
```
"""
function block_dot(bf::BlockFormat, a::AbstractVector, b::AbstractVector;
                   accumulate::FloatFormat = FP32)
    length(a) == length(b) || throw(DimensionMismatch("block_dot: length mismatch"))
    n = length(a)
    total = 0.0
    core_total = 0.0
    scale_prod_last = 1.0
    all_exact = true
    emax_elem = maxfinite(bf.elem)

    for i in 1:bf.K:n
        j = min(i + bf.K - 1, n)
        qa = quantize_block(bf, @view a[i:j])
        qb = quantize_block(bf, @view b[i:j])
        core = 0.0
        for k in eachindex(qa.elements)
            p = qa.elements[k] * qb.elements[k]
            quantize(accumulate, p) == p || (all_exact = false)
            core += p
        end
        Sp = qa.scale * qb.scale
        core_total += core
        scale_prod_last = Sp
        total = quantize(accumulate, total + quantize(accumulate, core * Sp))
    end

    exact = dot(collect(Float64, a), collect(Float64, b))
    BlockDotResult(total, exact, core_total, scale_prod_last, all_exact,
                   bf.K * emax_elem^2)
end

"""
    core_sum_bound(bf::BlockFormat) -> Float64

The provable ceiling on `|Σᵢ eᵃᵢeᵇᵢ|` within one block: `K · max(elem)²`.

For MXFP4 that is `32 × 6 × 6 = 1152` — a value that fits in a dozen integer bits
plus two fraction bits, which is precisely why the block's adder tree can be plain
narrow fixed point with **no rounding anywhere**."""
core_sum_bound(bf::BlockFormat) = bf.K * maxfinite(bf.elem)^2

"""
    block_gemm(bf::BlockFormat, A, B; accumulate=FP32) -> Matrix{Float64}

Matrix product with both operands block-quantized along the shared `k` axis:

```math
C_{jl} = \\sum_\\beta S^A_{j\\beta} S^B_{\\beta l} \\sum_{i} e^A_{j\\beta i} e^B_{\\beta l i}
```

Weights are quantized once (rows of `A`), activations per fresh block (columns of
`B`).  The scale fixups add `n/K` multiplies to `n` MACs — a 3.1% arithmetic overhead
at `K = 32` for the entire two-level machinery."""
function block_gemm(bf::BlockFormat, A::AbstractMatrix, B::AbstractMatrix;
                    accumulate::FloatFormat = FP32)
    m, n = size(A)
    n2, p = size(B)
    n == n2 || throw(DimensionMismatch("block_gemm: inner dimensions $n vs $n2"))
    C = zeros(Float64, m, p)
    for j in 1:m, l in 1:p
        C[j, l] = block_dot(bf, @view(A[j, :]), @view(B[:, l]); accumulate).value
    end
    C
end

"""
    scale_fixup_overhead(bf::BlockFormat) -> Float64

Fraction of extra arithmetic the block scales cost: one fixup per `K` MACs, i.e.
`1/K`.  0.031 for MXFP4."""
scale_fixup_overhead(bf::BlockFormat) = 1 / bf.K

"""
    storage_bytes(bf::BlockFormat, nvalues) -> Float64

Bytes needed to store `nvalues` values, scales included.

```jldoctest
julia> storage_bytes(MXFP4, 4096*4096) / 1e6      # a transformer's workhorse layer
8.912896
```
"""
storage_bytes(bf::BlockFormat, nvalues::Integer) = nvalues * bits_per_element(bf) / 8

"""
    storage_bytes(f::FloatFormat, nvalues) -> Float64"""
storage_bytes(f::FloatFormat, nvalues::Integer) = nvalues * nbits(f) / 8

"""
    compression_ratio(bf::BlockFormat, f::FloatFormat = FP32) -> Float64

How many times smaller `bf` is than `f`.  7.53× for MXFP4 against FP32, 3.76× against
FP16 — and since LLM token generation is memory-bound, that compression is to first
order a proportional decode speedup, before the 144×-smaller multipliers are even
counted."""
compression_ratio(bf::BlockFormat, f::FloatFormat = FP32) =
    nbits(f) / bits_per_element(bf)
