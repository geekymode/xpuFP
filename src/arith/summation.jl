# ---------------------------------------------------------------------------
# Summation schedules.  The format sets the per-operation error; the *schedule*
# sets how those errors accumulate — sequential is a random walk, tree is nearly
# flat, and compensated summation is nearly exact.
# ---------------------------------------------------------------------------

"""
    seq_sum(f::FloatFormat, x) -> Float64

Sequential (naive) summation in format `f`: one rounding per element, so the injected
noise powers sum and the SNR decays like `−10log₁₀N` — a random walk losing 3 dB per
doubling of the length."""
function seq_sum(f::FloatFormat, x::AbstractVector)
    acc = 0.0
    @inbounds for v in x
        acc = fadd(f, acc, v)
    end
    acc
end

"""
    tree_sum(f::FloatFormat, x) -> Float64

Pairwise (tree) summation: each datum passes through only `⌈log₂N⌉` roundings, so the
error law becomes `≈ SNRₑ − 10log₁₀log₂N` — nearly flat.  The decay in [`seq_sum`](@ref)
belongs to the schedule, not to the format."""
function tree_sum(f::FloatFormat, x::AbstractVector)
    n = length(x)
    n == 0 && return 0.0
    n == 1 && return quantize(f, x[1])
    buf = Float64[quantize(f, v) for v in x]
    while length(buf) > 1
        m = length(buf)
        out = Vector{Float64}(undef, cld(m, 2))
        @inbounds for i in 1:2:m
            out[cld(i, 2)] = i == m ? buf[i] : fadd(f, buf[i], buf[i+1])
        end
        buf = out
    end
    buf[1]
end

"""
    kahan_sum(f::FloatFormat, x) -> Float64

Kahan compensated summation: carry the rounding error of each addition in a second
variable and feed it back.  Costs four operations per element and buys back most of
the lost bits."""
function kahan_sum(f::FloatFormat, x::AbstractVector)
    s = 0.0; c = 0.0
    @inbounds for v in x
        y = quantize(f, quantize(f, v) - c)
        t = fadd(f, s, y)
        c = quantize(f, quantize(f, t - s) - y)
        s = t
    end
    s
end

"""
    fp_dot(f::FloatFormat, x, y; schedule=:seq, accumulate=f) -> Float64

Dot product computed entirely in format `f`, with the summation `schedule` one of
`:seq`, `:tree`, or `:kahan`.

`accumulate` selects the accumulator format independently of the operand format —
pass `FP32` with `f = E2M1` to model the deployed pattern (4-bit operands, wide
accumulator), which is exact because any product of two E2M1 values has at most four
significand bits and lands on the FP32 grid without rounding.

```jldoctest
julia> u = [1.5, 3.0, 0.5]; v = [1.5, -2.0, 6.0];

julia> fp_dot(E2M1, u, v; accumulate=FP32)      # deployed: exact
-0.75

julia> fp_dot(E2M1, u, v)                       # all-FP4: 33% error
-1.0
```
"""
function fp_dot(f::FloatFormat, x::AbstractVector, y::AbstractVector;
                schedule::Symbol = :seq, accumulate::FloatFormat = f)
    length(x) == length(y) || throw(DimensionMismatch("fp_dot: length mismatch"))
    prods = [quantize(accumulate, quantize(f, x[i]) * quantize(f, y[i])) for i in eachindex(x)]
    schedule === :seq   ? seq_sum(accumulate, prods) :
    schedule === :tree  ? tree_sum(accumulate, prods) :
    schedule === :kahan ? kahan_sum(accumulate, prods) :
    throw(ArgumentError("schedule must be :seq, :tree or :kahan"))
end

"""
    stagnation_trace(f::FloatFormat, addend, nsteps) -> Vector{Float64}

Accumulate `addend` into a running total `nsteps` times, entirely within format `f`,
returning the accumulator after every step.

The classic FP4 demonstration: eight additions of `0.5` in E2M1 give `0.5, 1, 1.5, 2,
2, 2, 2, 2` — four exact hops, then permanent stagnation, because `2 + 0.5 = 2.5`
lands on a midpoint and ties-to-even sends it straight back to `2`.  The general law:
once the accumulator reaches `ε⁻¹` times the addend, addition stops progressing.

```jldoctest
julia> stagnation_trace(E2M1, 0.5, 8)
8-element Vector{Float64}:
 0.5
 1.0
 1.5
 2.0
 2.0
 2.0
 2.0
 2.0
```
"""
function stagnation_trace(f::FloatFormat, addend::Real, nsteps::Integer)
    acc = 0.0
    out = Vector{Float64}(undef, nsteps)
    for i in 1:nsteps
        acc = fadd(f, acc, addend)
        out[i] = acc
    end
    out
end

"""
    stagnation_threshold(f::FloatFormat, addend) -> Float64

The accumulator value at which adding `addend` becomes a no-op: `addend / ε`, where
`ε` is the format's machine epsilon.  Above this the addend is below half an ulp and
is absorbed without trace."""
stagnation_threshold(f::FloatFormat, addend::Real) = abs(Float64(addend)) / machine_eps(f)
