# ---------------------------------------------------------------------------
# The output-stationary systolic array.
#
# The array is not an approximation of matrix multiplication; it is the same sum,
# scheduled in space-time.
# ---------------------------------------------------------------------------

"""
    SystolicRun

A complete output-stationary systolic multiply, cycle by cycle.

# Fields
- `A`, `B`, `C` — operands and the computed product.
- `snapshots::Vector{Matrix{Float64}}` — the accumulator grid after each cycle.
- `activity::Vector{Int}` — how many PEs actually multiplied on each cycle.
- `cycles::Int` — `K + M + P − 2`.
- `utilization::Float64` — `MPK / (MP·T) = K/(K+M+P−2)`.
- `exact::Bool` — whether the array's result equals the textbook triple loop.
"""
struct SystolicRun
    A::Matrix{Float64}
    B::Matrix{Float64}
    C::Matrix{Float64}
    snapshots::Vector{Matrix{Float64}}
    activity::Vector{Int}
    cycles::Int
    utilization::Float64
    exact::Bool
end

"""
    systolic_run(A, B) -> SystolicRun

Simulate `C = A·B` on an output-stationary systolic array, one PE per output entry.

Processing element `(i,j)` owns accumulator `C[i,j]`; each cycle it multiplies the `a`
arriving from its left by the `b` arriving from above, adds the product in, and passes
`a` rightward and `b` downward.  The only cleverness is the **skew**: row `i` of `A`
enters `i` cycles late and column `j` of `B` enters `j` cycles late, so at cycle `t`,
PE `(i,j)` receives exactly the matched pair

```math
a_{i,k} \\text{ and } b_{k,j}, \\qquad k = t - i - j
```

and the right operands meet at the right place with **no addressing logic at all** —
position in space plus delay in time *is* the index arithmetic.

```jldoctest
julia> r = systolic_run([1.0 2.0; 3.0 4.0], [5.0 6.0; 7.0 8.0]);

julia> r.C
2×2 Matrix{Float64}:
 19.0  22.0
 43.0  50.0

julia> r.cycles, r.activity
(4, [1, 3, 3, 1])
```
"""
function systolic_run(A::AbstractMatrix, B::AbstractMatrix)
    Af = Matrix{Float64}(A); Bf = Matrix{Float64}(B)
    M, K = size(Af); K2, P = size(Bf)
    K == K2 || throw(DimensionMismatch("systolic_run: inner dimensions $K vs $K2"))
    T = K + M + P - 2
    C = zeros(Float64, M, P)
    snaps = Matrix{Float64}[]
    act = Int[]
    for t in 0:T-1
        n = 0
        for i in 0:M-1, j in 0:P-1
            k = t - i - j
            if 0 <= k < K
                C[i+1, j+1] += Af[i+1, k+1] * Bf[k+1, j+1]
                n += 1
            end
        end
        push!(snaps, copy(C)); push!(act, n)
    end
    SystolicRun(Af, Bf, C, snaps, act, T, (M * P * K) / (M * P * T),
                C ≈ Af * Bf)
end

"""
    systolic_cycles(M, K, P) -> Int

The cycle count `T = K + M + P − 2`.  PE `(i,j)` cannot begin until data reaches it:
the head of row `i` takes `i` hops from the left edge and the head of column `j` takes
`j` hops from the top, so its working window is `[i+j, i+j+K−1]`, and the whole
computation ends when the farthest PE finishes.

```jldoctest
julia> systolic_cycles(2, 2, 2), systolic_cycles(4, 3, 5)
(4, 10)
```
"""
systolic_cycles(M::Integer, K::Integer, P::Integer) = Int(K + M + P - 2)

"""
    systolic_utilization(M, K, P) -> Float64

`K / (K + M + P − 2)` — the fraction of PE-cycles doing useful work.

Utilization → 1 as `K` grows, which is precisely why real deployments keep the
reduction depth long: marching over many `K = 32` MX blocks back-to-back keeps the
array in its dense steady state instead of forever ramping."""
systolic_utilization(M::Integer, K::Integer, P::Integer) =
    K / systolic_cycles(M, K, P)

"""
    systolic_start_cycle(i, j) -> Int

The cycle PE `(i,j)` begins work: its taxicab distance `i + j` from the corner.  Equal
start times form the diagonal wavefronts that sweep the array."""
systolic_start_cycle(i::Integer, j::Integer) = Int(i + j)
