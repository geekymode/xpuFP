# Hardware models

## The systolic array

A GEMM engine is the FMA datapath *tiled*: many MAC units arranged so operands are
fetched once and reused in flight.

```@docs
systolic_run
SystolicRun
systolic_cycles
systolic_utilization
systolic_start_cycle
```

Processing element ``(i,j)`` owns accumulator ``C_{ij}``; each cycle it multiplies the
``a`` arriving from its left by the ``b`` arriving from above, adds the product in, and
passes ``a`` rightward and ``b`` downward. The only cleverness is the **skew**: row
``i`` enters ``i`` cycles late and column ``j`` enters ``j`` cycles late, so at cycle
``t`` the PE receives exactly the matched pair with ``k = t - i - j``.

The right operands meet at the right place with **no addressing logic at all** —
position in space plus delay in time *is* the index arithmetic.

```jldoctest
julia> using xpuFP

julia> r = systolic_run([1.0 2.0; 3.0 4.0], [5.0 6.0; 7.0 8.0]);

julia> r.C
2×2 Matrix{Float64}:
 19.0  22.0
 43.0  50.0

julia> r.cycles, r.activity, r.exact
(4, [1, 3, 3, 1], true)
```

!!! note "The array is not an approximation"
    Substituting ``k = t-i-j`` over PE ``(i,j)``'s window turns its accumulated total
    into ``\sum_k A_{ik}B_{kj}`` — each product formed exactly once and accumulated in
    ascending ``k``, term for term and in the same order as the textbook triple loop.
    It is the same sum, scheduled in space-time.

```@example hw
using xpuFP, CairoMakie # hide
plot_systolic_activity(4, 3, 5)
```

Utilisation is ``K/(K+M+P-2)``, which ``\to 1`` as ``K`` grows. That is the quantitative
reason real deployments keep the reduction depth long — marching over many ``K = 32`` MX
blocks back-to-back keeps the array in its dense steady state instead of forever ramping.
