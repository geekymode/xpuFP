# API — Arithmetic

```@autodocs
Modules = [xpuFP]
Pages   = ["arith/datapath.jl", "arith/floatarith.jl",
           "arith/fixedarith.jl", "arith/summation.jl",
           "arith/algorithms.jl"]
Order   = [:type, :function, :constant]
```

## Accumulators

An accumulator is the worst case for carry propagation — `acc += x` with a loop-carried
dependence, and a register that widens as the sum grows. These run the conventional and
the carry-free schedules on the same terms and report what each cost; see
[Examples](@ref) for the measured comparison.

```@autodocs
Modules = [xpuFP]
Pages   = ["arith/accumulator.jl"]
Order   = [:type, :function, :constant]
```
