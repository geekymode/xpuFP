# API — Analysis

The measuring stick: a number format treated as a noisy channel, and the metrics that
decide between formats.

```@autodocs
Modules = [xpuFP]
Pages   = ["analysis/snr.jl"]
Order   = [:type, :function, :constant]
```

## Quick simulation

The sweep path: the same SNR as [`measure_snr`](@ref), streamed through a tabulated
element grid so a design-space question costs a fraction of a second rather than a
coffee break. See [Examples](@ref) for it applied end to end.

```@autodocs
Modules = [xpuFP]
Pages   = ["analysis/quicksim.jl"]
Order   = [:type, :function, :constant]
```

## Simulation and analytic estimates

A measured number and a derived number are worth more together than either alone:
agreement validates both, disagreement localises the error. [`simulate_snr`](@ref)
measures with a confidence interval, [`estimate_element_snr`](@ref) derives by
order-statistic quadrature, and each is a check on the other.

```@autodocs
Modules = [xpuFP]
Pages   = ["analysis/snrsim.jl"]
Order   = [:type, :function, :constant]
```

## Cost model — memory, cells, depth

Depth is what redundancy buys; these price the two currencies that decide whether the
trade is worth taking. The model is stated in full in the source header of
`analysis/costmodel.jl` and summarised in [`CostBudget`](@ref) — ratios between rows are
the intended reading, absolute numbers are first-order.

```@autodocs
Modules = [xpuFP]
Pages   = ["analysis/costmodel.jl"]
Order   = [:type, :function, :constant]
```

## MXFP4 and RR4

Once the weights are MXFP4 the operands are 4–14 bits wide, and the general-width
conclusions of the redundant-arithmetic chapters no longer hold. These derive the widths
from the format and price the alternatives on them.

```@autodocs
Modules = [xpuFP]
Pages   = ["analysis/mxfp4cost.jl"]
Order   = [:type, :function, :constant]
```

## Cycle model

Depth divided by a clock budget. Nothing here is meaningful without `levels_per_cycle`
— see [The cost model: how computations map to gate levels](@ref) §10 for why that
parameter decides whether a depth advantage becomes cycles or merely slack.

```@autodocs
Modules = [xpuFP]
Pages   = ["analysis/cyclemodel.jl"]
Order   = [:type, :function, :constant]
```

## Pipelining and the single-cycle multiply

Latency is what §10 computes; throughput is what a pipeline sells. These cut a depth into
stages, price the registers that buy the initiation interval, and walk the ladder of
techniques that shrink a multiply toward one cycle — ending at the conclusion that
narrowing the format beats every one of them. See
[The cost model: how computations map to gate levels](@ref) §11 and §12.

```@autodocs
Modules = [xpuFP]
Pages   = ["analysis/pipeline.jl"]
Order   = [:type, :function, :constant]
```

## High-level synthesis

What an HLS flow decides for you, and what it leaves in your hands: the `ap_int` widths
derived from a block format, the accumulator sizing that decides whether the inner loop
reaches `II = 1`, and the energy split that says whether any arithmetic choice matters.
See [HLS: which schemes survive a synthesis flow](@ref).

```@autodocs
Modules = [xpuFP]
Pages   = ["analysis/hls.jl"]
Order   = [:type, :function, :constant]
```

## Benchmark harness

One deliberate bias runs through this layer: no single number decides a format. Every
runner reports several metrics side by side.

```@autodocs
Modules = [xpuFP]
Pages   = ["analysis/benchmark.jl"]
Order   = [:type, :function, :constant]
```
