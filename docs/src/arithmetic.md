# Arithmetic

IEEE 754 demands **correct rounding**: every basic operation must behave as if
computed exactly and then rounded once to the nearest representable value. This
package takes that literally — each operation returns a trace of every pipeline stage.

!!! note "Why Float64 intermediates are sound"
    Every format here has at most 23 mantissa bits, so `Float64` satisfies the
    classical no-double-rounding condition ``53 \ge 2p + 2`` for all of them.
    Computing in `Float64` and rounding once therefore yields exactly the
    IEEE-mandated result.

## Datapath traces

```@docs
DatapathTrace
FPParts
unpack
Stage
relerror
rounding_ulps
```

## The four operations

```@docs
fpadd
fpmul
fpdiv
fpfma
fpmul_then_add
```

Addition must first make the exponents agree, and **the shift step is where
information is lost**: bits pushed off the right end survive only as guard/round/sticky
bits, and if the exponents differ by more than the significand width the small operand
vanishes entirely.

Multiplication needs no alignment. Because normalized significands lie in ``[1,2)``,
their product lies in ``[1,4)`` — so the normalize stage needs at most one right shift,
never more.

## Fixed-point arithmetic

```@docs
fxadd
fxmul
fxdiv
fxdot
```

## Summation schedules

The format sets the per-operation error; the **schedule** sets how those errors
accumulate.

```@docs
seq_sum
tree_sum
kahan_sum
fp_dot
```

## Stagnation

```@docs
stagnation_trace
stagnation_threshold
```

Sum eight copies of ``0.5`` with an FP4 accumulator: the first four additions are
exact, then ``2 + 0.5 = 2.5`` lands on a midpoint, ties-to-even returns it to ``2``,
and the accumulator is **stuck there forever**. The general law: once the accumulator
reaches ``\varepsilon^{-1}`` times the addend, addition stops progressing.

This is why nobody accumulates in FP4.
