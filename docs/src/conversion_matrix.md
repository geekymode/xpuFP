# Universal conversion

Every format family in the package converts to every other through one surface. The
pipeline is always the same three steps: decode the source to an exact real, round once
onto the target's grid, store. Decoding is exact in every scheme, so **all the loss
lives in the middle step**.

```@docs
AnyFormat
realvalue
represent
convert_format
```

```jldoctest
julia> using xpuFP

julia> convert_format(E2M1, FP32, 2.25)
2.0

julia> digit_string(convert_format(RR4, FP32, 6.5))
"12.2"

julia> convert_format(FP32, RR4, to_digits(RR4, 6.5))
6.5

julia> convert_format(FixedFormat(7, 8), FP32, -3.65)
-3.6484375
```

## Auditing a conversion

```@docs
ConversionReport
conversion_report
conversion_matrix
```

`conversion_report` answers not just *what came out* but *which edge of the target it
met* — overflow, underflow to zero, or a landing in the subnormal ramp.

!!! tip "Digit systems are lossless in both directions"
    Converting between `BINARY`, `CSD`, `RADIX4`, `RR4` and `RR4_MAX` is a
    **re-spelling**, not an approximation: `conversion_report(f, BINARY, x).exact` is
    `true` for every one of them. That is the structural difference from a float or
    block format, and it is why the redundant systems sit inside arithmetic units while
    minimal formats own everything at rest.

## Layouts across all families

```@docs
plot_digit_layout
plot_layout_comparison
```

```@example conv2
using xpuFP, CairoMakie # hide
plot_digit_layout(RR4; value = 16.1656375)
```

The value is held **exactly**: `16.1656375` as a `Float64` is a dyadic rational, and
since ``4 = 2^2`` every dyadic rational has a finite radix-4 expansion. The 27 digits are
not an approximation — they are what that float actually is.

```@example conv2
using xpuFP, CairoMakie # hide
plot_layout_comparison((BINARY, CSD, RADIX4, RR4, RR4_MAX), 231)
```

Read the nonzero counts: binary spends 6, CSD 4 (hence 3 adders instead of 5), and the
maximally redundant radix-4 alphabet needs no rewriting at all because `3` is already a
legal digit.

```@example conv2
using xpuFP, CairoMakie # hide
plot_bit_layout(FixedFormat(3, 4); value = 6.6875)
```

```@example conv2
using xpuFP, CairoMakie # hide
plot_bit_layout(INT8; value = -74)
```

The top cell carries the **negative** weight ``-2^{b-1}``, which is what makes
two's complement signed with no correction step anywhere — the same fact Booth's
recoding theorem exploits.
