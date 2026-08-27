# Examples

Seven runnable scripts live in `examples/`. Each is a self-contained study that prints
its findings as tables and — for the first three — writes figures to
`examples/figures/`. They are the package demonstrating itself: every number below was
produced by the script above it, not transcribed from the report.

Run them from the package root:

```console
$ julia --project=. examples/01_compare_schemes.jl
$ julia --project=. examples/02_outlier_study.jl
$ julia --project=. examples/03_dot_and_profile.jl
$ julia --project=. examples/04_quick_snr.jl
$ julia --project=. examples/05_accumulator.jl
$ julia --project=. examples/06_cost_savings.jl
$ julia --project=. examples/07_mxfp4_rr4.jl
$ julia --project=. examples/08_pipeline.jl
```

!!! note "First run is slow, and it is not the arithmetic"
    The first `using xpuFP` in a fresh environment precompiles the package and its
    dependencies (CairoMakie is the bulk of it) — a few minutes. After that the cache
    makes it a few seconds, until you edit `src/`. The measurements themselves are fast:
    example 4 does its entire seven-scheme, three-sweep, seven-distribution study in
    about four seconds of actual work.

Every script fixes its RNG seed, so re-running reproduces these tables exactly on the
same Julia version. Timing columns will differ with your machine.

| | Script | Answers | Figures | Runtime |
|---|:---|:---|---:|---:|
| 1 | `01_compare_schemes.jl` | Which FP4 scheme wins, on which data, at what cost | 4 | ~30 s |
| 2 | `02_outlier_study.jl` | What one outlier does to a block, and what rotation does about it | 4 | ~20 s |
| 3 | `03_dot_and_profile.jl` | Whether dot-product SNR decays with length (it does not) | 3 | ~20 s |
| 4 | `04_quick_snr.jl` | Sweeping the FP4 design space — the fast path | 0 | ~4 s |
| 5 | `05_accumulator.jl` | Whether carry-free RR4 addition actually scales | 0 | ~10 s |
| 6 | `06_cost_savings.jl` | What Booth, CSD and RR4 cost and save in memory, cells and depth | 0 | ~8 s |
| 7 | `07_mxfp4_rr4.jl` | With weights already MXFP4, what is RR4 worth? (mostly: nothing) | 0 | ~8 s |
| 8 | `08_pipeline.jl` | What pipeline registers buy, and how close a multiply gets to one cycle | 0 | ~5 s |

---

## 1 — Comparing every block-scaled scheme

```console
$ julia --project=. examples/01_compare_schemes.jl
```

Benchmarks seven schemes across the seven-distribution battery
([`test_distributions`](@ref)), three metrics deep, then repeats the whole thing under
Hadamard rotation. The point of three metrics is that no single number decides a
format: SNR is energy-weighted and blind to annihilated coordinates, the zeroed
fraction says nothing about the survivors' accuracy, and the median block hides the
unlucky blocks that actually break a network.

```
==============================================================================
  BLOCK-SCALED FP4 SCHEMES — 7 schemes × 7 distributions
==============================================================================

metric: snr
scheme            b/val    gaussian     laplace  student-t3   lognormal     outlier      sparse     uniform
-----------------------------------------------------------------------------------------------------------
NVFP4best16       4.500      21.568      21.524      21.422      21.177      21.138      26.424      21.835
NVFP4             4.500      20.408      20.687      20.791      20.686      21.036      24.974      19.915
XPFP4-32          4.250      20.733      20.354      19.813      19.381      18.355      24.481      21.398
MXFP4opt16        4.500      19.174      18.630      18.038      17.736      17.124      19.616      20.165
MXFP4best32       4.250      19.056      18.229      17.223      16.756      15.504      19.534      20.126
MXFP4opt32        4.250      19.051      18.209      17.206      16.715      15.486      19.427      20.118
MXFP4             4.250      18.815      17.968      16.977      16.475      15.095      18.018      16.800

metric: zeroed  (%)
scheme            b/val    gaussian     laplace  student-t3   lognormal     outlier      sparse     uniform
-----------------------------------------------------------------------------------------------------------
NVFP4best16       4.500       7.30%      12.92%      12.55%      10.13%      20.89%       2.82%       4.77%
NVFP4             4.500       6.76%      12.44%      12.24%       9.98%      20.66%       2.73%       3.96%
XPFP4-32          4.250       8.25%      15.17%      15.77%      15.21%      35.86%       4.65%       5.15%
MXFP4opt16        4.500       8.50%      15.00%      14.82%      13.50%      23.92%       3.50%       5.80%
MXFP4best32       4.250       9.43%      17.33%      18.29%      18.93%      40.06%       5.40%       6.15%
MXFP4opt32        4.250       9.44%      17.59%      18.58%      19.58%      40.33%       5.28%       6.10%
MXFP4             4.250       8.73%      16.11%      16.89%      17.03%      38.05%       4.69%       3.15%

metric: block_snr_p10
scheme            b/val    gaussian     laplace  student-t3   lognormal     outlier      sparse     uniform
-----------------------------------------------------------------------------------------------------------
NVFP4best16       4.500      20.245      20.172      20.200      20.003      20.008       0.000      20.541
NVFP4             4.500      18.619      18.816      18.887      18.942      18.919       0.000      18.340
XPFP4-32          4.250      19.856      19.366      19.054      18.474      17.333      21.252      20.420
MXFP4opt16        4.500      17.449      16.845      16.590      16.252      15.811       0.000      18.661
MXFP4best32       4.250      17.745      16.735      16.069      15.404      13.763      15.781      19.041
MXFP4opt32        4.250      17.741      16.691      16.064      15.318      13.725      15.660      19.018
MXFP4             4.250      17.355      16.383      15.789      15.056      13.268      14.259      15.729

--- the Pareto frontier on Gaussian data ---
  XPFP4-32       4.250 bits   20.733 dB
  NVFP4best16    4.500 bits   21.568 dB
```

Read across the `outlier` column of the first table: MXFP4 gives up 3.7 dB relative to
its Gaussian score, and the `zeroed` table explains where it went — **40 % of the
block's nonzero inputs are stored as exact zero**. The energy-weighted SNR still reads
a respectable 15.1 dB, because the outlier that caused the annihilation also carries
most of the energy. That gap between the two tables is the reason this script prints
three of them.

The `sparse` column of the `block_snr_p10` table shows `0.000` for the NVFP4 and
MXFP4opt16 rows: with 90 % exact zeros, the unluckiest tenth of blocks are all-zero,
and an all-zero block has no signal to have a ratio with.

Then the same sweep with a Hadamard rotation in front:

```
--- with Hadamard rotation ---
metric: snr
scheme            b/val    gaussian     laplace  student-t3   lognormal     outlier      sparse     uniform
-----------------------------------------------------------------------------------------------------------
H·NVFP4best16     4.500      21.584      21.552      21.617      21.572      22.225      25.406      21.616
H·XPFP4-32        4.250      20.757      20.728      20.849      20.810      21.458      22.816      20.708
H·NVFP4           4.500      20.452      20.323      20.189      20.071      18.870      23.902      20.503
H·MXFP4opt16      4.500      19.179      19.238      19.329      19.317      19.578      19.616      19.147
H·MXFP4best32     4.250      19.057      19.141      19.245      19.230      19.586      19.394      19.100
H·MXFP4opt32      4.250      19.052      19.135      19.240      19.223      19.571      19.335      19.098
H·MXFP4           4.250      18.807      18.861      18.939      18.953      19.301      18.501      18.970

metric: zeroed  (%)
scheme            b/val    gaussian     laplace  student-t3   lognormal     outlier      sparse     uniform
-----------------------------------------------------------------------------------------------------------
H·NVFP4best16     4.500       1.86%       2.73%       2.66%       1.93%       5.13%       4.21%       1.30%
H·XPFP4-32        4.250       0.50%       0.67%       0.69%       0.55%       1.34%       2.98%       0.39%
H·NVFP4           4.500       1.62%       2.33%       2.39%       1.88%       3.87%       3.98%       1.23%
H·MXFP4opt16      4.500       2.05%       2.96%       2.80%       2.36%       4.77%       3.99%       1.58%
H·MXFP4best32     4.250       0.57%       0.86%       0.78%       0.62%       1.95%       3.03%       0.42%
H·MXFP4opt32      4.250       0.56%       0.84%       0.77%       0.63%       1.93%       3.15%       0.43%
H·MXFP4           4.250       0.52%       0.75%       0.69%       0.56%       1.98%       3.36%       0.40%

figures → …/examples/figures
```

**The rotation's headline is in the second table, not the first.** Every row of the
rotated SNR table is nearly flat across distributions — the rotation makes the scheme
distribution-agnostic, which is the real result — but the zeroed fractions collapse by
an order of magnitude, `H·MXFP4` going from 38 % to 2 % on the outlier column. Rotation
converts *annihilation* into *spread*. Read [`SchemeMetrics`](@ref)'s note on
`worst_rel` before concluding it is free.

Figures: `01_grid.png`, `01_grid_rotated.png`, `01_pareto.png`, `01_schemes.png`.

---

## 2 — One outlier, and what rotation does about it

```console
$ julia --project=. examples/02_outlier_study.jl
```

Takes a single realistic block — 32 trained-weight-scale values with one 28× outlier
dropped in — and asks every scheme what it makes of it. Then grows the outlier and
watches the rest of the block die.

```
==============================================================================
  ONE BLOCK, ONE OUTLIER
==============================================================================
  max/median magnitude ratio: 28.5×

  scheme             SNR dB   zeroed    cosine     worst
  MXFP4               17.04    56.2%   0.99326     83.0%
  MXFP4opt32          17.04    56.2%   0.99326     83.0%
  NVFP4               21.74    34.4%   0.99671     71.6%
  XPFP4-32            18.67    56.2%   0.99326     71.6%
  H·MXFP4             17.70     3.1%   0.99217    512.0%
  H·XPFP4-32          21.62     0.0%   0.99733    406.0%
```

This table is the argument against reading SNR alone, in five columns. `MXFP4` reports
17.04 dB and a cosine of 0.993 — the block's *direction* survives to within about 6.7°,
which sounds fine — while **18 of its 32 values have been zeroed**. The energy is
intact because the outlier holds it; the information is not.

Compare the two rotated rows. `H·MXFP4` scores *worse* on `worst` (512 % relative error
on some single coordinate) yet zeroes almost nothing. That is not a contradiction: a
rotation spreads the block's quantization noise evenly over all 32 coordinates, so a
coordinate that was tiny to begin with acquires a large *relative* error even though it
was never annihilated. Whether that trade is worth taking depends on whether the
consumer of the block cares about coordinates or about directions.

```
--- rest-of-block SNR as the outlier grows (the 6 dB/octave law) ---
       R        MXFP4      H·MXFP4
       1        18.04        21.46
       2        18.04        18.57
       4        15.29        18.20
       8        10.70        14.77
      16         4.39        11.23
      32         0.06        4.11
      64         0.00        -0.63
```

Read the `MXFP4` column from `R=4` down: 15.29, 10.70, 4.39, 0.06. Each doubling of the
outlier costs the *other 31 elements* roughly 6 dB — one full bit — because the shared
scale is set by the maximum and every doubling of the maximum halves the resolution
available to everyone else. By `R=32` the rest of the block is gone. The `H·MXFP4`
column decays at the same slope but starts about 6 dB higher and stays ahead
throughout, which is the quantitative version of what the first table showed.

Figures: `02_rotation.png`, `02_scale_placement.png`, `02_reconstruction.png`,
`02_outlier_law.png`.

---

## 3 — Dot products and error profiles

```console
$ julia --project=. examples/03_dot_and_profile.jl
```

The question a hardware designer actually has: does accumulating a long dot product in
FP4 lose accuracy as it gets longer?

```
==============================================================================
  DOT-PRODUCT SNR vs LENGTH  (should be flat: the block tree is exact)
==============================================================================
  scheme                32       128       512      2048      8192
  MXFP4              15.92     15.38     16.14     16.21     15.39
  MXFP4opt32         16.19     15.57     16.34     16.30     15.39
  NVFP4              17.63     18.43     17.52     17.82     17.12
  XPFP4-32           17.72     18.10     17.84     18.14     17.44

--- every element product is exact; all error is representational ---
  MXFP4          products exact: true    core bound 1152
  MXFP4opt32     products exact: true    core bound 1152
  NVFP4          products exact: true    core bound 576
  XPFP4-32       products exact: true    core bound 1152
```

**No.** Every row is flat from N=32 to N=8192 — a 256× increase in length costs
nothing. The reason is in the second table: every element product is *exact*, so the
only error in the whole dot product is the representational error already present in
the inputs. Both signal and noise power grow linearly in N, and their ratio is
therefore length-invariant.

Compare each row against its element SNR from example 1 — MXFP4's 18.8 dB against the
~15.9 dB here. The 3 dB gap is the [`dot_snr_law`](@ref): writing ``\hat a = a(1+δ^a)``,
each product carries two independent noise sources instead of one, so error power
doubles while signal power does not. Exactly a factor of two, at every length.

`core bound` is the width the accumulator must carry for the core sum to stay exact —
1152 for a 32-element block of E2M1 products, 576 for NVFP4's 16.

Figures: `03_dot_snr.png`, `03_profiles.png`, `03_block_snr.png`.

---

## 4 — Sweeping the design space quickly

```console
$ julia --project=. examples/04_quick_snr.jl
$ julia --project=. examples/05_accumulator.jl
$ julia --project=. examples/06_cost_savings.jl
$ julia --project=. examples/07_mxfp4_rr4.jl
$ julia --project=. examples/08_pipeline.jl
```

The other three scripts answer questions about *shipped* formats. This one is for
asking new ones — [`quick_snr`](@ref) streams about ``10^8`` values a second and returns
the identical number [`measure_snr`](@ref) does, so a whole design sweep costs less than
a single figure.

### Comparing shipped formats against modified ones

[`fp4_variant`](@ref) builds a modified MXFP4 in one line; [`quick_compare`](@ref) runs
several formats over **one shared draw** of data, so the differences are differences in
the format, not in the luck of two independent samples.

```
==================================================================================
  1. THE SHIPPED FORMATS, plus what modifying them buys
==================================================================================
scheme                        bits    SNR dB  QSNR p10 QSNR min     gap  dB/bit  zeroed%  clip%
────────────────────────────────────────────────────────────────────────────────────────────────
MXFP4                         4.25    18.790    17.275   14.281  +1.515    4.42     8.78   2.34
NVFP4                         4.50    20.440    18.666   16.193  +1.774    4.54     6.78   3.47
FP4-K32-E8M0-OPT_SHIFT        4.25    19.043    17.709   14.516  +1.334    4.48     9.48   0.91
FP4-K32-E8M0-BEST_POW2        4.25    19.047    17.715   14.516  +1.332    4.48     9.50   0.82
FP4-K16-E8M0-MX_FLOOR_POW2    4.50    18.582    16.536   13.203  +2.046    4.13     7.46   4.71
FP4-K16-E4M3-MSE_OPTIMAL      4.50    21.588    20.290   18.862  +1.298    4.80     7.31   3.35
MXINT4                        4.25    17.575    15.702   10.279  +1.874    4.14    17.41   0.86

SNR dB is pooled (energy-weighted, the usual published QSNR).
QSNR p10/min are per-block, every block weighted equally; gap = pooled − p10.
```

Five readings worth having — the last two only visible once QSNR is in the table:

* The **optimized shift rule** buys +0.25 dB over MXFP4 for *one comparison against a
  constant* in the encoder — and lands within 0.004 dB of `BEST_POW2`, which pays a
  full block MSE evaluation for the same answer. That is the power-of-two class
  ceiling, reached for free.
* `FP4-K16-E8M0` — NVFP4's block length with MXFP4's scale rule — measures **18.58 dB,
  *worse* than MXFP4's 18.79** while costing more bits. So none of NVFP4's advantage
  comes from its shorter block. It is all the E4M3 scale format.
* `MXINT4` zeroes 17.4 % of its inputs against MXFP4's 8.8 %. The uniform grid spends
  its levels on the Gaussian's thin tails and starves the bulk.
* **`MXINT4`'s tail is far worse than its headline admits.** Its pooled SNR is 1.2 dB
  below MXFP4's, but its *worst block* is **10.28 dB against MXFP4's 14.28** — a 4 dB
  deficit that the energy-weighted column compresses to one. If a downstream layer is
  sensitive to its weakest block rather than its average one, that is the number that
  decides, and pooled SNR does not show it.
* **`FP4-K16-E4M3-MSE_OPTIMAL` wins on every column, including the `gap`.** It has both
  the highest floor (18.86 dB minimum block) and the *smallest* spread between headline
  and tenth percentile (+1.30 dB). It is not merely better on average — it is more
  uniformly good, which is a different and more useful property. Compare
  `FP4-K16-E8M0-MX_FLOOR_POW2`, whose +2.05 dB gap is the largest in the table: its
  headline flatters it most.

The `clip%` column is not zero for any rule, which is why it is printed:
`MX_FLOOR_POW2` only promises ``M/S ∈ [4,8)`` against a grid that stops at 6, and
`NV_MAXDIV` aims the maximum straight at the top code but must round ``M/6`` onto E4M3
— rounding *down* overloads. Read it as a dial, not an alarm.

### The block-length sweep, and a trap in it

```
==================================================================================
  2. BLOCK LENGTH — note the SNR column runs the OPPOSITE way to intuition
==================================================================================
    K    bits    SNR dB  eff bits   dB/bit
──────────────────────────────────────────
    4   6.000    18.139      3.01     3.02
    8   5.000    18.374      3.05     3.67
   16   4.500    18.581      3.09     4.13
   32   4.250    18.785      3.12     4.42
   64   4.125    18.935      3.14     4.59
  128   4.062    18.959      3.15     4.67
  256   4.031    18.929      3.14     4.70
```

The expectation is that a shorter block buys SNR, a shared scale fitting fewer values
better, paid for in bits. **The measurement says the opposite**: under `MX_FLOOR_POW2`,
short blocks cost more bits *and* score worse, all the way from 18.14 dB at ``K=4`` to
18.96 dB at ``K=128``.

Before believing that as a fact about block length, change one variable:

```
==================================================================================
  3. THE SAME SWEEP UNDER THE OPTIMIZED SHIFT RULE — the trend inverts back
==================================================================================
    K    bits    SNR dB  eff bits   dB/bit
──────────────────────────────────────────
    4   6.000    19.452      3.23     3.24
    8   5.000    19.314      3.21     3.86
   16   4.500    19.173      3.18     4.26
   32   4.250    19.042      3.16     4.48
   64   4.125    18.980      3.15     4.60
  128   4.062    18.945      3.15     4.66
  256   4.031    18.919      3.14     4.69
```

It reverses. So sweep 2's inversion was never about block length — it was the floor
rule wasting most of a binade, a fixed overhead that a longer block dilutes across more
elements. Fix the rule and short blocks win again, as intuition said. Two sweeps, four
seconds, and the confound is separated from the effect.

### Where each scheme stops working

```
==================================================================================
  4. DISTRIBUTION SENSITIVITY — where each scheme stops working
==================================================================================
distribution           MXFP4         NVFP4      MX+shift      XPFP4-32
──────────────────────────────────────────────────────────────────────
gaussian               18.78         20.42         19.05         20.77
uniform                16.83         19.85         20.10         21.38
laplace                17.98         20.66         18.23         20.38
student_t3             16.98         20.81         17.24         19.81
lognormal              16.56         20.68         16.74         19.40
sparse                 18.12         24.85         19.35         24.60
outlier                14.99         20.96         15.40         18.28
```

The shift rule is worth **+0.25 dB on Gaussians but +3.3 dB on uniforms**, and almost
nothing on the heavy-tailed rows. A scale rule is only ever as good as its data, and a
single Gaussian benchmark would have hidden both the win and the limit.

```
==================================================================================
  5. WHAT SNR HIDES — the zeroed fraction on the same data
==================================================================================
  MXFP4                    SNR   16.90 dB   zeroed  16.86 %   clipped  1.89 %
  NVFP4                    SNR   21.09 dB   zeroed  12.11 %   clipped  3.33 %
  MX+shift                 SNR   17.13 dB   zeroed  18.56 %   clipped  1.10 %
  XPFP4-32                 SNR   20.02 dB   zeroed  15.77 %   clipped  2.13 %
```

On heavy-tailed data the shift rule reads **+0.23 dB better than MXFP4 while zeroing
1.7 % more of the inputs**. The two metrics disagree about which format is preferable,
which is the whole reason [`QuickSNR`](@ref) reports both.

---

## 5 — An accumulator, and whether carry-free scales

```console
$ julia --project=. examples/05_accumulator.jl
$ julia --project=. examples/05_accumulator.jl 4096            # N terms
$ julia --project=. examples/05_accumulator.jl 1024 positive   # N and input kind
```

An accumulator is the worst case for carry propagation: `acc += x` N times, strictly
sequential, with the register widening as the sum grows. Conventional adders pay a carry
chain on every step and the chain gets longer; [`accumulate_rr4`](@ref) pays none.

The script takes **N and the input shape from the command line**, and every routine also
accepts a plain vector, so you can accumulate your own numbers
([`accumulator_inputs`](@ref) lists the generated shapes).

It opens with the conversion itself — [`rr4_recode`](@ref), the parallel binary→RR4
recoding in which every carry depends on **its own digit alone**:

```
      c_{i+1} = [ d_i ≥ 2 ]        z_i = d_i − 4·c_{i+1} + c_i

    i    d_i    c_out    c_in     z_i
    0      3        1       0      -1
    1      0        0       1       1
    2      3        1       0      -1
    3      1        0       1       2
    4      3        1       0      -1
    5      2        1       1      -1
```

No row waits on another row's output — that is what carry-free means. Note that
[`to_rr4`](@ref) reaches the same *value* by a sequential rewrite (`carry iff d + c_in >
2`), so the two disagree on **spelling** wherever `d = 2` meets no incoming carry, and
agree everywhere else:

```
         v  to_rr4        rr4_recode    same spelling?
      2931  1 1̄ 1̄ 2 1̄ 1 1̄  1 1̄ 1̄ 2 1̄ 1 1̄  true
        38  2 1 2         1 2̄ 2 2̄       false
      2666  2 2 1 2 2 2   1 1̄ 2̄ 2 1̄ 1̄ 2̄  false
```

Both always decode to the same number. That redundancy *is* the point.

Then the accumulators, on one shared set of terms:

```
accumulating 1024 terms, sequential schedule — final accumulator 15 bits
method                  add depth  exit CPA     total   per term    exact  vs RR4
────────────────────────────────────────────────────────────────────────────────
ripple                      13070         0     13070      12.76      yes  4.3× deeper
prefix                       4079         0      4079       3.98      yes  1.3× deeper
rr4                          3069         4      3073       3.00      yes  —
all methods returned the same value: 16697
```

**All three return the same value** — redundancy trades cost, never accuracy, and
`exact` is measured rather than assumed. Only the depth differs. RR4's `per term` is
3.00 because its add is depth 3 *whatever the accumulator width*, and the exit
conversion is 4 levels paid **once** across 1024 terms.

The scaling sweep:

```
      N    bits      ripple     prefix        RR4     ripple/RR4   prefix/RR4
     16      10         174         60         49           3.6×         1.2×
     64      13         800        251        193           4.1×         1.3×
    256      13        3133       1015        769           4.1×         1.3×
   1024      15       15844       4092       3073           5.2×         1.3×
   4096      16       59778      16359      12289           4.9×         1.3×
  16384      17      261947      66541      49154           5.3×         1.4×
```

But `N` is the wrong variable. Hold `N = 512` and vary the **width** instead:

```
   magnitude   bits     ripple    prefix       RR4      rip/RR4    pre/RR4
          10     12       5873      1994      1537         3.8×       1.3×
        1000     18       9162      2417      1538         6.0×       1.6×
     1000000     28      14259      2555      1538         9.3×       1.7×
  1000000000     38      19359      3056      1539        12.6×       2.0×
     10^15       58      29554      3066      1539        19.2×       2.0×
```

**The RR4 column does not move** — 1537 to 1539 across a 12→58 bit range. That is the
entire claim: per-add depth is a constant of the number system, not a function of the
word length. Ripple nearly quintuples over the same span.

!!! warning "Where it does *not* win"
    The script ends with this, and it belongs next to the tables above. Against ripple
    carry the win is large and grows with width (19× at 58 bits). Against a Kogge–Stone
    prefix network it is **1.2×–2.0×**, because a prefix adder is already logarithmic in
    the width — carry-free replaces `log w` with a constant, not something linear.

    And at small widths RR4 **loses**: the 16-term example accumulates into a 7-bit
    register, where prefix needs `⌈log₂ 7⌉ = 3` levels per add — exactly what RR4 needs
    — and RR4 still owes the exit conversion, coming out at 0.8×. The crossover sits
    near 8 bits.

    Redundancy buys width-independence, which is worth a great deal in a wide sequential
    accumulator and almost nothing in a narrow one. Trees narrow the gap further, since
    they help every method equally — redundancy is worth most exactly where you *cannot*
    tree, which is a loop-carried accumulator.

!!! note "Values are measured, gate counts are modelled"
    `value` and `exact` come from really running [`rr4_add`](@ref). `add_depth` and
    `exit_cpa` come from the cost model stated in [`AccumulatorRun`](@ref): ripple
    `w+1`, prefix `⌈log₂w⌉`, RR4 `3` at any width. Compare depths between methods here,
    not against silicon.

---

## 6 — Quantifying the savings: memory, cells, depth

```console
$ julia --project=. examples/06_cost_savings.jl
$ julia --project=. examples/07_mxfp4_rr4.jl
$ julia --project=. examples/08_pipeline.jl
```

Every other page reports **depth**, because depth is what redundancy buys. This one
prices the two currencies that decide whether to take the trade — **state** (flip-flops)
and **cells** (combinational area) — with [`cost_report`](@ref).

The verdicts genuinely differ per operation, so read them separately.

### Addition and accumulation — depth bought with memory

```
  ADDITION — one 64-bit add (baseline: ripple)
method              state b      cells    depth     state ×   cells ×   depth ×
ripple                   64         64       65       1.00      1.00      1.00
prefix                   64        448        6       1.00      7.00↑     0.09↓
carry_save              128         64        1       2.00↑     1.00      0.02↓
rr4_carry_free           96         96        3       1.50↑     1.50↑     0.05↓

  ACCUMULATION — 1024 terms into 64 bits (baseline: ripple)
ripple                   64         64    66560       1.00      1.00      1.00
prefix                   64        448     6144       1.00      7.00↑     0.09↓
rr4_carry_free           96         96     3078       1.50↑     1.50↑     0.05↓
```

**Redundancy never saves memory.** A `{-2..2}` digit needs 3 bits where a `{0..3}` digit
needs 2, so a carry-free register is **1.5×** the size; carry-save is **2×**, holding a
sum word and a carry word. That penalty is structural, and it is the first thing the
report prints.

What it buys is depth that does not move with width — and the accumulation table is
where that pays, because the exit conversion is amortised: `3 × 1024 + 6` levels against
ripple's `1024 × 65`, a **21.6×** reduction.

### Multiplication — Booth wins every currency at once

```
  MULTIPLICATION — 24×24 bits (baseline: shift-add)
shift_add                96       1728      144       1.00      1.00      1.00
wallace_plain          1152       1632       13      12.00↑     0.94↓     0.09↓
booth_wallace           624        999       12       6.50↑     0.58↓     0.08↓
```

Booth recoding is the one unambiguous win in the package: **0.58× the cells** and
**0.54× the partial-product storage** of plain Wallace, at slightly lower depth. It
escapes the 3-bits-per-digit penalty because the recoded digits are consumed
immediately, never stored in a register.

Note also the honest cost of the tree itself: `wallace_plain` holds **12×** the state of
a sequential shift-add, because every partial-product row is materialised at once. That
is what buys the 11× depth reduction.

Measured across widths — this table comes from [`booth_tree_saving`](@ref), not the
model:

```
  n bits   plain  booth    levels  levels     cells   cells       cell
            rows   rows     plain   booth     plain   booth      ratio
       8       8      5         4       3         6       3      0.50×
      24      24     13         7       5        22      11      0.50×
      53      53     27         9       7        51      25      0.49×
      64      64     33        10       8        62      31      0.50×
```

Rows halve exactly at every width; CSA levels fall by 1–2; cells follow the rows.

### CSD constants — the offline optimiser

```
    constant    binary   adds        CSD   adds      saved
         231         6      5          4      3          2
         255         8      7          2      1          6
        4095        12     11          2      1         10
       65535        16     15          2      1         14
  total adders: binary 60 → CSD 17   (0.28×, 43 saved)
```

Runs of ones are where CSD pays: `2^k − 1` needs `k` adders in binary and **1** in CSD.
This is affordable only because the search happens once at design time — precisely why
silicon ships Booth's fixed windows for variable operands and CSD's minimal strings for
coefficients.

### Where the memory savings actually are

```
  format         bits/val           MB    vs FP32
  FP32             32.000        67.11      1.00×
  FP16             16.000        33.55      2.00×
  E4M3              8.000        16.78      4.00×
  MXFP4             4.250         8.91      7.53×
  NVFP4             4.500         9.44      7.11×
```

One 4096×4096 weight matrix. **Narrowing the format saves 7.5×; a carry-free accumulator
downstream spends 1.5× more state.** Both are real, they act on different memories, and
only the first is measured in megabytes — [`format_memory_costs`](@ref) prints it beside
the datapath tables so the two are not confused for each other.

The script closes by composing all three on one MXFP4 dot product of length 4096:

```
  operand storage : 4096 values × 4.25 b = 2.1 KB   (FP32: 16.0 KB, 7.53× less)
  multiplier cells: E2M1 4×4 39 vs FP32 24×24 999   → 26× smaller
  core sum bound  : 1152, so the adder tree is exact fixed point — no rounding
  fixup accumulate: ripple 2176 levels vs carry-free 388   → 5.6× shallower
```

Three independent savings, in three different currencies, none a substitute for another.

!!! warning "First-order model, stated so it can be argued with"
    Values in examples 1–5 are computed by running the arithmetic. The **cost numbers
    here are a model**: ripple `w+1` levels and `w` cells, Kogge–Stone `⌈log₂w⌉` levels
    and `w⌈log₂w⌉` cells, RR4 `3` levels and `1.5w` cells, `rows × 2n` bits of
    partial-product storage, Booth selects charged at 1.3× a plain AND. It is calibrated
    to reproduce the structural facts — rows halve, carry-free is depth-3 — and to make
    the state penalty visible. It is not a substitute for synthesis. The full model is in
    the source header of `analysis/costmodel.jl`.

---

## 7 — MXFP4 weights: what is RR4 actually worth?

```console
$ julia --project=. examples/07_mxfp4_rr4.jl
$ julia --project=. examples/07_mxfp4_rr4.jl 8192
```

Examples 5 and 6 measure redundancy at general word widths, where carry-free addition's
constant depth 3 beats a ripple's `w+1` by a mile. **MXFP4 changes the widths, and that
changes the answer.** [`rr4_opportunity`](@ref) derives them from the format rather than
assuming them:

```
  E2M1 element ×2 → integers [-12,-8,-6,-4,-3,-2,-1,0,1,2,3,4,6,8,12]
  element      : 4 magnitude bits + sign = 5
  one product  : |m_a·m_b| ≤ 144 → 9 bits   (37 distinct products exist)
  K=32 core sum : |Σ| ≤ 4608 → 14 bits
```

Every operand in the inner loop is **4 to 14 bits**. At 14 bits a prefix adder is 4
levels deep and carry-free is 3 — a 1.3× margin, not the 20× a 64-bit sequential
accumulator shows. Three findings follow, and all three cut against RR4.

### The element multiply wants a ROM, not a multiplier

```
  method           rows    depth    cells  note
  array               5        6       25  5-row array, no recoding
  booth_r4            3        5       35  3 rows + recode; the recode costs as much as it saves
  lut                 0        2       64  64-entry ROM over 8 magnitudes; 37 distinct products exist
  rr4_digits          9       16       36  3×3 digit products — recoding a 4-bit operand is pure overhead
```

E2M1 has 8 magnitudes, so the entire multiplier is a **64-entry ROM at depth 2**. Booth
recoding a 5-bit operand spends as much on recode logic as it saves in rows.

### Carry-save already beats RR4 for the block reduction

```
  method                 levels    depth
  carry_save_tree             8       12
  rr4_tree                    5       19
  prefix_tree                 5       20
  sequential_prefix          31      124
```

A 3:2 compressor is **one** gate level; an RR4 add is **three**. A reduction tree never
needs its digits back in range, so RR4's bounded digits are a feature it pays for and
cannot use. This is why Wallace trees have used carry-save since 1964.

### Dot product and soft-max, scaling in N

```
         N |   dot conv    dot RR4 |  smax conv   smax RR4
        64 |         20         23 |         56         76
      1024 |         28         35 |         78        112
     16384 |         35         47 |        101        148
     65536 |         38         53 |        113        166
```

Both grow like `log N` — RR4 changes the constant, not the order — and **RR4 is above
the conventional column in every row of both kernels.**

Soft-max is worse than the dot product because of a structural fact:
[`sign_detect_depth`](@ref) is **0** for two's complement (the sign *is* the top bit)
and `⌈log₂⌈w/2⌉⌉` for a signed-digit string, which must priority-encode to its leading
non-zero digit. Soft-max opens with a max reduction, so **carry-free makes the subtract
cheaper and the sign test dearer, and they cancel.** Redundancy removes the carry chain
from `+` and leaves it in `<`.

### The Amdahl ceiling

```
         N |    dot addr.  dot ceiling smax ceiling
       256 |        37.5%        1.60×        1.34×
      4096 |        51.6%        2.07×        1.36×
     16384 |        57.1%        2.33×        1.36×
```

Set RR4's addition cost to **zero** and those are the speedups. They bound *any*
redundant scheme, because the remaining depth is products, exp, comparisons and the exit
CPA.

### The best case, and the incumbent it still loses to

Streaming accumulation of block results — loop-carried, no tree, wide accumulator — is
the shape carry-free was invented for:

```
   acc bits |  canonical        RR4 carry-save | RR4 vs canon   RR4 vs CS
         32 |        650        401        147 |       1.62×       0.37×
         64 |        777        402        148 |       1.93×       0.37×
```

Against a **canonical** adder RR4 wins 1.93× — a real result. Against **carry-save** it
loses 2.7×, because a 3:2 accumulator absorbs one term per gate level,
`(S,C)+x = csa(S,C,x)`, where RR4 needs three.

!!! warning "The honest framing: RR4 vs carry-save, never RR4 vs binary"
    Carry-save is itself a redundant scheme and it is the incumbent — it is what MAC
    accumulators have used for sixty years. On **depth** there is no configuration
    measured here in which RR4 beats it.

    What RR4 genuinely offers over carry-save is narrower than the depth story suggests
    but not nothing: **1.5w bits of state against 2w** (a 25 % saving), and a real
    positional representation whose digits can be shifted, truncated and indexed, which
    a `(sum, carry)` pair cannot until its exit CPA has run.

### What to do instead

The MXFP4 saving is already banked, and it is **memory**: 7.53× off the weights before
any arithmetic is considered, and token generation is memory-bound. On the arithmetic
side the available wins, in order of size, are the 64-entry ROM, the carry-save tree,
the E8M0 exponent-add scale, and the exact fixed-point core sum. **None of them is RR4.**

---

## 8 — Pipelining, and the single-cycle multiply

```console
$ julia --project=. examples/08_pipeline.jl
```

Two questions that follow from the depth model: what pipeline registers buy, and what
has to be given up to make a multiply fit in one cycle. No figures, about five seconds.

**The FP32 multiplier cut at three clock budgets.** The interesting column is `slack` —
level-budget bought and not used:

```
  FP32: depth 22 levels at L = 8
  stage   levels   slack   contains
  ────────────────────────────────────────────────────────────────────────
  S1           7       1   unpack + significand multiply (1/2)
  S2           7       1   significand multiply (2/2) + normalise
  S3           8       0   round + post-normalise + pack / specials

  latency 3 cy · II 1 · +116 flip-flops (2 cuts × 58 bits) · 2 levels of slack
```

Three stages, cut *through* the Wallace tree — the industrial shape, arrived at from the
gate counts rather than assumed. At `L = 16` it becomes two stages with 10 levels of
slack; at `L = 4`, seven stages with only 6.

**The reservation table**, six multiplies through the 3-stage unit:

```
  cycle           1  2  3  4  5  6  7  8
  ──────────────────────────────────────
  S1              A  B  C  D  E  F  .  .   7 lv
  S2              .  A  B  C  D  E  F  .   7 lv
  S3              .  .  A  B  C  D  E  F   8 lv
  retires         .  .  A  B  C  D  E  F

  latency 3 cy · II 1 · 6 items in 8 cy · 2.25× vs unpipelined · 75% utilised
```

**Pipelining never helps one operation:**

```
       1 items:      3 cy, speedup 1.0×
       2 items:      4 cy, speedup 1.5×
       8 items:     10 cy, speedup 2.4×
      32 items:     34 cy, speedup 2.82×
     128 items:    130 cy, speedup 2.95×
    4096 items:   4098 cy, speedup 3.0×

  90% of peak needs 19 items; 99% needs 198.
```

**How close a multiply gets to one cycle**, each rung cumulative:

```
  one FP32 × FP32 multiply at L = 16 levels/cycle
  technique                  depth   saved  cycles  1 cy?  what it costs
  ────────────────────────────────────────────────────────────────────────────
  IEEE baseline                 22       0       2     no  none — this is the reference
  + Booth radix-8               21       1       2     no  13→9 rows; one adder for the hard 3× multiple
  + carry-save product          15       6       1    yes  output is a redundant (48-bit sum, carry) pair, not a number
  + deferred rounding            8       7       1    yes  needs a wide fused accumulator; rounds once at the end
    table lookup (N/A)           8       0       1    yes  4611686018427387904 entries exceeds the 4096-entry limit

  single-cycle IEEE FP32 needs 22 levels = 44-66 FO4; a 15-25 FO4 core clock makes that 2-5 cycles.
```

The two rungs that get FP32 to one cycle both work by **not producing an IEEE result** —
the surviving unit emits an unrounded, unnormalised, redundant 48-bit intermediate. That
is a MAC array's internal wire, not a float.

**And the format beats all of it:**

```
  format      sig  IEEE lv   IEEE cy  best lv    best cy     LUT?
  ──────────────────────────────────────────────────────────────────
  E2M1          2        9         1        2          1      yes
  E4M3          4       12         1        4          1       no
  FP16         11       18         2        6          1       no
  BF16          8       16         1        5          1       no
  FP32         24       22         2        8          1       no
  FP64         53       26         2       10          1       no

FP32: every trick together, 22 → 8 levels (2.75×), and the result is no longer a float.
E2M1: the format change alone, 22 → 2 levels (11.0×), and it is still exact.
```

E2M1 is single-cycle as a plain IEEE multiplier, and 2 levels as a 64-entry ROM — one
cycle at any clock a person would build. The full derivation is
[The cost model: how computations map to gate levels](@ref) §11 and §12.

---

## Writing your own

The pieces the scripts are built from, in the order you would reach for them:

```julia
using xpuFP, Random

# one format, one number
quick_snr(MXFP4)                                  # 1e6 gaussians, ~12 ms
quick_snr(MXFP4; n = 10^7, dist = :student_t3)    # your data model
quick_snr(MXFP4, my_weights)                      # your actual data

# a modified format
fp4_variant(K = 16, scale = E4M3, rule = OPT_SHIFT)

# several formats on one shared draw
quick_compare([MXFP4, NVFP4, fp4_variant(rule = OPT_SHIFT)])

# a block-length sweep at fixed rule
quick_sweep(Ks = (8, 16, 32, 64), rule = OPT_SHIFT)

# the slow path, when you want a confidence interval and an analytic cross-check
simulate_snr(MXFP4; n = 40_000, trials = 12, rng = MersenneTwister(1))
```

And for the accumulator study:

```julia
rr4_recode(2931)                       # carry-free binary → RR4, depth 1
rr4_recode_table(2931)                 # its per-digit working

compare_accumulators(1024)             # 1024 random terms, all three methods
compare_accumulators(my_vector)        # or your own numbers
compare_accumulators(4096; kind = :positive, magnitude = 10^6)

accumulate_rr4(my_vector; schedule = :tree)
accumulate_carry(my_vector; adder = :prefix)
accumulator_scaling(Ns = (16, 64, 256, 1024))
```

And for the cost model:

```julia
cost_report(w = 64, n = 1024, mulbits = 24)   # all three operations at once
add_costs(64)                                  # or one at a time
accumulate_costs(1024, 64)
multiply_costs(24)
constant_multiply_costs(4095)                  # CSD adders saved
booth_tree_saving(53)                          # measured, not modelled
format_memory_costs(4096 * 4096)               # where the megabytes actually go
```

And for the MXFP4-vs-RR4 question:

```julia
rr4_opportunity()                              # the whole verdict, six sections
mxfp4_widths(MXFP4)                            # widths derived from the format
mxfp4_multiply_options()                       # array / Booth / LUT / RR4
mxfp4_reduction_options(32)                    # carry-save vs RR4 vs prefix
mxfp4_dot_costs(4096; schedule = :sequential)  # RR4's best case
mxfp4_softmax_costs(4096)                      # and its worst
sign_detect_depth(32; redundant = true)        # why comparison resists redundancy
```

!!! warning "`estimate_element_snr` is minutes, not milliseconds"
    [`estimate_element_snr`](@ref) derives the SNR by nested order-statistic quadrature
    rather than measuring it, and takes several **minutes** per call — it is a
    correctness check on the simulation, not a tool for sweeping. It agrees with
    [`quick_snr`](@ref) to 0.002 dB on MXFP4, which is precisely why the fast path can
    be trusted and used instead.

See [Error analysis](@ref) for what the metrics mean and
[API — Analysis](@ref) for the full signatures.
