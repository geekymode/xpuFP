# xpuFP.jl

A laboratory for the number representations accelerator hardware actually uses: fixed
point, the IEEE-style floating-point zoo down to four bits, the block-scaled microscaling
formats (MXFP4, NVFP4), the redundant systems inside every multiplier, and the error and
cost analysis that decides between them.

Every format is implemented **from the bit pattern up** — encode, decode, round, saturate
— so the arithmetic can be *watched* rather than trusted.

```julia
using xpuFP

FP32                       # a FloatFormat; show it for its vital statistics
grid(E2M1)                 # the *entire* FP4 number system, 15 distinct values
quantize(E2M1, 2.25)       # 2.0 — even the smallest interesting product rounds

fpadd(FP32, 12.0, 6.0)     # a full datapath trace, not just an answer

qb = quantize_block(MXFP4, randn(32))   # one MX block: scale + 32 nibbles
measure_snr(MXFP4, randn(200_000))      # ≈ 18.8 dB

quick_snr(MXFP4)                        # the same 18.8 dB, ~28× faster
quick_compare([MXFP4, NVFP4, fp4_variant(rule = OPT_SHIFT)])
```

## What's in it

| Layer | What it answers |
|:---|:---|
| **Formats** | What points does an `n`-bit format actually put on the real line? |
| **Arithmetic** | What does a datapath do, stage by stage, to produce this answer? |
| **Block formats** | MXFP4, NVFP4, MXINT4 and variants — scale rules, block sizes |
| **Error analysis** | SNR, effective bits, clipping, zeroing — measured and estimated |
| **Redundant systems** | CSD, Booth, RR4, carry-save, Wallace trees |
| **Cost model** | Gate levels, cells, state, cycles, pipelining — what redundancy buys |
| **HLS** | Which of those schemes survive a synthesis flow, and which don't |

## A few results it will argue with you about

- `quick_snr` is **bit-identical** to `measure_snr` (Δ = 0.00e+00) across every scale rule
  and distribution, at ~28× the speed.
- Under `MX_FLOOR_POW2`, **longer blocks measure better** (18.14 dB at K=4 → 18.96 at
  K=128) — the trend inverts under `OPT_SHIFT`.
- An FP32 multiply is **22 gate levels**, of which rounding is 5. Every microarchitectural
  trick together takes it to 8; changing the format to E2M1 takes it to 2.
- **Redundant arithmetic loses at 4 bits.** Booth recoding an MXFP4 element costs *more*
  cells than a plain array; a 64-entry table beats both.
- A float grid is a **piecewise-linear approximation of a log axis**, sagging 0.0861
  octaves at `m = 1/ln2 − 1` — the same constant that makes the fast inverse-square-root
  trick work.
- For an HLS/ASIC flow, **65–99 % of block-MAC energy is fetching the operands**, which
  reorders every arithmetic priority below it.

## Documentation

Built with [Documenter.jl](https://documenter.juliadocs.org/):

```console
$ julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
$ julia --project=docs docs/make.jl
$ open docs/build/index.html
```

25 pages, including a worked [Examples](docs/src/examples.md) page with real captured
output, a [cost model](docs/src/costmodel.md) that maps computations to gate levels, and
an [HLS](docs/src/hls.md) page on which schemes survive synthesis.

## Examples

Eight runnable scripts, each fixing its RNG seed so the tables reproduce exactly:

```console
$ julia --project=. examples/01_compare_schemes.jl   # which FP4 scheme wins, on what data
$ julia --project=. examples/04_quick_snr.jl         # sweeping the design space, fast
$ julia --project=. examples/05_accumulator.jl       # does carry-free RR4 actually scale?
$ julia --project=. examples/07_mxfp4_rr4.jl         # with MXFP4 weights, what is RR4 worth?
$ julia --project=. examples/08_pipeline.jl          # pipelining, and the 1-cycle multiply
```

## Tests

```console
$ julia --project=. -e 'using Pkg; Pkg.test()'
```

Every number quoted in the documentation is pinned by the suite, so the prose cannot
drift from the code without a test going red.

## Status

Under active development. The formats, arithmetic, block, analysis, cost-model and HLS
layers are implemented and verified; the figure suite and some conversion paths are still
being filled in.

## License

MIT — see [LICENSE](LICENSE).
