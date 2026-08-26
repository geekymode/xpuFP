# HLS: which schemes survive a synthesis flow

The [cost model](@ref "The cost model: how computations map to gate levels") prices
gates. This page answers a different question: **if the design is written in C++ and
handed to a high-level synthesis flow, which of those schemes can you still choose, and
which ones does the tool choose for you?**

The answer reorders the priorities substantially, and it reorders them again for an
advanced process node. The short version, before the detail:

> In HLS you do not select an adder architecture. You select **widths**, **recurrence
> structure**, and **how many bits move per useful operation** — and on a modern node the
> third one dominates the other two by an order of magnitude.

## 1. What HLS decides for you

You write `a + b` on an `ap_int<W>`. The datapath library picks ripple, carry-select or
prefix from your **clock constraint**; logic synthesis then re-maps it again. You never
named an architecture, and hand-coding one does not override the choice — it produces
bit-level code the scheduler can no longer restructure, and typically loses twice: once
on the architecture the tool would have picked, and once on the pipelining it can no
longer do.

```@example hls
using xpuFP # hide
hls_operator_table()
```

Read that table as a budget for your own attention. Four of the six rows are decided
for you. The two that are not — accumulation and memory — are where the design is won.

## 2. Two targets, opposite answers

Before anything else: **the gate-level rankings in the cost model are ASIC rankings, and
they invert on FPGA.** This matters because it is an easy and expensive misreading.

An FPGA has a dedicated carry chain — hard silicon, roughly one LUT delay per eight bits.
So for a 64-bit add:

| scheme | cells | depth | ASIC standard cell | FPGA |
|:---|---:|---:|:---|:---|
| ripple | 64 | 65 | slow, tiny | **fast** — the carry chain is free |
| prefix (Kogge–Stone) | 448 | 6 | fast, 7× the area | **slower**, and 7× the area |
| carry-save | 64 | 1 | fast, leaves a redundant pair | wastes the carry chain |
| RR4 carry-free | 96 | 3 | 3 levels, 96 bits of state | 96 LUT-based state bits |

Depth 65 against depth 6 is a true statement about gates and a **misleading prediction
about an FPGA**, where those 65 levels are about eight carry-chain hops. If you are
prototyping on FPGA before an ASIC, expect the prototype to mislead you about exactly
this, in exactly this direction.

The rest of this page assumes an **ASIC target**, because that is where the block-format
accelerator question actually lands.

## 3. What a 3 nm process changes

Three scaling asymmetries dominate, and all three push the same way.

**Logic scales; SRAM and wires do not.** Standard-cell density keeps improving across
recent nodes; SRAM bitcell density has nearly stalled, and interconnect resistance gets
*worse* as pitches shrink. So with every node, the ratio

```math
\frac{\text{energy to fetch an operand}}{\text{energy to compute with it}}
```

**increases**. Every argument on this page that favours moving fewer bits gets stronger
at 3 nm, and every argument that favours a cleverer adder gets weaker.

**The design is power-limited, not area-limited.** An XPU tile at 3 nm runs into a
thermal budget long before it runs out of transistors. That changes the objective
function: you are not minimising gates, you are minimising joules per useful result, and
the two rank schemes differently. A prefix adder that is 7× the area to save gate levels
is a *bad* trade when the levels were not on your critical path anyway.

**Flip-flops cost twice.** A register costs its own area and its share of a clock tree
that switches every cycle whether or not the data changes. This is the specific reason
redundant number systems age badly at an advanced node: keeping a 64-bit value in RR4
takes **96 bits of state**, and carry-save takes two words instead of one. In a
combinational tree that is free. In anything registered, it is a 50–100 % increase in
sequential power for a datapath that was not the bottleneck.

!!! note "The one redundancy that still pays"
    Carry-save **inside** a reduction, never resolved until the exit, is not a scheme you
    have to choose — it is what a Wallace tree already is, and it is what synthesis
    infers from an unrolled sum of products. You get it by writing the reduction plainly
    and letting the tool fuse it. What does *not* pay is carrying redundancy across a
    register boundary, or across a function call, or in memory.

## 4. Datatypes: the one lever with the largest coefficient

Every width below is derived from the format — from [`mxfp4_widths`](@ref) — not rounded
up to a machine word. That distinction is the whole game: `ap_int<14>` and `int32_t` hold
the same values for an MXFP4 block sum, and one of them costs 2.3× the flops, the wires
and the routing congestion.

```@example hls
hls_types(MXFP4; blocks = 128)
```

Four of these deserve comment.

**The element is an integer.** E2M1's grid times two is
`{0, ±1, ±2, ±3, ±4, ±6, ±8, ±12}` — a signed 5-bit integer. There is no exponent logic
anywhere in the inner loop of a block MAC; the only floating-point object in the design
is the per-block scale, and it is touched once per `K` elements.

**The product is 9 bits**, because `12 × 12 = 144`, and only **37 distinct products
exist** in the entire format.

**The block accumulator is 14 bits, and it is exact.** `K × 144 = 4608` fits. An MXFP4
block dot product is an *exact integer computation* — there is no rounding to model, no
error to budget, and no reason to use a float anywhere inside it. This is the single most
useful structural fact for an HLS implementation of a block format.

**The scale is a raw 8-bit exponent**, applied once per block. Not a multiply — a shift,
or an exponent add if you convert to float at the block boundary.

## 5. Adders

**Infer `+`. Do not hand-code.** The only lever you hold is width, and area and energy
are linear in it.

The cost model's four schemes are worth knowing so you can recognise what the tool did,
not so you can choose between them:

| scheme | cells | depth | when HLS produces it |
|:---|---:|---:|:---|
| ripple | 64 | 65 | loose clock constraint, area-directed |
| prefix | 448 | 6 | tight clock constraint on a critical path |
| carry-save | 64 | 1 | inside a reduction tree, automatically |
| RR4 carry-free | 96 | 3 | never — you would have to write it by hand |

The RR4 row is the honest one. Nothing in a synthesis flow will produce it, and writing
it yourself means 96 bits of state where 64 would do, in exchange for depth you probably
were not spending.

## 6. Multipliers

**Infer `*`.** The answer then splits by operand width.

At **8 bits and up**, the tool's datapath library produces a Booth-recoded Wallace tree
and synthesis maps it to optimised cells. Hand-coding Booth is reimplementing, worse,
what the library already does.

At **4 bits** — the case that matters for MXFP4 — the picture is different, and the
package prices it directly at the format's 5-bit signed element width:

```@example hls
for o in mxfp4_multiply_options(MXFP4)
    println(rpad(o.method, 12), " depth ", lpad(o.depth, 2),
            "  cells ", lpad(o.cells, 3), "  ", o.note)
end
```

**The table wins, and Booth loses outright**: radix-4 recoding costs *more* cells
(35 against 25) to buy a single gate level. RR4 digit recoding of a 4-bit operand is pure
overhead — 16 levels for 36 cells. A 64-entry ROM over the eight magnitudes, with the
sign as one XOR, is 2 levels, and it is exactly what synthesis infers from a small
constant lookup table.

This is the concrete form of a general result: **the redundant-arithmetic techniques in
this package are width-dependent, and at 4 bits there is nothing left for them to buy.**
They pay at 32 and 64 bits, in datapaths you are not building.

## 7. The MAC and the accumulator — the only real decision

Everything above is a lookup. This is the part that requires judgement, because it is the
one genuine recurrence in the design.

### Why a float accumulator will not schedule

```c++
float acc = 0;
for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
    acc += a[i] * b[i];        // <-- will not meet II=1
}
```

The loop-carried dependency on `acc` has the latency of a floating-point add — three to
five cycles — and the pragma asks for one. The tool will report that it cannot achieve
the initiation interval and will silently settle for `II = 4` or worse. This is the most
common performance bug in an HLS accelerator, and it is invisible in the C++.

### The three fixes, in order of preference

**First: accumulate in integer.** For a block format this is free, because the block sum
is already exact in `core_bits`. An integer add has single-cycle latency, so `II = 1`
falls out with no restructuring at all. **You get exactness and throughput from the same
decision** — which almost never happens.

**Second: partial accumulators.** Where you genuinely need floating point, unroll by four
to eight and keep independent accumulators, reducing them at the end. This is the
standard HLS idiom, and it is not a trade-off: the same restructuring measured on real
silicon in [§13b of the cost model](@ref "13b. Three accumulation schedules, timed and — the part the benchmark omits — measured for error")
was **8× faster and 5–41× more accurate** than the serial chain, because eight partial
sums each stay smaller and absorb less.

**Third, and rarely: redundant accumulation.** Only if the first two fail — a dependent
chain that must hit `II = 1` at a clock the tool cannot otherwise meet. Price the extra
state before you commit.

### Sizing the accumulator across blocks

Within one block the sum is exact in 14 bits because every element shares one scale.
*Across* blocks the scales differ, and you choose:

```@example hls
for N in (32, 256, 4096, 65536)
    a = hls_accumulator_bits(MXFP4; blocks = cld(N, 32))
    println("N = ", lpad(N, 6), "   blocks ", lpad(a.blocks, 5),
            "   per_block ", a.per_block, " bits (", a.per_block_roundings,
            " roundings)   wide_fixed ", a.wide_fixed, " bits (exact)")
end
```

Read the last two columns against each other, because this is a real design choice with a
clean answer:

- **`per_block`** — 14 bits, convert to float at each block boundary, accumulate in float.
  Costs `N/K` roundings instead of `N`, which is already a large win over element-wise
  accumulation, and it is what most implementations do.
- **`wide_fixed`** — hold the block sums in a fixed-point accumulator wide enough to span
  the scale range you actually use. **57 bits buys an exactly-rounded dot product of
  length 65 536**, with `II = 1`, and removes accumulation error from the error budget
  entirely.

Fifty-seven bits of accumulator per lane is a few hundred square microns at 3 nm. Against
that, it deletes a whole class of numerical risk, and it means your silicon and your
reference model agree bit for bit — which is worth more during bring-up than it looks on
a spreadsheet. **For an XPU I would take the wide accumulator.**

The `scale_span` keyword is the honest part of this: 32 octaves is a default, not a
measurement. E8M0's full range would need 275 bits, which is the classic Kulisch
accumulator and is impractical. Measure the scale range your workloads actually produce,
size the accumulator for that, and saturate beyond it.

## 8. Dot products

Within a block: fully unroll `K`, let the tool build the reduction tree, pipeline at
`II = 1`.

```@example hls
for o in mxfp4_reduction_options(32)
    println(rpad(o.method, 22), " levels ", lpad(o.levels, 2),
            "  depth ", lpad(o.depth, 3), "  ", o.note)
end
```

The gap that survives into HLS is **tree against sequential — 12 depth against 124**, and
you get it from `#pragma HLS UNROLL`, not from choosing a compressor. The difference
between the three tree rows is one the tool decides and you should not spend time on.

**The binding constraint is memory bandwidth, not arithmetic.** To sustain `K` MACs per
cycle you need `K` elements per cycle out of storage, and a reduction tree fed by a
single-port array simply idles. The pragma that decides whether the design works is
`ARRAY_PARTITION` / `ARRAY_RESHAPE`, not anything about the adders.

Scaling in `N` is then linear at constant `II`, and the wall is bandwidth.

## 9. Soft-max and the operations that are not MACs

Worth a sentence because they are where accelerators lose their advantage. Soft-max is
three streaming passes — max, exponentiate-and-sum, normalise — and its first phase is a
**comparison** reduction, not an addition. Redundancy helps addition and does not help
comparison: a redundant representation removes the carry chain from `+` and leaves it in
`<`, which is why [`sign_detect_depth`](@ref) is the asymmetry the cost model keeps
returning to.

In HLS terms: soft-max is bandwidth-bound and latency-bound, not arithmetic-bound. Fuse
it into the producer if you can, so the tensor is not written out and read back.

## 10. Memory, and why it settles the argument

```@example hls
hls_energy_compare(; formats = (MXFP4, NVFP4, MXINT4))
```

Units are relative — the ratios are the reading, not the magnitudes. The column that
matters is the last one.

**Between 65 % and 70 % of the energy of a block MAC is fetching the operands, even when
they come from local SRAM. From DRAM it is 98.7 %.** No arithmetic decision on this page
moves a design in which two thirds of the energy was spent before the multiplier saw the
data — and, from §3, that fraction *rises* at 3 nm rather than falling.

Which puts the priorities in an order that is uncomfortable if you like arithmetic:

1. **Do not fetch it.** Dataflow, tiling, fusion, reuse. Every operand read once instead
   of twice beats every scheme on this page combined.
2. **Fetch fewer bits.** MXFP4 is 4.25 bits per value against FP32's 32 — a **7.53×**
   reduction in the term that dominates. That is the actual case for block formats, and
   it is a memory argument, not an arithmetic one.
3. **Then** worry about the datapath.

Two HLS-specific cautions on the second point, because they are easy to lose by accident:

- **Pack the nibbles.** A naive `ap_uint<4> w[N]` can be padded to a byte per element by
  the memory-mapping step, throwing away half the win before you start. Use
  `ARRAY_RESHAPE`, or pack explicitly into a wide word.
- **Keep the scale on its own port.** One 8-bit scale and 32 nibbles sharing a single
  memory port serialises the block read — the array is idle waiting for a scalar.

## 11. The code, with every width derived

```@example hls
hls_pragmas(MXFP4; blocks = 128)
```

The pragma names follow Vitis HLS; the structure is the same in Catapult and Stratus.
The part that is actually derived — and the part worth copying — is the widths.

## 12. Summary: what to pick

| | Best scheme | Why |
|:---|:---|:---|
| **Adder** | inferred `+`, minimum derived width | you cannot pick the architecture; width is linear in everything |
| **Multiply ≥ 8 bit** | inferred `*` | the datapath library already does Booth–Wallace |
| **Multiply 4 bit** | **table lookup**, 2 levels | Booth costs *more* cells here; RR4 is pure overhead |
| **MAC** | integer, product never rounded | cheaper *and* exact |
| **Block accumulator** | `ap_int<14>`, exact | `K × 144` fits; no rounding inside a block |
| **Dot accumulator** | **wide fixed, ~57 bits** | exact at any `N`, `II = 1`, deletes an error class |
| **Reduction** | `UNROLL` + inferred tree | 12 depth against 124; the tree shape is the tool's job |
| **Redundant arithmetic** | **do not** | no HLS flow infers it; extra state costs double at 3 nm |
| **Memory** | MXFP4 packed, scale separate | 7.53×, and it is 65–99 % of the energy |

The uncomfortable conclusion for a page in this package: **redundant arithmetic — the
subject the cost model spends most of its effort on — is the wrong tool for an HLS flow.**
It pays in hand-built ASIC datapaths where you own the gates, at widths of 32 bits and
up. In HLS at 4 bits you own widths, recurrences and memory layout, and all of the gain
is there.

## What this page does not know

Stated plainly, because the difference between the two halves matters:

* Every **width, cell count and depth** above is computed by the package from the format
  definitions, and is pinned by the test suite.
* Every claim about **what a synthesis tool infers**, about carry chains, DSP mapping,
  pragma behaviour and scheduling, is architectural reasoning. No Vitis, Catapult or
  Stratus run backs it, and no synthesis report was read.
* The **energy weights are first-order and shaped by a 45 nm survey**, not by 3 nm PPA
  data. They are used only for ratios, the ratios are stated as ranges, and §3 argues
  they are conservative at an advanced node — but re-run [`hls_energy`](@ref) with your
  own numbers before you plan silicon around them.

The first bullet you can trust as arithmetic. The second and third are a starting
hypothesis to validate against your own flow.

The functions are [`hls_types`](@ref), [`hls_accumulator_bits`](@ref),
[`hls_operator_table`](@ref), [`hls_energy`](@ref), [`hls_energy_compare`](@ref),
[`hls_pragmas`](@ref) and [`hls_report`](@ref), with [`HLSDecl`](@ref),
[`HLS_CONTROL`](@ref) and [`ENERGY_WEIGHTS`](@ref) — all under
[API — Analysis](@ref).
