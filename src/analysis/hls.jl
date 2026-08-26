# ---------------------------------------------------------------------------
# High-level synthesis: what an HLS flow actually decides, and what it leaves
# to you.
#
# The rest of this analysis layer prices gates. HLS does not let you place
# gates — you write `a + b` and the datapath library picks an architecture from
# your clock constraint. So this file prices the three things you *do* control:
#
#   1. WIDTH      — the ap_int/ap_fixed declarations, derived from the format
#   2. RECURRENCE — whether the accumulator loop can reach II = 1
#   3. MEMORY     — how many bits move per useful arithmetic operation
#
# Everything here reuses the currencies of costmodel.jl (cells, state, depth)
# rather than inventing a parallel one, so an HLS estimate and a gate estimate
# can be read against each other.
# ---------------------------------------------------------------------------

"""
    HLSDecl

One synthesizable declaration: the C++ type an HLS flow should see for one role
in the datapath, with the range that forces the width.

Widths are **derived from the format**, never rounded up to a machine word. That is
the entire point: `ap_int<14>` and `int32_t` hold the same values for an MXFP4 block
sum, and one of them costs 2.3× the flops and wires.
"""
struct HLSDecl
    role::String
    decl::String
    bits::Int
    range::String
    note::String
end

function Base.show(io::IO, d::HLSDecl)
    @printf(io, "%-22s %-16s %3d bits  %-18s %s",
            d.role, d.decl, d.bits, d.range, d.note)
end

_ap(bits, signed) = signed ? "ap_int<$(bits)>" : "ap_uint<$(bits)>"

# stored width of one element or scale — FloatFormat and IntFormat both appear here
_stored_bits(f::FloatFormat) = (f.signed ? 1 : 0) + f.ebits + f.mbits
_stored_bits(f::IntFormat) = f.bits

"""
    hls_types(bf::BlockFormat = MXFP4; blocks::Integer = 1) -> Vector{HLSDecl}

The complete set of datapath declarations for a block-format MAC, derived from
[`mxfp4_widths`](@ref).

`blocks` is how many blocks are summed into one accumulator before conversion — the
dot-product length divided by `K`.

```jldoctest
julia> [d.bits for d in hls_types(MXFP4)]
5-element Vector{Int64}:
  5
  9
 14
 14
  8
```

The declarations that matter, and why:

- **element** — the E2M1 grid times two is an integer set, so an element is a signed
  5-bit integer, not a float. No exponent logic anywhere in the inner loop.
- **product** — 9 bits, because `12 × 12 = 144` is the largest magnitude.
- **block accumulator** — 14 bits holds `K × 144` exactly. **An MXFP4 block dot product
  is an exact integer computation**; there is no rounding to model inside a block.
- **scale** — E8M0 is a raw 8-bit exponent. It is applied once per block, not per element.
"""
function hls_types(bf::BlockFormat = MXFP4; blocks::Integer = 1)
    w = mxfp4_widths(bf)
    nb = max(1, Int(blocks))
    accb = w.core_bits + (nb > 1 ? ceil(Int, log2(nb)) : 0)
    [HLSDecl("element", _ap(w.elem_bits, true), w.elem_bits,
             "[-12, 12]", "E2M1 grid × 2 — an integer, not a float"),
     HLSDecl("product", _ap(w.product_bits, true), w.product_bits,
             "[-144, 144]", "$(w.distinct_products) distinct values exist"),
     HLSDecl("block accumulator", _ap(w.core_bits, true), w.core_bits,
             "[-$(w.core_sum_max), $(w.core_sum_max)]",
             "K = $(bf.K) products, summed EXACTLY"),
     HLSDecl("dot accumulator", _ap(accb, true), accb,
             "$(nb) block(s)", nb > 1 ? "+$(accb - w.core_bits) bits for $(nb) blocks" :
                                       "one block"),
     HLSDecl("block scale", _ap(bf.scale.ebits + bf.scale.mbits, false),
             bf.scale.ebits + bf.scale.mbits, "$(bf.scale.name)",
             "applied once per block, never per element")]
end

"""
    hls_accumulator_bits(bf::BlockFormat = MXFP4; blocks::Integer = 1,
                         scale_span = nothing) -> NamedTuple

How wide the accumulator has to be, for each of the three ways to sum across blocks.

Within one block the sum is exact in `core_bits` — every element shares one scale, so
the arithmetic is integer. **Across** blocks the scales differ, and you must choose:

| mode | width | rounding |
|:---|---:|:---|
| `per_block` | `core_bits`, then a float add per block | `N/K` roundings, not `N` |
| `wide_fixed` | `core_bits + span + log2(blocks)` | **none** — exact |
| `full_kulisch` | `core_bits + full exponent range` | none, and impractical |

`scale_span` is how many octaves of block scale you are willing to hold exactly. Real
tensors do not use E8M0's full 254-octave range; a span of 32–64 covers activations and
weights in practice, and the honest move is to **measure it** rather than assume it, then
saturate the rest.

The `wide_fixed` row is the design worth taking seriously on an XPU: a few tens of extra
flip-flops per lane buys an exactly-rounded dot product of any length, with `II = 1`,
and removes accumulation error from the error budget entirely.
"""
function hls_accumulator_bits(bf::BlockFormat = MXFP4; blocks::Integer = 1,
                              scale_span = nothing)
    w = mxfp4_widths(bf)
    nb = max(1, Int(blocks))
    lg = nb > 1 ? ceil(Int, log2(nb)) : 0
    span = scale_span === nothing ? 32 : Int(scale_span)
    full = 2^bf.scale.ebits - 2                     # E8M0: 254 usable octaves
    (core_bits = w.core_bits,
     blocks = nb,
     per_block = w.core_bits,
     per_block_roundings = nb,
     wide_fixed = w.core_bits + span + lg,
     wide_fixed_span = span,
     full_kulisch = w.core_bits + full + lg,
     exact_within_block = true)
end

"""
    HLS_CONTROL

What an HLS flow decides for you, and what it leaves in your hands — the table this
whole page turns on.

Writing `a + b` on an `ap_int<W>` hands the architecture to the datapath library, which
picks ripple / carry-select / prefix from the clock constraint. Hand-coding a prefix
adder does not override that; it produces bit-level code the scheduler can no longer
restructure, and usually loses twice.
"""
const HLS_CONTROL = [
    (op = "addition",      tool_picks = "ripple / CSA / prefix, from the clock constraint",
     you_control = "operand WIDTH",
     lever = "area and energy are linear in W; nothing else you write matters"),
    (op = "multiplication", tool_picks = "Booth radix-4 + Wallace, or a DSP/hard macro",
     you_control = "operand WIDTH, and whether it is a table",
     lever = "energy ≈ Wa·Wb — narrowing operands is quadratic"),
    (op = "MAC",           tool_picks = "fusion of the multiply into the reduction",
     you_control = "whether the product is rounded before accumulating",
     lever = "not rounding is cheaper AND more accurate"),
    (op = "accumulation",  tool_picks = "nothing — this is a recurrence",
     you_control = "the loop structure: II, partial accumulators, datatype",
     lever = "the single biggest latency decision in the design"),
    (op = "dot product",   tool_picks = "the reduction tree shape, given UNROLL",
     you_control = "UNROLL factor and ARRAY_PARTITION",
     lever = "memory bandwidth, not arithmetic, is the binding constraint"),
    (op = "memory",        tool_picks = "SRAM macro mapping",
     you_control = "packing, banking, and how many bits move per MAC",
     lever = "dominates everything above at any modern node"),
]

"""
    hls_operator_table(; io = stdout) -> Nothing

Print [`HLS_CONTROL`](@ref): per operation, what the tool decides and what you decide.
"""
function hls_operator_table(; io::IO = stdout)
    for r in HLS_CONTROL
        @printf(io, "  %-15s\n", uppercase(r.op))
        @printf(io, "     tool picks : %s\n", r.tool_picks)
        @printf(io, "     you control: %s\n", r.you_control)
        @printf(io, "     →            %s\n\n", r.lever)
    end
    nothing
end

"""
    ENERGY_WEIGHTS

First-order energy weights, relative to **one full-adder cell switching**.

| term | weight | what it stands for |
|:---|---:|:---|
| `cell` | `1` | one combinational full-adder-equivalent |
| `flop` | `3` | one flip-flop, including its share of the clock tree |
| `sram` | `40` | one bit read from a local SRAM macro |
| `dram` | `1300` | one bit from off-chip |

The *ratios* are the intended reading, and they come from the shape of the Horowitz
45 nm survey that [`plot_energy_bars`](@ref) draws: an SRAM read is one to two orders of
magnitude above an arithmetic op, and DRAM is another one to two above that.

!!! warning "These get worse, not better, at an advanced node"
    Logic energy scales with the process; **SRAM and wires largely do not**. Bitcell
    area has nearly stalled across recent nodes and interconnect RC gets worse, so at
    3 nm the `sram`/`cell` and `dram`/`cell` ratios are **larger** than these 45 nm-shaped
    weights, not smaller. Every conclusion below that favours moving fewer bits gets
    *stronger* on an advanced node. Treat the weights as a floor and re-run
    [`hls_energy`](@ref) with your own PPA numbers.
"""
const ENERGY_WEIGHTS = (cell = 1.0, flop = 3.0, sram = 40.0, dram = 1300.0)

"""
    hls_energy(bf::BlockFormat = MXFP4; blocks::Integer = 1, weights = ENERGY_WEIGHTS,
               from_dram::Bool = false, multiply = :lut) -> NamedTuple

Where the energy of one block MAC actually goes, in the relative units of
[`ENERGY_WEIGHTS`](@ref).

Counts, per block of `K` elements: the multiplies (via
[`mxfp4_multiply_options`](@ref)), the reduction to one sum (via
[`mxfp4_reduction_options`](@ref)), the accumulator flops, and the bits that had to be
read to feed it.

The number to look at is `memory_fraction`. **If it is above one half, no arithmetic
choice on this page can move your design** — and for every block format at every
sensible operating point, it is.
"""
function hls_energy(bf::BlockFormat = MXFP4; blocks::Integer = 1,
                    weights = ENERGY_WEIGHTS, from_dram::Bool = false,
                    multiply::Symbol = :lut)
    w = mxfp4_widths(bf)
    K = bf.K
    mo = mxfp4_multiply_options(bf)
    m = mo[findfirst(x -> x.method === multiply, mo)]
    red = mxfp4_reduction_options(K)
    rt = red[findfirst(x -> x.method === :carry_save_tree, red)]

    mul_cells = K * m.cells
    red_cells = (K - 1) * w.product_bits          # a 3:2 tree is ~one cell per bit merged
    acc = hls_accumulator_bits(bf; blocks)
    flops = acc.per_block + w.core_bits           # pipeline register + accumulator

    # bits that must be read to do this block once — the element may be a float
    # format (E2M1) or an integer one (INT4), so ask for its stored width
    bits_in = K * _stored_bits(bf.elem) + _stored_bits(bf.scale)
    memw = from_dram ? weights.dram : weights.sram

    e_mul = mul_cells * weights.cell
    e_red = red_cells * weights.cell
    e_flop = flops * weights.flop
    e_mem = bits_in * memw
    total = e_mul + e_red + e_flop + e_mem
    (format = bf.name, K = K, multiply = multiply,
     bits_read = bits_in, bits_per_element = bits_in / K,
     e_multiply = e_mul, e_reduce = e_red, e_registers = e_flop, e_memory = e_mem,
     e_total = total,
     arithmetic_fraction = (e_mul + e_red + e_flop) / total,
     memory_fraction = e_mem / total,
     source = from_dram ? :dram : :sram)
end

"""
    hls_energy_compare(; formats = (MXFP4, NVFP4), io = stdout) -> Nothing

The energy split for each block format, from SRAM and from DRAM — the table that decides
where optimisation effort belongs.
"""
function hls_energy_compare(; formats = (MXFP4, NVFP4), io::IO = stdout)
    @printf(io, "  %-8s %-6s %8s %10s %10s %10s %10s\n",
            "format", "src", "bits/el", "arith", "memory", "total", "mem %")
    println(io, "  " * "─"^72)
    for bf in formats, src in (false, true)
        e = hls_energy(bf; from_dram = src)
        @printf(io, "  %-8s %-6s %8.3f %10.0f %10.0f %10.0f %9.1f%%\n",
                e.format, src ? "DRAM" : "SRAM", e.bits_per_element,
                e.e_multiply + e.e_reduce + e.e_registers, e.e_memory, e.e_total,
                100 * e.memory_fraction)
    end
    nothing
end

"""
    hls_pragmas(bf::BlockFormat = MXFP4; blocks::Integer = 1, io = stdout) -> Nothing

Emit the datatype and pragma sketch for a block MAC — the shape of the code, with every
width taken from the format rather than guessed.

This is deliberately a *sketch*: the pragma names follow Vitis HLS, the structure is the
same in Catapult and Stratus, and the widths are the part that is actually derived.
"""
function hls_pragmas(bf::BlockFormat = MXFP4; blocks::Integer = 1, io::IO = stdout)
    w = mxfp4_widths(bf)
    acc = hls_accumulator_bits(bf; blocks)
    K = bf.K
    sb = bf.scale.ebits + bf.scale.mbits
    ty(t, b) = rpad("$(t)<$(b)>", 13)
    println(io, """
    // ---- types: every width derived from $(bf.name), none rounded to a word ----
    typedef $(ty("ap_int", w.elem_bits)) elem_t;   // $(bf.elem.name) grid x2: [-12, 12]
    typedef $(ty("ap_int", w.product_bits)) prod_t;   // [-144, 144]
    typedef $(ty("ap_int", w.core_bits)) core_t;   // K=$(K) products, EXACT
    typedef $(ty("ap_uint", sb)) scale_t;  // $(bf.scale.name), one per block
    typedef $(ty("ap_int", acc.wide_fixed)) acc_t;    // wide-fixed, $(acc.wide_fixed_span) octaves of scale span

    // ---- storage: pack the nibbles, keep the scale on its own port ----
    elem_t  w_mant[N];
    scale_t w_scale[N / $(K)];
    #pragma HLS ARRAY_RESHAPE   variable=w_mant  cyclic factor=$(K) dim=1
    #pragma HLS ARRAY_PARTITION variable=w_scale cyclic factor=2    dim=1

    // ---- one block: unroll fully, integer tree, exact ----
    core_t block_dot(const elem_t a[$(K)], const elem_t b[$(K)]) {
    #pragma HLS INLINE off
    #pragma HLS PIPELINE II=1
        core_t s = 0;
        for (int k = 0; k < $(K); ++k) {
        #pragma HLS UNROLL
            s += (prod_t)a[k] * (prod_t)b[k];   // infer it; do NOT hand-code Booth
        }
        return s;                                // no rounding happened
    }""")
    nothing
end

"""
    hls_report(bf::BlockFormat = MXFP4; blocks::Integer = 1, io = stdout) -> Nothing

Everything this file knows about one block format, in the order a designer needs it:
the declarations, the accumulator options, the control table, and the energy split.
"""
function hls_report(bf::BlockFormat = MXFP4; blocks::Integer = 1, io::IO = stdout)
    println(io, "  === $(bf.name), $(blocks) block(s) per accumulator ===\n")
    println(io, "  datatypes")
    for d in hls_types(bf; blocks); println(io, "    ", d); end
    a = hls_accumulator_bits(bf; blocks)
    println(io, "\n  accumulator options")
    @printf(io, "    %-14s %5d bits   %s\n", "per_block", a.per_block,
            "$(a.blocks) float rounding(s) across blocks")
    @printf(io, "    %-14s %5d bits   %s\n", "wide_fixed", a.wide_fixed,
            "EXACT over $(a.wide_fixed_span) octaves of scale span")
    @printf(io, "    %-14s %5d bits   %s\n", "full_kulisch", a.full_kulisch,
            "exact over all of $(bf.scale.name) — impractical, shown for scale")
    println(io, "\n  what HLS decides for you")
    hls_operator_table(; io)
    println(io, "\n  energy split (relative units; ratios are the reading)")
    hls_energy_compare(; formats = (bf,), io)
    nothing
end
