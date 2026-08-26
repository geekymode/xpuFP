# ---------------------------------------------------------------------------
# Gate levels → CYCLES.  A different question, and one that needs a new parameter.
#
# Depth is a property of the logic.  Cycles are a property of the logic AND the clock,
# so nothing here is meaningful without stating how many gate levels fit in one cycle.
# That parameter is `levels_per_cycle` (L) throughout, and it is the whole story:
#
#   combinational, pipelined   latency = ⌈D/L⌉ cycles, initiation interval 1
#   iterative (one step/cycle) latency = steps · ⌈d_step/L⌉, II the same
#
# The consequence runs against redundancy, and it is the point of this file:
#
#   A depth advantage converts into CYCLES only when the per-step depth EXCEEDS the
#   cycle budget.  Below that it converts into slack — clock headroom, or lower power
#   at the same clock — not into fewer cycles.  At L = 16 a 64-bit prefix adder (6
#   levels), an RR4 adder (3) and a carry-save adder (1) all take exactly ONE cycle,
#   and a sequential accumulator over N terms takes N cycles for all three.  Only the
#   ripple adder (65 levels) still costs 5.
#
# And in a throughput-oriented accelerator the picture tilts further: a pipelined MAC
# retires one result per cycle whatever its depth, so depth buys pipeline registers,
# not throughput.
#
# Depths come from `analysis/costmodel.jl` and `analysis/mxfp4cost.jl`; this file only
# divides them by a clock.  Every caveat on those models applies here, plus one more:
# the mapping assumes the logic can be cut cleanly at any level, which real retiming
# cannot always do.
# ---------------------------------------------------------------------------

"""    DEFAULT_LEVELS_PER_CYCLE

16 full-adder-equivalent levels in one cycle — a mid-range choice for a synchronous
datapath, and the default for every `*_cycles` function.

The number is a **budget, not a measurement**. Aggressive high-frequency designs run
nearer 8; deeply combinational or low-frequency ones run 32 or more. Because it decides
whether a depth advantage becomes cycles or merely slack, sweep it with
[`clock_sensitivity`](@ref) rather than trusting one value."""
const DEFAULT_LEVELS_PER_CYCLE = 16

"""
    cycles_for(depth::Integer; levels_per_cycle = DEFAULT_LEVELS_PER_CYCLE) -> Int

Gate levels to cycles: `⌈depth / L⌉`, never less than 1.

This is the only conversion in the package, and it is deliberately trivial — the
modelling content is in the *depth*, and in the choice of `L`.

```jldoctest
julia> cycles_for(65; levels_per_cycle = 16), cycles_for(3; levels_per_cycle = 16)
(5, 1)
```
"""
cycles_for(depth::Integer; levels_per_cycle::Integer = DEFAULT_LEVELS_PER_CYCLE) =
    max(1, cld(Int(depth), Int(levels_per_cycle)))

# ---------------------------------------------------------------------------
# What a "level" is in physical terms, and a worked FP32 multiply.
# ---------------------------------------------------------------------------

"""    FO4_PER_LEVEL

`(2, 3)` — the conventional range of **FO4 delays** in one full-adder-equivalent level.

FO4 (the delay of an inverter driving four identical inverters) is the standard
process-portable unit of logic delay. A full adder's sum path is two XORs, and an XOR
is roughly 1.5–2 FO4, so one level of this package's model lands at **2–3 FO4**:

| gate | ≈ FO4 |
|:---|---:|
| 2-input NAND / NOR | 1 |
| 2-input XOR | 1.5 – 2 |
| AOI prefix combine cell | 1 – 1.5 |
| 2:1 mux | 1 – 1.5 |
| **full adder / 3:2 compressor** | **2 – 3** |

Approximate and process-dependent — use it to sanity-check a clock budget, not to
predict a frequency."""
const FO4_PER_LEVEL = (2, 3)

"""
    fo4_range(levels::Integer) -> Tuple{Int,Int}

The FO4 delay a given number of levels corresponds to, as a `(low, high)` range from
[`FO4_PER_LEVEL`](@ref).

Useful for checking whether a `levels_per_cycle` budget is realistic. Aggressive CPU
pipelines run roughly **12–20 FO4** per cycle; throughput-oriented accelerator datapaths
are more relaxed, 20–40. So:

| `L` | FO4 per cycle | design point |
|---:|:---|:---|
| 4 | 8 – 12 | very aggressive |
| 8 | 16 – 24 | aggressive |
| 16 | 32 – 48 | relaxed / throughput-oriented |
| 32 | 64 – 96 | very relaxed |

The package default of 16 is therefore on the **relaxed** side. That matters, and
[`clock_sensitivity`](@ref) shows why: redundancy's depth win is invisible in cycles at
`L = 8` and `L = 16`, and only becomes a real cycle saving near `L = 4`.
"""
fo4_range(levels::Integer) = (Int(levels) * FO4_PER_LEVEL[1],
                              Int(levels) * FO4_PER_LEVEL[2])

"""
    float_multiply_stages(f::FloatFormat = FP32) -> Vector{NamedTuple}

The gate-level budget of one `f × f` floating-point multiply, stage by stage.

The stages are exactly those [`fpmul`](@ref) walks through when it executes the
arithmetic — unpack, sign, exponent add, significand multiply, normalise, round, pack —
so the cost model and the value model describe the same datapath.

Each row carries `critical`: whether the stage sits on the critical path. The sign and
exponent stages do not, because they run in parallel with a significand multiply that is
an order of magnitude deeper. **That is the whole shape of an FP multiplier: one big
integer multiply with a little exponent arithmetic alongside it.**

```julia
julia> sum(s.levels for s in float_multiply_stages(FP32) if s.critical)
22
```
"""
function float_multiply_stages(f::FloatFormat = FP32)
    p = f.mbits + 1                       # significand width, implicit bit included
    prod_w = 2p
    eb = f.ebits
    lg(n) = max(1, ceil(Int, log2(max(2, n))))
    mc = multiply_costs(p)
    booth = mc[findfirst(x -> x.method === :booth_wallace, mc)]
    [(stage = "unpack", levels = 1, critical = true,
      note = "split fields, prepend the implicit 1 — wiring plus a subnormal mux"),
     (stage = "sign", levels = 1, critical = false,
      note = "one XOR of the two sign bits; parallel with everything"),
     (stage = "exponent add", levels = lg(eb), critical = false,
      note = "$(eb)-bit add of E_a + E_b − bias, a small prefix adder; parallel"),
     (stage = "significand multiply", levels = booth.depth, critical = true,
      note = "$(p)×$(p) Booth-Wallace: 1 recode + tree + exit CPA over $(prod_w) bits"),
     (stage = "normalise", levels = 1, critical = true,
      note = "product is in [1,4), so a 1-bit conditional right shift — a 2:1 mux"),
     (stage = "round", levels = lg(p), critical = true,
      note = "round-to-nearest-even: conditional increment of $(p) bits"),
     (stage = "post-normalise", levels = 1, critical = true,
      note = "rounding can carry out (1.111→10.000): one more shift and exponent bump"),
     (stage = "pack / specials", levels = 2, critical = true,
      note = "Inf / NaN / zero / overflow selection muxes")]
end

"""
    float_multiply_depth(f::FloatFormat = FP32) -> Int

Total critical-path levels of one `f × f` multiply — the `critical` rows of
[`float_multiply_stages`](@ref), summed. **22 for FP32.**
"""
float_multiply_depth(f::FloatFormat = FP32) =
    sum(s.levels for s in float_multiply_stages(f) if s.critical)

"""
    CycleBudget

One implementation of one operation, priced in cycles.

# Fields
- `op`, `method`, `n` — what was run.
- `levels_per_cycle` — the clock budget the conversion used. **Every cycle figure is
  meaningless without it.**
- `depth` — gate levels, from the depth model.
- `latency` — cycles from input to result.
- `ii` — initiation interval: cycles before the unit can accept the next operand set.
  `1` for a pipelined combinational unit, `latency` for one that occupies itself.
- `throughput` — results per cycle, `1/ii`.
- `pipelined` — whether `ii < latency`.

!!! note "Latency and throughput are different questions"
    A pipelined multiplier with `latency = 3` and `ii = 1` retires one product every
    cycle after a 3-cycle fill. For a long dot product the `ii` column decides the
    runtime and the `latency` column decides only the ramp.
"""
struct CycleBudget
    op::Symbol
    method::Symbol
    n::Int
    levels_per_cycle::Int
    depth::Int
    latency::Int
    ii::Int
    throughput::Float64
    pipelined::Bool
    detail::String
end

_cb(op, meth, n, L, depth, ii, detail) = begin
    lat = cycles_for(depth; levels_per_cycle = L)
    CycleBudget(op, meth, Int(n), Int(L), Int(depth), lat, Int(ii), 1 / ii,
                ii < lat, detail)
end

"""
    add_cycles(w::Integer; levels_per_cycle = DEFAULT_LEVELS_PER_CYCLE) -> Vector{CycleBudget}

One `w`-bit addition, in cycles, for each schedule in [`add_costs`](@ref).

The table is where the central caveat of this file shows itself: at a normal clock
budget every scheme except ripple collapses to **one cycle**, because 6, 3 and 1 levels
all fit inside 16. Carry-free's depth advantage over a prefix adder is real in gate
levels and **invisible in cycles** at this `L`.
"""
function add_cycles(w::Integer; levels_per_cycle::Integer = DEFAULT_LEVELS_PER_CYCLE)
    [_cb(:add, b.method, w, levels_per_cycle, b.depth, 1, b.detail)
     for b in add_costs(w)]
end

"""
    multiply_cycles(n::Integer; levels_per_cycle = DEFAULT_LEVELS_PER_CYCLE) -> Vector{CycleBudget}

An `n × n` multiply, in cycles, for the schedules in [`multiply_costs`](@ref).

`shift_add` is modelled as **iterative** — one partial product per cycle, `n` cycles,
occupying the unit throughout — because that is what makes it small. The tree
multipliers are combinational and pipelined: `⌈D/L⌉` cycles of latency and one product
per cycle thereafter.

That difference dominates everything else in the table. Booth's depth saving over a
plain Wallace tree is 1 level, which at any sane `L` is **zero cycles**; what Booth
actually buys is area, and what the tree buys over shift-add is `n`-fold throughput.
"""
function multiply_cycles(n::Integer; levels_per_cycle::Integer = DEFAULT_LEVELS_PER_CYCLE)
    N = Int(n)
    out = CycleBudget[]
    for b in multiply_costs(N)
        if b.method === :shift_add
            # iterative: one row per cycle; each row's add must fit the budget
            per = cycles_for(max(1, b.depth ÷ max(1, N)); levels_per_cycle)
            steps = N * per
            push!(out, CycleBudget(:multiply, :shift_add, N, Int(levels_per_cycle),
                                   b.depth, steps, steps, 1 / steps, false,
                                   "$(N) rows, one per cycle × $(per) cycle(s) each"))
        else
            push!(out, _cb(:multiply, b.method, N, levels_per_cycle, b.depth, 1,
                           b.detail))
        end
    end
    out
end

"""
    accumulate_cycles(N::Integer, w::Integer; levels_per_cycle = DEFAULT_LEVELS_PER_CYCLE,
                      schedule = :sequential) -> Vector{CycleBudget}

Accumulating `N` terms into a `w`-bit register, in cycles.

A **sequential** accumulator has a loop-carried dependence, so it cannot be pipelined:
it costs `N × ⌈d/L⌉` cycles where `d` is the per-term add depth. This is the one place
in the package where a depth advantage converts directly into cycles — but **only when
`d > L`**. At `w = 64, L = 16` the ripple adder needs 5 cycles a term and everything
else needs 1, so carry-free ties prefix and carry-save at `N` cycles.

A **tree** schedule is `⌈log₂N⌉` combines deep and pipelines, so its latency is
`⌈D/L⌉` and its `ii` is 1.
"""
function accumulate_cycles(N::Integer, w::Integer;
                           levels_per_cycle::Integer = DEFAULT_LEVELS_PER_CYCLE,
                           schedule::Symbol = :sequential)
    Nn = Int(N); W = Int(w); L = Int(levels_per_cycle)
    per_term = Dict(:ripple => W + 1, :prefix => max(1, ceil(Int, log2(max(2, W)))),
                    :carry_save => 1, :rr4_carry_free => 3)
    exitcpa = max(1, ceil(Int, log2(max(2, W))))
    out = CycleBudget[]
    for meth in (:ripple, :prefix, :rr4_carry_free, :carry_save)
        d = per_term[meth]
        needs_exit = meth in (:rr4_carry_free, :carry_save)
        if schedule === :sequential
            pc = cycles_for(d; levels_per_cycle = L)
            tot = Nn * pc + (needs_exit ? cycles_for(exitcpa; levels_per_cycle = L) : 0)
            push!(out, CycleBudget(:accumulate, meth, Nn, L, d * Nn, tot, tot,
                                   1 / tot, false,
                                   "$(pc) cycle(s)/term × $(Nn)" *
                                   (needs_exit ? " + exit" : "")))
        elseif schedule === :tree
            lv = max(1, ceil(Int, log2(max(2, Nn))))
            D = lv * d + (needs_exit ? exitcpa : 0)
            push!(out, _cb(:accumulate, meth, Nn, L, D, 1,
                           "$(lv) tree levels × $(d)" * (needs_exit ? " + exit" : "")))
        else
            throw(ArgumentError("accumulate_cycles: schedule must be :sequential or :tree"))
        end
    end
    out
end

"""
    dot_cycles(N::Integer; levels_per_cycle = DEFAULT_LEVELS_PER_CYCLE, bf = MXFP4,
               acc_bits = 32, schedule = :tree, macs = nothing) -> NamedTuple

A length-`N` MXFP4 dot product, in cycles, for the three cross-block datapaths.

Two regimes, and they answer different questions:

* **Latency** — `⌈D/L⌉` from [`mxfp4_dot_costs`](@ref), assuming enough hardware to do
  every product at once. This is what the depth model converts to.
* **Throughput** — with `macs` multiply-accumulate units instead of `N`, the runtime is
  bound by `N/macs` cycles of issue, not by depth at all. Pass `macs` to see it.

In an accelerator the second regime is the real one, and in it **depth buys pipeline
registers rather than cycles**: a pipelined MAC array retires one product per cycle
whatever its latency.
"""
function dot_cycles(N::Integer; levels_per_cycle::Integer = DEFAULT_LEVELS_PER_CYCLE,
                    bf::BlockFormat = MXFP4, acc_bits::Integer = 32,
                    schedule::Symbol = :tree, macs = nothing)
    L = Int(levels_per_cycle)
    d = mxfp4_dot_costs(N; bf, acc_bits, schedule)
    lat(x) = cycles_for(x; levels_per_cycle = L)
    issue = macs === nothing ? nothing : cld(Int(N), Int(macs))
    (N = Int(N), levels_per_cycle = L, schedule = schedule, blocks = d.blocks,
     depth_conv = d.total_conv, depth_rr4 = d.total_rr4, depth_cs = d.total_cs,
     latency_conv = lat(d.total_conv), latency_rr4 = lat(d.total_rr4),
     latency_cs = lat(d.total_cs),
     macs = macs === nothing ? 0 : Int(macs), issue_cycles = issue,
     # with limited MACs the pipeline fill is amortised into the issue stream
     total_conv = issue === nothing ? lat(d.total_conv) : issue + lat(d.total_conv),
     total_rr4  = issue === nothing ? lat(d.total_rr4)  : issue + lat(d.total_rr4),
     total_cs   = issue === nothing ? lat(d.total_cs)   : issue + lat(d.total_cs))
end

"""
    softmax_cycles(N::Integer; levels_per_cycle = DEFAULT_LEVELS_PER_CYCLE, w = 16,
                   lanes = 1) -> NamedTuple

A length-`N` soft-max, in cycles, split into what the arithmetic costs and what the
**passes** cost.

Soft-max is inherently three passes over `N` — a max reduction, then `exp` and its sum,
then the normalising divide — and with `lanes` elements processed per cycle each pass
costs `⌈N/lanes⌉` cycles of streaming. Against that, the arithmetic depth is a fill
cost paid once per pass.

The returned `pass_bound` and `depth_bound` say which dominates. For any realistic `N`
soft-max is **pass-bound, not gate-bound**, and no choice of adder changes it — which
is the honest ceiling on what redundancy can do here.
"""
function softmax_cycles(N::Integer; levels_per_cycle::Integer = DEFAULT_LEVELS_PER_CYCLE,
                        w::Integer = 16, lanes::Integer = 1)
    L = Int(levels_per_cycle); Nn = Int(N); ln = max(1, Int(lanes))
    s = mxfp4_softmax_costs(Nn; w)
    lat(x) = cycles_for(x; levels_per_cycle = L)
    stream = cld(Nn, ln)
    passes = 3
    (N = Nn, levels_per_cycle = L, lanes = ln,
     depth_conv = s.total_conv, depth_rr4 = s.total_rr4,
     fill_conv = lat(s.total_conv), fill_rr4 = lat(s.total_rr4),
     stream_cycles = stream, passes = passes,
     pass_bound = passes * stream,
     depth_bound_conv = lat(s.total_conv), depth_bound_rr4 = lat(s.total_rr4),
     total_conv = passes * stream + lat(s.total_conv),
     total_rr4 = passes * stream + lat(s.total_rr4),
     arithmetic_share = lat(s.total_conv) / (passes * stream + lat(s.total_conv)))
end

# ---- reporting -------------------------------------------------------------

function _print_cycles(bs::Vector{CycleBudget}; io::IO = stdout, baseline = nothing)
    b0 = baseline === nothing ? nothing : bs[findfirst(x -> x.method === baseline, bs)]
    @printf(io, "  %-18s %8s %9s %8s %11s %8s  %s\n",
            "method", "depth", "latency", "II", "results/1k cy", "vs base", "note")
    println(io, "  " * "─"^92)
    for x in bs
        rel = b0 === nothing ? "     —  " :
              @sprintf("%6.2f× ", b0.latency / x.latency)
        @printf(io, "  %-18s %8d %7d cy %8d %11.1f %s  %s\n",
                String(x.method), x.depth, x.latency, x.ii,
                1000 * x.throughput, rel, x.detail)
    end
    bs
end

"""
    clock_sensitivity(; Ls = (4, 8, 16, 32, 64), w = 64, io = stdout) -> Nothing

The table that decides whether any of this matters: one `w`-bit addition, in cycles, as
the clock budget varies.

A depth advantage becomes a **cycle** advantage only where the depths straddle a
multiple of `L`. Read down the columns: at `L = 4` the four schedules differ; by
`L = 16` everything but ripple is one cycle; by `L = 64` even ripple has caught up.
Redundancy's depth win is real, and it turns into cycles only at aggressive clocks.
"""
function clock_sensitivity(; Ls = (4, 8, 16, 32, 64), w::Integer = 64, io::IO = stdout)
    println(io, "  one $(w)-bit addition, latency in cycles")
    @printf(io, "  %-18s", "levels/cycle →")
    for L in Ls; @printf(io, " %6d", L); end
    @printf(io, "   %8s\n", "depth")
    println(io, "  " * "─"^(18 + 7 * length(Ls) + 11))
    for b in add_costs(w)
        @printf(io, "  %-18s", String(b.method))
        for L in Ls
            @printf(io, " %6d", cycles_for(b.depth; levels_per_cycle = L))
        end
        @printf(io, "   %8d\n", b.depth)
    end
    nothing
end

"""
    cycle_report(; levels_per_cycle = DEFAULT_LEVELS_PER_CYCLE, w = 64, n = 1024,
                 mulbits = 24, dotN = 4096, io = stdout) -> NamedTuple

Every operation in the package, priced in cycles at one clock budget: addition,
multiplication, accumulation, dot product and soft-max.

Prints the clock-sensitivity table first, because no cycle count on the page means
anything without it.
"""
function cycle_report(; levels_per_cycle::Integer = DEFAULT_LEVELS_PER_CYCLE,
                      w::Integer = 64, n::Integer = 1024, mulbits::Integer = 24,
                      dotN::Integer = 4096, io::IO = stdout)
    L = Int(levels_per_cycle)
    println(io, "="^88)
    println(io, "  CYCLES = DEPTH ÷ CLOCK BUDGET.  Every number below assumes")
    println(io, "  levels_per_cycle = $(L); change it and the answers change.")
    println(io, "="^88)
    clock_sensitivity(; w, io)

    println(io, "\n", "="^88)
    println(io, "  ADDITION — one $(w)-bit add at L=$(L)")
    println(io, "="^88)
    a = _print_cycles(add_cycles(w; levels_per_cycle = L); io, baseline = :ripple)

    println(io, "\n", "="^88)
    println(io, "  MULTIPLICATION — $(mulbits)×$(mulbits) at L=$(L)")
    println(io, "="^88)
    m = _print_cycles(multiply_cycles(mulbits; levels_per_cycle = L); io,
                      baseline = :shift_add)

    println(io, "\n", "="^88)
    println(io, "  ACCUMULATION — $(n) terms into $(w) bits, sequential, at L=$(L)")
    println(io, "="^88)
    ac = _print_cycles(accumulate_cycles(n, w; levels_per_cycle = L); io,
                       baseline = :ripple)

    println(io, "\n", "="^88)
    println(io, "  DOT PRODUCT — MXFP4, N=$(dotN), at L=$(L)")
    println(io, "="^88)
    d = dot_cycles(dotN; levels_per_cycle = L)
    @printf(io, "  %-24s %8s %10s\n", "datapath", "depth", "latency")
    println(io, "  " * "─"^46)
    @printf(io, "  %-24s %8d %7d cy\n", "canonical", d.depth_conv, d.latency_conv)
    @printf(io, "  %-24s %8d %7d cy\n", "RR4", d.depth_rr4, d.latency_rr4)
    @printf(io, "  %-24s %8d %7d cy\n", "carry-save", d.depth_cs, d.latency_cs)
    println(io, "\n  ...but with finite hardware the issue stream dominates:")
    @printf(io, "  %-24s %10s %10s %10s\n", "MACs", "issue cy", "canon tot", "RR4 tot")
    println(io, "  " * "─"^58)
    for macs in (64, 256, 1024, dotN)
        dm = dot_cycles(dotN; levels_per_cycle = L, macs)
        @printf(io, "  %-24d %10d %10d %10d\n",
                macs, dm.issue_cycles, dm.total_conv, dm.total_rr4)
    end

    println(io, "\n", "="^88)
    println(io, "  SOFT-MAX — N=$(dotN) at L=$(L), three passes")
    println(io, "="^88)
    @printf(io, "  %-8s %10s %10s %10s %10s %12s\n",
            "lanes", "stream cy", "pass bound", "fill conv", "fill RR4", "arith share")
    println(io, "  " * "─"^66)
    for lanes in (1, 8, 32, 128)
        s = softmax_cycles(dotN; levels_per_cycle = L, lanes)
        @printf(io, "  %-8d %10d %10d %10d %10d %11.1f%%\n",
                lanes, s.stream_cycles, s.pass_bound, s.fill_conv, s.fill_rr4,
                100 * s.arithmetic_share)
    end
    println(io, "\n  Soft-max is pass-bound: three streaming passes over N dwarf the")
    println(io, "  arithmetic fill, and no adder choice moves the total.")
    (add = a, multiply = m, accumulate = ac, dot = d,
     softmax = softmax_cycles(dotN; levels_per_cycle = L))
end
