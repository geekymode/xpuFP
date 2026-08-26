# Pipelining, and how far a multiply can be pushed toward one cycle.
#
# cyclemodel.jl answers "how many cycles is this deep?"; this file answers the two
# questions that follow from it: what a pipeline actually buys you, and what you have
# to give up to make a multiply fit in a single cycle.

"""
    PipeStage

One stage of a pipeline: `levels` of logic between two register banks.

`slack` is `levels_per_cycle - levels` — budget bought and not used. Negative slack means
the stage does not fit and the block must be split internally or the clock relaxed.
"""
struct PipeStage
    index::Int
    name::String
    levels::Int
    slack::Int
end

"""
    Pipe

A pipelined datapath: a depth cut into stages, priced.

# Fields
- `depth` — total gate levels, unchanged by pipelining. **Registers do not make logic
  shallower; they make the clock shorter.**
- `latency` — cycles from input to result, equal to the number of stages.
- `ii` — initiation interval, `1` for a fully pipelined unit.
- `registers` — flip-flops added at the cuts, `(latency - 1) × width`. This is what the
  throughput costs.
- `slack` — total unused level-budget across all stages, the price of cutting at stage
  boundaries rather than wherever you like.
"""
struct Pipe
    name::String
    levels_per_cycle::Int
    depth::Int
    stages::Vector{PipeStage}
    latency::Int
    ii::Int
    width::Int
    registers::Int
    slack::Int
end

function _mkpipe(name, L, width, cuts::Vector{Tuple{String,Int}})
    stages = [PipeStage(i, nm, lv, Int(L) - lv) for (i, (nm, lv)) in enumerate(cuts)]
    depth = sum(s.levels for s in stages)
    Pipe(String(name), Int(L), depth, stages, length(stages), 1, Int(width),
         (length(stages) - 1) * Int(width), sum(s.slack for s in stages))
end

"""
    pipeline_plan(depth::Integer; levels_per_cycle = DEFAULT_LEVELS_PER_CYCLE,
                  name = "block", width = 64) -> Pipe

Cut a **homogeneous** block of `depth` levels — one that can be registered anywhere, like
a wide adder or a reduction tree — into the fewest stages that each fit the clock budget,
balanced so no stage is the odd one out.

`⌈depth/L⌉` stages, so the latency is exactly [`cycles_for`](@ref) — pipelining does not
change latency, it changes what happens on the cycles *after* the first.

```julia
julia> pipeline_plan(22; levels_per_cycle = 8).latency
3
```
"""
function pipeline_plan(depth::Integer; levels_per_cycle::Integer = DEFAULT_LEVELS_PER_CYCLE,
                       name::AbstractString = "block", width::Integer = 64)
    D, L = Int(depth), Int(levels_per_cycle)
    D > 0 || throw(ArgumentError("pipeline_plan: depth must be positive"))
    L > 0 || throw(ArgumentError("pipeline_plan: levels_per_cycle must be positive"))
    n = cld(D, L)
    base, extra = fldmod(D, n)               # spread the remainder over the first stages
    cuts = [("$(name) $(i)/$(n)", base + (i <= extra ? 1 : 0)) for i in 1:n]
    _mkpipe(name, L, width, cuts)
end

"""
    pipeline_cut(stages; levels_per_cycle = DEFAULT_LEVELS_PER_CYCLE, name = "unit",
                 width = 64) -> Pipe

Cut a datapath made of **named** stages — the rows of [`float_multiply_stages`](@ref),
say — by greedily packing them into cycles.

This is the realistic case, and it is where real pipelines lose budget: a 12-level
significand multiply cannot share a cycle with anything when `L = 16` leaves only 4
behind. The `slack` field counts exactly that loss.

A stage deeper than `L` is split internally into balanced pieces — a Wallace tree can be
registered between reduction levels even though it is one logical stage — and its pieces
are labelled `name (i/n)`. Without that, the model would report a stage that cannot
close timing.

Only `critical` stages consume budget; stages marked non-critical run in parallel and are
placed in whichever cycle their critical neighbour lands in.

!!! note "Greedy packing can cost a cycle"
    Unlike [`pipeline_plan`](@ref), this can exceed `cycles_for(depth)`. FP32 at `L = 4`
    takes **7 stages against an ideal 6**, because the 1-level unpack cannot share a cycle
    with a 4-level slice of the tree and burns a stage alone. That gap is what named stage
    boundaries cost, and it grows as `L` shrinks.
"""
function pipeline_cut(stages; levels_per_cycle::Integer = DEFAULT_LEVELS_PER_CYCLE,
                      name::AbstractString = "unit", width::Integer = 64)
    L = Int(levels_per_cycle)
    L > 0 || throw(ArgumentError("pipeline_cut: levels_per_cycle must be positive"))
    # flatten, splitting any stage too deep to close timing on its own
    flat = Tuple{String,Int}[]
    for s in stages
        get(s, :critical, true) || continue
        nm, lv = String(s.stage), Int(s.levels)
        if lv <= L
            push!(flat, (nm, lv))
        else
            n = cld(lv, L)
            base, extra = fldmod(lv, n)
            for i in 1:n
                push!(flat, ("$(nm) ($(i)/$(n))", base + (i <= extra ? 1 : 0)))
            end
        end
    end
    cuts = Tuple{String,Int}[]
    cur, acc = String[], 0
    for (nm, lv) in flat
        if !isempty(cur) && acc + lv > L
            push!(cuts, (join(cur, " + "), acc)); cur, acc = String[], 0
        end
        push!(cur, nm); acc += lv
    end
    isempty(cur) || push!(cuts, (join(cur, " + "), acc))
    _mkpipe(name, L, width, cuts)
end

"""
    pipeline_time(p::Pipe, items::Integer) -> Int

Cycles to push `items` operand sets through `p`: `latency + (items − 1) × ii`.

The first result costs `latency`; every one after it costs `ii`. That single line is the
entire argument for pipelining.
"""
pipeline_time(p::Pipe, items::Integer) = p.latency + (Int(items) - 1) * p.ii

"""
    pipeline_speedup(p::Pipe, items::Integer) -> Float64

Speedup over the same logic **not** pipelined.

The honest baseline is the same combinational block with no cut registers. It cannot meet
the clock budget, so it either runs at a clock `latency`× slower or is sequenced over
`latency` cycles — either way `items × latency` fast-clock cycles. Against that,

    speedup = items·latency / (latency + items − 1)

which is `1` at `items = 1` and approaches `latency` as the stream lengthens. **Pipelining
never helps a single operation.**
"""
pipeline_speedup(p::Pipe, items::Integer) =
    Int(items) * p.latency / pipeline_time(p, items)

"""
    pipeline_utilization(p::Pipe, items::Integer) -> Float64

Fraction of cycles that retire a result, `items / pipeline_time(p, items)`. The shortfall
from `1.0` is the fill and drain — `latency − 1` wasted cycles, paid once.
"""
pipeline_utilization(p::Pipe, items::Integer) = Int(items) / pipeline_time(p, items)

"""
    pipeline_breakeven(p::Pipe; fraction = 0.9) -> Int

How many items a stream needs before the pipeline reaches `fraction` of its asymptotic
speedup. Solves `items·L/(L+items−1) ≥ fraction·L`:

    items ≥ fraction·(latency − 1) / (1 − fraction)

At 90% of peak a 3-stage unit needs 18 items and a 22-stage unit needs 189. **Deep
pipelines are a bet on long streams**; a dot product of length 4096 wins that bet easily
and a scalar `a*b+c` in a control path never does.
"""
function pipeline_breakeven(p::Pipe; fraction::Real = 0.9)
    0 < fraction < 1 || throw(ArgumentError("pipeline_breakeven: fraction must be in (0,1)"))
    max(1, ceil(Int, fraction * (p.latency - 1) / (1 - fraction)))
end

"""
    float_multiply_pipeline(f::FloatFormat = FP32;
                            levels_per_cycle = DEFAULT_LEVELS_PER_CYCLE) -> Pipe

The FP multiplier of [`float_multiply_stages`](@ref), cut into pipeline stages at its
natural boundaries.

For FP32 at `L = 8` this gives the familiar 3-stage industrial shape, and the stage that
refuses to share a cycle is the 12-level significand multiply — which is why real FP
multipliers cut *inside* the Wallace tree rather than around it.
"""
float_multiply_pipeline(f::FloatFormat = FP32;
                        levels_per_cycle::Integer = DEFAULT_LEVELS_PER_CYCLE) =
    pipeline_cut(float_multiply_stages(f); levels_per_cycle,
                 name = f.name, width = 2 * (f.mbits + 1) + f.ebits + 2)

"""
    pipeline_timeline(p::Pipe, items::Integer = 5; io = stdout) -> Nothing

Draw the reservation table: stages down, cycles across, one letter per item in flight.

```
  cycle          1  2  3  4  5  6  7
  S1             A  B  C  D  E  .  .
  S2             .  A  B  C  D  E  .
  S3             .  .  A  B  C  D  E
  retires        .  .  A  B  C  D  E
```

Read the diagonal for latency and the bottom row for throughput.
"""
function pipeline_timeline(p::Pipe, items::Integer = 5; io::IO = stdout)
    M = Int(items)
    T = pipeline_time(p, M)
    letter(i) = ('A' + (i - 1) % 26)
    @printf(io, "  %-14s", "cycle")
    for c in 1:T; @printf(io, "%3d", c); end
    println(io)
    println(io, "  " * "─"^(14 + 3T))
    for s in p.stages
        @printf(io, "  S%-13d", s.index)
        for c in 1:T
            i = c - s.index + 1                      # item in this stage this cycle
            print(io, 1 <= i <= M ? "  $(letter(i))" : "  .")
        end
        @printf(io, "   %d lv\n", s.levels)
    end
    @printf(io, "  %-14s", "retires")
    for c in 1:T
        i = c - p.latency + 1
        print(io, 1 <= i <= M ? "  $(letter(i))" : "  .")
    end
    println(io)
    @printf(io, "\n  latency %d cy · II %d · %d items in %d cy · %.2f× vs unpipelined · %.0f%% utilised\n",
            p.latency, p.ii, M, T, pipeline_speedup(p, M), 100 * pipeline_utilization(p, M))
    nothing
end

"""
    pipeline_report(p::Pipe; items = (1, 4, 16, 64, 1024), io = stdout) -> Pipe

Stage table, register cost and the speedup curve for one pipeline.
"""
function pipeline_report(p::Pipe; items = (1, 4, 16, 64, 1024), io::IO = stdout)
    println(io, "  $(p.name): depth $(p.depth) levels at L = $(p.levels_per_cycle)")
    @printf(io, "  %-6s %7s %7s   %s\n", "stage", "levels", "slack", "contains")
    println(io, "  " * "─"^72)
    for s in p.stages
        @printf(io, "  S%-5d %7d %7d   %s\n", s.index, s.levels, s.slack, s.name)
    end
    @printf(io, "\n  latency %d cy · II %d · +%d flip-flops (%d cut%s × %d bits) · %d levels of slack\n\n",
            p.latency, p.ii, p.registers, p.latency - 1,
            p.latency == 2 ? "" : "s", p.width, p.slack)
    @printf(io, "  %-8s %10s %10s %12s\n", "items", "cycles", "speedup", "utilisation")
    println(io, "  " * "─"^44)
    for m in items
        @printf(io, "  %-8d %10d %9.2f× %11.0f%%\n", m, pipeline_time(p, m),
                pipeline_speedup(p, m), 100 * pipeline_utilization(p, m))
    end
    @printf(io, "\n  90%% of peak speedup needs %d items.\n", pipeline_breakeven(p))
    p
end

# ---- the one-cycle multiply ------------------------------------------------

"""
    MulRung

One rung of the ladder in [`one_cycle_multiply`](@ref): a technique, the depth after
applying it, and what it costs you.
"""
struct MulRung
    step::String
    depth::Int
    saved::Int
    cycles::Int
    fits::Bool
    cost::String
end

# 2^(2·(ebits+mbits)) entries, computed so wide formats do not overflow Int
_lut_bits(f::FloatFormat) = 2 * (f.ebits + f.mbits)
_lut_fits(f::FloatFormat, limit::Integer) = _lut_bits(f) <= 62 && (1 << _lut_bits(f)) <= Int(limit)
_lut_entries(f::FloatFormat) = _lut_bits(f) <= 62 ? string(1 << _lut_bits(f)) : "2^$(_lut_bits(f))"

"""
    one_cycle_multiply(f::FloatFormat = FP32;
                       levels_per_cycle = DEFAULT_LEVELS_PER_CYCLE,
                       lut_limit = 4096) -> Vector{MulRung}

**How close can one `f × f` multiply get to a single cycle, and what does each step
cost?** A cumulative ladder: every rung keeps the ones above it.

The rungs, in the order a designer would actually reach for them:

1. **IEEE baseline** — [`float_multiply_depth`](@ref). 22 levels for FP32.
2. **Booth radix-8** — a third fewer partial-product rows, so a shallower Wallace tree.
   Costs a hard `3×` multiple, precomputed with one adder.
3. **Carry-save product** — delete the exit carry-propagate adder and hand the consumer a
   redundant `(sum, carry)` pair. This is the single largest saving available, and it is
   free *only* if the next stage is an adder tree rather than a register file.
4. **Deferred rounding** — do not round or normalise the product at all; feed the wide
   product straight into a fused accumulator and round once, at the end. Removes the
   round, normalise and post-normalise stages, and is *more* accurate, not less.
5. **Table lookup** — for narrow formats the whole multiplier is a ROM. Available only
   while the table stays under `lut_limit` entries, which E2M1 and E4M3 clear and BF16
   does not.

The rung that is missing is the honest one: **relax the clock.** A multiply that will not
fit in `L` levels always fits in `depth` levels; you pay for it in frequency, everywhere
in the chip. [`one_cycle_clock`](@ref) prices that.

For FP32 at the default `L = 16`, rungs 3 and 4 together are what turn a 2-cycle multiply
into a 1-cycle one — and both of them work by *not producing an IEEE result*. That is the
real answer to "can a multiply be single-cycle": yes, if the thing it produces is allowed
to be a redundant, unrounded, wide intermediate. That is exactly what an MXFP4 MAC array
does.
"""
function one_cycle_multiply(f::FloatFormat = FP32;
                            levels_per_cycle::Integer = DEFAULT_LEVELS_PER_CYCLE,
                            lut_limit::Integer = 4096)
    L = Int(levels_per_cycle)
    p = f.mbits + 1
    lg(n) = max(1, ceil(Int, log2(max(2, n))))
    st = float_multiply_stages(f)
    lvl(nm) = st[findfirst(x -> x.stage == nm, st)].levels
    rows_r4 = cld(p + 1, 2)
    rows_r8 = cld(p + 1, 3)
    tree_r4 = length(reduction_schedule(rows_r4)) - 1
    tree_r8 = length(reduction_schedule(rows_r8)) - 1
    cpa = lg(2p)

    out = MulRung[]
    d = float_multiply_depth(f)
    push!(out, MulRung("IEEE baseline", d, 0, cycles_for(d; levels_per_cycle = L), d <= L,
                       "none — this is the reference"))

    save = max(0, tree_r4 - tree_r8)
    d -= save
    push!(out, MulRung("+ Booth radix-8", d, save, cycles_for(d; levels_per_cycle = L), d <= L,
                       "$(rows_r4)→$(rows_r8) rows; one adder for the hard 3× multiple"))

    d -= cpa
    push!(out, MulRung("+ carry-save product", d, cpa, cycles_for(d; levels_per_cycle = L), d <= L,
                       "output is a redundant ($(2p)-bit sum, carry) pair, not a number"))

    save = lvl("normalise") + lvl("round") + lvl("post-normalise")
    d -= save
    push!(out, MulRung("+ deferred rounding", d, save, cycles_for(d; levels_per_cycle = L), d <= L,
                       "needs a wide fused accumulator; rounds once at the end"))

    ent = _lut_entries(f)
    if _lut_fits(f, lut_limit)
        save = d - 2
        d = 2
        push!(out, MulRung("+ table lookup", d, save, cycles_for(d; levels_per_cycle = L), d <= L,
                           "$(ent)-entry ROM over the magnitudes; sign is one XOR"))
    else
        push!(out, MulRung("  table lookup (N/A)", d, 0, cycles_for(d; levels_per_cycle = L), d <= L,
                           "$(ent) entries exceeds the $(lut_limit)-entry limit"))
    end
    out
end

"""
    one_cycle_clock(f::FloatFormat = FP32) -> NamedTuple

The other way to make a multiply single-cycle: make the cycle long enough.

Reports the levels-per-cycle each rung of [`one_cycle_multiply`](@ref) demands, and the
same figure in [`fo4_range`](@ref) FO4 — the unit that survives a process change.

The calibration that matters: high-performance cores budget roughly **15–25 FO4** per
cycle. FP32's 22 levels are 44–66 FO4, so a single-cycle IEEE FP32 multiply costs a clock
2–5× slower than the rest of the machine. **That is why FP multipliers are pipelined and
not made single-cycle** — and why the stripped-down rungs, which land near 10 levels, are
the interesting ones.
"""
function one_cycle_clock(f::FloatFormat = FP32)
    rungs = one_cycle_multiply(f; levels_per_cycle = 1)
    (format = f.name,
     rungs = [(step = r.step, levels_needed = r.depth, fo4 = fo4_range(r.depth))
              for r in rungs],
     baseline_levels = rungs[1].depth,
     baseline_fo4 = fo4_range(rungs[1].depth),
     typical_core_fo4 = (15, 25),
     baseline_cycles_at_core_clock = (cld(fo4_range(rungs[1].depth)[1], 25),
                                      cld(fo4_range(rungs[1].depth)[2], 15)))
end

"""
    one_cycle_report(f::FloatFormat = FP32;
                     levels_per_cycle = DEFAULT_LEVELS_PER_CYCLE, io = stdout)

Print the [`one_cycle_multiply`](@ref) ladder with the clock each rung would need.
"""
function one_cycle_report(f::FloatFormat = FP32;
                          levels_per_cycle::Integer = DEFAULT_LEVELS_PER_CYCLE,
                          io::IO = stdout)
    L = Int(levels_per_cycle)
    println(io, "  one $(f.name) × $(f.name) multiply at L = $(L) levels/cycle")
    @printf(io, "  %-24s %7s %7s %7s %6s  %s\n",
            "technique", "depth", "saved", "cycles", "1 cy?", "what it costs")
    println(io, "  " * "─"^100)
    for r in one_cycle_multiply(f; levels_per_cycle = L)
        @printf(io, "  %-24s %7d %7d %7d %6s  %s\n", r.step, r.depth, r.saved,
                r.cycles, r.fits ? "yes" : "no", r.cost)
    end
    c = one_cycle_clock(f)
    @printf(io, "\n  single-cycle IEEE %s needs %d levels = %d-%d FO4;",
            f.name, c.baseline_levels, c.baseline_fo4[1], c.baseline_fo4[2])
    @printf(io, " a %d-%d FO4 core clock makes that %d-%d cycles.\n",
            c.typical_core_fo4[1], c.typical_core_fo4[2],
            c.baseline_cycles_at_core_clock[1], c.baseline_cycles_at_core_clock[2])
    nothing
end

"""
    one_cycle_formats(; levels_per_cycle = DEFAULT_LEVELS_PER_CYCLE,
                      formats = (E2M1, E4M3, FP16, BF16, FP32, FP64),
                      io = stdout) -> Vector{NamedTuple}

Which formats already multiply in one cycle, and which need the ladder.

The dividing line is the point of the whole exercise: **E2M1 is single-cycle at any clock
anyone builds**, by table lookup and with room to spare, while FP32 is not single-cycle
even after every trick. Narrowing the format buys more than every microarchitectural
technique combined — which is the case for MXFP4 stated in cycles.
"""
function one_cycle_formats(; levels_per_cycle::Integer = DEFAULT_LEVELS_PER_CYCLE,
                           formats = (E2M1, E4M3, FP16, BF16, FP32, FP64),
                           io::IO = stdout)
    L = Int(levels_per_cycle)
    rows = NamedTuple[]
    @printf(io, "  %-8s %6s %8s %9s %8s %10s %8s\n",
            "format", "sig", "IEEE lv", "IEEE cy", "best lv", "best cy", "LUT?")
    println(io, "  " * "─"^66)
    for f in formats
        rungs = one_cycle_multiply(f; levels_per_cycle = L)
        best = rungs[end]
        lut = _lut_fits(f, 4096)
        r = (format = f.name, sig = f.mbits + 1, ieee_levels = rungs[1].depth,
             ieee_cycles = rungs[1].cycles, best_levels = best.depth,
             best_cycles = best.cycles, lut = lut, one_cycle = best.fits)
        push!(rows, r)
        @printf(io, "  %-8s %6d %8d %9d %8d %10d %8s\n", r.format, r.sig,
                r.ieee_levels, r.ieee_cycles, r.best_levels, r.best_cycles,
                lut ? "yes" : "no")
    end
    rows
end
