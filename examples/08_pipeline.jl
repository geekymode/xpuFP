# 08 — Pipelining, and how close a multiply gets to one cycle.
#
#     julia --project=. examples/08_pipeline.jl
#
# Two questions that follow from the depth model: what do pipeline registers buy, and
# what has to be given up to make a multiply single-cycle?

using xpuFP

hdr(s) = (println("\n", "="^78); println("  ", s); println("="^78))

# ---------------------------------------------------------------------------
hdr("1. One FP32 multiply, cut into pipeline stages at three clock budgets")

for L in (4, 8, 16)
    println("\n--- L = $(L) levels/cycle ---")
    pipeline_report(float_multiply_pipeline(FP32; levels_per_cycle = L);
                    items = (1, 4, 16, 64, 1024))
end

# ---------------------------------------------------------------------------
hdr("2. The reservation table — six multiplies through the 3-stage unit")

pipeline_timeline(float_multiply_pipeline(FP32; levels_per_cycle = 8), 6)

println("""
  Down a column: three multiplies are in flight at once on cycle 4.
  Along a row:   item A enters at cycle 1 and retires at cycle 3.
  The dot triangles are fill and drain — latency-1 cycles, paid once.""")

# ---------------------------------------------------------------------------
hdr("3. Pipelining never helps one operation, and always helps a stream")

p = float_multiply_pipeline(FP32; levels_per_cycle = 8)
for m in (1, 2, 8, 32, 128, 4096)
    println("  $(lpad(m, 6)) items: $(lpad(pipeline_time(p, m), 6)) cy, ",
            "speedup $(round(pipeline_speedup(p, m); digits = 2))×")
end
println("\n  90% of peak needs $(pipeline_breakeven(p)) items; ",
        "99% needs $(pipeline_breakeven(p; fraction = 0.99)).")

# ---------------------------------------------------------------------------
hdr("4. The ladder toward a single-cycle multiply")

one_cycle_report(FP32)
println()
one_cycle_report(E2M1)

# ---------------------------------------------------------------------------
hdr("5. Narrowing the format beats every technique in the ladder")

rows = one_cycle_formats()

let f32 = rows[findfirst(r -> r.format == "FP32", rows)],
    e21 = rows[findfirst(r -> r.format == "E2M1", rows)]
    println("""
  FP32: every trick together, $(f32.ieee_levels) → $(f32.best_levels) levels \
($(round(f32.ieee_levels / f32.best_levels; digits = 2))×), and the result is no longer a float.
  E2M1: the format change alone, $(f32.ieee_levels) → $(e21.best_levels) levels \
($(round(f32.ieee_levels / e21.best_levels; digits = 2))×), and it is still exact.""")
end

# ---------------------------------------------------------------------------
hdr("6. The clock a single-cycle IEEE multiply would demand")

for f in (E2M1, E4M3, BF16, FP32, FP64)
    c = one_cycle_clock(f)
    println("  $(rpad(c.format, 6)) $(lpad(c.baseline_levels, 3)) levels = ",
            "$(c.baseline_fo4[1])-$(c.baseline_fo4[2]) FO4 → ",
            "$(c.baseline_cycles_at_core_clock[1])-$(c.baseline_cycles_at_core_clock[2]) ",
            "cycles at a 15-25 FO4 core clock")
end
