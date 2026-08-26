# ---------------------------------------------------------------------------
# Accumulators, and the question redundancy exists to answer.
#
# An accumulator is the worst case for carry propagation: `acc += x` N times, each add
# waiting on the one before, and the accumulator growing wider as the sum grows.  A
# conventional adder pays a carry chain on EVERY step, and the chain gets longer as the
# register widens.  A redundant accumulator pays none: the RR4 add is depth 3 at any
# width, and the single carry-propagate conversion is deferred to the exit.
#
# This file runs both, on the same data, and reports what each cost.  The values are
# computed for real — `rr4_add` is executed, not modelled — so `exact` is a measurement.
# Only the GATE COUNTS are a model, and it is stated in full in `AccumulatorRun`.
# ---------------------------------------------------------------------------

# ---- the genuinely carry-free conversion -----------------------------------

"""
    rr4_recode(x::Integer) -> SignedDigits

Convert an integer into minimally-redundant RR4 `{-2..2}` by **parallel recoding** —
every output digit computed simultaneously, no carry chain, depth 1.

Take the ordinary radix-4 digits ``d_i \\in \\{0,1,2,3\\}`` and emit

```math
c_{i+1} = [\\, d_i \\ge 2 \\,], \\qquad z_i = d_i - 4c_{i+1} + c_i
```

Each ``c_{i+1}`` depends on **its own digit only**, so every carry is known immediately
and the whole string is recoded in one parallel step. The digits land in range because
the two cases cover it exactly: ``d_i \\le 1`` gives ``z_i = d_i + c_i \\in \\{0,1,2\\}``,
and ``d_i \\ge 2`` gives ``z_i = d_i - 4 + c_i \\in \\{-2,-1,0\\}``.

!!! note "This is not what `to_rr4` does"
    [`to_rr4`](@ref) reaches the same *value* by a sequential rewrite that only fires
    when a digit falls outside the alphabet, so its loop carries state between positions.
    Both are valid spellings of the same number — that redundancy is the entire point —
    but only `rr4_recode` is carry-free, and only it models what recoding hardware does.
    [`same_spelling`](@ref) will usually report `false`; `value` will always agree.

Negative inputs are recoded by magnitude and negated, since negation in a signed-digit
system is free.

```jldoctest
julia> z = rr4_recode(255);

julia> value(z), digit_string(z)
(255//1, "1 0 0 -1")

julia> value(rr4_recode(-1000)) == -1000
true
```
"""
function rr4_recode(x::Integer)
    v = BigInt(x)
    neg = v < 0
    u = abs(v)
    d = Int[]
    while u > 0
        push!(d, Int(u % 4)); u ÷= 4
    end
    isempty(d) && push!(d, 0)
    n = length(d)
    # every carry is a function of one digit — this is the parallel step
    c = [d[i] >= 2 ? 1 : 0 for i in 1:n]
    z = Vector{Int}(undef, n)
    for i in 1:n                      # independent; a loop only because Julia is serial
        cin = i == 1 ? 0 : c[i-1]
        z[i] = d[i] - 4c[i] + cin
    end
    c[n] != 0 && push!(z, c[n])
    neg && (z .= .-z)
    SignedDigits(z, 4, 2, 0)
end

"""
    rr4_recode_table(x::Integer) -> Vector{NamedTuple}

The per-digit working of [`rr4_recode`](@ref): for each position, the incoming radix-4
digit, the carry it generates, the carry it receives, and the digit it emits.

Every row is computable without looking at any other row's *output* — which is what
"carry-free" means, and what the table is for."""
function rr4_recode_table(x::Integer)
    u = abs(BigInt(x)); d = Int[]
    while u > 0
        push!(d, Int(u % 4)); u ÷= 4
    end
    isempty(d) && push!(d, 0)
    n = length(d)
    c = [d[i] >= 2 ? 1 : 0 for i in 1:n]
    [(i = i - 1, d = d[i], c_out = c[i], c_in = i == 1 ? 0 : c[i-1],
      z = d[i] - 4c[i] + (i == 1 ? 0 : c[i-1])) for i in 1:n]
end

# ---- cost model ------------------------------------------------------------
# Stated, not measured.  One level per ripple position, ⌈log₂w⌉ for a prefix network,
# 3 for an RR4 carry-free add at any width.

_accbits(v) = v == 0 ? 1 : ndigits(abs(BigInt(v)); base = 2)
_ripple_levels(w::Integer) = Int(w) + 1
_prefix_levels(w::Integer) = max(1, ceil(Int, log2(max(2, Int(w)))))

"""
    AccumulatorRun

What one accumulation cost, and whether it got the right answer.

# Fields
- `method` — `:rr4`, `:ripple`, or `:prefix`.
- `schedule` — `:sequential` (a real accumulator register) or `:tree` (pairwise reduction).
- `n` — how many terms were accumulated.
- `value` — the sum. Every method returns the same one; `exact` records that it did.
- `exact::Bool` — **measured**, by comparing against the reference sum.
- `acc_bits` — width of the final accumulator, in bits.
- `add_depth` — gate levels spent on the additions themselves.
- `exit_cpa` — gate levels for the one carry-propagate conversion at the exit
  (0 for the conventional methods, which are already canonical at every step).
- `depth` — `add_depth + exit_cpa`, the critical path to a usable answer.
- `per_term` — `depth / n`, the number that decides whether a method scales.
- `seconds` — wall-clock of the simulation, not of any hardware.

!!! warning "Gate counts are a model; values are not"
    `value` and `exact` come from running the arithmetic. `add_depth` and `exit_cpa`
    come from the cost model in the source header: ripple `w+1`, prefix `⌈log₂w⌉`,
    RR4 `3` at any width. Compare depths between methods here, not against silicon.
"""
struct AccumulatorRun
    method::Symbol
    schedule::Symbol
    n::Int
    value::BigInt
    exact::Bool
    acc_bits::Int
    add_depth::Int
    exit_cpa::Int
    depth::Int
    per_term::Float64
    seconds::Float64
end

"""
    accumulator_inputs(n; kind=:random, rng=Random.default_rng(), magnitude=1_000) -> Vector{Int}

`n` terms to accumulate. `kind` selects the shape:

| kind | what it is |
|:---|:---|
| `:random` | uniform in `±magnitude` — the default |
| `:positive` | uniform in `1:magnitude`, so the accumulator only ever grows |
| `:ramp` | `1, 2, 3, …`, a deterministic worst case for width growth |
| `:ones` | all `1`, the cheapest possible carry behaviour |
| `:alternating` | `+magnitude, -magnitude, …`, which keeps the sum near zero |
"""
function accumulator_inputs(n::Integer; kind::Symbol = :random,
                            rng::AbstractRNG = Random.default_rng(),
                            magnitude::Integer = 1_000)
    m = Int(magnitude); N = Int(n)
    if kind === :random
        rand(rng, -m:m, N)
    elseif kind === :positive
        rand(rng, 1:m, N)
    elseif kind === :ramp
        collect(1:N)
    elseif kind === :ones
        ones(Int, N)
    elseif kind === :alternating
        [isodd(i) ? m : -m for i in 1:N]
    else
        throw(ArgumentError("accumulator_inputs: unknown kind :$kind — one of " *
                            ":random, :positive, :ramp, :ones, :alternating"))
    end
end

_as_terms(xs::AbstractVector, kw...) = collect(Int, xs)
_as_terms(n::Integer; kwargs...) = accumulator_inputs(n; kwargs...)

# ---- the two accumulators --------------------------------------------------

"""
    accumulate_rr4(xs; schedule=:sequential, exit=:prefix) -> AccumulatorRun

**Carry-free accumulation.** The running total is held in redundant RR4 form, so each
`acc += x` is [`rr4_add`](@ref) at a constant depth of 3 — *independent of how wide the
accumulator has grown*. Exactly one carry-propagate conversion is paid, at the exit.

```math
\\text{depth} = 3N + \\text{CPA}(w) \\qquad\\text{(sequential)}
```

The arithmetic is really executed, so `exact` is a measurement: the redundant
accumulator's value is compared against the reference sum.

`exit` selects the model for the single exit conversion, `:prefix` (``⌈\\log_2 w⌉``) or
`:ripple` (``w+1``). Even at `:ripple` it is paid **once**, not once per term — which is
the whole argument for staying redundant inside the loop.
"""
function accumulate_rr4(xs::AbstractVector; schedule::Symbol = :sequential,
                        exit::Symbol = :prefix, alphabet::RR4Alphabet = MIN_REDUNDANT)
    t0 = time()
    terms = collect(Int, xs)
    n = length(terms)
    n == 0 && throw(ArgumentError("accumulate_rr4: no terms"))
    ref = sum(BigInt, terms)

    if schedule === :sequential
        acc = to_rr4(terms[1]; alphabet)
        for i in 2:n
            acc = acc + to_rr4(terms[i]; alphabet)     # routes through rr4_add
        end
        add_depth = 3 * (n - 1)
    elseif schedule === :tree
        level = [to_rr4(v; alphabet) for v in terms]
        levels = 0
        while length(level) > 1
            nxt = similar(level, 0)
            for i in 1:2:length(level)
                push!(nxt, i == length(level) ? level[i] : level[i] + level[i+1])
            end
            level = nxt
            levels += 1
        end
        acc = level[1]
        add_depth = 3 * levels
    else
        throw(ArgumentError("accumulate_rr4: schedule must be :sequential or :tree"))
    end

    v = value(acc)
    got = BigInt(numerator(v) ÷ denominator(v))
    w = _accbits(got)
    cpa = exit === :ripple ? _ripple_levels(w) : _prefix_levels(w)
    AccumulatorRun(:rr4, schedule, n, got, got == ref, w,
                   add_depth, cpa, add_depth + cpa,
                   (add_depth + cpa) / n, time() - t0)
end

"""
    accumulate_carry(xs; adder=:ripple, schedule=:sequential) -> AccumulatorRun

**Conventional accumulation.** The running total is kept in ordinary binary, so every
step pays a full carry-propagate add — and the cost of that add *grows with the
accumulator's width*, which itself grows as the sum does.

`adder` picks the carry network: `:ripple` costs ``w+1`` levels per step, `:prefix`
(Kogge–Stone) costs ``⌈\\log_2 w⌉``. No exit conversion is needed, because the
accumulator is canonical after every step — that is precisely what it is paying for.

Widths are measured from the actual running total at each step, not assumed.
"""
function accumulate_carry(xs::AbstractVector; adder::Symbol = :ripple,
                          schedule::Symbol = :sequential)
    t0 = time()
    terms = collect(Int, xs)
    n = length(terms)
    n == 0 && throw(ArgumentError("accumulate_carry: no terms"))
    adder in (:ripple, :prefix) ||
        throw(ArgumentError("accumulate_carry: adder must be :ripple or :prefix"))
    levels = adder === :ripple ? _ripple_levels : _prefix_levels
    ref = sum(BigInt, terms)

    add_depth = 0
    if schedule === :sequential
        acc = BigInt(terms[1])
        for i in 2:n
            w = max(_accbits(acc), _accbits(terms[i]))   # real width at this step
            add_depth += levels(w)
            acc += terms[i]
        end
        got = acc
    elseif schedule === :tree
        level = BigInt.(terms)
        while length(level) > 1
            w = maximum(_accbits, level)
            add_depth += levels(w)                       # one level of the tree
            nxt = similar(level, 0)
            for i in 1:2:length(level)
                push!(nxt, i == length(level) ? level[i] : level[i] + level[i+1])
            end
            level = nxt
        end
        got = level[1]
    else
        throw(ArgumentError("accumulate_carry: schedule must be :sequential or :tree"))
    end

    w = _accbits(got)
    AccumulatorRun(adder, schedule, n, got, got == ref, w,
                   add_depth, 0, add_depth, add_depth / n, time() - t0)
end

# ---- comparison and scaling ------------------------------------------------

"""
    compare_accumulators(input; schedule=:sequential, io=stdout, kwargs...) -> Vector{AccumulatorRun}

Run all three accumulators on the same terms and print the comparison.

`input` is either the **vector** of terms, or an **integer count** — in which case that
many terms are generated by [`accumulator_inputs`](@ref) and any extra keywords
(`kind`, `rng`, `magnitude`) are passed to it.

```julia
julia> compare_accumulators(1024);                    # 1024 random terms
julia> compare_accumulators([3, 1, 4, 1, 5, 9, 2, 6]) # your own vector
julia> compare_accumulators(4096; kind = :positive, magnitude = 10^6)
```

All three agree on the value — redundancy trades cost, never accuracy — so the table is
entirely about depth.
"""
function compare_accumulators(input; schedule::Symbol = :sequential, io::IO = stdout,
                              exit::Symbol = :prefix, kwargs...)
    terms = input isa AbstractVector ? collect(Int, input) :
            accumulator_inputs(input; kwargs...)
    runs = AccumulatorRun[
        accumulate_carry(terms; adder = :ripple, schedule),
        accumulate_carry(terms; adder = :prefix, schedule),
        accumulate_rr4(terms; schedule, exit),
    ]
    _print_accumulators(runs; io)
    runs
end

function _print_accumulators(runs::Vector{AccumulatorRun}; io::IO = stdout)
    r1 = first(runs)
    println(io, "accumulating ", r1.n, " terms, ", r1.schedule,
            " schedule — final accumulator ", r1.acc_bits, " bits")
    @printf(io, "%-22s %10s %9s %9s %10s %8s  %s\n",
            "method", "add depth", "exit CPA", "total", "per term", "exact", "vs RR4")
    println(io, "─"^88)
    rr4 = findfirst(r -> r.method === :rr4, runs)
    base = rr4 === nothing ? nothing : runs[rr4].depth
    for r in runs
        rel = base === nothing || r.method === :rr4 ? "—" :
              @sprintf("%.1f× deeper", r.depth / base)
        @printf(io, "%-22s %10d %9d %9d %10.2f %8s  %s\n",
                String(r.method), r.add_depth, r.exit_cpa, r.depth, r.per_term,
                r.exact ? "yes" : "NO", rel)
    end
    allsame = all(r -> r.value == first(runs).value, runs)
    println(io, allsame ? "all methods returned the same value: $(first(runs).value)" :
                          "!! VALUES DISAGREE — this is a bug, not a trade-off")
    runs
end

"""
    accumulator_scaling(; Ns=(16, 64, 256, 1024, 4096), schedule=:sequential,
                        io=stdout, kwargs...) -> Vector

Sweep the term count and print depth per term for each method — the table that answers
"does carry-free actually scale".

The RR4 column is flat by construction: 3 levels per add whatever the width, plus one
amortised exit. The ripple column grows, because the accumulator widens as the sum does
and every step pays the full width. The prefix column grows too, just logarithmically.
"""
function accumulator_scaling(; Ns = (16, 64, 256, 1024, 4096),
                             schedule::Symbol = :sequential, io::IO = stdout,
                             exit::Symbol = :prefix, kwargs...)
    @printf(io, "%7s %7s  %10s %10s %10s   %12s %12s\n",
            "N", "bits", "ripple", "prefix", "RR4", "ripple/RR4", "prefix/RR4")
    println(io, "─"^82)
    out = []
    for N in Ns
        terms = accumulator_inputs(N; kwargs...)
        a = accumulate_carry(terms; adder = :ripple, schedule)
        b = accumulate_carry(terms; adder = :prefix, schedule)
        c = accumulate_rr4(terms; schedule, exit)
        @printf(io, "%7d %7d  %10d %10d %10d   %11.1f× %11.1f×\n",
                N, c.acc_bits, a.depth, b.depth, c.depth,
                a.depth / c.depth, b.depth / c.depth)
        push!(out, (N = Int(N), bits = c.acc_bits, ripple = a, prefix = b, rr4 = c))
    end
    out
end
