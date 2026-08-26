# ---------------------------------------------------------------------------
# Weight analysis: how FEW nonzero digits can a value be spelled with, and how does
# that scale?
#
# The cheapest spelling is the one that matters for a constant multiplier — one
# adder/subtractor per nonzero digit — so the minimum, not the count, is the
# engineering quantity.  Both the minimum and the full weight distribution come from a
# dynamic program over the carry, because enumerating (up to F(n+1)) spellings to find
# the cheapest is hopeless past a dozen digits.
# ---------------------------------------------------------------------------

"""
    minimal_weight(x; method=:minimally_redundant, ndigits=nothing, fracdigits=nothing)
        -> Int

The **fewest nonzero digits** any spelling of `x` can use — i.e. the cheapest the value
can be as a constant multiplier, at one adder per nonzero digit beyond the first.

Computed by a shortest-path dynamic program over the carry state, not by enumeration, so
it stays fast at widths where the number of spellings runs to hundreds of thousands.
Returns `-1` when the value does not fit in `ndigits`.

```jldoctest
julia> minimal_weight(231), weight(to_rr4(231))    # the one-pass form is not minimal
(4, 5)

julia> minimal_weight(1000), weight(to_rr4(1000))
(3, 4)

julia> minimal_weight(231; method = :maximally_redundant)
4
```
"""
function minimal_weight(x; method = :minimally_redundant, ndigits = nothing,
                        fracdigits = nothing)
    alp = _method_alphabet(method)
    a = alphabet_halfwidth(alp)
    n = ndigits === nothing ? min_ndigits(x; alphabet = alp, fracdigits) : Int(ndigits)
    N, _ = dyadic_parts(x; fracdigits)
    states = Dict{BigInt,Int}(BigInt(N) => 0)
    for _ in 1:n
        nxt = Dict{BigInt,Int}()
        for (rem, w) in states, d in -a:a
            r2 = rem - d
            mod(r2, 4) == 0 || continue
            k = r2 ÷ 4
            w2 = w + (d != 0)
            (!haskey(nxt, k) || nxt[k] > w2) && (nxt[k] = w2)
        end
        states = nxt
    end
    get(states, BigInt(0), -1)
end

"""
    weight_distribution(x; method=:minimally_redundant, ndigits=nothing,
                        fracdigits=nothing) -> Vector{BigInt}

How many spellings have each nonzero count: element `k+1` is the number of spellings
with exactly `k` nonzero digits.

The whole distribution comes from one dynamic program — the per-state generating
polynomial in the weight variable — so it costs no more than counting does, and it sums
to [`count_rr4_representations`](@ref).

```jldoctest
julia> d = weight_distribution(231);

julia> sum(d) == count_rr4_representations(231)
true

julia> findfirst(!iszero, d) - 1 == minimal_weight(231)
true
```
"""
function weight_distribution(x; method = :minimally_redundant, ndigits = nothing,
                             fracdigits = nothing)
    alp = _method_alphabet(method)
    a = alphabet_halfwidth(alp)
    n = ndigits === nothing ? min_ndigits(x; alphabet = alp, fracdigits) : Int(ndigits)
    N, _ = dyadic_parts(x; fracdigits)
    states = Dict{BigInt,Vector{BigInt}}(BigInt(N) => [big(1)])
    for _ in 1:n
        nxt = Dict{BigInt,Vector{BigInt}}()
        for (rem, poly) in states, d in -a:a
            r2 = rem - d
            mod(r2, 4) == 0 || continue
            k = r2 ÷ 4
            sh = d != 0 ? 1 : 0
            tgt = get!(nxt, k, BigInt[])
            while length(tgt) < length(poly) + sh
                push!(tgt, big(0))
            end
            for (i, c) in enumerate(poly)
                tgt[i + sh] += c
            end
        end
        states = nxt
    end
    get(states, BigInt(0), BigInt[])
end

"""
    WeightStats

Summary statistics for the minimal-weight question over a collection of values.

# Fields
`method`, `ndigits`, `nsamples`, `min_weight`, `mean_weight`, `median_weight`,
`max_weight`, `density` (= `mean_weight / ndigits`), `canonical_mean`,
`canonical_density`, `saving` (fraction by which the one-pass conversion exceeds the
minimum).
"""
struct WeightStats
    method::Symbol
    ndigits::Int
    nsamples::Int
    min_weight::Int
    mean_weight::Float64
    median_weight::Float64
    max_weight::Int
    density::Float64
    canonical_mean::Float64
    canonical_density::Float64
    saving::Float64
end

"""
    weight_stats(values; method=:minimally_redundant, ndigits=nothing) -> WeightStats

Quantify the minimal-weight behaviour over a collection of values: the distribution of
minimum weights, the resulting **density** (nonzeros per digit), and how much the
one-pass conversion leaves on the table.

```julia
julia> weight_stats(rand(1:10^6, 200))
```
"""
function weight_stats(values; method = :minimally_redundant, ndigits = nothing)
    alp = _method_alphabet(method)
    n = ndigits === nothing ?
        maximum(min_ndigits(v; alphabet = alp) for v in values) : Int(ndigits)
    mins = Int[]; canon = Int[]
    for v in values
        w = minimal_weight(v; method, ndigits = n)
        w < 0 && continue
        push!(mins, w)
        push!(canon, weight(to_rr4(v; alphabet = alp, ndigits = n)))
    end
    isempty(mins) && throw(ArgumentError("no value fits in $(n) digits"))
    mw = Statistics.mean(mins); cw = Statistics.mean(canon)
    WeightStats(_alphabet_method(alp), n, length(mins), minimum(mins), mw,
                Statistics.median(mins), maximum(mins), mw / n, cw, cw / n,
                cw == 0 ? 0.0 : (cw - mw) / cw)
end

"""
    weight_scaling(widths; method=:minimally_redundant, samples=200, rng=default_rng())
        -> Vector{NamedTuple}

Measure how the minimal-weight **density** scales with word length, by sampling random
values at each width.

Values are drawn from `[0, a(4ⁿ−1)/(r−1)]`, the range the alphabet can actually
represent in `n` digits — sampling beyond it silently mixes in values that do not fit.

Measured limits (400 samples, widths to 128):

| alphabet | minimal-weight density |
|:---|---:|
| `{-2..2}` minimally redundant | ``\\to 2/3 \\approx 0.667`` |
| `{-3..3}` maximally redundant | ``\\to 3/5 = 0.600`` |
| radix-2 `{-1,0,1}` (CSD/NAF) | ``1/3`` exactly, the classical result |

More redundancy buys a lower density, as it must: the alphabets are nested, so the
minimum over the larger set cannot exceed the minimum over the smaller.

```julia
julia> weight_scaling([16, 32, 64])
```
"""
function weight_scaling(widths; method = :minimally_redundant, samples::Integer = 200,
                        rng::AbstractRNG = Random.default_rng())
    alp = _method_alphabet(method)
    a = alphabet_halfwidth(alp)
    out = NamedTuple{(:ndigits, :density, :mean_weight, :canonical_density, :saving),
                     Tuple{Int,Float64,Float64,Float64,Float64}}[]
    for n in widths
        nn = Int(n)
        hi = BigInt(a) * (BigInt(4)^nn - 1) ÷ 3
        vals = [rand(rng, big(0):hi) for _ in 1:samples]
        st = weight_stats(vals; method, ndigits = nn)
        push!(out, (ndigits = nn, density = st.density, mean_weight = st.mean_weight,
                    canonical_density = st.canonical_density, saving = st.saving))
    end
    out
end

function Base.show(io::IO, ::MIME"text/plain", s::WeightStats)
    println(io, "WeightStats  (", s.method, ", ", s.ndigits, " digits, ", s.nsamples, " values)")
    println(io, "  minimal weight  : min ", s.min_weight, "   median ", s.median_weight,
            "   mean ", round(s.mean_weight, digits = 3), "   max ", s.max_weight)
    println(io, "  density         : ", round(s.density, digits = 4), " nonzeros per digit")
    println(io, "  one-pass conv.  : ", round(s.canonical_mean, digits = 3),
            " mean weight (density ", round(s.canonical_density, digits = 4), ")")
    print(io,   "  searching saves : ", round(100 * s.saving, digits = 1),
          "% of the nonzero digits over the one-pass form")
end

# ---------------------------------------------------------------------------
# Canonical forms.
#
# "Canonical" means a rule that picks exactly ONE spelling per value, restoring the
# uniqueness redundancy gave away.  At radix 2 the non-adjacent form does this
# beautifully — it exists for every value, is unique, and is minimum weight, which is
# why CSD deserves the name.  At radix 4 that construction FAILS: most values have no
# non-adjacent spelling at all (the minimal density is ~2/3, well above the 1/2 that
# non-adjacency would force).  So radix-4 canonical forms must be defined some other
# way, and the choice has to be stated rather than assumed.
# ---------------------------------------------------------------------------

# minimum nonzero digits needed to drive `rem` to zero in exactly `levels` positions
function _minw_to_zero(rem::BigInt, levels::Int, a::Int, memo::Dict{Tuple{BigInt,Int},Int})
    levels == 0 && return rem == 0 ? 0 : typemax(Int) ÷ 2
    key = (rem, levels)
    haskey(memo, key) && return memo[key]
    reach = BigInt(a) * (BigInt(4)^levels - 1) ÷ 3
    if abs(rem) > reach
        memo[key] = typemax(Int) ÷ 2
        return memo[key]
    end
    best = typemax(Int) ÷ 2
    for d in -a:a
        r2 = rem - d
        mod(r2, 4) == 0 || continue
        w = (d != 0) + _minw_to_zero(r2 ÷ 4, levels - 1, a, memo)
        w < best && (best = w)
    end
    memo[key] = best
    best
end

"""
    canonical_rr4(x; method=:minimally_redundant, rule=:minweight, ndigits=nothing,
                  fracdigits=nothing) -> SignedDigits

A **canonical** redundant radix-4 form: a rule that selects exactly one spelling per
value, restoring the uniqueness that redundancy gave away.

Radix 4 has no single agreed canonical form, so the rule is explicit:

- `:minweight` (default) — the fewest nonzero digits, with ties broken by preferring the
  **smaller digit at the less significant position first**. Deterministic and unique, and
  it is the form you want for a constant multiplier, since it minimises the adder count.
- `:nonadjacent` — the spelling with no two adjacent nonzero digits. Unique *when it
  exists*, but at radix 4 it usually does not: about 85% of values have none. Throws if
  the value has no such form; test first with [`has_nonadjacent_form`](@ref).
- `:onepass` — whatever [`to_rr4`](@ref) produces, i.e. the non-redundant `{0..3}` digits
  with out-of-alphabet digits rewritten by a single borrow pass. Cheap and reproducible,
  but **not** minimum weight — it exceeds the minimum for about 28% of values.

```jldoctest
julia> weight(canonical_rr4(231)), weight(to_rr4(231))
(4, 5)

julia> value(canonical_rr4(1000)) == 1000
true

julia> weight(canonical_rr4(1000; rule = :onepass))
4
```

!!! note "Why CSD *is* canonical and this needs a stated rule"
    At radix 2 the non-adjacent form exists for every value, is unique, and is minimum
    weight — three properties in one construction, which is what makes "the" CSD
    well defined. At radix 4 the first property fails, so minimality and uniqueness have
    to be arranged separately.
"""
function canonical_rr4(x; method = :minimally_redundant, rule::Symbol = :minweight,
                       ndigits = nothing, fracdigits = nothing)
    alp = _method_alphabet(method)
    a = alphabet_halfwidth(alp)
    n = ndigits === nothing ? min_ndigits(x; alphabet = alp, fracdigits) : Int(ndigits)
    N, e = dyadic_parts(x; fracdigits)

    if rule === :onepass
        return to_rr4(x; alphabet = alp, fracdigits, ndigits = n)
    elseif rule === :nonadjacent
        f = nonadjacent_rr4(x; method, ndigits = n, fracdigits)
        f === nothing && throw(ArgumentError(
            "$(x) has no non-adjacent radix-4 form in {-$(a)..$(a)}; " *
            "use rule = :minweight (about 85% of values have no NAF at this radix)"))
        return f
    elseif rule !== :minweight
        throw(ArgumentError("rule must be :minweight, :nonadjacent or :onepass"))
    end

    memo = Dict{Tuple{BigInt,Int},Int}()
    target = _minw_to_zero(BigInt(N), n, a, memo)
    target >= typemax(Int) ÷ 2 && throw(ArgumentError(
        "$(x) does not fit in $(n) digits over {-$(a)..$(a)}"))
    # walk forward, always taking the smallest digit that keeps the global minimum
    digits = Int[]
    rem = BigInt(N)
    remaining = target
    for lvl in n:-1:1
        chosen = nothing
        for d in -a:a
            r2 = rem - d
            mod(r2, 4) == 0 || continue
            cost = (d != 0) + _minw_to_zero(r2 ÷ 4, lvl - 1, a, memo)
            if cost == remaining
                chosen = d
                break                       # -a:a ascending ⇒ smallest digit wins ties
            end
        end
        chosen === nothing && error("canonical_rr4: no digit preserves the minimum")
        push!(digits, chosen)
        rem = (rem - chosen) ÷ 4
        remaining -= (chosen != 0)
    end
    SignedDigits(digits, 4, a, e)
end

"""
    nonadjacent_rr4(x; method=:minimally_redundant, ndigits=nothing, fracdigits=nothing)
        -> Union{SignedDigits,Nothing}

The radix-4 spelling with **no two adjacent nonzero digits**, or `nothing` when none
exists.

At radix 2 this is the CSD/NAF and always exists. At radix 4 it usually does not — the
minimal-weight density is about 2/3, well above the 1/2 that non-adjacency would demand,
so most values simply cannot be spread out that thinly. When it does exist it is unique
and minimum weight (verified exhaustively for values below 600).

```jldoctest
julia> nonadjacent_rr4(4) |> digit_string
"10"

julia> nonadjacent_rr4(231) === nothing
true
```
"""
function nonadjacent_rr4(x; method = :minimally_redundant, ndigits = nothing,
                         fracdigits = nothing)
    alp = _method_alphabet(method)
    n = ndigits === nothing ?
        min_ndigits(x; alphabet = alp, fracdigits) + 2 : Int(ndigits)
    for r in all_rr4_representations(x, n; method, fracdigits)
        has_adjacent_nonzeros(r) || return r
    end
    nothing
end

"""
    has_nonadjacent_form(x; method=:minimally_redundant, ndigits=nothing) -> Bool

Whether `x` admits a non-adjacent radix-4 spelling.  About 15% of values do, in either
alphabet — which is precisely why the CSD construction does not carry over to radix 4."""
has_nonadjacent_form(x; kwargs...) = nonadjacent_rr4(x; kwargs...) !== nothing
