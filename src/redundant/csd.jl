# ---------------------------------------------------------------------------
# Canonical Signed Digit / Non-Adjacent Form.
#
# The cashier's trick: paying a 999 bill by handing over 1000 and receiving 1 is
# writing 999 = 1000 − 1.  Binary CSD is the identical move on runs of ones:
# 7 = 0111 becomes 1001̄ (8 − 1).
# ---------------------------------------------------------------------------

"""
    csd(x::Integer) -> SignedDigits

The canonical signed-digit (non-adjacent form) spelling of `x`, over the digit set
`{-1, 0, 1}` in radix 2.

Remarkably, finding the minimum-weight representation involves **no search at all**.
A single greedy pass from the least significant bit produces it:

- `x` even → emit `0`, halve.
- `x ≡ 1 (mod 4)` → this `1` is isolated: emit `+1` and keep it.
- `x ≡ 3 (mod 4)` → this `1` starts a run: emit `-1`, turning the whole run into a
  single borrow that surfaces at the run's far end.

Subtract the emitted digit, halve, repeat.  The greedy output is not merely good: the
non-adjacent form is **provably minimum-weight over every signed-binary spelling**,
so the linear scan already delivers the global optimum (see
[`verify_csd_minimal`](@ref)).

```jldoctest
julia> sd = csd(231);

julia> digit_string(sd)          # 256 − 32 + 8 − 1
"1001̄01001̄"

julia> weight(sd), adders(sd)    # 3 adders instead of binary's 5
(4, 3)

julia> digit_string(csd(27))     # 32 − 4 − 1
"1001̄01̄"
```
"""
function csd(x::Integer)
    ds = Int[]
    # keep the working value at the caller's width: `Int(x)` silently overflowed for
    # BigInt inputs, which are exactly what wide-operand experiments produce
    v = x
    neg = v < 0
    v = abs(v)
    while v > 0
        if iseven(v)
            push!(ds, 0)
            v = v >> 1
        else
            d = (mod(v, 4) == 1) ? 1 : -1
            push!(ds, d)
            v = (v - d) >> 1
        end
    end
    isempty(ds) && push!(ds, 0)
    neg && (ds .= .-ds)
    SignedDigits(ds, 2, 1)
end

"""    naf(x)

Alias for [`csd`](@ref).  In elliptic-curve cryptography the same object is called
the *non-adjacent form*, where it drops the expected point additions of a random
256-bit scalar from 128 to ``256/3 \\approx 85`` — a third of the dominant cost, from
a change of spelling."""
const naf = csd

"""
    csd_trace(x::Integer) -> Vector{NamedTuple}

The greedy scan, step by step: the running value, its residue mod 4, the digit
emitted, and why.  This is the table in the report's CSD section.

```jldoctest
julia> for r in csd_trace(27); println(r.x, "  mod4=", r.mod4, "  → ", r.digit, "   ", r.why); end
27  mod4=3  → -1   run starts: borrow
14  mod4=2  → 0   even
7  mod4=3  → -1   run starts: borrow
4  mod4=0  → 0   even
2  mod4=2  → 0   even
1  mod4=1  → 1   isolated one: keep
```
"""
function csd_trace(x::Integer)
    out = NamedTuple{(:x, :mod4, :digit, :why),Tuple{Int,Int,Int,String}}[]
    v = abs(x)
    while v > 0
        m = Int(mod(v, 4))
        if iseven(v)
            push!(out, (x = Int(v), mod4 = m, digit = 0, why = "even"))
            v = v >> 1
        else
            d = (m == 1) ? 1 : -1
            why = d == 1 ? "isolated one: keep" : "run starts: borrow"
            push!(out, (x = Int(v), mod4 = m, digit = d, why = why))
            v = (v - d) >> 1
        end
    end
    out
end

"""
    binary_weight(x::Integer) -> Int

Number of set bits — the adder count of a naive shift-and-add constant multiplier.
A random `n`-bit constant has `n/2` expected nonzero bits against CSD's `n/3 + O(1)`,
a guaranteed-average 33% cut."""
binary_weight(x::Integer) = count_ones(abs(x))

"""
    all_signed_spellings(x::Integer, ndig::Integer) -> Vector{SignedDigits}

Every signed-binary spelling of `x` that fits in `ndig` digits, sorted by weight.

The landscape shows why the canonical form matters twice over: cost varies wildly
(three adders up to eight for spellings of the very same number), and even the
*minimum* need not be unique — the non-adjacency rule is what breaks the tie
canonically.

```jldoctest
julia> sp = all_signed_spellings(231, 9);

julia> length(sp)                       # the full census
31

julia> minimum(weight, sp)              # the two best spellings
4

julia> count(s -> weight(s) == 4, sp)
2

julia> only(filter(s -> weight(s) == 4 && !has_adjacent_nonzeros(s), sp)) |> digit_string
"1001̄01001̄"
```
"""
function all_signed_spellings(x::Integer, ndig::Integer)
    out = SignedDigits[]
    digits = zeros(Int, ndig)
    function rec(pos::Int, remaining::Int)
        if pos > ndig
            remaining == 0 && push!(out, SignedDigits(copy(digits), 2, 1))
            return
        end
        # bound: the highest magnitude the remaining positions can still reach
        maxreach = 0
        for k in pos:ndig
            maxreach += 1 << (k - 1)
        end
        abs(remaining) > maxreach && return
        for d in (-1, 0, 1)
            w = 1 << (pos - 1)
            digits[pos] = d
            rec(pos + 1, remaining - d * w)
        end
        digits[pos] = 0
    end
    rec(1, Int(x))
    sort!(out; by = weight)
    out
end

"""
    verify_csd_minimal(upto::Integer) -> Bool

Check exhaustively that the greedy scan's weight equals the true minimum over *all*
signed-binary spellings, for every `x ≤ upto`.

The minimum is computed by a dynamic program over the two carry states, which is
independent of the greedy rule — so agreement is real evidence, not a tautology.

```jldoctest
julia> verify_csd_minimal(2000)
true
```
"""
function verify_csd_minimal(upto::Integer)
    for x in 1:Int(upto)
        weight(csd(x)) == min_signed_weight(x) || return false
    end
    true
end

"""
    min_signed_weight(x::Integer) -> Int

The minimum number of nonzero signed-binary digits needed to spell `x`, by dynamic
programming over the carry state — an independent check on [`csd`](@ref)."""
function min_signed_weight(x::Integer)
    # state: (value still to represent).  At each bit we may emit -1, 0, +1 provided
    # the parity works out; memoise on the remaining value.
    memo = Dict{Int,Int}()
    function rec(v::Int)
        v == 0 && return 0
        haskey(memo, v) && return memo[v]
        r = if iseven(v)
            rec(v >> 1)
        else
            # the "+1" branch must strictly decrease, or v = 1 recurses into itself
            a = rec((v - 1) >> 1)
            b = ((v + 1) >> 1) < v ? rec((v + 1) >> 1) : typemax(Int) - 1
            1 + min(a, b)
        end
        memo[v] = r
        r
    end
    rec(abs(Int(x)))
end

"""
    csd_multiplier_taps(x::Integer) -> Vector{Tuple{Int,Int}}

The `(shift, sign)` taps of a CSD constant multiplier: one per nonzero digit.  Shifts
are free (wiring), so `y = c·x` costs one adder/subtractor per tap beyond the first.

```jldoctest
julia> csd_multiplier_taps(231)          # 2^8 − 2^5 + 2^3 − 2^0
4-element Vector{Tuple{Int64, Int64}}:
 (0, -1)
 (3, 1)
 (5, -1)
 (8, 1)
```
"""
function csd_multiplier_taps(x::Integer)
    sd = csd(x)
    [(i - 1, sd.digits[i]) for i in eachindex(sd.digits) if sd.digits[i] != 0]
end

"""
    csd_multiply(c::Integer, x::Integer) -> Int

Evaluate `c·x` through the CSD tap structure — shifts and adds only, no multiplier.
Exact by construction: CSD is a *re-spelling*, not an approximation, so this loses
nothing, ever.

```jldoctest
julia> csd_multiply(231, 1234) == 231 * 1234
true
```
"""
function csd_multiply(c::Integer, x::Integer)
    s = 0
    for (sh, sg) in csd_multiplier_taps(c)
        s += sg * (Int(x) << sh)
    end
    s
end

"""
    csd_density(nbits::Integer; trials=10_000, rng=Random.default_rng()) -> NamedTuple

Measure the expected nonzero density of binary versus CSD over random `nbits`
constants.  The theory says the density drops from `1/2` to `1/3`; this measures it.

```julia
julia> csd_density(16; trials=20_000)
(binary = 0.4999, csd = 0.3335, binary_weight = 8.0, csd_weight = 5.7, saving = 0.29)
```
"""
function csd_density(nbits::Integer; trials::Integer = 10_000,
                     rng::AbstractRNG = Random.default_rng())
    bw = 0.0; cw = 0.0
    for _ in 1:trials
        x = rand(rng, 1:(1 << nbits) - 1)
        bw += binary_weight(x)
        cw += weight(csd(x))
    end
    bw /= trials; cw /= trials
    (binary = bw / nbits, csd = cw / (nbits + 1),
     binary_weight = bw, csd_weight = cw, saving = 1 - cw / bw)
end
