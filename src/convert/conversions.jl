# ---------------------------------------------------------------------------
# Conversions between representations.
#
# The organising idea real machines live by: let FP32 be the HUB.  Values enter it
# once (one half-ulp, paid at the door) and every other representation is an
# EXCURSION — words leave the hub, are converted into some internal format chosen on
# merit for one operation, and convert back before anything else sees them.
#
# The decisive question is not "which format is best" but, per excursion: does the
# returning word have the same bits it would have had without the trip?
# ---------------------------------------------------------------------------

# `convert_format` now lives in convert/universal.jl, where it spans every format
# family rather than just the scalar ones.

"""
    narrow(to::FloatFormat, from::FloatFormat, x) -> NamedTuple

A traced narrowing conversion (`FP32 → BF16`, `FP32 → E2M1`, …), reporting where the
value landed and why.

`FP32 → BF16` is the extreme case: BF16 is simply FP32 with the bottom 16 fraction
bits chopped off, so the exponent — and therefore the range — is untouched and *all*
the loss is precision.

Returned fields: `value`, `exact`, `relerror`, `overflowed`, `underflowed`,
`became_subnormal`, `round_ulps`.

```jldoctest
julia> r = narrow(BF16, FP32, 3.14159);

julia> r.value, round(r.relerror, sigdigits=3)
(3.140625, 0.00296)

julia> narrow(FP16, FP32, 1.0e30).overflowed
true
```
"""
function narrow(to::FloatFormat, from::FloatFormat, x::Real)
    src = quantize(from, x)
    v = quantize(to, src)
    (value = v,
     exact = src,
     relerror = src == 0 ? abs(v) : abs(v - src) / abs(src),
     overflowed = (isfinite(src) && !isfinite(v)) || (to.saturate && abs(src) > maxfinite(to)),
     underflowed = src != 0 && v == 0,
     became_subnormal = v != 0 && abs(v) < minnormal(to),
     round_ulps = _round_ulps(to, src, v))
end

"""
    int_to_float(f::FloatFormat, n::Integer; width=32) -> NamedTuple

The `int → float` conversion, traced the way hardware performs it: a count-leading-
zeros to find the exponent, a shift into the significand window, and one rounding.

The exponent is found **without any logarithm**: the position of the top set bit *is*
the exponent, `e = (width−1) − CLZ(x)` — an `O(log width)` binary search in gates,
five levels for 32 bits.

Lossless precisely for integers below `2^(mbits+1)`; the first casualty for FP32 is
`16_777_217`.

```jldoctest
julia> r = int_to_float(FP32, 1440);

julia> r.clz, r.exponent, r.fraction, r.exact
(21, 10, 3407872, true)

julia> string(r.code, base=16)
"44b40000"

julia> int_to_float(FP32, 16_777_217).exact
false
```
"""
function int_to_float(f::FloatFormat, n::Integer; width::Integer = 32)
    v = Int(n); a = abs(v)
    topbit = a == 0 ? 0 : (8 * sizeof(Int) - leading_zeros(a))
    clz = a == 0 ? Int(width) : Int(width) - topbit
    e = a == 0 ? 0 : topbit - 1
    q = quantize(f, v)
    m = a == 0 ? 0.0 : a / exp2(e)
    F = a == 0 ? 0 : round(Int, (m - 1) * nmancodes(f))
    (value = q, clz = clz, exponent = e, mantissa = m, fraction = F,
     code = encode(f, v), exact = q == Float64(v),
     lossless_below = 1 << (f.mbits + 1))
end

"""
    float_to_fixed(fx::FixedFormat, f::FloatFormat, x) -> NamedTuple

Convert a float into fixed point: scale by `2ⁿ`, round, clamp.  Reports whether the
value saturated — the failure mode fixed point has that floating point does not."""
function float_to_fixed(fx::FixedFormat, f::FloatFormat, x::Real)
    src = quantize(f, x)
    v = quantize(fx, src)
    (value = v, exact = src,
     relerror = src == 0 ? abs(v) : abs(v - src) / abs(src),
     saturated = src > fxmax(fx) || src < fxmin(fx))
end

"""
    fixed_to_float(f::FloatFormat, fx::FixedFormat, I::Integer) -> Float64

Read a stored fixed-point integer back as a float."""
fixed_to_float(f::FloatFormat, fx::FixedFormat, I::Integer) = quantize(f, decode(fx, I))

# ---- the monotone-encoding gift -------------------------------------------

"""
    nextafter(f::FloatFormat, x::Real) -> Float64

The next representable value above `x` — implemented, as in real systems, by an
**integer increment of the bit pattern**.

This works because FP encodings are *monotone*: for non-negative values a larger bit
pattern always means a larger value, and one increment always means one grid step.  At
the bottom seam that step is a tiny `2⁻¹⁴⁹`; at the top seam a gargantuan `2¹⁰⁴`; but
each is exactly one ulp of its own neighbourhood.

```jldoctest
julia> nextafter(FP32, 1.0) - 1.0 == machine_eps(FP32)
true

julia> nextafter(E2M1, 2.0)
3.0
```
"""
function nextafter(f::FloatFormat, x::Real)
    v = quantize(f, x)
    isnan(v) && return NaN
    v == 0 && return minsubnormal(f)
    c = encode(f, v)
    v > 0 ? decode(f, c + 1) : decode(f, c - 1)
end

"""
    prevafter(f::FloatFormat, x::Real) -> Float64

The next representable value below `x`."""
function prevafter(f::FloatFormat, x::Real)
    v = quantize(f, x)
    isnan(v) && return NaN
    v == 0 && return -minsubnormal(f)
    c = encode(f, v)
    v > 0 ? decode(f, c - 1) : decode(f, c + 1)
end

"""
    encodings_monotone(f::FloatFormat) -> Bool

Verify the monotonicity invariant exhaustively for a narrow format.

```jldoctest
julia> encodings_monotone(E2M1), encodings_monotone(E4M3), encodings_monotone(FP16)
(true, true, true)
```
"""
function encodings_monotone(f::FloatFormat)
    nbits(f) <= 20 || throw(ArgumentError("exhaustive check refuses $(nbits(f))-bit formats"))
    prev = -Inf
    top = (1 << (f.ebits + f.mbits)) - 1
    for c in 0:top
        v = decode(f, c)
        isfinite(v) || continue
        v > prev || return false
        prev = v
    end
    true
end

"""
    top_seam(f::FloatFormat) -> NamedTuple

The boundary where the normals hand over to `+Inf`, measured in **both** spaces.

In *encoding* space `x_max` and `+Inf` are adjacent integers.  In *value* space the
next grid point would be exactly `2^(emax+1)` — but writing it needs the reserved
exponent, so infinity occupies the slot of a **phantom grid point** one ordinary ulp
above `x_max`.  Under round-to-nearest a result need not reach that phantom point to
overflow: it need only *round* to it, so the threshold is the midpoint.

```jldoctest
julia> t = top_seam(FP32);

julia> t.xmax == maxfinite(FP32), t.phantom == 2.0^128
(true, true)

julia> t.overflow_threshold == 2.0^128 - 2.0^103
true
```
"""
function top_seam(f::FloatFormat)
    xm = maxfinite(f)
    u = ulp(f, xm)
    phantom = exp2(emax(f) + 1)
    (xmax = xm, xmax_code = encode(f, xm), ulp = u, phantom = phantom,
     overflow_threshold = phantom - u / 2,
     codes_adjacent = true)
end

"""
    bottom_seam(f::FloatFormat) -> NamedTuple

The subnormal/normal boundary.  Unlike the top seam, the spacing here is **identical
on both sides**: pinning the subnormal scale at `2^emin` makes the last subnormal and
the first normal exactly one subnormal step apart — the same step as every rung around
them.  The ladder crosses without a seam you could feel.

```jldoctest
julia> b = bottom_seam(FP32);

julia> b.gap == minsubnormal(FP32), b.codes_differ_by == 1
(true, true)
```
"""
function bottom_seam(f::FloatFormat)
    mn = minnormal(f)
    largest_sub = prevafter(f, mn)
    (largest_subnormal = largest_sub, smallest_normal = mn,
     gap = mn - largest_sub,
     subnormal_step = minsubnormal(f),
     codes_differ_by = Int(encode(f, mn)) - Int(encode(f, largest_sub)),
     seamless = (mn - largest_sub) == minsubnormal(f))
end

# ---- redundant-domain conversions ------------------------------------------

"""
    to_carrysave(x::Integer) -> Tuple{Int,Int}

Enter the carry-save domain: any `(u, t)` pair with `u + t = x`.  The canonical entry
is `(x, 0)`; the representation is redundant, so this is *a* spelling."""
to_carrysave(x::Integer) = (Int(x), 0)

"""
    from_carrysave(u::Integer, t::Integer) -> Int

Leave the carry-save domain — the single carry-propagate addition that is the **exit
toll**, paid once at the foot of a whole reduction tree rather than once per operand."""
from_carrysave(u::Integer, t::Integer) = Int(u) + Int(t)

"""
    rr4_to_canonical(sd::SignedDigits) -> Vector{Int}

Convert a redundant radix-4 string back to the unique canonical `{0..r−1}` form — one
full carry propagation, the *exit CPA*.

This is the toll every redundant system pays: conventional binary at the boundaries,
redundancy in the interior, one conversion at each seam.

```jldoctest
julia> rr4_to_canonical(to_rr4(49))      # 49 = 3·16 + 0·4 + 1
3-element Vector{Int64}:
 1
 0
 3
```
"""
function rr4_to_canonical(sd::SignedDigits)
    v = value(sd)
    v >= 0 || throw(ArgumentError("canonical radix-$(sd.radix) form needs a non-negative value"))
    to_radix(v, sd.radix)
end

"""
    minimal_to_maximal(sd::SignedDigits) -> SignedDigits

Widen a minimally-redundant `{-2..2}` string into the maximal `{-3..3}` alphabet —
verbatim, since the alphabets are strictly nested.  Every minimal spelling *is* a
maximal spelling; they are different coordinate systems on the same integers."""
function minimal_to_maximal(sd::SignedDigits)
    sd.maxdigit <= 2 || throw(ArgumentError("expected a minimal-alphabet string"))
    SignedDigits(copy(sd.digits), sd.radix, 3)
end

"""
    maximal_to_minimal(sd::SignedDigits) -> SignedDigits

Narrow a maximal `{-3..3}` string into `{-2..2}` under one bounded borrow pass: any
`±3` becomes `∓1` with a `±1` handed one place up.

```jldoctest
julia> sd = SignedDigits([3, 3, 1], 4, 3);

julia> m = maximal_to_minimal(sd);

julia> value(m) == value(sd), m.maxdigit
(true, 2)
```
"""
function maximal_to_minimal(sd::SignedDigits)
    out = Int[]; carry = 0
    for d in sd.digits
        v = d + carry; carry = 0
        if v > 2
            v -= 4; carry = 1
        elseif v < -2
            v += 4; carry = -1
        end
        push!(out, v)
    end
    while carry != 0
        v = carry; carry = 0
        if v > 2
            v -= 4; carry = 1
        elseif v < -2
            v += 4; carry = -1
        end
        push!(out, v)
    end
    SignedDigits(out, sd.radix, 2)
end

# ---- the hub and its excursions --------------------------------------------

"""
    ExcursionClass

The verdict on a round trip out of the FP32 hub and back.

- `INVISIBLE` — the returning word is **bit-identical** to what the hub would have
  produced alone.  The Booth/RR4-and-carry-save interior of a multiplier computes the
  exact product before its single rounding, so it returns precisely the correctly
  rounded result IEEE 754 mandates; a CSD constant multiplier likewise, since a
  re-spelling is exact.  These change cost and nothing else — take every one.
- `VISIBLE_BETTER` — different bits, because they are *better*.  Deviating from
  reference bits is a semantics decision, not a correctness failure, and it has
  precedent: the FMA is exactly such an excursion, standardised in IEEE 754-2008.
- `VISIBLE_WORSE` — different, cheaper, worse.  The MXFP4 trip.  Because it begins and
  ends at hub words its damage is *measurable in isolation*, which is how one decides
  that GEMM operands may take this trip while the softmax may not.
"""
@enum ExcursionClass INVISIBLE VISIBLE_BETTER VISIBLE_WORSE

"""
    Excursion

The record of one round trip: what came back, whether the bits changed, and by how
much the answer moved against the truth."""
struct Excursion
    name::String
    hub::FloatFormat
    result::Float64
    reference::Float64
    truth::Float64
    bits_identical::Bool
    class::ExcursionClass
end

relerror(e::Excursion) = e.truth == 0 ? abs(e.result) : abs(e.result - e.truth) / abs(e.truth)

"""
    excursion(name, hub, result, reference, truth) -> Excursion

Classify a round trip by comparing `result` (what the excursion returned) against
`reference` (what the hub alone would produce) and `truth` (the exact answer)."""
function excursion(name::AbstractString, hub::FloatFormat, result::Real,
                   reference::Real, truth::Real)
    r = quantize(hub, result); ref = quantize(hub, reference)
    same = encode(hub, r) == encode(hub, ref)
    cls = same ? INVISIBLE :
          (abs(r - truth) < abs(ref - truth) ? VISIBLE_BETTER : VISIBLE_WORSE)
    Excursion(String(name), hub, r, ref, Float64(truth), same, cls)
end

"""
    multiplier_excursion(hub::FloatFormat, a, b) -> Excursion

The **invisible** excursion: the Booth/carry-save core computes `a×b` exactly in its
double-width register, and the single exit rounding lands bit-for-bit on the mandated
result.

The report's poetic instance is `0.1f × 10f`.  The hub holds `0x3DCCCCCD` (0.1 rounded
*up*); the core forms `1 + 2⁻²⁶` exactly, an amount the hub cannot represent; the exit
rounding snaps it to `0x3F800000 = 1.0` exactly.  The error `0.1` picked up entering
the hub is precisely cancelled by the rounding on the way back.

```jldoctest
julia> e = multiplier_excursion(FP32, 0.1, 10.0);

julia> e.result, e.bits_identical, e.class
(1.0, true, INVISIBLE)
```
"""
function multiplier_excursion(hub::FloatFormat, a::Real, b::Real)
    va, vb = quantize(hub, a), quantize(hub, b)
    exact = va * vb
    res = quantize(hub, exact)
    excursion("multiplier core (Booth + carry-save)", hub, res, res, exact)
end

"""
    redundant_dot_excursion(hub::FloatFormat, x, y) -> Excursion

The **visible-better** excursion: keep the accumulator redundant across every term and
round once at the exit, against the per-step hub chain that rounds `N` times.

The report's demonstration sums `1, 2⁻²⁵, 2⁻²⁵, 2⁻²⁵`.  Each grain is below the
half-ulp at 1, so the per-step chain absorbs all three and returns exactly `1.0`.  The
excursion holds `1 + 3·2⁻²⁵` exactly; the accumulated grains now total 1.5 ulps, over
the threshold, and the returned word is `1 + 2⁻²³` — provably the correctly rounded
true sum.  Deferring the rounding did not merely save carries; it saved *information*.

```jldoctest
julia> e = redundant_dot_excursion(FP32, [1.0, 2.0^-25, 2.0^-25, 2.0^-25], ones(4));

julia> e.reference == 1.0, e.result == 1.0 + 2.0^-23, e.class
(true, true, VISIBLE_BETTER)
```
"""
function redundant_dot_excursion(hub::FloatFormat, x::AbstractVector, y::AbstractVector)
    xs = [quantize(hub, v) for v in x]
    ys = [quantize(hub, v) for v in y]
    truth = sum(xs[i] * ys[i] for i in eachindex(xs))
    chain = 0.0
    for i in eachindex(xs)
        chain = quantize(hub, chain + quantize(hub, xs[i] * ys[i]))
    end
    excursion("dot product, accumulator kept redundant", hub, quantize(hub, truth), chain, truth)
end

"""
    mx_excursion(hub::FloatFormat, bf::BlockFormat, x, y) -> Excursion

The **visible-worse** excursion: the MXFP4 trip, returning a word measurably far from
truth in exchange for `7.5×` fewer bits and `144×` smaller multipliers."""
function mx_excursion(hub::FloatFormat, bf::BlockFormat, x::AbstractVector, y::AbstractVector)
    xs = [quantize(hub, v) for v in x]
    ys = [quantize(hub, v) for v in y]
    truth = dot(xs, ys)
    r = block_dot(bf, xs, ys; accumulate = hub)
    excursion("MXFP4 GEMM operands", hub, r.value, quantize(hub, truth), truth)
end

function Base.show(io::IO, ::MIME"text/plain", e::Excursion)
    verdict = e.class == INVISIBLE ? "INVISIBLE — bit-identical; take it unconditionally" :
              e.class == VISIBLE_BETTER ? "VISIBLE, BETTER — different bits, closer to truth" :
              "VISIBLE, WORSE — different, cheaper, further from truth"
    println(io, "─"^72)
    println(io, "  Excursion: ", e.name, "        hub: ", e.hub.name)
    println(io, "─"^72)
    @printf(io, "  returned  : %-24.17g 0x%s\n", e.result, string(encode(e.hub, e.result), base=16))
    @printf(io, "  hub alone : %-24.17g 0x%s\n", e.reference, string(encode(e.hub, e.reference), base=16))
    @printf(io, "  truth     : %.17g\n", e.truth)
    @printf(io, "  rel error : %.4g\n", relerror(e))
    println(io, "─"^72)
    print(io, "  ", verdict)
end
