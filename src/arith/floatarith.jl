# ---------------------------------------------------------------------------
# The four datapaths: +, ×, ÷, and the fused MAC.
# ---------------------------------------------------------------------------

"""
    fpadd(f::FloatFormat, a, b) -> DatapathTrace

Floating-point addition, traced: align the smaller operand, add the significands,
renormalise, round once.

The alignment shift is where information is lost — bits pushed off the right end
survive only as guard/round/sticky, and if the exponents differ by more than the
significand width the small operand vanishes entirely.

```jldoctest
julia> t = fpadd(FP32, 12.0, 6.0);

julia> t.result, t.info.align_shift, t.info.norm_shift
(18.0, 1, 1)

julia> fpadd(FP32, 1.0e8, 1.0).result     # absorption: the 1 is below half an ulp
1.0e8
```
"""
function fpadd(f::FloatFormat, a::Real, b::Real)
    va, vb = quantize(f, a), quantize(f, b)
    pa, pb = unpack(f, va), unpack(f, vb)
    exact = Float64(va) + Float64(vb)
    result = quantize(f, exact)

    # order operands so `hi` has the larger exponent — that is the one the
    # alignment shifter leaves alone
    swapped = pb.e > pa.e
    hi, lo = swapped ? (pb, pa) : (pa, pb)
    Δe = (hi.iszero || lo.iszero) ? 0 : hi.e - lo.e
    shifted = lo.significand / exp2(Δe)
    sgn_hi = hi.sign == 1 ? -1.0 : 1.0
    sgn_lo = lo.sign == 1 ? -1.0 : 1.0
    raw = sgn_hi * hi.significand + sgn_lo * shifted

    norm_e = hi.e
    norm_shift = 0
    if raw != 0
        k = binade_exponent(raw)
        norm_shift = k
        norm_e = hi.e + k
    end
    norm_sig = raw == 0 ? 0.0 : raw / exp2(norm_shift)

    stages = Stage[
        Stage("unpack a", "$(pa.sign == 1 ? "-" : "+") $(_sigstr(f, pa.significand)) × $(_expstr(pa.e))", va),
        Stage("unpack b", "$(pb.sign == 1 ? "-" : "+") $(_sigstr(f, pb.significand)) × $(_expstr(pb.e))", vb),
        Stage("align", "shift smaller right by Δe = $(Δe)  ⇒  $(_sigstr(f, abs(shifted), bits=f.mbits+4)) × $(_expstr(hi.e))", nothing),
        Stage("add significands", "$(round(raw, sigdigits=10)) × $(_expstr(hi.e))", nothing),
        Stage("normalize", norm_shift == 0 ? "already in [1,2) — no shift" :
              (norm_shift > 0 ? "carry out: shift right $(norm_shift), e += $(norm_shift)" :
                                "cancellation: shift left $(-norm_shift), e -= $(-norm_shift)"),
              nothing),
        Stage("round", "to $(f.mbits + 1) significand bits, ties to even", result),
    ]

    info = (align_shift = Δe, norm_shift = norm_shift,
            raw_significand = raw, norm_significand = norm_sig, norm_exp = norm_e,
            absorbed = (result == va && vb != 0) || (result == vb && va != 0),
            round_ulps = _round_ulps(f, exact, result), swapped = swapped)
    DatapathTrace(:add, f, [va, vb], stages, result, exact, info)
end

"""
    fpsub(f::FloatFormat, a, b) -> DatapathTrace

Subtraction, i.e. `fpadd(f, a, -b)`.  Watch the `norm_shift` field go strongly
negative when the operands are close: that is catastrophic cancellation, the
normaliser shifting correct leading digits out and rounded trailing digits in."""
fpsub(f::FloatFormat, a::Real, b::Real) = fpadd(f, a, -Float64(b))

"""
    fpmul(f::FloatFormat, a, b) -> DatapathTrace

Multiplication: no alignment needed — XOR the signs, add the exponents, multiply the
significands, then renormalise by at most one right shift.

Because normalized significands lie in `[1,2)`, their product lies in `[1,4)`, so the
normalise stage never needs more than a single shift.

```jldoctest
julia> fpmul(FP32, 6.0, 6.0).result
36.0

julia> fpmul(E2M1, 1.5, 1.5).result       # 2.25 is not on the FP4 grid
2.0
```
"""
function fpmul(f::FloatFormat, a::Real, b::Real)
    va, vb = quantize(f, a), quantize(f, b)
    pa, pb = unpack(f, va), unpack(f, vb)
    exact = Float64(va) * Float64(vb)
    result = quantize(f, exact)

    sigprod = pa.significand * pb.significand
    esum = pa.e + pb.e
    norm_shift = (sigprod >= 2 && sigprod != 0) ? 1 : 0
    norm_sig = sigprod / exp2(norm_shift)
    Esum = pa.E + pb.E - f.bias

    stages = Stage[
        Stage("unpack a", "$(_sigstr(f, pa.significand)) × $(_expstr(pa.e))", va),
        Stage("unpack b", "$(_sigstr(f, pb.significand)) × $(_expstr(pb.e))", vb),
        Stage("sign", "s = s_a ⊕ s_b = $(pa.sign ⊻ pb.sign)", nothing),
        Stage("exponent add", "e = $(pa.e) + $(pb.e) = $(esum)   (stored: E = $(pa.E) + $(pb.E) − $(f.bias) = $(Esum))", nothing),
        Stage("significand multiply", "$(_sigstr(f, pa.significand)) × $(_sigstr(f, pb.significand)) = $(round(sigprod, sigdigits=10))", nothing),
        Stage("normalize", norm_shift == 1 ? "product ≥ 2: shift right 1, e += 1" : "product in [1,2) — no shift", nothing),
        Stage("round", "to $(f.mbits + 1) significand bits, ties to even", result),
    ]

    info = (exp_sum = esum, stored_E = Esum, sig_product = sigprod,
            norm_shift = norm_shift, norm_significand = norm_sig,
            round_ulps = _round_ulps(f, exact, result))
    DatapathTrace(:mul, f, [va, vb], stages, result, exact, info)
end

"""
    fpdiv(f::FloatFormat, a, b) -> DatapathTrace

Division: subtract the exponents, XOR the signs, divide the significands by binary
long division.  Since both significands lie in `[1,2)` the quotient lies in
`(1/2, 2)`, so normalisation shifts *left* by at most one.

The trace's `quotient_bits` field carries the long-division digit string — one bit
per position, exactly as an SRT recurrence would emit them (albeit here in radix 2
and without the redundant quotient digits real dividers use).

```jldoctest
julia> t = fpdiv(FP32, 9.0, 6.0);

julia> t.result, t.info.norm_shift
(1.5, -1)
```
"""
function fpdiv(f::FloatFormat, a::Real, b::Real)
    va, vb = quantize(f, a), quantize(f, b)
    pa, pb = unpack(f, va), unpack(f, vb)
    exact = Float64(va) / Float64(vb)
    result = quantize(f, exact)

    sigq = pb.significand == 0 ? Inf : pa.significand / pb.significand
    ediff = pa.e - pb.e
    norm_shift = (isfinite(sigq) && sigq != 0 && sigq < 1) ? -1 : 0
    norm_sig = sigq / exp2(norm_shift)

    # binary long division of the significands, for the trace
    qbits = Char[]
    if isfinite(sigq) && pb.significand != 0
        rem = pa.significand
        d = pb.significand
        for _ in 0:(f.mbits + 2)
            if rem >= d
                push!(qbits, '1'); rem -= d
            else
                push!(qbits, '0')
            end
            rem *= 2
        end
    end
    qstr = isempty(qbits) ? "" : string(qbits[1]) * "." * String(qbits[2:end])

    stages = Stage[
        Stage("unpack a", "$(_sigstr(f, pa.significand)) × $(_expstr(pa.e))", va),
        Stage("unpack b", "$(_sigstr(f, pb.significand)) × $(_expstr(pb.e))", vb),
        Stage("sign", "s = s_a ⊕ s_b = $(pa.sign ⊻ pb.sign)", nothing),
        Stage("exponent subtract", "e = $(pa.e) − $(pb.e) = $(ediff)", nothing),
        Stage("significand divide", "long division ⇒ $(qstr)  = $(round(sigq, sigdigits=10))", nothing),
        Stage("normalize", norm_shift == -1 ? "quotient < 1: shift left 1, e −= 1" : "quotient in [1,2) — no shift", nothing),
        Stage("round", "to $(f.mbits + 1) significand bits, ties to even", result),
    ]

    info = (exp_diff = ediff, sig_quotient = sigq, quotient_bits = qstr,
            norm_shift = norm_shift, norm_significand = norm_sig,
            round_ulps = _round_ulps(f, exact, result))
    DatapathTrace(:div, f, [va, vb], stages, result, exact, info)
end

"""
    fpfma(f::FloatFormat, a, b, c) -> DatapathTrace

Fused multiply–add `a*b + c`: the double-width product is kept **exact**, `c` is
added to it at full width, and the result is rounded **once**, at the very end.

Contrast [`fpmul_then_add`](@ref), which rounds twice.  The difference is the entire
advantage of the FMA, and it is why error-free algorithms — Kahan summation,
double-double arithmetic, accurate dot products — are built on it.

```jldoctest
julia> f4 = FloatFormat("toy", 4, 4);       # 4 fraction bits, as in the report

julia> fpfma(f4, 1.125, 1.375, -1.5).result        # exact
0.046875

julia> fpmul_then_add(f4, 1.125, 1.375, -1.5).result   # the first rounding kills it
0.0625
```
"""
function fpfma(f::FloatFormat, a::Real, b::Real, c::Real)
    va, vb, vc = quantize(f, a), quantize(f, b), quantize(f, c)
    prod_exact = Float64(va) * Float64(vb)
    exact = prod_exact + Float64(vc)
    result = quantize(f, exact)

    stages = Stage[
        Stage("unpack", "a = $(va), b = $(vb), c = $(vc)", nothing),
        Stage("multiply (exact)", "a × b = $(prod_exact)  — kept at full $(2*(f.mbits+1))-bit width, NOT rounded", prod_exact),
        Stage("align c", "shift c against the double-width product", vc),
        Stage("wide add", "a×b + c = $(exact)  (exact)", exact),
        Stage("round ONCE", "to $(f.mbits + 1) significand bits", result),
    ]
    info = (product_exact = prod_exact, roundings = 1,
            round_ulps = _round_ulps(f, exact, result))
    DatapathTrace(:fma, f, [va, vb, vc], stages, result, exact, info)
end

"""
    fpmul_then_add(f::FloatFormat, a, b, c) -> DatapathTrace

The unfused `a*b + c`: round the product, *then* add and round again — two roundings.
Provided for comparison against [`fpfma`](@ref)."""
function fpmul_then_add(f::FloatFormat, a::Real, b::Real, c::Real)
    va, vb, vc = quantize(f, a), quantize(f, b), quantize(f, c)
    prod_exact = Float64(va) * Float64(vb)
    prod_rounded = quantize(f, prod_exact)
    exact = prod_exact + Float64(vc)
    sum_exact = Float64(prod_rounded) + Float64(vc)
    result = quantize(f, sum_exact)

    stages = Stage[
        Stage("unpack", "a = $(va), b = $(vb), c = $(vc)", nothing),
        Stage("multiply", "a × b = $(prod_exact)", prod_exact),
        Stage("ROUND #1", "product → $(prod_rounded)   (discarded: $(prod_exact - prod_rounded))", prod_rounded),
        Stage("add c", "$(prod_rounded) + $(vc) = $(sum_exact)", sum_exact),
        Stage("ROUND #2", "to $(f.mbits + 1) significand bits", result),
    ]
    info = (product_exact = prod_exact, product_rounded = prod_rounded, roundings = 2,
            first_round_error = prod_exact - prod_rounded,
            round_ulps = _round_ulps(f, exact, result))
    DatapathTrace(:mul_then_add, f, [va, vb, vc], stages, result, exact, info)
end

# ---- convenience scalar wrappers -------------------------------------------

"""    fadd(f, a, b), fmul(f, a, b), fdiv(f, a, b), ffma(f, a, b, c)

Value-only versions of the traced operations — the correctly rounded result of one
operation in format `f`, with no trace built.  Use these in loops."""
fadd(f::FloatFormat, a::Real, b::Real) = quantize(f, quantize(f, a) + quantize(f, b))
fmul(f::FloatFormat, a::Real, b::Real) = quantize(f, quantize(f, a) * quantize(f, b))
fdiv(f::FloatFormat, a::Real, b::Real) = quantize(f, quantize(f, a) / quantize(f, b))
ffma(f::FloatFormat, a::Real, b::Real, c::Real) =
    quantize(f, quantize(f, a) * quantize(f, b) + quantize(f, c))

# ---- pretty printing -------------------------------------------------------

function Base.show(io::IO, ::MIME"text/plain", t::DatapathTrace)
    opname = Dict(:add => "ADD", :mul => "MULTIPLY", :div => "DIVIDE",
                  :fma => "FUSED MULTIPLY-ADD", :mul_then_add => "UNFUSED MULTIPLY-ADD")[t.op]
    println(io, "─"^72)
    println(io, "  ", opname, "  in ", t.fmt.name,
            "   inputs: ", join(string.(t.inputs), ", "))
    println(io, "─"^72)
    w = maximum(length(s.label) for s in t.stages)
    for (i, s) in enumerate(t.stages)
        @printf(io, "  %d. %-*s │ %s\n", i, w, s.label, s.detail)
    end
    println(io, "─"^72)
    @printf(io, "  exact  : %.17g\n", t.exact)
    @printf(io, "  result : %.17g\n", t.result)
    re = relerror(t)
    @printf(io, "  rel err: %.4g", re)
    if re == 0
        print(io, "   (exact)")
    end
    println(io)
    ru = rounding_ulps(t)
    isnan(ru) || @printf(io, "  rounding moved the answer by %+.4f ulp\n", ru)
    print(io, "─"^72)
end

Base.show(io::IO, t::DatapathTrace) =
    print(io, "DatapathTrace(:", t.op, ", ", t.fmt.name, ", ", t.inputs, " → ", t.result, ")")
