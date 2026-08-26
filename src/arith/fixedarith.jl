# ---------------------------------------------------------------------------
# Fixed-point arithmetic: ordinary integer hardware, plus scale bookkeeping.
# ---------------------------------------------------------------------------

"""
    fxadd(f::FixedFormat, a, b; mode=SATURATE) -> Float64

Add two Q`m`.`n` values.  When both operands share a format the underlying integers
add directly and the result is already in Q`m`.`n` — this is why fixed point runs on
a plain integer ALU.  The only hazard is overflow, handled per `mode`.

```jldoctest
julia> fxadd(FixedFormat(3,4), 2.5, 1.25)
3.75

julia> fxadd(FixedFormat(7,8), 100.0, 100.0)          # saturates
127.99609375

julia> fxadd(FixedFormat(7,8), 100.0, 100.0; mode=WRAP)   # wraps, catastrophically
-56.0
```
"""
function fxadd(f::FixedFormat, a::Real, b::Real; mode::OverflowMode = SATURATE)
    I = encode(f, a; mode) + encode(f, b; mode)
    decode(f, _clampcode(f, I, mode))
end

"""    fxsub(f, a, b; mode=SATURATE) -> Float64"""
function fxsub(f::FixedFormat, a::Real, b::Real; mode::OverflowMode = SATURATE)
    I = encode(f, a; mode) - encode(f, b; mode)
    decode(f, _clampcode(f, I, mode))
end

"""
    fxmul(f::FixedFormat, a, b; mode=SATURATE) -> Float64

Multiply two Q`m`.`n` values.  Multiplying two scaled integers doubles the scale —
`(a·2ⁿ)(b·2ⁿ) = ab·2²ⁿ` — so the raw product lives in Q`2m`.`2n` and must be shifted
right by `n` (with rounding) to return to Q`m`.`n`.

```jldoctest
julia> fxmul(FixedFormat(3,4), 1.5, 2.25)     # 24 × 36 = 864, >> 4 = 54 = 3.375
3.375
```
"""
function fxmul(f::FixedFormat, a::Real, b::Real; mode::OverflowMode = SATURATE)
    raw = encode(f, a; mode) * encode(f, b; mode)     # Q(2m).(2n)
    I = _shift_round(raw, f.n)
    decode(f, _clampcode(f, I, mode))
end

"""
    fxdiv(f::FixedFormat, a, b; mode=SATURATE) -> Float64

Divide two Q`m`.`n` values: pre-shift the dividend left by `n` before the integer
divide — the mirror image of [`fxmul`](@ref)'s post-shift."""
function fxdiv(f::FixedFormat, a::Real, b::Real; mode::OverflowMode = SATURATE)
    Ib = encode(f, b; mode)
    Ib == 0 && throw(DivideError())
    num = encode(f, a; mode) << f.n
    I = _div_round(num, Ib)
    decode(f, _clampcode(f, I, mode))
end

# arithmetic right shift with round-to-nearest, ties away from zero
function _shift_round(raw::Integer, n::Integer)
    n == 0 && return Int(raw)
    half = 1 << (n - 1)
    raw >= 0 ? Int((raw + half) >> n) : -Int((-raw + half) >> n)
end

function _div_round(num::Integer, den::Integer)
    q, r = divrem(num, den)
    2 * abs(r) >= abs(den) || return Int(q)
    Int(q + (sign(num) * sign(den)))
end

"""
    fxdot(f::FixedFormat, x, y; mode=SATURATE, wide=true) -> Float64

Fixed-point dot product.  With `wide=true` the products accumulate at full Q`2m`.`2n`
width and are shifted back only once at the end — the fixed-point analogue of a wide
accumulator, and the only sane way to do it.  With `wide=false` every product is
shifted and clamped back to Q`m`.`n` immediately, which is where the errors come from.
"""
function fxdot(f::FixedFormat, x::AbstractVector, y::AbstractVector;
               mode::OverflowMode = SATURATE, wide::Bool = true)
    length(x) == length(y) || throw(DimensionMismatch("fxdot: length mismatch"))
    if wide
        acc = 0
        for i in eachindex(x)
            acc += encode(f, x[i]; mode) * encode(f, y[i]; mode)
        end
        return decode(f, _clampcode(f, _shift_round(acc, f.n), mode))
    else
        acc = 0.0
        for i in eachindex(x)
            acc = fxadd(f, acc, fxmul(f, x[i], y[i]; mode); mode)
        end
        return acc
    end
end
