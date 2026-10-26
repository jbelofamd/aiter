# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""All-reduce + residual add + RMSNorm (+ fp8 group quant): the all-reduce of
comms/all_reduce.py with one row in each program, so that the receiver of a row
can normalize it before it stores anything."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.gfx1250.comms.all_reduce import (
    _hi,
    _lo,
    _pack,
    _receive,
    _send,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_all_reduce_rmsnorm_repr = make_kernel_repr(
    "_all_reduce_rmsnorm_kernel", ["RANK", "LINES", "GEMMA", "QUANT"]
)


@gluon.jit(repr=_all_reduce_rmsnorm_repr, do_not_specialize=["tokens"])
def _all_reduce_rmsnorm_kernel(
    x,
    residual,
    weight,
    out,
    residual_out,
    q,
    scale,
    inboxes,
    calls,
    err,
    tokens,
    eps,
    RANK: gl.constexpr,
    LINES: gl.constexpr,
    SLOT: gl.constexpr,
    MAXIT: gl.constexpr,
    GEMMA: gl.constexpr,
    QUANT: gl.constexpr,
):
    """For each row: s = all-reduce(x) + residual, residual_out = s and
    out = s * rsqrt(mean(s * s) + eps) * w, with w = 1 + weight if GEMMA. With
    QUANT, q = out as fp8 with one fp32 scale for each 128 values. x, residual,
    weight, out and residual_out are words of two bf16 values."""
    n_words = tokens * 2 * LINES
    pid = gl.program_id(0)
    if pid < tokens:
        _send(x, inboxes, calls, pid, n_words, RANK, LINES, SLOT)
    else:
        row = pid - tokens
        w0, w1, w2, w3, word = _receive(
            x, inboxes, calls, err, row, n_words, RANK, LINES, SLOT, MAXIT
        )
        # The all-reduce result is bf16, as an all-reduce returns it. The sum with
        # the residual stays fp32 for the norm, as in aiter's add_rmsnorm.
        ar_lo = (_lo(w0) + _lo(w1) + _lo(w2) + _lo(w3)).to(gl.bfloat16)
        ar_hi = (_hi(w0) + _hi(w1) + _hi(w2) + _hi(w3)).to(gl.bfloat16)
        res = gl.load(residual + word)
        s_lo = ar_lo.to(gl.float32) + _lo(res)
        s_hi = ar_hi.to(gl.float32) + _hi(res)
        gl.store(residual_out + word, _pack(s_lo, s_hi))

        square_sum = gl.sum(gl.sum(s_lo * s_lo + s_hi * s_hi, axis=1), axis=0)
        rstd = gl.rsqrt(square_sum / (4 * LINES) + eps)
        w = gl.load(weight + word - row * 2 * LINES)
        w_lo = _lo(w)
        w_hi = _hi(w)
        if GEMMA:
            w_lo += 1.0
            w_hi += 1.0
        y_lo = s_lo * rstd * w_lo
        y_hi = s_hi * rstd * w_hi
        gl.store(out + word, _pack(y_lo, y_hi))

        if QUANT:
            # A group of 128 values is 32 lines of 2 words. 448 is the fp8 maximum.
            WORDS: gl.constexpr = gl.BlockedLayout(
                [1, 2], [32, 1], [gl.num_warps(), 1], [1, 0]
            )
            a = gl.reshape(gl.maximum(gl.abs(y_lo), gl.abs(y_hi)), [LINES // 32, 32, 2])
            group_scale = gl.maximum(gl.max(gl.max(a, axis=2), axis=1), 1e-10) / 448
            s = gl.broadcast(gl.expand_dims(gl.expand_dims(group_scale, 1), 2), a)[0]
            s = gl.convert_layout(gl.reshape(s, [LINES, 2]), WORDS, assert_trivial=True)
            gl.store(q + 2 * word, (y_lo / s).to(gl.float8e4nv))
            gl.store(q + 2 * word + 1, (y_hi / s).to(gl.float8e4nv))
            i = word - row * 2 * LINES  # a group is 64 words
            gl.store(scale + row * (LINES // 32) + i // 64, s, mask=i % 64 == 0)
