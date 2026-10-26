# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.distributed as dist

from aiter.ops.triton._gluon_kernels.gfx1250.comms.fused.all_reduce_rmsnorm_quant import (
    _all_reduce_rmsnorm_kernel,
)
from aiter.ops.triton.comms.all_reduce import _LINES, _MAXIT
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def all_reduce_rmsnorm_quant(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    gemma: bool,
    quant: bool,
    group,
    inboxes,
    calls,
):
    """
    residual_out = all-reduce(x) + residual and out = RMSNorm(residual_out), with
    one kernel launch on gfx1250. It replaces an all-reduce and the add + RMSNorm
    kernel after it. For decode sizes: RCCL + add_rmsnorm is faster above 1024
    tokens. All ranks get the same bits.

    Args:
        x, residual (torch.Tensor): bf16 (tokens, hidden). hidden is a power of 2
            and at least 256. residual is the same on all ranks.
        weight (torch.Tensor): bf16 (hidden,). Gemma RMSNorm multiplies by
            1 + weight.
        quant (bool): Also return out as fp8, with one fp32 scale for each 128
            values.
        inboxes, calls: The result of
            ``all_reduce_init(group, max_tokens * hidden * 2)``, for this op only.

    Returns:
        tuple: (out, residual_out), and with quant also (q, scale).
    """
    tokens, hidden = x.shape
    assert x.dtype == residual.dtype == weight.dtype == torch.bfloat16
    assert residual.shape == x.shape and weight.shape == (hidden,)
    assert hidden >= 256 and hidden & (hidden - 1) == 0
    slot = 4 * _LINES * ((calls.numel() - 1) // 2)  # as in all_reduce_init
    assert tokens * hidden <= slot, "x is larger than max_bytes"

    _LOGGER.info("ALL_REDUCE_RMSNORM_QUANT [gluon]: x=%s quant=%s", x.shape, quant)

    out = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    q, scale = out, out  # not used without quant
    if quant:
        q = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
        scale = torch.empty(
            (tokens, hidden // 128), dtype=torch.float32, device=x.device
        )
    lines = hidden // 4  # a row is hidden / 4 lines of 16 bytes
    _all_reduce_rmsnorm_kernel[(2 * tokens,)](
        x.view(torch.int32),
        residual.view(torch.int32),
        weight.view(torch.int32),
        out.view(torch.int32),
        residual_out.view(torch.int32),
        q,
        scale,
        inboxes,
        calls,
        calls[-1:],
        tokens,
        eps,
        RANK=dist.get_rank(group),
        LINES=lines,
        SLOT=slot,
        MAXIT=_MAXIT,
        GEMMA=gemma,
        QUANT=quant,
        num_warps=min(16, lines // 32),  # one line for each lane
    )
    if quant:
        return out, residual_out, q, scale
    return out, residual_out
