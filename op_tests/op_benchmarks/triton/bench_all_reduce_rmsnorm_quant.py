# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon all-reduce + residual add + RMSNorm (+ fp8 group quant) against RCCL +
add_rmsnorm, CAR + add_rmsnorm and the aiter fused CAR path on 4 gfx1250 GPUs.
The time is the slowest rank, in a HIP graph."""

from multiprocessing import Pool, set_start_method

import torch
import torch.distributed as dist
import triton

from aiter import rmsnorm2d_fwd_with_add
from aiter.dist.communication_op import (
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_fused_allreduce_rmsnorm,
)
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    graph_capture,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.ops.triton.comms.all_reduce import all_reduce_free, all_reduce_init
from aiter.ops.triton.comms.fused.all_reduce_rmsnorm_quant import (
    all_reduce_rmsnorm_quant,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_parser,
    override_parser_action,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    do_bench_aiter_triton_synced,
    generate_rotating_buffers_pool,
    get_caller_name_no_ext,
)

WORLD_SIZE = 4
EPS = 1e-6
TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
METHODS = {
    "gluon": "Gluon fused",
    "gluon_fp8": "Gluon fused + fp8",
    "rccl": "RCCL + add_rmsnorm",
    "car": "CAR + add_rmsnorm",
    "car_fused": "aiter fused (CAR)",
}


def _method_fn(method, workspace, weight, out, res_out, use_car):
    """One call of ``method`` on inputs ``x`` and ``residual``."""
    if method.startswith("gluon"):
        quant = method == "gluon_fp8"
        return lambda x, residual: all_reduce_rmsnorm_quant(
            x, residual, weight, EPS, True, quant, *workspace
        )

    def call(x, residual):
        use_car(method != "rccl")
        if method == "car":
            ar = tensor_model_parallel_all_reduce(x)
            rmsnorm2d_fwd_with_add(out, ar, residual, res_out, weight, EPS, True, 0)
        else:
            # with CAR off, the fused API runs RCCL and then add_rmsnorm
            tensor_model_parallel_fused_allreduce_rmsnorm(
                x, residual, weight, EPS, gemma_norm=True
            )

    return call


def _worker(rank, init_method, args):
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=WORLD_SIZE,
        rank=rank,
        distributed_init_method=init_method,
        local_rank=rank,
    )
    ensure_model_parallel_initialized(WORLD_SIZE, 1)
    tp = get_tp_group()
    ca = tp.device_communicator.ca_comm
    car_max = ca._car_max_size

    def use_car(on):
        # AITER_CUSTOM_AR_MAX_SIZE=0 (RCCL for every size) is _car_max_size == 0
        ca._car_max_size = car_max if on else 0

    hidden = args.hidden
    max_bytes = max(args.tokens) * hidden * 2
    workspace = (tp.cpu_group, *all_reduce_init(tp.cpu_group, max_bytes))
    benchmark = triton.testing.Benchmark(
        x_names=["tokens"],
        x_vals=args.tokens,
        line_arg="method",
        line_vals=args.methods,
        line_names=[METHODS[m] for m in args.methods],
        ylabel="Time_(us)",
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench_all_reduce_rmsnorm_quant(tokens, method):
        def gen():
            x = torch.randn((tokens, hidden), dtype=torch.bfloat16, device=device)
            res = torch.randn((tokens, hidden), dtype=torch.bfloat16, device=device)
            return x, res

        weight = torch.randn((hidden,), dtype=torch.bfloat16, device=device)
        out = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device)
        res_out = torch.empty_like(out)
        rot_bufs = generate_rotating_buffers_pool(
            [gen],
            [("x", "residual")],
            args.rot_bufs_target_mb,
            args.rot_bufs_min_pool_len,
        )
        fn = _method_fn(method, workspace, weight, out, res_out, use_car)
        # graph_capture(): the CAR paths capture as they do in a model
        with graph_capture():
            ms = do_bench_aiter_triton_synced(
                fn,
                rot_bufs,
                args.graph_warmup_iters,
                args.graph_kernels_per_graph,
                args.graph_num_replays,
                args.graph_warmup_replays,
                tp.cpu_group,
            )
        us = torch.tensor([ms * 1000.0], dtype=torch.float64)
        dist.all_reduce(us, op=dist.ReduceOp.MAX, group=tp.cpu_group)
        return us.item()

    try:
        bench_all_reduce_rmsnorm_quant.run(
            save_path="." if args.o and rank == 0 else None, print_data=rank == 0
        )
    finally:
        use_car(True)
        torch.cuda.synchronize(device)
        all_reduce_free(tp.cpu_group, workspace[1])
        destroy_model_parallel()
        destroy_distributed_environment()


def parse_args(args: list[str] | None = None):
    parser = get_parser(kernel_name="AllReduceRMSNormQuant")
    override_parser_action(parser, "--metric", choices=["time"], default="time")
    override_parser_action(parser, "--graph-kernels-per-graph", default=100)
    override_parser_action(parser, "--rot-bufs-target-mb", default=4)
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=TOKENS,
        help="Token counts (rows of the hidden states).",
    )
    parser.add_argument("--hidden", type=int, default=2048, help="Hidden size.")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHODS),
        default=list(METHODS),
        help="Paths to time.",
    )
    parser.add_argument(
        "-o", action="store_true", help="Write performance results to CSV file"
    )
    return parser.parse_args(args=args)


def main(args: list[str] | None = None) -> None:
    parsed_args = parse_args(args=args)
    assert torch.cuda.device_count() >= WORLD_SIZE, "needs 4 GPUs"
    set_start_method("spawn", force=True)
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(processes=WORLD_SIZE) as pool:
        rets = [
            pool.apply_async(_worker, args=(rank, init_method, parsed_args))
            for rank in range(WORLD_SIZE)
        ]
        for r in rets:
            r.get()


if __name__ == "__main__":
    main()
