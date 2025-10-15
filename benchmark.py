# ruff: noqa: T201

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import torch

from tests.all_to_all_utils import MoEConfig, RankTestData
from tests.distributed_utils import (
    ProcessGroupInfo,
    parallel_launch,
    parallel_launch_from_env,
)

logger = logging.getLogger(__name__)


@torch.inference_mode()
def bench_all_to_all(
    pgi: ProcessGroupInfo,
    dp_size: int,
    moe: MoEConfig,
) -> tuple[tuple[int, ...], torch.Tensor]:
    device = pgi.device
    num_dp = pgi.world_size // dp_size
    dp_rank = pgi.rank // dp_size

    # Generate the same rank data for each DP group
    rng = torch.Generator()
    rng.manual_seed(dp_rank + 1)
    rank_data = RankTestData(moe, rng, use_max_tokens=True)

    # Allocate symmetric memory
    num_local_experts = moe.num_experts // pgi.world_size

    hidden_dim_bytes_with_scale = moe.hidden_dim // dp_size * moe.in_dtype.itemsize
    if moe.in_dtype.itemsize == 1:
        hidden_dim_bytes_with_scale += (
            (moe.hidden_dim // dp_size + moe.block_size - 1)
            // moe.block_size
            * torch.float32.itemsize
        )

    a2a_shape = (
        pgi.world_size,
        num_local_experts,
        moe.max_num_tokens * hidden_dim_bytes_with_scale,
    )
    a2a_tensor = torch.empty(
        a2a_shape,
        dtype=torch.uint8,
        device=device,
    )
    a2a_out_tensor = torch.empty_like(a2a_tensor)

    # Compute stats
    dispatch_bytes = (
        rank_data.num_tokens * moe.experts_per_token * hidden_dim_bytes_with_scale
    )
    combine_bytes = (
        rank_data.num_tokens
        * moe.experts_per_token
        * (moe.hidden_dim // dp_size)
        * moe.out_dtype.itemsize
    )
    a2a_bytes = a2a_tensor.numel() * a2a_tensor.element_size()

    # Benchmark launcher
    def run() -> tuple[float, ...]:
        num_samples = 10
        events = [
            [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            for _ in range(num_samples)
        ]

        torch_stream_ = torch.cuda.current_stream()

        for e0, e1 in events:
            torch.distributed.barrier()

            e0.record(torch_stream_)

            torch.distributed.all_to_all_single(a2a_out_tensor, a2a_tensor)

            e1.record(torch_stream_)

        # Get latency
        torch_stream_.synchronize()
        sum_a2a_us = 0.0
        for e0, e1 in events:
            sum_a2a_us += e0.elapsed_time(e1) * 1e3
        a2a_us = sum_a2a_us / num_samples
        a2a_gbps = a2a_bytes / a2a_us / 1e3
        return (
            a2a_us,
            a2a_gbps,
        )

    # Warmup
    num_warmup = 10
    with torch.cuda.nvtx.range("warmup"):
        for _ in range(num_warmup):
            run()

    # Benchmark
    num_repeat = 20
    torch.distributed.barrier()
    with torch.cuda.nvtx.range("bench"):
        result = torch.tensor([run() for _ in range(num_repeat)])

    if pgi.rank == 0:
        print(f"Completed!")

    return (
        (dispatch_bytes, combine_bytes, a2a_bytes),
        result,
    )


def _worker_bench_all_to_all(
    pgi: ProcessGroupInfo,
    dp_size: int,
    in_dtype_str: str,
    out_dtype_str: str,
) -> None:
    num_ranks = pgi.world_size
    global_rank = pgi.rank
    local_rank = pgi.local_rank

    torch.cuda.set_device(local_rank)

    in_dtype = getattr(torch, in_dtype_str)
    out_dtype = getattr(torch, out_dtype_str)
    assert isinstance(in_dtype, torch.dtype)
    configs = [
        # V2-Lite:  64 Experts, 6 Experts per Token, 2048 Hidden Dim
        MoEConfig(64, 6, 2048, 1, in_dtype, out_dtype),
        MoEConfig(64, 6, 2048, 4, in_dtype, out_dtype),
        MoEConfig(64, 6, 2048, 8, in_dtype, out_dtype),
        MoEConfig(64, 6, 2048, 16, in_dtype, out_dtype),
        MoEConfig(64, 6, 2048, 32, in_dtype, out_dtype),
        MoEConfig(64, 6, 2048, 64, in_dtype, out_dtype),
        MoEConfig(64, 6, 2048, 128, in_dtype, out_dtype),
        # R1     : 256 Experts, 8 Experts per Token, 7168 Hidden Dim
        MoEConfig(256, 8, 7168, 1, in_dtype, out_dtype),
        MoEConfig(256, 8, 7168, 4, in_dtype, out_dtype),
        MoEConfig(256, 8, 7168, 8, in_dtype, out_dtype),
        MoEConfig(256, 8, 7168, 16, in_dtype, out_dtype),
        MoEConfig(256, 8, 7168, 32, in_dtype, out_dtype),
        MoEConfig(256, 8, 7168, 64, in_dtype, out_dtype),
        MoEConfig(256, 8, 7168, 128, in_dtype, out_dtype),
    ]

    header = [
        "E",
        "E/tok",
        "tok",
        "dim",
        "Torch_lat",
        "Torch_bw",
        "Torch_bytes",
    ]

    outpath = (
        Path(__file__).resolve().parents[0]
        / "data"
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_all_to_all.tsv"
    )
    f_out = None
    if pgi.rank == 0:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        f_out = outpath.open("w")

        line = f"EP={pgi.world_size} DP={pgi.world_size // dp_size}"
        print(line)
        f_out.write(line + "\n")

        line = "\t".join(header)
        print(line)
        f_out.write(line + "\n")

    for i, config in enumerate(configs):
        if pgi.world_size > config.num_experts:
            continue

        meta, result = bench_all_to_all(pgi, dp_size, config)
        dispatch_bytes, combine_bytes, a2a_bytes = meta
        if pgi.rank == 0:
            row: dict[str, str] = {
                "E": f"{config.num_experts}",
                "E/tok": f"{config.experts_per_token}",
                "tok": f"{config.max_num_tokens}",
                "dim": f"{config.hidden_dim}",
                "Torch_lat": f"{result[:, 0].mean():4.1f}μs ± {result[:, 0].std():4.1f}μs",
                "Torch_bw": f"{result[:, 1].mean():2.3f}GB/s",
                "Torch_bytes": f"{a2a_bytes}",
            }
            assert list(row.keys()) == header
            line = "\t".join(row[h] for h in header)
            print(line)
            assert f_out is not None
            f_out.write(line + "\n")
            f_out.flush()

    if f_out is not None:
        f_out.close()
        print("Saved to", outpath)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument(
        "--in-dtype",
        choices=["bfloat16", "float16", "float8_e4m3fn"],
        default="float8_e4m3fn",
    )
    parser.add_argument(
        "--out-dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
    )
    args = parser.parse_args()
    dp_size = int(args.dp_size)
    in_dtype = str(args.in_dtype)
    out_dtype = str(args.out_dtype)

    print("Starting benchmark...")
    print(f"Arguments: dp_size={dp_size}, in_dtype={in_dtype}, out_dtype={out_dtype}")

    if "MASTER_ADDR" in os.environ:
        parallel_launch_from_env(_worker_bench_all_to_all, dp_size, in_dtype, out_dtype)
    else:
        world_size = torch.cuda.device_count()
        parallel_launch(
            world_size, _worker_bench_all_to_all, dp_size, in_dtype, out_dtype
        )

    print("Benchmark completed!")


if __name__ == "__main__":
    main()