#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch evaluate.py with accelerate for multi-GPU / multi-node evaluation.",
    )
    parser.add_argument("--node-id", type=int, required=True, help="Machine rank for this node.")
    parser.add_argument("--num-machines", type=int, required=True, help="Total number of nodes.")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=1,
        help="Processes (usually GPUs) per node.",
    )
    parser.add_argument(
        "--main-process-ip",
        type=str,
        required=True,
        help="IP or hostname of rank-0 node.",
    )
    parser.add_argument(
        "--main-process-port",
        type=int,
        default=29500,
        help="Rendezvous port on rank-0 node.",
    )
    parser.add_argument(
        "eval_args",
        nargs=argparse.REMAINDER,
        help="Hydra overrides forwarded to evaluate.py (e.g. checkpoint_dir=... evaluator=rk4).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parent
    eval_script = repo_root / "evaluate.py"

    eval_args = list(args.eval_args)
    if eval_args and eval_args[0] == "--":
        eval_args = eval_args[1:]

    # Mirror topology into the Hydra config so resolved_config.yaml is self-describing.
    accelerate_overrides = [
        f"accelerate.num_machines={args.num_machines}",
        f"accelerate.num_processes={args.num_processes}",
        f"accelerate.machine_rank={args.node_id}",
        f"accelerate.node_id={args.node_id}",
        f"accelerate.main_process_ip={args.main_process_ip}",
        f"accelerate.main_process_port={args.main_process_port}",
    ]

    cmd = [
        "accelerate",
        "launch",
        "--num_machines",
        str(args.num_machines),
        "--machine_rank",
        str(args.node_id),
        "--num_processes",
        str(args.num_processes),
        "--main_process_ip",
        args.main_process_ip,
        "--main_process_port",
        str(args.main_process_port),
        str(eval_script),
        *accelerate_overrides,
        *eval_args,
    ]

    subprocess.run(cmd, check=True, cwd=repo_root)


if __name__ == "__main__":
    main()
