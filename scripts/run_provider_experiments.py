#!/usr/bin/env python3
"""执行 provider 对比实验并写出标准化 JSON 结果。"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hepilot.metrics import run_provider_experiments  # noqa: E402


def build_arg_parser():
    """声明 provider 实验的输入 benchmark、输出目录与采样参数。"""
    parser = argparse.ArgumentParser(description="Run Claude/GPT provider experiments for Hepilot benchmark tasks.")
    parser.add_argument("--benchmark-path", default="benchmarks/coding_tasks.json", help="Path to benchmark task JSON.")
    parser.add_argument("--workspace-root", default="artifacts/provider-workspaces", help="Workspace root for provider experiment copies.")
    parser.add_argument("--artifact-root", default="artifacts/provider-artifacts", help="Directory to store provider benchmark artifacts.")
    parser.add_argument("--output-json", required=True, help="Path to output provider experiment JSON.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max output tokens per provider run.")
    return parser


def main(argv=None):
    """运行 provider 实验并落盘结果。"""
    args = build_arg_parser().parse_args(argv)
    payload = run_provider_experiments(
        benchmark_path=args.benchmark_path,
        workspace_root=args.workspace_root,
        artifact_root=args.artifact_root,
        max_new_tokens=args.max_new_tokens,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
