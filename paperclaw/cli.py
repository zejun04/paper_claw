from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="抓取并整理机器人方向的 arXiv 论文")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch = subparsers.add_parser("fetch", help="抓取候选论文并生成 Markdown")
    fetch.add_argument("--days", type=int, help="覆盖配置中的回补天数")
    fetch.add_argument(
        "--dry-run",
        action="store_true",
        help="只查询并展示候选，不调用模型或写文件",
    )
    fetch.add_argument("--root", type=Path, default=Path.cwd(), help="仓库根目录")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        result = run(args.root.resolve(), days=args.days, dry_run=args.dry_run)
    except Exception as exc:
        print(f"paper-claw 运行失败: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))
