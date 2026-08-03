#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""命令行入口：导入数据、回测、输出快乐8预测报告。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许 python kl8/run.py 直接运行
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kl8.predict import run_pipeline  # noqa: E402
from kl8 import db  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="快乐8统计分析与预测")
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="开奖 CSV 路径（默认 data/kl8/快乐8_近100期开奖数据.csv）",
    )
    p.add_argument(
        "--import-only",
        action="store_true",
        help="仅导入数据库，不预测",
    )
    p.add_argument(
        "--rebuild",
        action="store_true",
        help="删除旧版数据库后从 CSV 重建，再预测",
    )
    args = p.parse_args(argv)

    if args.import_only:
        if args.rebuild:
            n = db.rebuild_from_csv(args.csv or db.CSV_PATH)
            print(f"已删除旧库并重建，导入 {n} 期到 {db.DB_PATH}")
        else:
            n = db.bootstrap_from_csv(args.csv or db.CSV_PATH)
            print(f"已导入 {n} 期到 {db.DB_PATH}")
        return 0

    result = run_pipeline(args.csv, rebuild=args.rebuild)
    print(result["report"])
    print(f"\n[OK] 报告已写入: {result['report_path']}")
    print(f"[OK] 数据库: {db.DB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
