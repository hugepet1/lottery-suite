#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快乐8 SQLite 数据库：导入 CSV、持久化开奖、预测与回测记录。"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

POOL = 80
DRAW_COUNT = 20

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "kl8"
DB_PATH = DATA_DIR / "kl8.db"
CSV_PATH = DATA_DIR / "快乐8_近100期开奖数据.csv"


def ensure_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    ensure_dir()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: Optional[sqlite3.Connection] = None) -> sqlite3.Connection:
    own = conn is None
    if own:
        conn = connect()
    assert conn is not None
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS draws (
            period INTEGER PRIMARY KEY,
            draw_date TEXT NOT NULL,
            n01 INTEGER NOT NULL, n02 INTEGER NOT NULL, n03 INTEGER NOT NULL,
            n04 INTEGER NOT NULL, n05 INTEGER NOT NULL, n06 INTEGER NOT NULL,
            n07 INTEGER NOT NULL, n08 INTEGER NOT NULL, n09 INTEGER NOT NULL,
            n10 INTEGER NOT NULL, n11 INTEGER NOT NULL, n12 INTEGER NOT NULL,
            n13 INTEGER NOT NULL, n14 INTEGER NOT NULL, n15 INTEGER NOT NULL,
            n16 INTEGER NOT NULL, n17 INTEGER NOT NULL, n18 INTEGER NOT NULL,
            n19 INTEGER NOT NULL, n20 INTEGER NOT NULL,
            numbers_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            target_period INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            scheme TEXT NOT NULL,
            numbers_json TEXT NOT NULL,
            weights_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            period INTEGER NOT NULL,
            draw_numbers_json TEXT NOT NULL,
            pred_numbers_json TEXT NOT NULL,
            hits_json TEXT NOT NULL,
            hit_count INTEGER NOT NULL,
            fail_reasons_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS model_weights (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            weights_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS backtest_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            as_of_period INTEGER NOT NULL,
            window_size INTEGER NOT NULL,
            avg_hit REAL NOT NULL,
            max_hit INTEGER NOT NULL,
            min_hit INTEGER NOT NULL,
            hit10 INTEGER NOT NULL,
            hit9 INTEGER NOT NULL,
            hit8 INTEGER NOT NULL,
            detail_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.commit()
    if own:
        return conn
    return conn


def _parse_nums(row: Sequence[str]) -> Optional[List[int]]:
    nums: List[int] = []
    # columns: period, date, n1..n20, optional joined
    for i in range(2, 22):
        if i >= len(row):
            return None
        cell = (row[i] or "").strip()
        if not cell:
            return None
        try:
            n = int(cell)
        except ValueError:
            return None
        if n < 1 or n > POOL:
            return None
        nums.append(n)
    if len(set(nums)) != DRAW_COUNT:
        return None
    return sorted(nums)


def import_csv(
    csv_path: Path = CSV_PATH,
    db_path: Path = DB_PATH,
    *,
    max_period: Optional[int] = None,
) -> int:
    """导入 CSV，返回写入/更新条数。

    max_period: 若指定，仅保留 period <= max_period 的开奖（用于截断后期数据）。
    """
    conn = init_db(connect(db_path))
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 不存在: {csv_path}")

    # 尝试多编码
    text = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            text = csv_path.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError(f"无法解码 CSV: {csv_path}")

    reader = csv.reader(text.splitlines())
    count = 0
    for row in reader:
        if not row or not row[0].strip().isdigit():
            continue
        period = int(row[0].strip())
        if max_period is not None and period > max_period:
            continue
        date = (row[1] if len(row) > 1 else "").strip()
        nums = _parse_nums(row)
        if nums is None:
            continue
        cols = [f"n{i:02d}" for i in range(1, 21)]
        placeholders = ", ".join(["?"] * (2 + 20 + 1))
        col_sql = "period, draw_date, " + ", ".join(cols) + ", numbers_json"
        conn.execute(
            f"INSERT OR REPLACE INTO draws ({col_sql}) VALUES ({placeholders})",
            [period, date, *nums, json.dumps(nums)],
        )
        count += 1
    conn.commit()
    conn.close()
    return count


def load_draws(db_path: Path = DB_PATH) -> List[Dict]:
    conn = init_db(connect(db_path))
    rows = conn.execute(
        "SELECT period, draw_date, numbers_json FROM draws ORDER BY period ASC"
    ).fetchall()
    conn.close()
    out: List[Dict] = []
    for r in rows:
        out.append(
            {
                "period": int(r["period"]),
                "date": r["draw_date"],
                "numbers": tuple(json.loads(r["numbers_json"])),
            }
        )
    return out


def upsert_draw(
    period: int,
    numbers: Sequence[int],
    draw_date: str = "",
    db_path: Path = DB_PATH,
) -> None:
    """写入/更新单期开奖。"""
    nums = sorted(int(x) for x in numbers)
    if len(nums) != DRAW_COUNT or len(set(nums)) != DRAW_COUNT:
        raise ValueError(f"开奖号码必须为 {DRAW_COUNT} 个不重复号码")
    if any(n < 1 or n > POOL for n in nums):
        raise ValueError("号码超出 01-80")
    conn = init_db(connect(db_path))
    cols = [f"n{i:02d}" for i in range(1, 21)]
    placeholders = ", ".join(["?"] * (2 + 20 + 1))
    col_sql = "period, draw_date, " + ", ".join(cols) + ", numbers_json"
    conn.execute(
        f"INSERT OR REPLACE INTO draws ({col_sql}) VALUES ({placeholders})",
        [int(period), draw_date or "", *nums, json.dumps(nums)],
    )
    conn.commit()
    conn.close()


def save_prediction(
    target_period: int,
    scheme: str,
    numbers: Sequence[int],
    weights: Dict[str, float],
    created_at: str,
    db_path: Path = DB_PATH,
) -> None:
    conn = init_db(connect(db_path))
    conn.execute(
        """
        INSERT INTO predictions (target_period, created_at, scheme, numbers_json, weights_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            target_period,
            created_at,
            scheme,
            json.dumps([int(x) for x in numbers]),
            json.dumps(weights),
        ],
    )
    conn.commit()
    conn.close()


def save_review(
    period: int,
    draw_numbers: Sequence[int],
    pred_numbers: Sequence[int],
    hits: Sequence[int],
    fail_reasons: Dict,
    created_at: str,
    db_path: Path = DB_PATH,
) -> None:
    conn = init_db(connect(db_path))
    conn.execute(
        """
        INSERT INTO reviews (
            period, draw_numbers_json, pred_numbers_json, hits_json,
            hit_count, fail_reasons_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            period,
            json.dumps([int(x) for x in draw_numbers]),
            json.dumps([int(x) for x in pred_numbers]),
            json.dumps([int(x) for x in hits]),
            len(hits),
            json.dumps(fail_reasons, ensure_ascii=False),
            created_at,
        ],
    )
    conn.commit()
    conn.close()


def load_latest_review(period: int, db_path: Path = DB_PATH) -> Optional[Dict]:
    """读取某期最近一次复盘记录，供报告在「等待下期开奖」时重放。"""
    conn = init_db(connect(db_path))
    row = conn.execute(
        """
        SELECT draw_numbers_json, pred_numbers_json, hits_json, hit_count,
               fail_reasons_json, created_at
        FROM reviews
        WHERE period = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        [period],
    ).fetchone()
    conn.close()
    if not row:
        return None
    fail = json.loads(row["fail_reasons_json"] or "{}")
    pred = json.loads(row["pred_numbers_json"])
    hits = json.loads(row["hits_json"])
    schemes = fail.get("schemes") or {}
    scheme_details = {}
    for key, meta in schemes.items():
        hr = meta.get("hit_rate") or ""
        scheme_hits = meta.get("hits") or []
        scheme_details[key] = {
            "hits": scheme_hits,
            "hit_count": len(scheme_hits),
            "hit_rate": hr,
        }
    return {
        "draw_numbers": json.loads(row["draw_numbers_json"]),
        "pred_numbers": pred,
        "hits": hits,
        "hit_count": int(row["hit_count"]),
        "hit_rate": f"{len(hits)}/{len(pred) if pred else 10}",
        "fail_reasons": fail.get("reasons") or {},
        "primary_reasons": fail.get("primary") or [],
        "scheme_key": fail.get("scheme_key"),
        "scheme_details": scheme_details,
        "jin_dan": fail.get("jin_dan"),
        "complementary": fail.get("complementary"),
        "note": "（复盘已落库，本次为等待下期开奖时的报告重放）",
        "created_at": row["created_at"],
    }


def save_weights(weights: Dict[str, float], updated_at: str, db_path: Path = DB_PATH) -> None:
    conn = init_db(connect(db_path))
    conn.execute(
        """
        INSERT INTO model_weights (id, weights_json, updated_at)
        VALUES (1, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            weights_json = excluded.weights_json,
            updated_at = excluded.updated_at
        """,
        [json.dumps(weights), updated_at],
    )
    conn.commit()
    conn.close()


def load_weights(db_path: Path = DB_PATH) -> Optional[Dict[str, float]]:
    conn = init_db(connect(db_path))
    row = conn.execute("SELECT weights_json FROM model_weights WHERE id = 1").fetchone()
    conn.close()
    if not row:
        return None
    return {k: float(v) for k, v in json.loads(row["weights_json"]).items()}


def save_backtest(result: Dict, created_at: str, db_path: Path = DB_PATH) -> None:
    conn = init_db(connect(db_path))
    conn.execute(
        """
        INSERT INTO backtest_runs (
            as_of_period, window_size, avg_hit, max_hit, min_hit,
            hit10, hit9, hit8, detail_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            int(result["as_of_period"]),
            int(result["window_size"]),
            float(result["avg_hit"]),
            int(result["max_hit"]),
            int(result["min_hit"]),
            int(result["hit10"]),
            int(result["hit9"]),
            int(result["hit8"]),
            json.dumps(result.get("detail", {}), ensure_ascii=False),
            created_at,
        ],
    )
    conn.commit()
    conn.close()


def latest_prediction_for_period(period: int, db_path: Path = DB_PATH) -> Optional[Dict]:
    conn = init_db(connect(db_path))
    row = conn.execute(
        """
        SELECT * FROM predictions
        WHERE target_period = ?
        ORDER BY id DESC LIMIT 1
        """,
        [period],
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {
        "target_period": int(row["target_period"]),
        "scheme": row["scheme"],
        "numbers": json.loads(row["numbers_json"]),
        "weights": json.loads(row["weights_json"]),
        "created_at": row["created_at"],
    }


def delete_database(db_path: Path = DB_PATH) -> bool:
    """删除旧版 SQLite 数据库文件（若存在）。"""
    existed = db_path.exists()
    if existed:
        db_path.unlink()
    # 清理可能的旁路文件
    for suffix in ("-wal", "-shm", "-journal"):
        p = Path(str(db_path) + suffix)
        if p.exists():
            p.unlink()
    return existed


def _export_runtime_state(
    db_path: Path = DB_PATH,
    *,
    max_period: Optional[int] = None,
) -> Dict:
    """导出权重与复盘，供删库重建后恢复。

    max_period: 若指定，仅导出 period <= max_period 的复盘记录。
    """
    state: Dict = {"weights": None, "reviews": [], "backtests": []}
    if not db_path.exists():
        return state
    conn = connect(db_path)
    try:
        row = conn.execute(
            "SELECT weights_json, updated_at FROM model_weights WHERE id = 1"
        ).fetchone()
        if row:
            state["weights"] = {
                "weights": json.loads(row["weights_json"]),
                "updated_at": row["updated_at"],
            }
        # 表可能不存在于损坏库
        try:
            if max_period is not None:
                reviews = conn.execute(
                    """
                    SELECT period, draw_numbers_json, pred_numbers_json, hits_json,
                           hit_count, fail_reasons_json, created_at
                    FROM reviews WHERE period <= ? ORDER BY id ASC
                    """,
                    [max_period],
                ).fetchall()
            else:
                reviews = conn.execute(
                    """
                    SELECT period, draw_numbers_json, pred_numbers_json, hits_json,
                           hit_count, fail_reasons_json, created_at
                    FROM reviews ORDER BY id ASC
                    """
                ).fetchall()
            state["reviews"] = [dict(r) for r in reviews]
        except sqlite3.Error:
            pass
    finally:
        conn.close()
    return state


def _restore_runtime_state(state: Dict, db_path: Path = DB_PATH) -> None:
    if not state:
        return
    conn = init_db(connect(db_path))
    try:
        w = state.get("weights")
        if w:
            conn.execute(
                """
                INSERT INTO model_weights (id, weights_json, updated_at)
                VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    weights_json = excluded.weights_json,
                    updated_at = excluded.updated_at
                """,
                [json.dumps(w["weights"]), w["updated_at"]],
            )
        for r in state.get("reviews") or []:
            conn.execute(
                """
                INSERT INTO reviews (
                    period, draw_numbers_json, pred_numbers_json, hits_json,
                    hit_count, fail_reasons_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    r["period"],
                    r["draw_numbers_json"],
                    r["pred_numbers_json"],
                    r["hits_json"],
                    r["hit_count"],
                    r["fail_reasons_json"],
                    r["created_at"],
                ],
            )
        conn.commit()
    finally:
        conn.close()


def purge_draws_after(max_period: int, db_path: Path = DB_PATH) -> int:
    """删除 period > max_period 的开奖及相关预测/复盘记录。"""
    if not db_path.exists():
        return 0
    conn = init_db(connect(db_path))
    try:
        cur = conn.execute("DELETE FROM draws WHERE period > ?", [max_period])
        deleted = cur.rowcount
        conn.execute("DELETE FROM predictions WHERE target_period > ?", [max_period + 1])
        conn.execute("DELETE FROM reviews WHERE period > ?", [max_period])
        conn.commit()
        return int(deleted or 0)
    finally:
        conn.close()


def rebuild_from_csv(
    csv_path: Path = CSV_PATH,
    db_path: Path = DB_PATH,
    *,
    max_period: Optional[int] = None,
) -> int:
    """删除旧库后从 CSV 重建；保留权重与（截断范围内的）复盘记录。"""
    ensure_dir()
    state = _export_runtime_state(db_path, max_period=max_period)
    delete_database(db_path)
    n = import_csv(csv_path, db_path, max_period=max_period)
    _restore_runtime_state(state, db_path)
    return n


def bootstrap_from_csv(
    csv_path: Path = CSV_PATH,
    *,
    max_period: Optional[int] = None,
) -> int:
    ensure_dir()
    return import_csv(csv_path, max_period=max_period)
