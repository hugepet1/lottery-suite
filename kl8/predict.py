#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快乐8 走势图数据库 + 规则预测

玩法：01–80 开出 20 个号码（顺序无关）。
本模块：
1. 历史开奖入库（JSON / CSV）
2. 基本走势图遗漏矩阵（红号命中 + 灰色遗漏值）
3. 按「三空打中间 / 四空打两边 / 六空打连子 / 封口斜连 /
   冷热搭配 / 遗漏周期 / 连号必出 / 四区对称 / 跨度定胆」生成 3 组选十
"""

from __future__ import annotations

import csv
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

POOL_MAX = 80
DRAW_COUNT = 20
PICK_COUNT = 10  # 选十
ZONES = ((1, 20), (21, 40), (41, 60), (61, 80))
ZONE_NAMES = ("一区1-20", "二区21-40", "三区41-60", "四区61-80")


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def bundled_data_dir() -> Path | None:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "data" / "kl8"
    return None


ROOT = app_root()
DATA_DIR = ROOT / "data" / "kl8"
DATA_PATH = DATA_DIR / "history.json"
LAST_PRED_PATH = DATA_DIR / "last_prediction.json"
COMPARE_LOG_PATH = DATA_DIR / "compare_log.json"
WEIGHTS_PATH = DATA_DIR / "algo_weights.json"

RULE_KEYS = [
    "gap3",
    "gap4",
    "gap6",
    "seal_diag",
    "hot_cold",
    "omit_cycle",
    "consec",
    "zone_sym",
    "span_sym",
]
ALGO_NAMES = {
    "gap3": "三空打中间",
    "gap4": "四空打两边",
    "gap6": "六空打连子",
    "seal_diag": "封口斜连",
    "hot_cold": "冷热搭配",
    "omit_cycle": "遗漏周期",
    "consec": "连号必出",
    "zone_sym": "四区对称",
    "span_sym": "跨度定胆",
}
DEFAULT_RULE_WEIGHTS = {
    "gap3": 0.14,
    "gap4": 0.12,
    "gap6": 0.12,
    "seal_diag": 0.11,
    "hot_cold": 0.12,
    "omit_cycle": 0.10,
    "consec": 0.11,
    "zone_sym": 0.10,
    "span_sym": 0.08,
}

Draw = Dict[str, object]
Nums = Tuple[int, ...]
Cell = Dict[str, object]
TrendRow = Dict[str, object]


def ensure_data_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists():
        return
    seed = bundled_data_dir()
    if seed and (seed / "history.json").exists():
        shutil.copy2(seed / "history.json", DATA_PATH)
        return
    # 若有随包 CSV，自动导入
    csv_seed = None
    for p in (DATA_DIR / "快乐8_近100期开奖数据.csv",):
        if p.exists():
            csv_seed = p
            break
    if seed:
        for p in seed.glob("*.csv"):
            csv_seed = p
            break
    if csv_seed is not None:
        import_csv(csv_seed)


def _norm_nums(nums: Sequence[int], need: int = DRAW_COUNT) -> List[int]:
    out = sorted({int(x) for x in nums})
    if len(out) != need:
        raise ValueError(f"需要 {need} 个不重复号码，收到 {nums}")
    if any(x < 1 or x > POOL_MAX for x in out):
        raise ValueError(f"号码需在 01-{POOL_MAX:02d}：{out}")
    return out


def load_history(path: Path = DATA_PATH) -> List[Draw]:
    ensure_data_files()
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    draws: List[Draw] = []
    for item in raw:
        nums = item.get("nums") or item.get("numbers") or item.get("balls")
        if nums is None:
            continue
        draws.append(
            {
                "period": int(item["period"]),
                "date": str(item.get("date", "")),
                "nums": tuple(_norm_nums(nums, DRAW_COUNT)),
            }
        )
    draws.sort(key=lambda x: int(x["period"]))
    return draws


def save_history(draws: Sequence[Draw], path: Path = DATA_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(draws, key=lambda x: int(x["period"]))
    payload = [
        {
            "period": int(d["period"]),
            "date": d.get("date", ""),
            "nums": list(d["nums"]),
        }
        for d in ordered
    ]
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def add_draw(period: int, nums: Sequence[int], date: str = "") -> List[Draw]:
    draws = load_history()
    for d in draws:
        if int(d["period"]) == period:
            raise ValueError(f"期号 {period} 已存在")
    draws.append({"period": period, "date": date, "nums": tuple(_norm_nums(nums))})
    save_history(draws)
    return draws


def delete_draw(period: int) -> List[Draw]:
    draws = load_history()
    kept = [d for d in draws if int(d["period"]) != period]
    if len(kept) == len(draws):
        raise ValueError(f"未找到期号 {period}")
    save_history(kept)
    return kept


def delete_latest() -> Tuple[Draw, List[Draw]]:
    draws = load_history()
    if not draws:
        raise ValueError("没有可删除的数据")
    removed = draws[-1]
    kept = draws[:-1]
    save_history(kept)
    return removed, kept


def clear_all_history() -> None:
    save_history([])


def fmt_nums(nums: Sequence[int]) -> str:
    return " ".join(f"{int(x):02d}" for x in nums)


def load_algo_weights() -> Dict[str, float]:
    if WEIGHTS_PATH.exists():
        try:
            with WEIGHTS_PATH.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            rules = raw.get("rules", raw)
            out = dict(DEFAULT_RULE_WEIGHTS)
            for k in RULE_KEYS:
                if k in rules:
                    out[k] = float(rules[k])
            s = sum(out.values()) or 1.0
            return {k: v / s for k, v in out.items()}
        except Exception:
            pass
    return dict(DEFAULT_RULE_WEIGHTS)


def save_algo_weights(weights: Dict[str, float]) -> None:
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with WEIGHTS_PATH.open("w", encoding="utf-8") as f:
        json.dump({"rules": weights}, f, ensure_ascii=False, indent=2)


def reset_algo_weights() -> Dict[str, float]:
    w = dict(DEFAULT_RULE_WEIGHTS)
    save_algo_weights(w)
    return w


# ---------------------------------------------------------------------------
# CSV 导入（兼容桌面「快乐8_近100期开奖数据.csv」）
# ---------------------------------------------------------------------------


def _parse_num_token(s: str) -> List[int]:
    text = str(s).strip()
    for ch in ["，", ",", " ", "-", "|", "/", "\t", ";", "、"]:
        text = text.replace(ch, " ")
    out: List[int] = []
    for p in text.split():
        digits = "".join(c for c in p if c.isdigit())
        if digits:
            out.append(int(digits))
    return out


def import_csv(csv_path: str | Path) -> List[Draw]:
    """
    支持常见列布局：
    - 期号,开奖日期,号码1..号码20
    - 期号,日期,开奖号码（空格/逗号分隔 20 个）
    - 期号 + 20 个号码列（无表头也可）
    """
    path = Path(csv_path)
    existing = {int(d["period"]): d for d in load_history()}

    # 尝试 utf-8-sig / gbk
    raw = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            raw = path.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if raw is None:
        raise RuntimeError(f"无法解码 CSV：{path}")

    # 去掉可能的 BOM 空白行
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return load_history()

    reader = csv.reader(lines)
    rows = list(reader)
    header = [c.strip() for c in rows[0]]
    has_header = any(
        any(k in h for k in ("期", "号", "日", "period", "date", "开奖"))
        for h in header
    )
    start = 1 if has_header else 0

    # 识别列
    period_idx = 0
    date_idx: Optional[int] = None
    num_cols: List[int] = []
    bulk_idx: Optional[int] = None

    if has_header:
        for i, h in enumerate(header):
            hl = h.lower().strip()
            # 先识别日期，避免「开奖日期」被「期」误匹配为期号
            if hl in ("date", "time") or any(
                k == h or h.endswith(k) or h.startswith(k)
                for k in ("开奖日期", "日期", "时间")
            ):
                date_idx = i
                continue
            if hl == "period" or h in ("期号", "期数", "期") or h.startswith("期号"):
                period_idx = i
                continue
            if any(k in h for k in ("号码", "开奖号", "开奖号码", "红球")):
                # 单个聚合列 or 号码1..
                base = "".join(ch for ch in h if not ch.isdigit()).rstrip("_-")
                if base in ("号码", "号", "开奖号码", "开奖号", "红球") and any(
                    ch.isdigit() for ch in h
                ):
                    num_cols.append(i)
                elif base in ("号码", "开奖号码", "开奖号", "红球", "开奖号码列"):
                    bulk_idx = i
                else:
                    bulk_idx = i
                continue
            if h.isdigit() or (len(h) <= 3 and h.replace("0", "").isdigit()):
                num_cols.append(i)
        if not num_cols and bulk_idx is None:
            # 兜底：期号后全部当号码
            num_cols = list(range(period_idx + 1, len(header)))
            if date_idx in num_cols:
                num_cols.remove(date_idx)

    for row in rows[start:]:
        if not row or len(row) <= period_idx:
            continue
        try:
            period_raw = str(row[period_idx]).strip()
            period_digits = "".join(c for c in period_raw if c.isdigit())
            if not period_digits:
                continue
            period = int(period_digits)
            date = ""
            if date_idx is not None and date_idx < len(row):
                date = str(row[date_idx]).strip()[:10]
            # 防御：若「期号」实为 YYYYMMDD 而邻列才是真正期号，则纠正
            if len(period_digits) == 8 and 20200101 <= period <= 20991231:
                for alt_i, cell in enumerate(row):
                    if alt_i == period_idx:
                        continue
                    alt = "".join(c for c in str(cell) if c.isdigit())
                    if len(alt) == 7 and alt.startswith("20"):
                        if not date:
                            date = f"{period_digits[:4]}-{period_digits[4:6]}-{period_digits[6:8]}"
                        period = int(alt)
                        break
                else:
                    # 无法纠正的日期型期号跳过，避免污染库
                    continue

            nums: List[int] = []
            if num_cols:
                for i in num_cols:
                    if i < len(row) and str(row[i]).strip():
                        nums.extend(_parse_num_token(row[i]))
            elif bulk_idx is not None and bulk_idx < len(row):
                nums = _parse_num_token(row[bulk_idx])
            else:
                # 无表头：第 0 列期号，第 1 列可能是日期
                rest = row[1:]
                if rest and ("-" in str(rest[0]) or "/" in str(rest[0])):
                    date = str(rest[0]).strip()[:10]
                    rest = rest[1:]
                for cell in rest:
                    nums.extend(_parse_num_token(cell))

            nums = _norm_nums(nums, DRAW_COUNT)
            existing[period] = {"period": period, "date": date, "nums": tuple(nums)}
        except Exception:
            continue

    draws = sorted(existing.values(), key=lambda x: int(x["period"]))
    save_history(draws)
    return draws


# ---------------------------------------------------------------------------
# 走势图遗漏矩阵
# ---------------------------------------------------------------------------


def compute_omissions(draws: Sequence[Draw]) -> Dict[int, int]:
    """当前遗漏：距上一次开出的期数（刚开出为 0）。"""
    omit = {k: len(draws) for k in range(1, POOL_MAX + 1)}
    n = len(draws)
    for i, d in enumerate(draws):
        for x in map(int, d["nums"]):
            omit[x] = n - 1 - i
    return omit


def build_trend_matrix(
    draws: Sequence[Draw],
    recent: Optional[int] = None,
) -> List[TrendRow]:
    """
    构建基本走势图数据。
    每一行：期号 + 1..80 单元格 {num, hit, omit}
    omit = 截至该期连续未开出的期数（开出当行 omit=0）。
    """
    ordered = sorted(draws, key=lambda x: int(x["period"]))
    if recent is not None and recent > 0:
        ordered = ordered[-recent:]

    last_hit_idx = {k: -1 for k in range(1, POOL_MAX + 1)}
    # 需要全局索引以延续遗漏：若只显示近期，用完整历史校正起点
    all_ordered = sorted(draws, key=lambda x: int(x["period"]))
    start_period = int(ordered[0]["period"]) if ordered else None
    prefix = []
    if start_period is not None:
        prefix = [d for d in all_ordered if int(d["period"]) < start_period]
    for i, d in enumerate(prefix):
        for x in map(int, d["nums"]):
            last_hit_idx[x] = i
    base = len(prefix)

    rows: List[TrendRow] = []
    for j, d in enumerate(ordered):
        idx = base + j
        hit_set = set(map(int, d["nums"]))
        cells: List[Cell] = []
        for num in range(1, POOL_MAX + 1):
            if num in hit_set:
                omit = 0
                last_hit_idx[num] = idx
                hit = True
            else:
                hit = False
                omit = idx - last_hit_idx[num] if last_hit_idx[num] >= 0 else idx + 1
            cells.append({"num": num, "hit": hit, "omit": omit})
        rows.append(
            {
                "period": int(d["period"]),
                "date": d.get("date", ""),
                "nums": tuple(d["nums"]),
                "cells": cells,
            }
        )
    return rows


def find_gaps(drawn: Iterable[int]) -> List[Tuple[int, int]]:
    """在 1-80 上找未开出的连续空位区间 [lo, hi]（含端点）。"""
    s = set(map(int, drawn))
    gaps: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for n in range(1, POOL_MAX + 1):
        if n not in s:
            if start is None:
                start = n
        else:
            if start is not None:
                gaps.append((start, n - 1))
                start = None
    if start is not None:
        gaps.append((start, POOL_MAX))
    return gaps


def consecutive_groups(nums: Iterable[int]) -> List[List[int]]:
    arr = sorted(set(map(int, nums)))
    if not arr:
        return []
    groups = [[arr[0]]]
    for x in arr[1:]:
        if x == groups[-1][-1] + 1:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups


def zone_of(n: int) -> int:
    for i, (a, b) in enumerate(ZONES):
        if a <= n <= b:
            return i
    return -1


def symmetric(n: int) -> int:
    """跨区对称号：n + s = 80 → s = 80 - n；1↔79 等。常用 n+s=80。"""
    return 80 - int(n)


def complementary_tail(n: int) -> int:
    """同区尾数互补启发：尾数互补到 10（如 03→07）。"""
    tail = n % 10
    if tail == 0:
        comp = 5
    else:
        comp = (10 - tail) % 10
        if comp == 0:
            comp = 5
    base = n - tail
    cand = base + comp
    if cand == n:
        cand = base + ((comp + 2) % 10)
    z = zone_of(n)
    lo, hi = ZONES[z]
    if lo <= cand <= hi and cand != n:
        return cand
    # 回退同区邻号
    for d in (1, -1, 2, -2):
        if lo <= n + d <= hi:
            return n + d
    return n


# ---------------------------------------------------------------------------
# 规则候选打分
# ---------------------------------------------------------------------------


def score_gap3(gaps: Sequence[Tuple[int, int]]) -> Dict[int, float]:
    """三空打中间：空位长度=3，取中间号。"""
    sc: Dict[int, float] = defaultdict(float)
    for lo, hi in gaps:
        length = hi - lo + 1
        if length == 3:
            mid = (lo + hi) // 2
            sc[mid] += 3.0
            sc[lo] += 0.6
            sc[hi] += 0.6
        elif length == 5:
            # 近似三空结构，取中心
            sc[(lo + hi) // 2] += 1.2
    return sc


def score_gap4(gaps: Sequence[Tuple[int, int]]) -> Dict[int, float]:
    """四空打两边：空位长度=4，取两端相邻号。"""
    sc: Dict[int, float] = defaultdict(float)
    for lo, hi in gaps:
        length = hi - lo + 1
        if length == 4:
            if lo - 1 >= 1:
                sc[lo - 1] += 2.8
            if hi + 1 <= POOL_MAX:
                sc[hi + 1] += 2.8
            sc[lo] += 1.0
            sc[hi] += 1.0
        elif length == 5:
            if lo - 1 >= 1:
                sc[lo - 1] += 1.2
            if hi + 1 <= POOL_MAX:
                sc[hi + 1] += 1.2
    return sc


def score_gap6(gaps: Sequence[Tuple[int, int]]) -> Dict[int, float]:
    """六空打连子：空位≥6，重点布局内部连号对（如 50-57→51-52 或 55-56）。"""
    sc: Dict[int, float] = defaultdict(float)
    for lo, hi in gaps:
        length = hi - lo + 1
        if length >= 6:
            # 前连子 / 后连子 / 中部连子；大空位再加密四分位点连对
            mid = (lo + hi) // 2
            pairs = [
                (lo + 1, lo + 2),
                (hi - 2, hi - 1),
                (mid, mid + 1),
            ]
            if length >= 8:
                q1 = lo + max(1, length // 4)
                q3 = hi - max(1, length // 4)
                pairs.extend([(q1, q1 + 1), (q3 - 1, q3)])
            boost = 2.2 + 0.15 * min(length - 6, 8)
            for a, b in pairs:
                if lo <= a < b <= hi:
                    sc[a] += boost
                    sc[b] += boost
            # 空位内部轻铺，便于组内形成连号
            for n in range(lo + 1, hi):
                sc[n] += 0.35
    return sc


def score_seal_diag(draws: Sequence[Draw]) -> Dict[int, float]:
    """封口斜连：上期末位邻号 + 斜向等差延伸。"""
    sc: Dict[int, float] = defaultdict(float)
    if not draws:
        return sc
    last = list(map(int, draws[-1]["nums"]))
    # 末位（最大号）及其邻号
    end = max(last)
    for n in (end - 1, end + 1, end - 2, end + 2):
        if 1 <= n <= POOL_MAX and n not in last:
            sc[n] += 2.5 if abs(n - end) == 1 else 1.2
    # 最小号封口
    start = min(last)
    for n in (start - 1, start + 1):
        if 1 <= n <= POOL_MAX and n not in last:
            sc[n] += 1.4

    # 斜连：最近若干期，找公差约 11 的斜向（如 08-19-30 → 41/52）
    recent = draws[-6:]
    positions: Dict[int, List[int]] = defaultdict(list)
    for i, d in enumerate(recent):
        for x in map(int, d["nums"]):
            positions[x].append(i)

    # 扫描可能的等差链（公差 9~13，常见斜线）
    for step in (9, 10, 11, 12, 13):
        for seed in range(1, POOL_MAX - 2 * step + 1):
            chain = [seed, seed + step, seed + 2 * step]
            hits = 0
            for c in chain:
                # 是否在近几期出现过
                if any(c in map(int, d["nums"]) for d in recent):
                    hits += 1
            if hits >= 2:
                nxt = seed + 3 * step
                if 1 <= nxt <= POOL_MAX:
                    sc[nxt] += 1.8 + 0.4 * hits
                nxt2 = seed + 4 * step
                if 1 <= nxt2 <= POOL_MAX:
                    sc[nxt2] += 1.0
    return sc


def score_hot_cold(draws: Sequence[Draw]) -> Dict[int, float]:
    """冷热搭配：短期高频优先追踪，兼顾回补。"""
    sc: Dict[int, float] = defaultdict(float)
    if not draws:
        return sc
    window3 = draws[-3:]
    window10 = draws[-10:]
    c3 = Counter()
    c10 = Counter()
    for d in window3:
        c3.update(map(int, d["nums"]))
    for d in window10:
        c10.update(map(int, d["nums"]))
    omit = compute_omissions(draws)
    for n in range(1, POOL_MAX + 1):
        # 3 期出现 ≥2 次 → 热号追踪
        if c3[n] >= 2:
            sc[n] += 3.0
        elif c3[n] == 1:
            sc[n] += 1.0
        # 10 期热度
        sc[n] += 0.35 * c10[n]
        # 过冷适度回补（但不盲目追超冷）
        o = omit[n]
        if 8 <= o <= 18:
            sc[n] += 1.4
        elif 19 <= o <= 24:
            sc[n] += 0.8
    return sc


def score_omit_cycle(draws: Sequence[Draw]) -> Dict[int, float]:
    """遗漏周期：25–40 期遗漏结合分阶段补仓（中等权重，避免重仓）。"""
    sc: Dict[int, float] = defaultdict(float)
    omit = compute_omissions(draws)
    for n, o in omit.items():
        if 25 <= o <= 40:
            # 越靠近 30–35 越高，但仍克制
            sc[n] += 1.6 + 0.04 * (40 - abs(o - 32))
        elif 18 <= o <= 24:
            sc[n] += 0.7
        elif o > 40:
            sc[n] += 0.35  # 超冷轻仓观察
    return sc


def score_consec(draws: Sequence[Draw]) -> Dict[int, float]:
    """连号必出：关注中段(21-40)连号，长连号追两端/对称。"""
    sc: Dict[int, float] = defaultdict(float)
    if not draws:
        return sc
    last = list(map(int, draws[-1]["nums"]))
    groups = consecutive_groups(last)
    for g in groups:
        if len(g) >= 2:
            # 追两端延伸
            left, right = g[0] - 1, g[-1] + 1
            if 1 <= left <= POOL_MAX:
                sc[left] += 2.0 + 0.3 * (len(g) - 2)
            if 1 <= right <= POOL_MAX:
                sc[right] += 2.0 + 0.3 * (len(g) - 2)
            # 对称号
            for x in g:
                s = symmetric(x)
                if 1 <= s <= POOL_MAX and s not in last:
                    sc[s] += 1.1
        if len(g) >= 3:
            for x in g:
                sc[symmetric(x)] += 0.8

    # 中段优先：在 21-40 内找「半连」潜力（邻号遗漏小）
    omit = compute_omissions(draws)
    for n in range(21, 40):
        if omit[n] <= 3 and omit[n + 1] <= 3:
            sc[n] += 1.3
            sc[n + 1] += 1.3
    return sc


def score_zone_sym(draws: Sequence[Draw]) -> Dict[int, float]:
    """四区对称 / 同区遗漏尾数互补 / 跨区对称组合。"""
    sc: Dict[int, float] = defaultdict(float)
    if not draws:
        return sc
    last = set(map(int, draws[-1]["nums"]))
    omit = compute_omissions(draws)

    for zi, (lo, hi) in enumerate(ZONES):
        zone_hits = sorted(n for n in last if lo <= n <= hi)
        # 同区尾数互补：已出 03、05 → 补 07
        tails = {n % 10 for n in zone_hits}
        for n in zone_hits:
            c = complementary_tail(n)
            if c not in last:
                sc[c] += 1.8
        # 区内遗漏较大的号轻补
        cold = sorted(
            (omit[n], n) for n in range(lo, hi + 1) if n not in last
        )
        for o, n in cold[-4:]:
            if o >= 6:
                sc[n] += 0.9 + min(o, 20) * 0.03

        # 跨区对称：对已出号的对称号加权
        for n in zone_hits:
            s = symmetric(n)
            if 1 <= s <= POOL_MAX and s not in last:
                sc[s] += 2.0

    # 四区出号均衡：偏少的区抬分
    counts = [sum(1 for n in last if lo <= n <= hi) for lo, hi in ZONES]
    avg = sum(counts) / 4
    for zi, (lo, hi) in enumerate(ZONES):
        if counts[zi] < avg - 0.5:
            for n in range(lo, hi + 1):
                if n not in last:
                    sc[n] += 0.55
    return sc


def score_span_sym(draws: Sequence[Draw]) -> Dict[int, float]:
    """跨度定胆 + 对称增效。"""
    sc: Dict[int, float] = defaultdict(float)
    if not draws:
        return sc
    last = list(map(int, draws[-1]["nums"]))
    span = max(last) - min(last)
    # 以跨度为带宽，聚焦中段区间
    mid = (min(last) + max(last)) / 2
    half = max(8, span / 2)
    lo = max(1, int(mid - half))
    hi = min(POOL_MAX, int(mid + half))
    for n in range(lo, hi + 1):
        sc[n] += 1.0
    # 近中心加权
    for n in range(max(1, int(mid - 6)), min(POOL_MAX, int(mid + 6)) + 1):
        sc[n] += 0.8

    # 对称号组合效率
    for n in last:
        s = symmetric(n)
        if 1 <= s <= POOL_MAX and s not in last:
            sc[s] += 1.6
    # 历史平均跨度附近再定胆
    spans = [max(map(int, d["nums"])) - min(map(int, d["nums"])) for d in draws[-20:]]
    if spans:
        avg_span = sum(spans) / len(spans)
        target_lo = max(1, int(40 - avg_span / 2))
        target_hi = min(POOL_MAX, int(40 + avg_span / 2))
        for n in range(target_lo, target_hi + 1):
            sc[n] += 0.45
    return sc


def normalize(scores: Dict[int, float]) -> Dict[int, float]:
    s = sum(max(0.0, v) for v in scores.values())
    if s <= 0:
        return {k: 1.0 / POOL_MAX for k in range(1, POOL_MAX + 1)}
    return {k: max(0.0, v) / s for k, v in scores.items()}


def blend_rule_scores(
    draws: Sequence[Draw],
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[int, float], Dict[str, Dict[int, float]], dict]:
    """融合各规则得分，并返回分析摘要。"""
    w = weights or load_algo_weights()
    last_nums = list(map(int, draws[-1]["nums"])) if draws else []
    gaps = find_gaps(last_nums)
    sources = {
        "gap3": score_gap3(gaps),
        "gap4": score_gap4(gaps),
        "gap6": score_gap6(gaps),
        "seal_diag": score_seal_diag(draws),
        "hot_cold": score_hot_cold(draws),
        "omit_cycle": score_omit_cycle(draws),
        "consec": score_consec(draws),
        "zone_sym": score_zone_sym(draws),
        "span_sym": score_span_sym(draws),
    }
    # 补全键并归一
    normed = {}
    for k, sc in sources.items():
        full = {n: float(sc.get(n, 0.0)) for n in range(1, POOL_MAX + 1)}
        normed[k] = normalize(full)

    blend = {n: 0.0 for n in range(1, POOL_MAX + 1)}
    for k in RULE_KEYS:
        wk = w.get(k, DEFAULT_RULE_WEIGHTS[k])
        for n in blend:
            blend[n] += wk * normed[k][n]
    blend = normalize(blend)

    omit = compute_omissions(draws)
    analysis = {
        "period": int(draws[-1]["period"]) if draws else None,
        "last_nums": last_nums,
        "gaps": gaps,
        "gap_lens": [(lo, hi, hi - lo + 1) for lo, hi in gaps],
        "consec_groups": [g for g in consecutive_groups(last_nums) if len(g) >= 2],
        "span": (max(last_nums) - min(last_nums)) if last_nums else 0,
        "zone_counts": [
            sum(1 for n in last_nums if lo <= n <= hi) for lo, hi in ZONES
        ],
        "omit": omit,
        "hot3": [
            n
            for n, c in Counter(
                x for d in draws[-3:] for x in map(int, d["nums"])
            ).items()
            if c >= 2
        ]
        if len(draws) >= 3
        else [],
    }
    return blend, normed, analysis


# ---------------------------------------------------------------------------
# 生成 3 组选十
# ---------------------------------------------------------------------------


def _ensure_consec_pair(picks: List[int], pool_scores: Dict[int, float]) -> List[int]:
    """保证至少有一组连号；若无则替换最低分号插入最佳连对。"""
    groups = consecutive_groups(picks)
    if any(len(g) >= 2 for g in groups):
        return sorted(picks)
    # 找分数最高的连号对，且至少一端已在候选附近
    best = None
    best_s = -1.0
    for a in range(1, POOL_MAX):
        b = a + 1
        s = pool_scores.get(a, 0) + pool_scores.get(b, 0)
        if s > best_s:
            best_s = s
            best = (a, b)
    if best is None:
        return sorted(picks)
    chosen = set(picks)
    for x in best:
        if x not in chosen:
            # 踢掉不破坏新区的最低分号
            victims = sorted(chosen, key=lambda n: pool_scores.get(n, 0))
            for v in victims:
                if v not in best:
                    chosen.remove(v)
                    chosen.add(x)
                    break
    # 若仍不足 10 个
    while len(chosen) < PICK_COUNT:
        for n, _ in sorted(pool_scores.items(), key=lambda kv: -kv[1]):
            if n not in chosen:
                chosen.add(n)
                break
    while len(chosen) > PICK_COUNT:
        v = min(chosen, key=lambda n: pool_scores.get(n, 0))
        chosen.remove(v)
    return sorted(chosen)


def _balance_zones(picks: List[int], scores: Dict[int, float]) -> List[int]:
    """四区尽量 2–3 个，避免单区堆叠。"""
    chosen = set(picks)
    counts = [sum(1 for n in chosen if lo <= n <= hi) for lo, hi in ZONES]
    # 过多区削减，过少区补充
    for zi, c in enumerate(counts):
        lo, hi = ZONES[zi]
        while c > 4:
            victims = [n for n in chosen if lo <= n <= hi]
            victims.sort(key=lambda n: scores.get(n, 0))
            if not victims:
                break
            chosen.remove(victims[0])
            c -= 1
        while c < 1:
            cands = [
                n
                for n in range(lo, hi + 1)
                if n not in chosen
            ]
            if not cands:
                break
            cands.sort(key=lambda n: -scores.get(n, 0))
            # 若已满，先踢别区最高堆叠
            if len(chosen) >= PICK_COUNT:
                # 找最多的区踢一个
                zc = [sum(1 for n in chosen if a <= n <= b) for a, b in ZONES]
                mz = max(range(4), key=lambda i: zc[i])
                if zc[mz] <= 1:
                    break
                a, b = ZONES[mz]
                victim = min(
                    (n for n in chosen if a <= n <= b),
                    key=lambda n: scores.get(n, 0),
                )
                chosen.remove(victim)
            chosen.add(cands[0])
            c += 1
    while len(chosen) < PICK_COUNT:
        for n, _ in sorted(scores.items(), key=lambda kv: -kv[1]):
            if n not in chosen:
                chosen.add(n)
                break
    while len(chosen) > PICK_COUNT:
        chosen.remove(min(chosen, key=lambda n: scores.get(n, 0)))
    return sorted(chosen)


def _pick_from_scores(
    scores: Dict[int, float],
    k: int = PICK_COUNT,
    exclude: Optional[Iterable[int]] = None,
    prefer: Optional[Iterable[int]] = None,
    rng: Optional[random.Random] = None,
) -> List[int]:
    rng = rng or random.Random(0)
    ban = set(exclude or [])
    pref = [n for n in (prefer or []) if n not in ban and 1 <= n <= POOL_MAX]
    chosen: List[int] = []
    for n in pref:
        if n not in chosen:
            chosen.append(n)
        if len(chosen) >= k:
            return sorted(chosen[:k])

    items = [(n, max(1e-12, scores.get(n, 0))) for n in range(1, POOL_MAX + 1) if n not in ban and n not in chosen]
    items.sort(key=lambda x: -x[1])
    # 顶部确定性选取 + 轻微扰动，保证组间差异
    top = items[: max(k * 3, 24)]
    while len(chosen) < k and top:
        weights = [s for _, s in top]
        total = sum(weights) or 1.0
        r = rng.random() * total
        acc = 0.0
        idx = 0
        for i, (_, s) in enumerate(top):
            acc += s
            if r <= acc:
                idx = i
                break
        n = top[idx][0]
        chosen.append(n)
        top.pop(idx)
    return sorted(chosen[:k])


def generate_three_groups(
    draws: Sequence[Draw],
    weights: Optional[Dict[str, float]] = None,
    target_period: Optional[int] = None,
) -> dict:
    """
    生成 3 组选十：
    A 空位主导（三空/四空/六空 + 连号）
    B 斜连/对称/四区
    C 冷热 + 遗漏周期 + 跨度定胆

    target_period: 若指定，结果标注该目标期（回测时用历史截止期的下一期）。
    """
    if len(draws) < 5:
        raise RuntimeError("历史数据太少，至少需要 5 期")

    w = weights or load_algo_weights()
    blend, normed, analysis = blend_rule_scores(draws, w)
    gaps = analysis["gaps"]

    # A：空位规则候选池
    prefer_a: List[int] = []
    for lo, hi in gaps:
        length = hi - lo + 1
        if length == 3:
            prefer_a.append((lo + hi) // 2)
        elif length == 4:
            if lo - 1 >= 1:
                prefer_a.append(lo - 1)
            if hi + 1 <= POOL_MAX:
                prefer_a.append(hi + 1)
        elif length >= 6:
            mid = (lo + hi) // 2
            prefer_a.extend([lo + 1, lo + 2, hi - 2, hi - 1, mid, mid + 1])
            if length >= 8:
                q1 = lo + max(1, length // 4)
                q3 = hi - max(1, length // 4)
                prefer_a.extend([q1, q1 + 1, q3 - 1, q3])
    score_a = {
        n: 0.45 * blend[n]
        + 0.20 * normed["gap3"][n]
        + 0.15 * normed["gap4"][n]
        + 0.15 * normed["gap6"][n]
        + 0.05 * normed["consec"][n]
        for n in range(1, POOL_MAX + 1)
    }
    g1 = _pick_from_scores(score_a, prefer=prefer_a, rng=random.Random(11))
    g1 = _balance_zones(g1, score_a)
    g1 = _ensure_consec_pair(g1, score_a)

    # B：斜连 + 对称 + 四区
    prefer_b: List[int] = []
    last = analysis["last_nums"]
    if last:
        end = max(last)
        prefer_b.extend([x for x in (end - 1, end + 1) if 1 <= x <= POOL_MAX])
        for n in last:
            s = symmetric(n)
            if 1 <= s <= POOL_MAX and s not in last:
                prefer_b.append(s)
            prefer_b.append(complementary_tail(n))
    score_b = {
        n: 0.35 * blend[n]
        + 0.25 * normed["seal_diag"][n]
        + 0.25 * normed["zone_sym"][n]
        + 0.15 * normed["consec"][n]
        for n in range(1, POOL_MAX + 1)
    }
    g2 = _pick_from_scores(score_b, prefer=prefer_b, exclude=[], rng=random.Random(22))
    # 与第 1 组保持差异
    overlap = len(set(g1) & set(g2))
    if overlap >= 7:
        g2 = _pick_from_scores(score_b, prefer=prefer_b, exclude=g1[:4], rng=random.Random(23))
    g2 = _balance_zones(g2, score_b)
    g2 = _ensure_consec_pair(g2, score_b)

    # C：冷热 + 遗漏 + 跨度
    prefer_c = list(analysis.get("hot3") or [])
    omit = analysis["omit"]
    prefer_c.extend(
        n for n, o in sorted(omit.items(), key=lambda kv: -kv[1]) if 25 <= o <= 40
    )
    score_c = {
        n: 0.30 * blend[n]
        + 0.30 * normed["hot_cold"][n]
        + 0.25 * normed["omit_cycle"][n]
        + 0.15 * normed["span_sym"][n]
        for n in range(1, POOL_MAX + 1)
    }
    ban_c = list(set(g1) & set(g2))
    g3 = _pick_from_scores(score_c, prefer=prefer_c, exclude=ban_c[:3], rng=random.Random(33))
    g3 = _balance_zones(g3, score_c)
    g3 = _ensure_consec_pair(g3, score_c)

    groups = [
        {
            "name": "A组·空位连号",
            "focus": "三空打中间 / 四空打两边 / 六空打连子 + 连号必出",
            "nums": g1,
            "score": sum(score_a[n] for n in g1),
        },
        {
            "name": "B组·斜连对称",
            "focus": "封口斜连 / 四区对称 / 跨区对称号",
            "nums": g2,
            "score": sum(score_b[n] for n in g2),
        },
        {
            "name": "C组·冷热跨度",
            "focus": "冷热搭配 / 遗漏周期 / 跨度定胆",
            "nums": g3,
            "score": sum(score_c[n] for n in g3),
        },
    ]

    # 胆码：三组交集 + 融合分 Top
    counter = Counter(n for g in groups for n in g["nums"])
    dan = [n for n, c in counter.most_common() if c >= 2]
    top_blend = sorted(blend, key=blend.get, reverse=True)
    for n in top_blend:
        if len(dan) >= 4:
            break
        if n not in dan:
            dan.append(n)
    dan = dan[:4]

    # 拖码建议（选七以上胆拖）
    tuo = [n for n in top_blend if n not in dan][:12]

    tip_lines = _build_tips(analysis, groups, dan)
    base_period = int(draws[-1]["period"])
    resolved_target = int(target_period) if target_period is not None else base_period + 1

    return {
        "target_period": resolved_target,
        "base_period": base_period,
        "groups": groups,
        "dan": dan[:4],
        "tuo": tuo,
        "blend_top": top_blend[:20],
        "analysis": {
            "gaps": analysis["gap_lens"],
            "consec_groups": analysis["consec_groups"],
            "span": analysis["span"],
            "zone_counts": analysis["zone_counts"],
            "hot3": analysis["hot3"],
            "omit_25_40": [n for n, o in omit.items() if 25 <= o <= 40],
        },
        "tips": tip_lines,
        "weights": w,
    }


def history_before(period: int, draws: Optional[Sequence[Draw]] = None) -> List[Draw]:
    """截取严格小于目标期的历史，用于回测/定点预测。"""
    all_draws = list(draws) if draws is not None else load_history()
    prior = [d for d in all_draws if int(d["period"]) < int(period)]
    prior.sort(key=lambda x: int(x["period"]))
    return prior


def predict_for_period(
    period: int,
    draws: Optional[Sequence[Draw]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> dict:
    """
    用目标期之前的全部历史生成 3 组选十。
    若库中已有该期开奖，附加 actual / hit 对比字段。
    """
    all_draws = list(draws) if draws is not None else load_history()
    prior = history_before(period, all_draws)
    if len(prior) < 5:
        raise RuntimeError(f"期号 {period} 之前历史不足 5 期（现有 {len(prior)}）")
    result = generate_three_groups(prior, weights=weights, target_period=period)
    actual = next((d for d in all_draws if int(d["period"]) == int(period)), None)
    if actual is not None:
        actual_nums = list(map(int, actual["nums"]))
        actual_set = set(actual_nums)
        result["actual"] = actual_nums
        result["compare"] = []
        for g in result["groups"]:
            hit = sorted(set(g["nums"]) & actual_set)
            result["compare"].append(
                {
                    "name": g["name"],
                    "hit_count": len(hit),
                    "hit_nums": hit,
                }
            )
        dan_hit = sorted(set(result["dan"]) & actual_set)
        result["dan_hit"] = dan_hit
    return result


def export_trend_database(
    draws: Optional[Sequence[Draw]] = None,
    recent: Optional[int] = None,
    path: Optional[Path] = None,
) -> Path:
    """导出基本走势图数据库（期号 × 01-80 命中/遗漏）到 JSON。"""
    all_draws = list(draws) if draws is not None else load_history()
    rows = build_trend_matrix(all_draws, recent=recent)
    payload = {
        "pool_max": POOL_MAX,
        "draw_count": DRAW_COUNT,
        "periods": len(rows),
        "from_period": rows[0]["period"] if rows else None,
        "to_period": rows[-1]["period"] if rows else None,
        "rows": [
            {
                "period": r["period"],
                "date": r.get("date", ""),
                "nums": list(r["nums"]),
                "cells": [
                    {"num": c["num"], "hit": bool(c["hit"]), "omit": int(c["omit"])}
                    for c in r["cells"]
                ],
            }
            for r in rows
        ],
    }
    out = path or (DATA_DIR / "trend_matrix.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out


def _build_tips(analysis: dict, groups: list, dan: List[int]) -> List[str]:
    tips: List[str] = []
    gap_lens = analysis.get("gap_lens") or []
    g3 = [g for g in gap_lens if g[2] == 3]
    g4 = [g for g in gap_lens if g[2] == 4]
    g6 = [g for g in gap_lens if g[2] >= 6]
    if g3:
        tips.append(
            "三空打中间："
            + "；".join(f"{a:02d}-{b:02d}→选{(a+b)//2:02d}" for a, b, _ in g3[:4])
        )
    if g4:
        tips.append(
            "四空打两边："
            + "；".join(
                f"{a:02d}-{b:02d}→选{max(1,a-1):02d}/{min(80,b+1):02d}"
                for a, b, _ in g4[:4]
            )
        )
    if g6:
        tips.append(
            "六空打连子："
            + "；".join(f"{a:02d}-{b:02d}布局连号" for a, b, _ in g6[:3])
        )
    consec = analysis.get("consec_groups") or []
    if consec:
        tips.append(
            "上期连号："
            + "，".join("-".join(f"{x:02d}" for x in g) for g in consec[:5])
            + "（可追两端或对称）"
        )
    tips.append(
        f"上期跨度 {analysis.get('span', 0)}；四区分布 "
        + "/".join(str(c) for c in analysis.get("zone_counts") or [])
    )
    hot = analysis.get("hot3") or []
    if hot:
        tips.append("短期热号：" + fmt_nums(sorted(hot)))
    cold = analysis.get("omit_25_40") if "omit_25_40" in analysis else None
    # analysis in generate already nested; handle both
    tips.append(f"推荐胆码（选五/选六定胆）：{fmt_nums(dan)}")
    tips.append("选七以上建议胆拖：定 3–4 胆 + 拖码，控制成本。")
    _ = groups
    _ = cold
    return tips


class TrendPredictor:
    """快乐8 规则集成预测器。"""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights if weights is not None else load_algo_weights()
        self.draws: List[Draw] = []
        self.blend: Dict[int, float] = {}
        self.analysis: dict = {}

    def fit(self, draws: Sequence[Draw]) -> None:
        self.draws = list(draws)
        self.blend, _, self.analysis = blend_rule_scores(self.draws, self.weights)

    def predict_groups(self, target_period: Optional[int] = None) -> dict:
        return generate_three_groups(
            self.draws, self.weights, target_period=target_period
        )

    def predict_period(self, period: int) -> dict:
        return predict_for_period(period, self.draws, self.weights)

    def trend(self, recent: int = 50) -> List[TrendRow]:
        return build_trend_matrix(self.draws, recent=recent)


def save_last_prediction(result: dict) -> None:
    LAST_PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target_period": result["target_period"],
        "base_period": result["base_period"],
        "groups": [
            {"name": g["name"], "focus": g["focus"], "nums": list(g["nums"])}
            for g in result["groups"]
        ],
        "dan": list(result["dan"]),
        "tuo": list(result.get("tuo", [])),
    }
    with LAST_PRED_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_last_prediction() -> Optional[dict]:
    if not LAST_PRED_PATH.exists():
        return None
    try:
        with LAST_PRED_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def compare_prediction_to_draw(
    actual_nums: Sequence[int],
    period: Optional[int] = None,
) -> dict:
    """对比上次 3 组选十与开奖（选十命中个数）。"""
    pred = load_last_prediction()
    if not pred:
        raise ValueError("没有上次预测记录")
    actual = set(_norm_nums(actual_nums, DRAW_COUNT))
    rows = []
    for g in pred.get("groups", []):
        nums = [int(x) for x in g["nums"]]
        hit = sorted(set(nums) & actual)
        rows.append(
            {
                "name": g.get("name", ""),
                "nums": nums,
                "hit_count": len(hit),
                "hit_nums": hit,
            }
        )
    dan = [int(x) for x in pred.get("dan", [])]
    dan_hit = sorted(set(dan) & actual)

    # 按命中微调权重
    weights = load_algo_weights()
    focus_map = {
        "A组·空位连号": ["gap3", "gap4", "gap6", "consec"],
        "B组·斜连对称": ["seal_diag", "zone_sym", "consec"],
        "C组·冷热跨度": ["hot_cold", "omit_cycle", "span_sym"],
    }
    best = max(rows, key=lambda r: r["hit_count"]) if rows else None
    worst = min(rows, key=lambda r: r["hit_count"]) if rows else None
    adjusted = False
    if best and worst and best["hit_count"] != worst["hit_count"]:
        for k in focus_map.get(best["name"], []):
            weights[k] = weights.get(k, 0.1) * 1.08
        for k in focus_map.get(worst["name"], []):
            weights[k] = weights.get(k, 0.1) * 0.94
        s = sum(weights.values()) or 1.0
        weights = {k: v / s for k, v in weights.items()}
        save_algo_weights(weights)
        adjusted = True

    result = {
        "period": period or pred.get("target_period"),
        "actual": sorted(actual),
        "groups": rows,
        "dan": dan,
        "dan_hit": dan_hit,
        "weights": weights,
        "adjusted": adjusted,
    }
    # 追加日志
    log = []
    if COMPARE_LOG_PATH.exists():
        try:
            with COMPARE_LOG_PATH.open("r", encoding="utf-8") as f:
                log = json.load(f)
        except Exception:
            log = []
    log.append(
        {
            "period": result["period"],
            "hits": [r["hit_count"] for r in rows],
            "dan_hit": len(dan_hit),
            "adjusted": adjusted,
        }
    )
    with COMPARE_LOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(log[-200:], f, ensure_ascii=False, indent=2)
    return result


def format_compare_text(result: dict) -> str:
    lines = [
        f"期号 {result.get('period')} 对比",
        f"开奖：{fmt_nums(result['actual'])}",
        "",
    ]
    for g in result["groups"]:
        lines.append(
            f"{g['name']} 命中 {g['hit_count']}/10：{fmt_nums(g['nums'])}"
        )
        lines.append(f"  命中号：{fmt_nums(g['hit_nums']) or '-'}")
    lines.append(f"胆码命中：{fmt_nums(result['dan_hit']) or '-'} / {fmt_nums(result['dan'])}")
    return "\n".join(lines)


def format_prediction_text(result: dict) -> str:
    lines = [
        f"快乐8 选十预测 → 目标期 {result['target_period']}（基于 {result['base_period']} 期）",
        "",
    ]
    for i, g in enumerate(result["groups"], 1):
        lines.append(f"{i}. {g['name']}")
        lines.append(f"   思路：{g['focus']}")
        lines.append(f"   号码：{fmt_nums(g['nums'])}")
        lines.append("")
    lines.append(f"胆码：{fmt_nums(result['dan'])}")
    lines.append(f"拖码：{fmt_nums(result.get('tuo', [])[:10])}")
    lines.append("")
    lines.append("分析要点：")
    for t in result.get("tips", []):
        lines.append(f"· {t}")
    return "\n".join(lines)


def format_compare_inline(result: dict) -> str:
    if "compare" not in result:
        return ""
    lines = ["", "=== 与开奖对比 ==="]
    if result.get("actual"):
        lines.append(f"开奖：{fmt_nums(result['actual'])}")
    for row in result["compare"]:
        lines.append(
            f"{row['name']} 命中 {row['hit_count']}/10：{fmt_nums(row['hit_nums']) or '-'}"
        )
    if "dan_hit" in result:
        lines.append(
            f"胆码命中：{fmt_nums(result['dan_hit']) or '-'} / {fmt_nums(result['dan'])}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="快乐8 走势图选十预测")
    parser.add_argument(
        "--period",
        type=int,
        default=None,
        help="目标期号（用该期之前历史预测；默认预测最新期+1）",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="先导入 CSV 再预测（如 快乐8_近100期开奖数据.csv）",
    )
    parser.add_argument(
        "--export-trend",
        action="store_true",
        help="导出基本走势图数据库 trend_matrix.json",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="保存为 last_prediction.json",
    )
    args = parser.parse_args()

    ensure_data_files()
    if args.csv:
        draws = import_csv(args.csv)
        print(f"已导入 CSV：{len(draws)} 期")
    else:
        draws = load_history()
    print(f"历史 {len(draws)} 期")
    if draws:
        print("最新", draws[-1]["period"], fmt_nums(draws[-1]["nums"]))

    if args.export_trend:
        out = export_trend_database(draws)
        print(f"已导出走势图数据库：{out}")

    if args.period is not None:
        result = predict_for_period(args.period, draws)
    else:
        model = TrendPredictor()
        model.fit(draws)
        result = model.predict_groups()
    print(format_prediction_text(result))
    print(format_compare_inline(result))
    if args.save:
        save_last_prediction(result)
        # 定点回测结果另存一份
        stamp = DATA_DIR / f"prediction_{result['target_period']}.json"
        with stamp.open("w", encoding="utf-8") as f:
            payload = {
                "target_period": result["target_period"],
                "base_period": result["base_period"],
                "groups": [
                    {"name": g["name"], "focus": g["focus"], "nums": list(g["nums"])}
                    for g in result["groups"]
                ],
                "dan": list(result["dan"]),
                "tuo": list(result.get("tuo", [])),
                "analysis": result.get("analysis"),
                "tips": result.get("tips"),
                "actual": result.get("actual"),
                "compare": result.get("compare"),
                "dan_hit": result.get("dan_hit"),
            }
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"已保存：{LAST_PRED_PATH} / {stamp}")

    trend = build_trend_matrix(
        history_before(result["target_period"], draws)
        if args.period is not None
        else draws,
        recent=30,
    )
    print(f"走势图行数 {len(trend)}，列 80")
    if trend:
        last = trend[-1]
        hits = [c["num"] for c in last["cells"] if c["hit"]]
        print("末行命中", fmt_nums(hits))
