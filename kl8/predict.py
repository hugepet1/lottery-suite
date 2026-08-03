#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快乐8预测引擎（01-80 选 20，用户每期预测 10 个号码）

说明：开奖为近似均匀随机，本模块仅做统计启发式研究，无法保证中奖。
核心思路：多因子评分 + 反马尔可夫（低转移续开）+ 冷热/区间/奇偶形态约束。
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import db

POOL = 80
DRAW_N = 20
PICK_N = 10
LONG_TERM = (2, 11, 14, 27, 39, 49, 54, 62, 69, 75)

DEFAULT_WEIGHTS = {
    "freq": 0.20,
    "gap": 0.20,
    "hotcold": 0.15,
    "zone": 0.15,
    "oddeven_size": 0.10,
    "consec": 0.10,
    "noise": 0.10,
}

WEIGHT_NAMES = {
    "freq": "频率模型",
    "gap": "遗漏模型",
    "hotcold": "冷热转换",
    "zone": "区间概率",
    "oddeven_size": "奇偶大小",
    "consec": "连号模型",
    "noise": "随机扰动",
}

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "kl8"
LAST_PRED_PATH = DATA_DIR / "last_prediction.json"
REPORT_PATH = DATA_DIR / "latest_report.txt"
HISTORY_JSON = DATA_DIR / "history.json"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _norm_weights(w: Dict[str, float]) -> Dict[str, float]:
    keys = list(DEFAULT_WEIGHTS)
    cleaned = {k: max(0.01, float(w.get(k, DEFAULT_WEIGHTS[k]))) for k in keys}
    s = sum(cleaned.values())
    return {k: v / s for k, v in cleaned.items()}


def load_weights() -> Dict[str, float]:
    stored = db.load_weights()
    if stored:
        return _norm_weights(stored)
    return dict(DEFAULT_WEIGHTS)


def save_weights(weights: Dict[str, float]) -> Dict[str, float]:
    w = _norm_weights(weights)
    db.save_weights(w, _now())
    return w


def ensure_database(csv_path: Optional[Path] = None) -> List[Dict]:
    """建立/刷新数据库，并同步 history.json。"""
    db.ensure_dir()
    path = csv_path or db.CSV_PATH
    if path.exists():
        db.import_csv(path)
    draws = db.load_draws()
    if draws:
        payload = [
            {
                "period": d["period"],
                "date": d["date"],
                "numbers": list(d["numbers"]),
            }
            for d in draws
        ]
        HISTORY_JSON.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return draws


# ---------------------------------------------------------------------------
# 基础统计
# ---------------------------------------------------------------------------

def appearance_counts(draws: Sequence[Dict], window: Optional[int] = None) -> Counter:
    subset = draws[-window:] if window else draws
    c: Counter = Counter()
    for d in subset:
        c.update(d["numbers"])
    return c


def current_gaps(draws: Sequence[Dict]) -> Dict[int, int]:
    """号码当前遗漏期数（最近一期出现则为 0）。"""
    gaps = {n: len(draws) for n in range(1, POOL + 1)}
    for i, d in enumerate(reversed(draws)):
        for n in d["numbers"]:
            if gaps[n] == len(draws):
                gaps[n] = i
        if all(v < len(draws) for v in gaps.values()):
            # 仍可能有从未出现的号，继续扫完
            pass
    return gaps


def recent_trend(draws: Sequence[Dict], short: int = 10, long: int = 30) -> Dict[int, float]:
    """近期频率相对长期频率的差值（上涨为正）。"""
    if not draws:
        return {n: 0.0 for n in range(1, POOL + 1)}
    short_c = appearance_counts(draws, short)
    long_c = appearance_counts(draws, long)
    s_n = max(1, min(short, len(draws)))
    l_n = max(1, min(long, len(draws)))
    out = {}
    for n in range(1, POOL + 1):
        sf = short_c.get(n, 0) / s_n
        lf = long_c.get(n, 0) / l_n
        out[n] = sf - lf
    return out


def zone_of(n: int) -> int:
    # 4 区：1-20 / 21-40 / 41-60 / 61-80
    return (n - 1) // 20


def is_big(n: int) -> bool:
    return n >= 41


def consecutive_pairs(nums: Sequence[int]) -> int:
    s = sorted(nums)
    return sum(1 for a, b in zip(s, s[1:]) if b == a + 1)


# ---------------------------------------------------------------------------
# 反马尔可夫：低续开 / 转移稀有优先
# ---------------------------------------------------------------------------

def build_transition(draws: Sequence[Dict]) -> Dict[int, Counter]:
    """P(下一期出现 j | 本期出现 i) 的计数。"""
    trans: Dict[int, Counter] = defaultdict(Counter)
    for a, b in zip(draws, draws[1:]):
        nxt = set(b["numbers"])
        for i in a["numbers"]:
            for j in nxt:
                trans[i][j] += 1
    return trans


def anti_markov_scores(draws: Sequence[Dict]) -> Dict[int, float]:
    """
    反马尔可夫评分：对「上一期号码」的高转移续开给予惩罚，
    对历史上较少跟随上一期集合出现的号码加分（均值回归/反相关）。
    """
    if len(draws) < 2:
        return {n: 50.0 for n in range(1, POOL + 1)}
    trans = build_transition(draws)
    last = set(draws[-1]["numbers"])
    # 对每个候选 j，累加其从 last 集合被“跟随”的强度
    follow = Counter()
    total_from = Counter()
    for i in last:
        for j, c in trans[i].items():
            follow[j] += c
            total_from[i] += c
    max_f = max(follow.values()) if follow else 1
    # 上一期已开出的号码略降权（反续开）
    scores = {}
    for n in range(1, POOL + 1):
        f = follow.get(n, 0) / max_f
        # 低跟随 => 高分
        base = (1.0 - f) * 100.0
        if n in last:
            base *= 0.72
        scores[n] = base
    return scores


# ---------------------------------------------------------------------------
# 单号多因子评分（100 分制）
# ---------------------------------------------------------------------------

def score_numbers(
    draws: Sequence[Dict],
    weights: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> Dict[int, Dict[str, float]]:
    """返回每个号码的分项分与综合分。"""
    w = _norm_weights(weights or load_weights())
    rng = random.Random(seed)
    n_draw = max(1, len(draws))
    freq_all = appearance_counts(draws)
    freq_20 = appearance_counts(draws, 20)
    freq_10 = appearance_counts(draws, 10)
    gaps = current_gaps(draws)
    trend = recent_trend(draws)
    am = anti_markov_scores(draws)

    # 理论频率：每期开 20/80 = 0.25
    expect = 0.25 * n_draw
    # 区间历史占比
    zone_hist = Counter(zone_of(n) for d in draws for n in d["numbers"])
    zone_total = sum(zone_hist.values()) or 1
    zone_share = {z: zone_hist[z] / zone_total for z in range(4)}
    # 理想每区 25%
    zone_need = {z: max(0.0, 0.25 - zone_share.get(z, 0)) for z in range(4)}

    # 奇偶大小历史偏差（用于形态微调）
    odd_rate = sum(1 for d in draws for n in d["numbers"] if n % 2 == 1) / (
        n_draw * DRAW_N
    )
    big_rate = sum(1 for d in draws for n in d["numbers"] if is_big(n)) / (
        n_draw * DRAW_N
    )

    # 连号：号码作为连号成员出现的频率
    consec_hit = Counter()
    for d in draws:
        s = sorted(d["numbers"])
        for a, b in zip(s, s[1:]):
            if b == a + 1:
                consec_hit[a] += 1
                consec_hit[b] += 1

    # 热号过热惩罚阈值
    hot_cut = sorted(freq_20.values(), reverse=True)[9] if freq_20 else 0

    detail: Dict[int, Dict[str, float]] = {}
    for n in range(1, POOL + 1):
        # 频率：贴近期望略高分，过热略降
        f = freq_all.get(n, 0)
        freq_score = 100.0 * math.exp(-abs(f - expect) / max(3.0, expect * 0.35))
        if freq_20.get(n, 0) >= hot_cut and hot_cut > 0:
            freq_score *= 0.85

        # 遗漏：中等遗漏加分，极端遗漏略降（长期异常）
        g = gaps[n]
        # 理论几何遗漏均值约 3（开出概率 0.25）— 快乐8单号开出概率 20/80=0.25，期望遗漏 3
        # 但实际用户更关心相对遗漏排名；用平滑峰
        gap_score = 100.0 * math.exp(-((g - 4) ** 2) / 32.0)
        if g >= 18:
            gap_score *= 0.7  # 长期异常

        # 冷热转换：近期趋冷后回升 / 过热回落
        t = trend[n]
        f10 = freq_10.get(n, 0)
        f20 = freq_20.get(n, 0)
        if f20 <= 3 and g >= 5:
            hotcold = 70 + min(30, g * 2)  # 冷号回补潜力
        elif f10 >= 4 and t > 0:
            hotcold = 75 + min(20, f10 * 3)  # 热号延续
        else:
            hotcold = 50 + t * 80
        hotcold = max(0.0, min(100.0, hotcold))

        # 区间平衡
        z = zone_of(n)
        zone_score = 40 + zone_need[z] * 240
        zone_score = max(0.0, min(100.0, zone_score))

        # 奇偶大小：补短板
        oe = 55.0
        if odd_rate > 0.55 and n % 2 == 0:
            oe += 20
        elif odd_rate < 0.45 and n % 2 == 1:
            oe += 20
        else:
            oe += 5
        if big_rate > 0.55 and not is_big(n):
            oe += 15
        elif big_rate < 0.45 and is_big(n):
            oe += 15
        oe = min(100.0, oe)

        # 连号模型：适度连号成员加分，但避免过度追连
        cscore = 40 + min(40, consec_hit.get(n, 0) * 3)
        # 与上一期相邻号略加权（形态）
        if draws:
            last = set(draws[-1]["numbers"])
            if (n - 1) in last or (n + 1) in last:
                cscore += 12
        cscore = min(100.0, cscore)

        # 随机扰动（可复现）
        noise = rng.uniform(35, 75)

        # 反马尔可夫并入频率/趋势侧
        am_s = am[n]

        parts = {
            "freq": 0.6 * freq_score + 0.4 * am_s,
            "gap": gap_score,
            "hotcold": hotcold,
            "zone": zone_score,
            "oddeven_size": oe,
            "consec": cscore,
            "noise": noise,
        }
        total = sum(parts[k] * w[k] for k in w)
        # 连续过热额外降分
        if freq_10.get(n, 0) >= 5:
            total *= 0.9
        parts["total"] = max(0.0, min(100.0, total))
        parts["gap_periods"] = float(g)
        parts["appear_all"] = float(f)
        parts["appear_20"] = float(f20)
        parts["trend"] = float(t)
        detail[n] = parts
    return detail


def top_n(scores: Dict[int, Dict[str, float]], n: int = 10, key: str = "total") -> List[int]:
    ranked = sorted(scores.keys(), key=lambda x: scores[x][key], reverse=True)
    return ranked[:n]


# ---------------------------------------------------------------------------
# 组合生成：4 组方案 + 反马尔可夫变体
# ---------------------------------------------------------------------------

def _balance_pick(
    ranked: Sequence[int],
    scores: Dict[int, Dict[str, float]],
    n: int = PICK_N,
    prefer_odd: Optional[float] = 0.5,
    prefer_big: Optional[float] = 0.5,
    max_zone: int = 4,
) -> List[int]:
    """从候选中贪心选取，兼顾奇偶/大小/区间。"""
    chosen: List[int] = []
    zone_c = Counter()
    for cand in ranked:
        if len(chosen) >= n:
            break
        z = zone_of(cand)
        if zone_c[z] >= max_zone:
            continue
        trial = chosen + [cand]
        if len(trial) >= 4:
            odd = sum(1 for x in trial if x % 2 == 1) / len(trial)
            big = sum(1 for x in trial if is_big(x)) / len(trial)
            if prefer_odd is not None and abs(odd - prefer_odd) > 0.35 and len(trial) < n:
                # 仍可放行前期
                if scores[cand]["total"] < 55:
                    continue
            if prefer_big is not None and abs(big - prefer_big) > 0.35 and len(trial) < n:
                if scores[cand]["total"] < 55:
                    continue
        chosen.append(cand)
        zone_c[z] += 1
    if len(chosen) < n:
        for cand in ranked:
            if cand not in chosen:
                chosen.append(cand)
            if len(chosen) >= n:
                break
    return sorted(chosen)


def predict_groups(
    draws: Sequence[Dict],
    weights: Optional[Dict[str, float]] = None,
    seed: int = 2026,
) -> Dict[str, List[int]]:
    w = _norm_weights(weights or load_weights())
    scores = score_numbers(draws, w, seed=seed)
    am = anti_markov_scores(draws)
    gaps = current_gaps(draws)
    freq_20 = appearance_counts(draws, 20)
    trend = recent_trend(draws)

    # 综合排名
    ranked_total = sorted(range(1, POOL + 1), key=lambda n: scores[n]["total"], reverse=True)
    # 反马尔可夫排名
    ranked_am = sorted(range(1, POOL + 1), key=lambda n: am[n], reverse=True)
    # 冷号：高遗漏
    ranked_cold = sorted(range(1, POOL + 1), key=lambda n: (gaps[n], -freq_20.get(n, 0)), reverse=True)
    # 热号：近 20 期高频 + 趋势上行
    ranked_hot = sorted(
        range(1, POOL + 1),
        key=lambda n: (freq_20.get(n, 0) * 2 + max(0, trend[n]) * 10 + scores[n]["total"] * 0.01),
        reverse=True,
    )

    # 方案1：纯反马尔可夫链 Top10（带区间平衡）
    scheme1 = _balance_pick(ranked_am, scores, PICK_N, max_zone=3)

    # 方案2：反马尔可夫 + 热冷平衡（AM 与综合分融合）
    blend = {
        n: 0.55 * am[n] + 0.45 * scores[n]["total"] for n in range(1, POOL + 1)
    }
    ranked_blend = sorted(blend.keys(), key=lambda n: blend[n], reverse=True)
    scheme2 = _balance_pick(ranked_blend, scores, PICK_N, max_zone=3)

    # 方案3：反马尔可夫 + 冷号回补
    cold_blend = {
        n: 0.5 * am[n] + 0.35 * scores[n]["gap"] + 0.15 * scores[n]["hotcold"]
        for n in range(1, POOL + 1)
    }
    ranked_cb = sorted(cold_blend.keys(), key=lambda n: cold_blend[n], reverse=True)
    scheme3 = _balance_pick(ranked_cb, scores, PICK_N, max_zone=3)

    # 组1：综合模型
    g1 = _balance_pick(ranked_total, scores, PICK_N, max_zone=3)
    # 组2：冷号回补
    g2 = _balance_pick(ranked_cold, scores, PICK_N, max_zone=4)
    # 组3：热号延续
    g3 = _balance_pick(ranked_hot, scores, PICK_N, max_zone=4)
    # 组4：机器随机优化（高分池蒙特卡洛）
    rng = random.Random(seed + 7)
    pool = ranked_total[:28]
    best = None
    best_s = -1.0
    for _ in range(800):
        pick = sorted(rng.sample(pool, PICK_N))
        s = sum(scores[n]["total"] for n in pick)
        # 形态奖励
        odd = sum(1 for x in pick if x % 2 == 1)
        big = sum(1 for x in pick if is_big(x))
        zones = len({zone_of(x) for x in pick})
        cons = consecutive_pairs(pick)
        s += (5 - abs(odd - 5)) * 3
        s += (5 - abs(big - 5)) * 3
        s += zones * 4
        s += (2 - abs(cons - 2)) * 2
        if s > best_s:
            best_s = s
            best = pick
    g4 = best or g1

    return {
        "core": g1,
        "group1_ensemble": g1,
        "group2_cold": g2,
        "group3_hot": g3,
        "group4_random_opt": g4,
        "scheme1_anti_markov": scheme1,
        "scheme2_am_hotcold": scheme2,
        "scheme3_am_cold": scheme3,
    }


# ---------------------------------------------------------------------------
# 长期守号
# ---------------------------------------------------------------------------

def analyze_long_term(draws: Sequence[Dict]) -> Dict:
    gaps = current_gaps(draws)
    freq_all = appearance_counts(draws)
    freq_20 = appearance_counts(draws, 20)
    freq_10 = appearance_counts(draws, 10)
    trend = recent_trend(draws)
    scores = score_numbers(draws)
    items = []
    total_score = 0.0
    for n in LONG_TERM:
        sc = scores[n]["total"]
        # 守号评分：综合分 + 稳定性
        appear = freq_all.get(n, 0)
        rate = appear / max(1, len(draws))
        # 理论 0.25
        stability = 100 * math.exp(-abs(rate - 0.25) / 0.12)
        keep = 0.5 * sc + 0.3 * stability + 0.2 * max(0, 100 - gaps[n] * 4)
        if gaps[n] >= 15:
            keep *= 0.85
        keep = max(0.0, min(100.0, keep))
        total_score += keep
        direction = "上升" if trend[n] > 0.02 else ("下降" if trend[n] < -0.02 else "平稳")
        items.append(
            {
                "number": n,
                "appear_all": appear,
                "appear_20": freq_20.get(n, 0),
                "appear_10": freq_10.get(n, 0),
                "gap": gaps[n],
                "trend": trend[n],
                "trend_label": direction,
                "score": round(keep, 1),
                "keep": True,  # 不允许直接删除
                "note": "继续保留观察",
            }
        )
    avg = total_score / len(LONG_TERM)
    up = sum(1 for x in items if x["trend_label"] == "上升")
    down = sum(1 for x in items if x["trend_label"] == "下降")
    if up > down + 2:
        overall_trend = "整体偏强"
    elif down > up + 2:
        overall_trend = "整体偏弱"
    else:
        overall_trend = "整体震荡"
    return {
        "numbers": list(LONG_TERM),
        "items": items,
        "score": round(avg, 1),
        "trend": overall_trend,
        "up": up,
        "down": down,
    }


# ---------------------------------------------------------------------------
# 复盘
# ---------------------------------------------------------------------------

def review_prediction(
    draw_numbers: Sequence[int],
    pred_numbers: Sequence[int],
    draws_before: Sequence[Dict],
) -> Dict:
    ds = set(int(x) for x in draw_numbers)
    ps = [int(x) for x in pred_numbers]
    hits = sorted(n for n in ps if n in ds)
    gaps = current_gaps(draws_before) if draws_before else {}
    freq_20 = appearance_counts(draws_before, 20) if draws_before else Counter()
    miss = [n for n in ps if n not in ds]

    reasons = {
        "遗漏判断错误": 0,
        "冷热判断错误": 0,
        "区域分布错误": 0,
        "奇偶比例错误": 0,
        "大小比例错误": 0,
        "连号遗漏错误": 0,
        "模型权重错误": 0,
    }
    # 启发式归因
    for n in miss:
        g = gaps.get(n, 0)
        if g <= 1:
            reasons["遗漏判断错误"] += 1
        if freq_20.get(n, 0) >= 6:
            reasons["冷热判断错误"] += 1
        elif freq_20.get(n, 0) <= 2 and g < 8:
            reasons["冷热判断错误"] += 1
    pred_zones = Counter(zone_of(n) for n in ps)
    draw_zones = Counter(zone_of(n) for n in ds)
    if sum(abs(pred_zones[z] - draw_zones.get(z, 0) * PICK_N / DRAW_N) for z in range(4)) > 4:
        reasons["区域分布错误"] += 2
    pred_odd = sum(1 for n in ps if n % 2 == 1)
    draw_odd = sum(1 for n in ds if n % 2 == 1)
    if abs(pred_odd / PICK_N - draw_odd / DRAW_N) > 0.2:
        reasons["奇偶比例错误"] += 2
    pred_big = sum(1 for n in ps if is_big(n))
    draw_big = sum(1 for n in ds if is_big(n))
    if abs(pred_big / PICK_N - draw_big / DRAW_N) > 0.2:
        reasons["大小比例错误"] += 2
    if abs(consecutive_pairs(ps) - consecutive_pairs(ds) * PICK_N / DRAW_N) > 1.5:
        reasons["连号遗漏错误"] += 2
    if len(hits) <= 2:
        reasons["模型权重错误"] += 3
    elif len(hits) == 3:
        reasons["模型权重错误"] += 1

    # 归一说明
    ranked_reasons = sorted(reasons.items(), key=lambda x: x[1], reverse=True)
    primary = [k for k, v in ranked_reasons if v > 0][:4] or ["样本随机波动为主"]

    return {
        "draw_numbers": sorted(ds),
        "pred_numbers": sorted(ps),
        "hits": hits,
        "hit_count": len(hits),
        "hit_rate": f"{len(hits)}/{PICK_N}",
        "fail_reasons": reasons,
        "primary_reasons": primary,
    }


def adjust_weights_from_review(
    weights: Dict[str, float],
    review: Dict,
) -> Tuple[Dict[str, float], List[str]]:
    w = dict(_norm_weights(weights))
    notes: List[str] = []
    reasons = review["fail_reasons"]
    hit = review["hit_count"]

    def bump(key: str, delta: float, why: str) -> None:
        w[key] = max(0.03, w[key] + delta)
        notes.append(why)

    if reasons.get("遗漏判断错误", 0) >= 2:
        bump("gap", 0.04, "遗漏判断偏差 → 提高遗漏模型权重")
        bump("freq", -0.02, "同步略降频率权重")
    if reasons.get("冷热判断错误", 0) >= 2:
        bump("hotcold", 0.04, "冷热判断偏差 → 提高冷热转换权重")
    if reasons.get("区域分布错误", 0) >= 2:
        bump("zone", 0.03, "区间分布偏差 → 提高区间概率权重")
    if reasons.get("奇偶比例错误", 0) >= 2 or reasons.get("大小比例错误", 0) >= 2:
        bump("oddeven_size", 0.03, "奇偶/大小偏差 → 提高形态权重")
    if reasons.get("连号遗漏错误", 0) >= 2:
        bump("consec", 0.03, "连号形态偏差 → 提高连号权重")
    if reasons.get("模型权重错误", 0) >= 2:
        bump("noise", 0.02, "命中偏低 → 略增随机扰动防过拟合")
        # 向默认回归一点
        for k in w:
            w[k] = 0.85 * w[k] + 0.15 * DEFAULT_WEIGHTS[k]
        notes.append("权重向默认值部分回归，降低过拟合")

    if hit >= 4:
        # 强化当前主力
        bump("freq", 0.02, f"命中 {hit} 个表现尚可 → 微调强化频率/遗漏")
        bump("gap", 0.01, "同步微调遗漏")

    w = _norm_weights(w)
    if not notes:
        notes.append("本期偏差在随机波动范围内，权重小幅保持")
    return w, notes


# ---------------------------------------------------------------------------
# 回测
# ---------------------------------------------------------------------------

def backtest(
    draws: Sequence[Dict],
    window: int = 50,
    weights: Optional[Dict[str, float]] = None,
    scheme: str = "group1_ensemble",
) -> Dict:
    """用前 i 期预测第 i+1 期的 10 码，统计最近 window 期命中。"""
    if len(draws) < window + 15:
        window = max(10, len(draws) - 15)
    w = _norm_weights(weights or load_weights())
    hits_list: List[int] = []
    start = len(draws) - window
    for i in range(start, len(draws)):
        hist = draws[:i]
        actual = set(draws[i]["numbers"])
        groups = predict_groups(hist, w, seed=1000 + i)
        pred = groups.get(scheme) or groups["group1_ensemble"]
        hit = sum(1 for n in pred if n in actual)
        hits_list.append(hit)
    if not hits_list:
        return {
            "as_of_period": draws[-1]["period"] if draws else 0,
            "window_size": 0,
            "avg_hit": 0.0,
            "max_hit": 0,
            "min_hit": 0,
            "hit10": 0,
            "hit9": 0,
            "hit8": 0,
            "detail": {},
        }
    return {
        "as_of_period": draws[-1]["period"],
        "window_size": len(hits_list),
        "avg_hit": round(sum(hits_list) / len(hits_list), 3),
        "max_hit": max(hits_list),
        "min_hit": min(hits_list),
        "hit10": sum(1 for x in hits_list if x >= 10),
        "hit9": sum(1 for x in hits_list if x == 9),
        "hit8": sum(1 for x in hits_list if x == 8),
        "detail": {
            "scheme": scheme,
            "hits": hits_list,
            "expected_random": round(10 * 20 / 80, 3),  # 2.5
        },
    }


def optimize_weights_by_backtest(draws: Sequence[Dict], window: int = 50) -> Tuple[Dict[str, float], Dict]:
    """简易网格/扰动搜索，最大化平均命中。"""
    base = load_weights()
    best_w = dict(base)
    best = backtest(draws, window, best_w)
    best_avg = best["avg_hit"]
    rng = random.Random(7)
    keys = list(DEFAULT_WEIGHTS)
    for _ in range(40):
        trial = dict(best_w)
        k = rng.choice(keys)
        trial[k] *= rng.uniform(0.7, 1.35)
        trial = _norm_weights(trial)
        bt = backtest(draws, window, trial)
        if bt["avg_hit"] > best_avg or (
            bt["avg_hit"] == best_avg and bt["max_hit"] > best["max_hit"]
        ):
            best_avg = bt["avg_hit"]
            best_w = trial
            best = bt
    return best_w, best


# ---------------------------------------------------------------------------
# 分析摘要
# ---------------------------------------------------------------------------

def build_analysis(draws: Sequence[Dict], weights: Dict[str, float]) -> Dict:
    scores = score_numbers(draws, weights)
    gaps = current_gaps(draws)
    freq = appearance_counts(draws)
    freq_20 = appearance_counts(draws, 20)
    trend = recent_trend(draws)

    freq_rank = sorted(range(1, POOL + 1), key=lambda n: freq.get(n, 0), reverse=True)
    gap_rank = sorted(range(1, POOL + 1), key=lambda n: gaps[n], reverse=True)
    hot = sorted(range(1, POOL + 1), key=lambda n: freq_20.get(n, 0), reverse=True)[:10]
    cold = gap_rank[:10]
    rising = sorted(range(1, POOL + 1), key=lambda n: trend[n], reverse=True)[:10]
    falling = sorted(range(1, POOL + 1), key=lambda n: trend[n])[:10]
    core = top_n(scores, 10)

    return {
        "periods": len(draws),
        "latest_period": draws[-1]["period"] if draws else None,
        "latest_date": draws[-1]["date"] if draws else "",
        "latest_numbers": list(draws[-1]["numbers"]) if draws else [],
        "freq_rank": [(n, freq.get(n, 0)) for n in freq_rank],
        "gap_rank": [(n, gaps[n]) for n in gap_rank],
        "hot_top10": [(n, freq_20.get(n, 0)) for n in hot],
        "cold_top10": [(n, gaps[n]) for n in cold],
        "rising": [(n, round(trend[n], 4)) for n in rising],
        "falling": [(n, round(trend[n], 4)) for n in falling],
        "core10": core,
        "scores": scores,
        "weights": weights,
    }


def fmt_nums(nums: Sequence[int]) -> str:
    return " ".join(f"{int(n):02d}" for n in nums)


# ---------------------------------------------------------------------------
# 主流程：复盘 → 调权 → 回测 → 预测 → 报告
# ---------------------------------------------------------------------------

def load_last_prediction() -> Optional[Dict]:
    if not LAST_PRED_PATH.exists():
        return None
    return json.loads(LAST_PRED_PATH.read_text(encoding="utf-8"))


def save_last_prediction(payload: Dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LAST_PRED_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_pipeline(csv_path: Optional[Path] = None) -> Dict:
    draws = ensure_database(csv_path)
    if len(draws) < 20:
        raise RuntimeError(f"历史数据不足（{len(draws)} 期），至少需要约 20 期。")

    weights = load_weights()
    last_pred = load_last_prediction()
    latest = draws[-1]
    review_block = None
    adjust_notes: List[str] = ["首次运行或无上一期预测，保持默认/当前权重"]

    # 复盘：若有针对 latest.period 的预测
    if last_pred and int(last_pred.get("target_period", -1)) == int(latest["period"]):
        pred_nums = last_pred.get("numbers") or last_pred.get("core") or []
        review_block = review_prediction(
            latest["numbers"], pred_nums, draws[:-1]
        )
        weights, adjust_notes = adjust_weights_from_review(weights, review_block)
        weights = save_weights(weights)
        db.save_review(
            period=latest["period"],
            draw_numbers=latest["numbers"],
            pred_numbers=pred_nums,
            hits=review_block["hits"],
            fail_reasons={
                "reasons": review_block["fail_reasons"],
                "primary": review_block["primary_reasons"],
            },
            created_at=_now(),
        )
    elif last_pred:
        # 预测期号不匹配：仍展示最近预测与最新开奖的对比供参考
        pred_nums = last_pred.get("numbers") or last_pred.get("core") or []
        review_block = review_prediction(
            latest["numbers"], pred_nums, draws[:-1]
        )
        review_block["note"] = (
            f"上一份预测目标期为 {last_pred.get('target_period')}，"
            f"与最新开奖期 {latest['period']} 不一致，仅作参考对比"
        )
        adjust_notes = ["预测期号不一致，权重暂不按本次复盘强制调整"]

    # 每累计 50 期回测并优化
    bt = None
    if len(draws) >= 50:
        opt_w, bt = optimize_weights_by_backtest(draws, window=min(50, len(draws) - 15))
        if bt["avg_hit"] >= backtest(draws, min(50, len(draws) - 15), weights)["avg_hit"]:
            weights = save_weights(opt_w)
            adjust_notes.append(
                f"完成近 {bt['window_size']} 期回测优化：平均命中 {bt['avg_hit']}"
            )
        db.save_backtest(bt, _now())
    else:
        bt = backtest(draws, max(10, len(draws) // 2), weights)
        db.save_backtest(bt, _now())

    analysis = build_analysis(draws, weights)
    groups = predict_groups(draws, weights)
    long_term = analyze_long_term(draws)
    next_period = int(latest["period"]) + 1

    # 主推荐：方案2（反马尔可夫+热冷平衡）作为对外核心之一，同时保留综合组
    primary = groups["scheme2_am_hotcold"]

    pred_payload = {
        "target_period": next_period,
        "based_on_period": latest["period"],
        "created_at": _now(),
        "numbers": primary,
        "core": groups["core"],
        "groups": {k: v for k, v in groups.items()},
        "weights": weights,
    }
    save_last_prediction(pred_payload)
    for name, nums in groups.items():
        db.save_prediction(next_period, name, nums, weights, _now())

    report = render_report(
        draws=draws,
        analysis=analysis,
        groups=groups,
        long_term=long_term,
        weights=weights,
        adjust_notes=adjust_notes,
        review=review_block,
        backtest_result=bt,
        next_period=next_period,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    return {
        "draws": draws,
        "analysis": analysis,
        "groups": groups,
        "long_term": long_term,
        "weights": weights,
        "review": review_block,
        "backtest": bt,
        "next_period": next_period,
        "report": report,
        "report_path": str(REPORT_PATH),
    }


def render_report(
    draws: Sequence[Dict],
    analysis: Dict,
    groups: Dict[str, List[int]],
    long_term: Dict,
    weights: Dict[str, float],
    adjust_notes: Sequence[str],
    review: Optional[Dict],
    backtest_result: Optional[Dict],
    next_period: int,
) -> str:
    latest = draws[-1]
    lines: List[str] = []
    a = lines.append
    a("━━━━━━━━━━━━")
    a(f"快乐8 第{next_period}期预测")
    a("（研究用途：无法保证中奖，请理性看待命中概率）")
    a("")
    a("一、上一期复盘：")
    a(f"开奖期号：{latest['period']}（{latest['date']}）")
    a(f"开奖号码：{fmt_nums(latest['numbers'])}")
    if review:
        a(f"预测号码：{fmt_nums(review['pred_numbers'])}")
        a(f"命中：{fmt_nums(review['hits']) if review['hits'] else '无'}")
        a(f"命中率：{review['hit_rate']}")
        if review.get("note"):
            a(f"说明：{review['note']}")
        a("失败原因：")
        for r in review["primary_reasons"]:
            a(f"  - {r}（评分 {review['fail_reasons'].get(r, 0)}）")
    else:
        a("预测号码：无（首次建模，尚无上一期预测可复盘）")
        a("命中：—")
        a("命中率：—")
        a("失败原因：—")
    a("")
    a("━━━━━━━━━━━━")
    a("")
    a("二、模型调整：")
    a("调整内容：")
    for note in adjust_notes:
        a(f"  - {note}")
    a("新的权重：")
    for k, v in weights.items():
        a(f"  {WEIGHT_NAMES.get(k, k)}：{v*100:.1f}%")
    if backtest_result:
        a("")
        a(
            f"回测（近{backtest_result['window_size']}期，期望随机命中约 "
            f"{backtest_result.get('detail', {}).get('expected_random', 2.5)}）："
        )
        a(f"  平均命中：{backtest_result['avg_hit']}")
        a(f"  最高命中：{backtest_result['max_hit']}")
        a(f"  最低命中：{backtest_result['min_hit']}")
        a(f"  10中10次数：{backtest_result['hit10']}")
        a(f"  9中10次数：{backtest_result['hit9']}")
        a(f"  8中10次数：{backtest_result['hit8']}")
    a("")
    a("━━━━━━━━━━━━")
    a("")
    a("三、长期守号：")
    a(fmt_nums(long_term["numbers"]))
    a(f"评分：{long_term['score']} / 100")
    a(f"变化趋势：{long_term['trend']}（上升{long_term['up']} / 下降{long_term['down']}）")
    a("明细：")
    for it in long_term["items"]:
        a(
            f"  {it['number']:02d} | 出现{it['appear_all']}次 "
            f"(近20:{it['appear_20']}, 近10:{it['appear_10']}) | "
            f"遗漏{it['gap']}期 | {it['trend_label']} | "
            f"评分{it['score']} | {it['note']}"
        )
    a("")
    a("━━━━━━━━━━━━")
    a("")
    a("四、下一期预测")
    a(f"目标期号：{next_period}")
    a(f"基于历史：近 {analysis['periods']} 期（最新开奖 {analysis['latest_period']}）")
    a("")
    a(f"方案1 反马尔可夫链：{fmt_nums(groups['scheme1_anti_markov'])}")
    a(f"方案2 反马尔可夫链+热冷平衡：{fmt_nums(groups['scheme2_am_hotcold'])}")
    a(f"方案3 反马尔可夫链+冷号回补：{fmt_nums(groups['scheme3_am_cold'])}")
    a("")
    a("【四组预测】")
    a(f"第一组 综合模型推荐：{fmt_nums(groups['group1_ensemble'])}")
    a(f"第二组 冷号回补模型：{fmt_nums(groups['group2_cold'])}")
    a(f"第三组 热号延续模型：{fmt_nums(groups['group3_hot'])}")
    a(f"第四组 机器随机优化：{fmt_nums(groups['group4_random_opt'])}")
    a("")
    a(f"推荐10个核心号码：{fmt_nums(sorted(analysis['core10']))}")
    a("")
    a("━━━━━━━━━━━━")
    a("")
    a("五、最近100期数据分析")
    a(f"样本期数：{analysis['periods']}")
    a("号码出现次数 TOP10：")
    for n, c in analysis["freq_rank"][:10]:
        a(f"  {n:02d}: {c}")
    a("号码出现次数 BOTTOM10：")
    for n, c in analysis["freq_rank"][-10:]:
        a(f"  {n:02d}: {c}")
    a("当前遗漏 TOP10：")
    for n, g in analysis["gap_rank"][:10]:
        a(f"  {n:02d}: 遗漏 {g} 期")
    a("热号 TOP10（近20期）：")
    for n, c in analysis["hot_top10"]:
        a(f"  {n:02d}: {c} 次")
    a("冷号 TOP10（按遗漏）：")
    for n, g in analysis["cold_top10"]:
        a(f"  {n:02d}: 遗漏 {g} 期")
    a("近期上涨号码：")
    a("  " + fmt_nums([n for n, _ in analysis["rising"]]))
    a("近期下降号码：")
    a("  " + fmt_nums([n for n, _ in analysis["falling"]]))
    a("")
    a("号码综合评分 TOP20：")
    ranked = sorted(
        analysis["scores"].keys(),
        key=lambda n: analysis["scores"][n]["total"],
        reverse=True,
    )
    for n in ranked[:20]:
        s = analysis["scores"][n]
        a(
            f"  {n:02d}: {s['total']:.1f} "
            f"(频{s['freq']:.0f}/遗{s['gap']:.0f}/冷热{s['hotcold']:.0f}/"
            f"区{s['zone']:.0f}/形态{s['oddeven_size']:.0f}/连{s['consec']:.0f})"
        )
    a("")
    a("━━━━━━━━━━━━")
    a("数据库位置：data/kl8/kl8.db")
    a("报告文件：data/kl8/latest_report.txt")
    a("━━━━━━━━━━━━")
    return "\n".join(lines) + "\n"


def main() -> None:
    result = run_pipeline()
    print(result["report"])


if __name__ == "__main__":
    main()
