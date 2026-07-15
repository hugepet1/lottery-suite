#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
排列3 / 3D 直选预测（多算法集成 + 对比修正）

玩法：1号位/2号位/3号位，各 0-9；数字相同且顺序一致才中奖。
录入对比后，按各算法准确率从高到低自动修正权重。
"""

from __future__ import annotations

import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # lottery_suite/ is parent of pl3/ or dlt/
    return Path(__file__).resolve().parent.parent


def bundled_data_dir() -> Path | None:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "data" / "pl3"
    return None


ROOT = app_root()
DATA_DIR = ROOT / "data" / "pl3"
DATA_PATH = DATA_DIR / "history.json"
MODEL_PATH = DATA_DIR / "model.json"
LAST_PRED_PATH = DATA_DIR / "last_prediction.json"
COMPARE_LOG_PATH = DATA_DIR / "compare_log.json"
WEIGHTS_PATH = DATA_DIR / "algo_weights.json"

Digits = Tuple[int, int, int]
Draw = Dict[str, object]

DIGIT_KEYS = ["freq", "pos_markov", "softmax", "gap", "ema", "ar", "bayes", "skip", "rnn"]
COMBO_KEYS = ["digit_blend", "markov", "pattern", "montecarlo"]
ALGO_NAMES = {
    "freq": "加权频率",
    "pos_markov": "逐位马尔可夫",
    "softmax": "Softmax回归",
    "gap": "遗漏冷热",
    "ema": "EMA趋势",
    "ar": "AR自回归",
    "bayes": "贝叶斯融合",
    "skip": "间隔规律",
    "rnn": "迷你RNN",
    "digit_blend": "各位融合分",
    "markov": "整注马尔可夫",
    "pattern": "形态匹配",
    "montecarlo": "蒙特卡洛",
}
DEFAULT_DIGIT_WEIGHTS = {
    "freq": 0.14,
    "pos_markov": 0.12,
    "softmax": 0.14,
    "gap": 0.12,
    "ema": 0.10,
    "ar": 0.10,
    "bayes": 0.10,
    "skip": 0.08,
    "rnn": 0.10,
}
DEFAULT_COMBO_WEIGHTS = {
    "digit_blend": 0.40,
    "markov": 0.20,
    "pattern": 0.20,
    "montecarlo": 0.20,
}


def ensure_data_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists():
        return
    seed = bundled_data_dir()
    if seed and (seed / "history.json").exists():
        shutil.copy2(seed / "history.json", DATA_PATH)


def load_history(path: Path = DATA_PATH) -> List[Draw]:
    ensure_data_files()
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    draws: List[Draw] = []
    for item in data:
        digits = tuple(int(x) for x in item["digits"])
        if len(digits) != 3 or any(d < 0 or d > 9 for d in digits):
            raise ValueError(f"非法号码: {item}")
        draws.append({"period": int(item["period"]), "digits": list(digits)})
    draws.sort(key=lambda x: int(x["period"]))
    return draws


def save_history(draws: Sequence[Draw], path: Path = DATA_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(draws, key=lambda x: int(x["period"]))
    with path.open("w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)


def add_draw(period: int, p1: int, p2: int, p3: int) -> List[Draw]:
    draws = load_history()
    for d in draws:
        if int(d["period"]) == period:
            raise ValueError(f"期号 {period} 已存在: {d['digits']}")
    for x in (p1, p2, p3):
        if x < 0 or x > 9:
            raise ValueError("每位数字必须是 0-9")
    draws.append({"period": period, "digits": [p1, p2, p3]})
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
    removed, kept = draws[-1], draws[:-1]
    save_history(kept)
    return removed, kept


def clear_all_history() -> None:
    save_history([])


def fmt_digits(digits: Sequence[int]) -> str:
    return " ".join(str(int(d)) for d in digits)


def softmax(scores: Dict[object, float], temperature: float = 1.0) -> Dict[object, float]:
    if not scores:
        return {}
    t = max(temperature, 1e-6)
    mx = max(scores.values())
    exps = {k: math.exp((v - mx) / t) for k, v in scores.items()}
    s = sum(exps.values()) or 1.0
    return {k: v / s for k, v in exps.items()}


def normalize(scores: Dict[object, float]) -> Dict[object, float]:
    s = sum(max(0.0, v) for v in scores.values())
    if s <= 0:
        n = len(scores) or 1
        return {k: 1.0 / n for k in scores}
    return {k: max(0.0, v) / s for k, v in scores.items()}


def digits_key(digits: Sequence[int]) -> str:
    return "".join(str(int(d)) for d in digits)


def parse_key(key: str) -> Digits:
    return int(key[0]), int(key[1]), int(key[2])


def combo_product(probs: List[Dict[int, float]], combo: Digits) -> float:
    return probs[0][combo[0]] * probs[1][combo[1]] * probs[2][combo[2]]


def _clamp_normalize(weights: Dict[str, float], min_w: float = 0.04) -> Dict[str, float]:
    clamped = {k: max(min_w, float(v)) for k, v in weights.items()}
    s = sum(clamped.values()) or 1.0
    return {k: v / s for k, v in clamped.items()}


def load_algo_weights(path: Path = WEIGHTS_PATH) -> dict:
    data = {
        "digit": dict(DEFAULT_DIGIT_WEIGHTS),
        "combo": dict(DEFAULT_COMBO_WEIGHTS),
        "updates": 0,
        "history": [],
    }
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            for k in DIGIT_KEYS:
                if k in raw.get("digit", {}):
                    data["digit"][k] = float(raw["digit"][k])
            for k in COMBO_KEYS:
                if k in raw.get("combo", {}):
                    data["combo"][k] = float(raw["combo"][k])
            data["updates"] = int(raw.get("updates", 0))
            data["history"] = list(raw.get("history", []))[-50:]
        except Exception:
            pass
    data["digit"] = _clamp_normalize(data["digit"], 0.04)
    data["combo"] = _clamp_normalize(data["combo"], 0.05)
    return data


def save_algo_weights(data: dict, path: Path = WEIGHTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def reset_algo_weights(path: Path = WEIGHTS_PATH) -> dict:
    data = {
        "digit": dict(DEFAULT_DIGIT_WEIGHTS),
        "combo": dict(DEFAULT_COMBO_WEIGHTS),
        "updates": 0,
        "history": [],
    }
    save_algo_weights(data, path)
    return data


# ---------------------------------------------------------------------------
# 子模型
# ---------------------------------------------------------------------------

class PositionFrequencyModel:
    def __init__(self, half_life: float = 18.0):
        self.half_life = half_life
        self.pos_counts = [Counter() for _ in range(3)]

    def fit(self, draws: Sequence[Draw]) -> None:
        self.pos_counts = [Counter() for _ in range(3)]
        n = len(draws)
        for i, draw in enumerate(draws):
            w = 0.5 ** ((n - 1 - i) / self.half_life)
            for p in range(3):
                self.pos_counts[p][int(draw["digits"][p])] += w

    def digit_probs(self) -> List[Dict[int, float]]:
        return [
            normalize({d: float(self.pos_counts[p].get(d, 0.0)) + 0.5 for d in range(10)})
            for p in range(3)
        ]

    def score_combo(self, combo: Digits) -> float:
        return combo_product(self.digit_probs(), combo)


class MarkovComboModel:
    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self.trans: Dict[str, Counter] = defaultdict(Counter)
        self.global_next = Counter()
        self.last_key: Optional[str] = None

    def fit(self, draws: Sequence[Draw]) -> None:
        self.trans = defaultdict(Counter)
        self.global_next = Counter()
        self.last_key = None
        if not draws:
            return
        keys = [digits_key(d["digits"]) for d in draws]
        for i in range(len(keys) - 1):
            self.trans[keys[i]][keys[i + 1]] += 1
            self.global_next[keys[i + 1]] += 1
        self.last_key = keys[-1]

    def predict_distribution(self) -> Dict[str, float]:
        if self.last_key is None:
            return {}
        row = self.trans.get(self.last_key, Counter())
        candidates = set(row) | set(self.global_next)
        if not candidates:
            return {}
        scores = {
            c: row.get(c, 0) + self.alpha * (self.global_next.get(c, 0) + 1) for c in candidates
        }
        return normalize(scores)


class PositionMarkovModel:
    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha
        self.trans = [defaultdict(Counter) for _ in range(3)]
        self.last: Optional[Digits] = None

    def fit(self, draws: Sequence[Draw]) -> None:
        self.trans = [defaultdict(Counter) for _ in range(3)]
        self.last = None
        if not draws:
            return
        seq = [tuple(int(x) for x in d["digits"]) for d in draws]
        for i in range(len(seq) - 1):
            for p in range(3):
                self.trans[p][seq[i][p]][seq[i + 1][p]] += 1
        self.last = seq[-1]

    def digit_probs(self) -> List[Dict[int, float]]:
        if self.last is None:
            return [{d: 0.1 for d in range(10)} for _ in range(3)]
        out = []
        for p in range(3):
            row = self.trans[p].get(self.last[p], Counter())
            out.append(normalize({d: row.get(d, 0) + self.alpha for d in range(10)}))
        return out

    def score_combo(self, combo: Digits) -> float:
        return combo_product(self.digit_probs(), combo)


class SoftmaxDigitModel:
    def __init__(self, window: int = 5, lr: float = 0.08, epochs: int = 40, seed: int = 42):
        self.window = window
        self.lr = lr
        self.epochs = epochs
        self.rng = random.Random(seed)
        self.weights = None
        self.bias = None
        self.feat_dim = 0
        self.history_digits: List[Digits] = []

    def _feat(self, history: Sequence[Digits]) -> List[float]:
        feats: List[float] = []
        for digs in history:
            for d in digs:
                onehot = [0.0] * 10
                onehot[d] = 1.0
                feats.extend(onehot)
            feats.append(sum(digs) / 27.0)
            feats.append((max(digs) - min(digs)) / 9.0)
            feats.append(sum(x % 2 for x in digs) / 3.0)
        feats.extend(d / 9.0 for d in history[-1])
        return feats

    def fit(self, draws: Sequence[Draw]) -> None:
        seq = [tuple(int(x) for x in d["digits"]) for d in draws]
        self.history_digits = seq
        if len(seq) <= self.window:
            self.weights = None
            return
        xs, ys = [], []
        for i in range(self.window, len(seq)):
            xs.append(self._feat(seq[i - self.window : i]))
            ys.append(seq[i])
        self.feat_dim = len(xs[0])
        self.weights = [
            [[self.rng.uniform(-0.05, 0.05) for _ in range(self.feat_dim)] for _ in range(10)]
            for _ in range(3)
        ]
        self.bias = [[0.0] * 10 for _ in range(3)]
        for _ in range(self.epochs):
            order = list(range(len(xs)))
            self.rng.shuffle(order)
            for idx in order:
                x, y = xs[idx], ys[idx]
                for pos in range(3):
                    logits = [
                        self.bias[pos][c]
                        + sum(self.weights[pos][c][j] * x[j] for j in range(self.feat_dim))
                        for c in range(10)
                    ]
                    mx = max(logits)
                    exps = [math.exp(v - mx) for v in logits]
                    den = sum(exps) or 1.0
                    probs = [e / den for e in exps]
                    for c in range(10):
                        grad = probs[c] - (1.0 if c == y[pos] else 0.0)
                        self.bias[pos][c] -= self.lr * grad
                        for j in range(self.feat_dim):
                            self.weights[pos][c][j] -= self.lr * grad * x[j]

    def digit_probs(self) -> List[Dict[int, float]]:
        if self.weights is None or len(self.history_digits) < self.window:
            return [{d: 0.1 for d in range(10)} for _ in range(3)]
        x = self._feat(self.history_digits[-self.window :])
        out = []
        for pos in range(3):
            logits = [
                self.bias[pos][c] + sum(self.weights[pos][c][j] * x[j] for j in range(self.feat_dim))
                for c in range(10)
            ]
            mx = max(logits)
            exps = [math.exp(v - mx) for v in logits]
            den = sum(exps) or 1.0
            out.append({c: exps[c] / den for c in range(10)})
        return out

    def score_combo(self, combo: Digits) -> float:
        return combo_product(self.digit_probs(), combo)


class GapHotColdModel:
    def fit(self, draws: Sequence[Draw]) -> None:
        n = len(draws)
        window = min(20, n)
        self.gaps = [{d: n for d in range(10)} for _ in range(3)]
        self.recent = [Counter() for _ in range(3)]
        for i, draw in enumerate(draws):
            for p in range(3):
                d = int(draw["digits"][p])
                self.gaps[p][d] = n - 1 - i
                if i >= n - window:
                    self.recent[p][d] += 1

    def digit_probs(self) -> List[Dict[int, float]]:
        out = []
        for p in range(3):
            scores = {
                d: 1.0 + 0.15 * min(self.gaps[p][d], 12) + 0.35 * self.recent[p].get(d, 0)
                for d in range(10)
            }
            out.append(normalize(scores))
        return out

    def score_combo(self, combo: Digits) -> float:
        return combo_product(self.digit_probs(), combo)


class PatternModel:
    def fit(self, draws: Sequence[Draw]) -> None:
        self.sum_hist = Counter()
        self.oe_hist = Counter()
        self.span_hist = Counter()
        n = len(draws)
        for i, draw in enumerate(draws):
            digs = [int(x) for x in draw["digits"]]
            w = 0.5 ** ((n - 1 - i) / 25.0)
            self.sum_hist[sum(digs)] += w
            self.oe_hist[sum(x % 2 for x in digs)] += w
            self.span_hist[max(digs) - min(digs)] += w
        self.sum_p = normalize({k: float(v) for k, v in self.sum_hist.items()})
        self.oe_p = normalize({k: float(v) for k, v in self.oe_hist.items()})
        self.span_p = normalize({k: float(v) for k, v in self.span_hist.items()})

    def score_combo(self, combo: Digits) -> float:
        return (
            self.sum_p.get(sum(combo), 1e-4)
            * self.oe_p.get(sum(x % 2 for x in combo), 1e-4)
            * self.span_p.get(max(combo) - min(combo), 1e-4)
        )


class EMATrendModel:
    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha
        self.ema = [{d: 0.1 for d in range(10)} for _ in range(3)]

    def fit(self, draws: Sequence[Draw]) -> None:
        self.ema = [{d: 0.1 for d in range(10)} for _ in range(3)]
        a = self.alpha
        for draw in draws:
            for p in range(3):
                hit = int(draw["digits"][p])
                for d in range(10):
                    self.ema[p][d] = a * (1.0 if d == hit else 0.0) + (1 - a) * self.ema[p][d]

    def digit_probs(self) -> List[Dict[int, float]]:
        return [normalize(dict(self.ema[p])) for p in range(3)]

    def score_combo(self, combo: Digits) -> float:
        return combo_product(self.digit_probs(), combo)


class AutoRegressiveModel:
    def __init__(self, order: int = 3):
        self.order = order
        self.coefs = [[0.5, 0.3, 0.2] for _ in range(3)]
        self.seq: List[Digits] = []

    def fit(self, draws: Sequence[Draw]) -> None:
        self.seq = [tuple(int(x) for x in d["digits"]) for d in draws]
        o = self.order
        if len(self.seq) <= o + 2:
            return
        for p in range(3):
            ys = [self.seq[i][p] for i in range(o, len(self.seq))]
            xs = [[self.seq[i - k - 1][p] for k in range(o)] for i in range(o, len(self.seq))]
            xtx = [[0.0] * o for _ in range(o)]
            xty = [0.0] * o
            for row, y in zip(xs, ys):
                for i in range(o):
                    xty[i] += row[i] * y
                    for j in range(o):
                        xtx[i][j] += row[i] * row[j]
            for i in range(o):
                xtx[i][i] += 1.0
            a = [r[:] + [xty[i]] for i, r in enumerate(xtx)]
            for i in range(o):
                piv = a[i][i] or 1e-9
                for j in range(i, o + 1):
                    a[i][j] /= piv
                for k in range(o):
                    if k == i:
                        continue
                    factor = a[k][i]
                    for j in range(i, o + 1):
                        a[k][j] -= factor * a[i][j]
            self.coefs[p] = [a[i][o] for i in range(o)]

    def digit_probs(self) -> List[Dict[int, float]]:
        if len(self.seq) < self.order:
            return [{d: 0.1 for d in range(10)} for _ in range(3)]
        out = []
        for p in range(3):
            pred = sum(c * self.seq[-(k + 1)][p] for k, c in enumerate(self.coefs[p]))
            pred = max(0.0, min(9.0, pred))
            out.append(normalize({d: math.exp(-((d - pred) ** 2) / 2.2) for d in range(10)}))
        return out

    def score_combo(self, combo: Digits) -> float:
        return combo_product(self.digit_probs(), combo)


class BayesianFusionModel:
    def __init__(self, prior: float = 1.0):
        self.prior = prior
        self.counts = [Counter() for _ in range(3)]

    def fit(self, draws: Sequence[Draw]) -> None:
        self.counts = [Counter() for _ in range(3)]
        for i, draw in enumerate(draws):
            w = 1.0 + 0.02 * i
            for p in range(3):
                self.counts[p][int(draw["digits"][p])] += w

    def digit_probs(self) -> List[Dict[int, float]]:
        return [
            normalize({d: self.prior + float(self.counts[p].get(d, 0.0)) for d in range(10)})
            for p in range(3)
        ]

    def score_combo(self, combo: Digits) -> float:
        return combo_product(self.digit_probs(), combo)


class SkipIntervalModel:
    def fit(self, draws: Sequence[Draw]) -> None:
        self.seq = [tuple(int(x) for x in d["digits"]) for d in draws]
        self.interval_pref = [Counter() for _ in range(3)]
        last_pos = [{d: None for d in range(10)} for _ in range(3)]
        for i, digs in enumerate(self.seq):
            for p in range(3):
                d = digs[p]
                if last_pos[p][d] is not None:
                    self.interval_pref[p][i - last_pos[p][d]] += 1
                last_pos[p][d] = i
        self.current_gap = [
            {d: (len(self.seq) - last_pos[p][d]) if last_pos[p][d] is not None else len(self.seq) for d in range(10)}
            for p in range(3)
        ]

    def digit_probs(self) -> List[Dict[int, float]]:
        out = []
        for p in range(3):
            total = sum(self.interval_pref[p].values()) or 1
            scores = {}
            for d in range(10):
                g = self.current_gap[p][d]
                scores[d] = (
                    0.2
                    + self.interval_pref[p].get(g, 0) / total
                    + 0.05 * self.interval_pref[p].get(g - 1, 0) / total
                    + 0.05 * self.interval_pref[p].get(g + 1, 0) / total
                )
            out.append(normalize(scores))
        return out

    def score_combo(self, combo: Digits) -> float:
        return combo_product(self.digit_probs(), combo)


class MiniRNNModel:
    def __init__(self, hidden: int = 12, lr: float = 0.05, epochs: int = 20, seed: int = 7):
        self.h = hidden
        self.lr = lr
        self.epochs = epochs
        self.rng = random.Random(seed)
        self.seq: List[Digits] = []
        self.Wxh = self.Whh = self.Why = self.bh = self.by = None

    def _init_params(self) -> None:
        r = self.rng.uniform
        self.Wxh = [[r(-0.1, 0.1) for _ in range(30)] for _ in range(self.h)]
        self.Whh = [[r(-0.1, 0.1) for _ in range(self.h)] for _ in range(self.h)]
        self.Why = [[r(-0.1, 0.1) for _ in range(self.h)] for _ in range(30)]
        self.bh = [0.0] * self.h
        self.by = [0.0] * 30

    def _encode(self, digs: Digits) -> List[float]:
        v = [0.0] * 30
        for p, d in enumerate(digs):
            v[p * 10 + d] = 1.0
        return v

    def _tanh(self, x: float) -> float:
        if x > 20:
            return 1.0
        if x < -20:
            return -1.0
        e2 = math.exp(2 * x)
        return (e2 - 1) / (e2 + 1)

    def _forward(self, inputs: Sequence[List[float]]):
        hprev = [0.0] * self.h
        hs = []
        for x in inputs:
            hnew = []
            for i in range(self.h):
                s = self.bh[i]
                s += sum(self.Wxh[i][j] * x[j] for j in range(30))
                s += sum(self.Whh[i][j] * hprev[j] for j in range(self.h))
                hnew.append(self._tanh(s))
            hprev = hnew
            hs.append(hprev)
        return hs

    def fit(self, draws: Sequence[Draw]) -> None:
        self.seq = [tuple(int(x) for x in d["digits"]) for d in draws]
        if len(self.seq) < 6:
            self.Wxh = None
            return
        self._init_params()
        win = 4
        for _ in range(self.epochs):
            for i in range(win, len(self.seq)):
                xs = [self._encode(self.seq[j]) for j in range(i - win, i)]
                target = self.seq[i]
                hs = self._forward(xs)
                hlast = hs[-1]
                for p in range(3):
                    logits = [
                        self.by[p * 10 + c]
                        + sum(self.Why[p * 10 + c][j] * hlast[j] for j in range(self.h))
                        for c in range(10)
                    ]
                    mx = max(logits)
                    exps = [math.exp(v - mx) for v in logits]
                    den = sum(exps) or 1.0
                    probs = [e / den for e in exps]
                    for c in range(10):
                        grad = probs[c] - (1.0 if c == target[p] else 0.0)
                        idx = p * 10 + c
                        self.by[idx] -= self.lr * grad
                        for j in range(self.h):
                            self.Why[idx][j] -= self.lr * grad * hlast[j]

    def digit_probs(self) -> List[Dict[int, float]]:
        if self.Wxh is None or len(self.seq) < 4:
            return [{d: 0.1 for d in range(10)} for _ in range(3)]
        xs = [self._encode(self.seq[j]) for j in range(len(self.seq) - 4, len(self.seq))]
        hlast = self._forward(xs)[-1]
        out = []
        for p in range(3):
            logits = [
                self.by[p * 10 + c] + sum(self.Why[p * 10 + c][j] * hlast[j] for j in range(self.h))
                for c in range(10)
            ]
            mx = max(logits)
            exps = [math.exp(v - mx) for v in logits]
            den = sum(exps) or 1.0
            out.append({c: exps[c] / den for c in range(10)})
        return out

    def score_combo(self, combo: Digits) -> float:
        return combo_product(self.digit_probs(), combo)


class MonteCarloModel:
    def __init__(self, samples: int = 600, seed: int = 99):
        self.samples = samples
        self.seed = seed
        self.hit = Counter()

    def fit_from_probs(self, probs: List[Dict[int, float]]) -> None:
        rng = random.Random(self.seed)
        self.hit = Counter()
        for _ in range(self.samples):
            combo = []
            for p in range(3):
                keys = list(range(10))
                weights = [max(1e-9, probs[p][d] * rng.uniform(0.7, 1.3)) for d in keys]
                s = sum(weights)
                r = rng.random() * s
                acc = 0.0
                chosen = 0
                for d, w in zip(keys, weights):
                    acc += w
                    if r <= acc:
                        chosen = d
                        break
                combo.append(chosen)
            self.hit[digits_key(combo)] += 1

    def score_combo(self, combo: Digits) -> float:
        return (self.hit.get(digits_key(combo), 0) + 1) / (self.samples + 1000)


# ---------------------------------------------------------------------------
# 集成
# ---------------------------------------------------------------------------

class EnsemblePredictor:
    def __init__(self, weights: Optional[dict] = None):
        self.freq = PositionFrequencyModel()
        self.markov = MarkovComboModel()
        self.pos_markov = PositionMarkovModel()
        self.nn = SoftmaxDigitModel(epochs=40)
        self.gap = GapHotColdModel()
        self.pattern = PatternModel()
        self.ema = EMATrendModel()
        self.ar = AutoRegressiveModel()
        self.bayes = BayesianFusionModel()
        self.skip = SkipIntervalModel()
        self.rnn = MiniRNNModel()
        self.mc = MonteCarloModel()
        self.draws: List[Draw] = []
        self.weights = weights if weights is not None else load_algo_weights()
        self.blend_probs: List[Dict[int, float]] = []

    def digit_sources(self) -> Dict[str, List[Dict[int, float]]]:
        return {
            "freq": self.freq.digit_probs(),
            "pos_markov": self.pos_markov.digit_probs(),
            "softmax": self.nn.digit_probs(),
            "gap": self.gap.digit_probs(),
            "ema": self.ema.digit_probs(),
            "ar": self.ar.digit_probs(),
            "bayes": self.bayes.digit_probs(),
            "skip": self.skip.digit_probs(),
            "rnn": self.rnn.digit_probs(),
        }

    def fit(self, draws: Sequence[Draw]) -> None:
        self.draws = list(draws)
        for m in (
            self.freq, self.markov, self.pos_markov, self.nn, self.gap,
            self.pattern, self.ema, self.ar, self.bayes, self.skip, self.rnn,
        ):
            m.fit(draws)
        sources = self.digit_sources()
        dw = self.weights["digit"]
        blend = []
        for p in range(3):
            scores = {d: 0.0 for d in range(10)}
            for key in DIGIT_KEYS:
                for d in range(10):
                    scores[d] += dw[key] * sources[key][p][d]
            blend.append(normalize(scores))
        self.blend_probs = blend
        self.mc.fit_from_probs(blend)

    def _candidate_pool(self, size: int = 500) -> List[Digits]:
        pool: set[Digits] = set()
        for d in self.draws:
            pool.add(tuple(int(x) for x in d["digits"]))  # type: ignore
        tops = []
        for p in range(3):
            tops.append(sorted(self.blend_probs[p], key=self.blend_probs[p].get, reverse=True)[:6])
        for a in tops[0]:
            for b in tops[1]:
                for c in tops[2]:
                    pool.add((a, b, c))
        for key in self.markov.predict_distribution():
            pool.add(parse_key(key))
        for key, _ in self.mc.hit.most_common(40):
            pool.add(parse_key(key))
        rng = random.Random(7)
        while len(pool) < min(size, 1000):
            pool.add((rng.randint(0, 9), rng.randint(0, 9), rng.randint(0, 9)))
        return list(pool)

    def predict(self, top_n: int = 10) -> List[Tuple[str, float, Dict[str, float]]]:
        if len(self.draws) < 8:
            raise RuntimeError("历史数据太少，至少需要 8 期")
        markov_dist = self.markov.predict_distribution()
        cw = self.weights["combo"]
        scored = []
        for combo in self._candidate_pool():
            key = digits_key(combo)
            detail = {
                "digit_blend": combo_product(self.blend_probs, combo),
                "markov": markov_dist.get(key, 1e-9),
                "pattern": self.pattern.score_combo(combo),
                "montecarlo": self.mc.score_combo(combo),
            }
            score = (
                cw["digit_blend"] * math.log(detail["digit_blend"] + 1e-12)
                + cw["markov"] * math.log(detail["markov"] + 1e-12)
                + cw["pattern"] * math.log(detail["pattern"] + 1e-12)
                + cw["montecarlo"] * math.log(detail["montecarlo"] + 1e-12)
            )
            scored.append((key, score, detail))
        prob_map = softmax({k: s for k, s, _ in scored}, temperature=0.9)
        ranked = sorted(
            ((k, prob_map[k], d) for k, _, d in scored),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_n]

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "n_draws": len(self.draws),
            "last_period": self.draws[-1]["period"] if self.draws else None,
            "last_digits": self.draws[-1]["digits"] if self.draws else None,
            "weights": self.weights,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 对比 & 权重修正
# ---------------------------------------------------------------------------

def save_last_prediction(
    target_period: int,
    preds: Sequence[Tuple[str, float, Dict[str, float]]],
    based_on_period: Optional[int] = None,
) -> None:
    ensure_data_files()
    top = []
    for i, (key, prob, _) in enumerate(preds[:10], 1):
        top.append({"rank": i, "digits": [int(key[0]), int(key[1]), int(key[2])], "prob": float(prob)})
    payload = {
        "target_period": int(target_period),
        "based_on_period": based_on_period,
        "top": top,
    }
    with LAST_PRED_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_last_prediction() -> Optional[dict]:
    if not LAST_PRED_PATH.exists():
        return None
    with LAST_PRED_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _score_digit_algo(probs: List[Dict[int, float]], actual: Digits) -> dict:
    hits = sum(1 for p in range(3) if max(probs[p], key=probs[p].get) == actual[p])
    like = sum(probs[p].get(actual[p], 0) for p in range(3))
    pick = [max(probs[p], key=probs[p].get) for p in range(3)]
    return {
        "pos_hit": hits,
        "total_hit": hits,
        "accuracy": hits / 3.0,
        "likelihood": like,
        "pick": pick,
    }


def _score_combo_top10(
    scored: List[Tuple[str, float]],
    actual: Digits,
) -> dict:
    actual_key = digits_key(actual)
    ranked = sorted(scored, key=lambda x: x[1], reverse=True)[:10]
    best_hit = -1
    best_key = ""
    totals = []
    exact = 0
    for key, _ in ranked:
        digs = parse_key(key)
        hit = sum(1 for i in range(3) if digs[i] == actual[i])
        totals.append(hit)
        if key == actual_key:
            exact = 1
        if hit > best_hit:
            best_hit = hit
            best_key = key
    return {
        "pos_hit": best_hit if best_hit >= 0 else 0,
        "total_hit": best_hit if best_hit >= 0 else 0,
        "accuracy": (best_hit / 3.0) if best_hit >= 0 else 0.0,
        "likelihood": (sum(totals) / len(totals)) if totals else 0.0,
        "exact_in_top10": exact,
        "pick": list(parse_key(best_key)) if best_key else [0, 0, 0],
    }


def _target_weights_from_ranking(scores: Dict[str, dict], keys: Sequence[str]) -> Dict[str, float]:
    ordered = sorted(
        keys,
        key=lambda k: (scores[k]["accuracy"], scores[k].get("likelihood", 0.0)),
        reverse=True,
    )
    n = len(ordered)
    rank_raw = {k: float(n - i) for i, k in enumerate(ordered)}
    acc_raw = {k: scores[k]["accuracy"] + 0.08 for k in keys}
    mixed = {
        k: 0.55 * normalize(acc_raw)[k] + 0.45 * normalize(rank_raw)[k] for k in keys
    }
    return normalize(mixed)


def correct_algorithms_from_draw(
    actual_digits: Sequence[int],
    actual_period: Optional[int] = None,
    ema_alpha: float = 0.40,
    force: bool = False,
) -> dict:
    old = load_algo_weights()
    already = (
        not force
        and actual_period is not None
        and bool(old.get("history"))
        and old["history"][-1].get("period") == actual_period
    )
    history = load_history()
    if actual_period is not None:
        train = [d for d in history if int(d["period"]) < int(actual_period)]
    else:
        train = history[:-1] if history else []
    if len(train) < 8:
        raise ValueError("用于算法修正的历史不足 8 期。")

    actual = tuple(int(x) for x in actual_digits)
    model = EnsemblePredictor(
        weights={
            "digit": dict(DEFAULT_DIGIT_WEIGHTS),
            "combo": dict(DEFAULT_COMBO_WEIGHTS),
            "updates": 0,
            "history": [],
        }
    )
    model.fit(train)

    digit_scores = {
        k: _score_digit_algo(probs, actual) for k, probs in model.digit_sources().items()
    }

    cands = model._candidate_pool()
    markov_dist = model.markov.predict_distribution()
    combo_lists: Dict[str, List[Tuple[str, float]]] = {k: [] for k in COMBO_KEYS}
    for combo in cands:
        key = digits_key(combo)
        combo_lists["digit_blend"].append((key, combo_product(model.blend_probs, combo)))
        combo_lists["markov"].append((key, markov_dist.get(key, 1e-9)))
        combo_lists["pattern"].append((key, model.pattern.score_combo(combo)))
        combo_lists["montecarlo"].append((key, model.mc.score_combo(combo)))
    combo_scores = {k: _score_combo_top10(combo_lists[k], actual) for k in COMBO_KEYS}

    digit_rank = sorted(
        DIGIT_KEYS,
        key=lambda k: (digit_scores[k]["accuracy"], digit_scores[k]["likelihood"]),
        reverse=True,
    )
    combo_rank = sorted(
        COMBO_KEYS,
        key=lambda k: (combo_scores[k]["accuracy"], combo_scores[k]["likelihood"]),
        reverse=True,
    )

    if already:
        new_digit, new_combo = dict(old["digit"]), dict(old["combo"])
        updates = int(old.get("updates", 0))
    else:
        tn = _target_weights_from_ranking(digit_scores, DIGIT_KEYS)
        tc = _target_weights_from_ranking(combo_scores, COMBO_KEYS)
        new_digit = {
            k: (1 - ema_alpha) * old["digit"][k] + ema_alpha * tn[k] for k in DIGIT_KEYS
        }
        new_combo = {
            k: (1 - ema_alpha) * old["combo"][k] + ema_alpha * tc[k] for k in COMBO_KEYS
        }
        new_digit = _clamp_normalize(new_digit, 0.04)
        new_combo = _clamp_normalize(new_combo, 0.05)
        updates = int(old.get("updates", 0)) + 1
        payload = {
            "digit": new_digit,
            "combo": new_combo,
            "updates": updates,
            "history": list(old.get("history", [])),
        }
        payload["history"].append(
            {
                "period": actual_period,
                "digit_rank": [ALGO_NAMES[k] for k in digit_rank],
                "combo_rank": [ALGO_NAMES[k] for k in combo_rank],
                "digit_weights": new_digit,
                "combo_weights": new_combo,
            }
        )
        payload["history"] = payload["history"][-50:]
        save_algo_weights(payload)

    ranking_rows = []
    for i, k in enumerate(digit_rank, 1):
        ranking_rows.append(
            {
                "group": "各位算法",
                "rank": i,
                "key": k,
                "name": ALGO_NAMES[k],
                "accuracy": digit_scores[k]["accuracy"],
                "total_hit": digit_scores[k]["total_hit"],
                "old_weight": old["digit"][k],
                "new_weight": new_digit[k],
                "delta": new_digit[k] - old["digit"][k],
                "pick": digit_scores[k]["pick"],
            }
        )
    for i, k in enumerate(combo_rank, 1):
        ranking_rows.append(
            {
                "group": "组合算法",
                "rank": i,
                "key": k,
                "name": ALGO_NAMES[k],
                "accuracy": combo_scores[k]["accuracy"],
                "total_hit": combo_scores[k]["total_hit"],
                "old_weight": old["combo"][k],
                "new_weight": new_combo[k],
                "delta": new_combo[k] - old["combo"][k],
                "pick": combo_scores[k]["pick"],
            }
        )

    return {
        "digit_rank": digit_rank,
        "combo_rank": combo_rank,
        "ranking_rows": ranking_rows,
        "old_weights": {"digit": old["digit"], "combo": old["combo"]},
        "new_weights": {"digit": new_digit, "combo": new_combo},
        "updates": updates,
        "skipped_duplicate": already,
    }


def compare_prediction_to_draw(
    actual_digits: Sequence[int],
    prediction: Optional[dict] = None,
    actual_period: Optional[int] = None,
    correct_weights: bool = True,
) -> dict:
    pred = prediction if prediction is not None else load_last_prediction()
    if not pred or not pred.get("top"):
        raise ValueError("没有可对比的上次预测，请先点击「开始预测」生成 Top10。")

    actual = [int(x) for x in actual_digits]
    rows = []
    for item in pred["top"][:10]:
        digs = [int(x) for x in item["digits"]]
        pos_hit = sum(1 for i in range(3) if digs[i] == actual[i])
        exact = digs == actual
        rows.append(
            {
                "rank": int(item["rank"]),
                "digits": digs,
                "pos_hit": pos_hit,
                "exact": exact,
                "accuracy": pos_hit / 3.0,
                "hit_pos": [i + 1 for i in range(3) if digs[i] == actual[i]],
            }
        )

    best = max(rows, key=lambda r: (r["exact"], r["pos_hit"], -r["rank"]))
    avg_hit = sum(r["pos_hit"] for r in rows) / len(rows)
    avg_acc = sum(r["accuracy"] for r in rows) / len(rows)
    exact_count = sum(1 for r in rows if r["exact"])

    period_match = True
    target_period = pred.get("target_period")
    if actual_period is not None and target_period is not None:
        period_match = int(actual_period) == int(target_period)

    correction = None
    if correct_weights:
        try:
            correction = correct_algorithms_from_draw(actual, actual_period=actual_period)
        except ValueError:
            correction = None

    result = {
        "target_period": target_period,
        "actual_period": actual_period,
        "period_match": period_match,
        "actual_digits": actual,
        "rows": rows,
        "best": best,
        "summary": {
            "top_n": len(rows),
            "avg_pos_hit": avg_hit,
            "avg_accuracy": avg_acc,
            "best_pos_hit": best["pos_hit"],
            "best_accuracy": best["accuracy"],
            "best_rank": best["rank"],
            "exact_in_top10": exact_count,
            "exact_hit": best["exact"],
        },
        "correction": correction,
    }

    log = []
    if COMPARE_LOG_PATH.exists():
        try:
            with COMPARE_LOG_PATH.open("r", encoding="utf-8") as f:
                log = json.load(f)
        except Exception:
            log = []
    entry = {
        "actual_period": actual_period,
        "target_period": target_period,
        "best_pos_hit": best["pos_hit"],
        "best_accuracy": best["accuracy"],
        "avg_accuracy": avg_acc,
        "exact_in_top10": exact_count,
    }
    if correction:
        entry["digit_rank"] = [ALGO_NAMES[k] for k in correction["digit_rank"]]
        entry["combo_rank"] = [ALGO_NAMES[k] for k in correction["combo_rank"]]
    log.append(entry)
    with COMPARE_LOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(log[-100:], f, ensure_ascii=False, indent=2)
    return result


def format_compare_text(result: dict) -> str:
    s = result["summary"]
    lines = [
        f"开奖期号：{result.get('actual_period')}    预测目标期号：{result.get('target_period')}",
        f"开奖号码：1号位={result['actual_digits'][0]} 2号位={result['actual_digits'][1]} 3号位={result['actual_digits'][2]}",
        "",
        f"【Top10 最佳一注】第{s['best_rank']}名：位对位命中 {s['best_pos_hit']}/3  准确率 {s['best_accuracy']:.1%}"
        + ("  【整注命中】" if s["exact_hit"] else ""),
        f"【Top10 平均】位命中 {s['avg_pos_hit']:.2f}/3  平均准确率 {s['avg_accuracy']:.1%}  整注命中数 {s['exact_in_top10']}",
        "",
        "各排名明细：",
    ]
    if not result.get("period_match", True):
        lines.insert(2, "注意：录入期号与上次预测目标期号不一致。")
    for r in result["rows"]:
        lines.append(
            f"  #{r['rank']}  {fmt_digits(r['digits'])}  → 命中 {r['pos_hit']}/3"
            f"（号位{r['hit_pos'] or '无'}）准确率 {r['accuracy']:.1%}"
            + (" 整注命中" if r["exact"] else "")
        )
    corr = result.get("correction")
    if corr:
        lines.extend(["", "【算法准确率排名 → 权重修正】"])
        for row in corr["ranking_rows"]:
            sign = "+" if row["delta"] >= 0 else ""
            lines.append(
                f"  [{row['group']}] #{row['rank']} {row['name']}  "
                f"命中{row['total_hit']}/3 准确率{row['accuracy']:.1%}  "
                f"权重 {row['old_weight']:.1%} → {row['new_weight']:.1%} ({sign}{row['delta']:.1%})"
            )
        lines.append(f"累计修正次数：{corr['updates']}")
    return "\n".join(lines)


if __name__ == "__main__":
    draws = load_history()
    print(f"历史 {len(draws)} 期")
    model = EnsemblePredictor()
    model.fit(draws)
    for i, (k, p, _) in enumerate(model.predict(5), 1):
        print(i, k[0], k[1], k[2], f"{p:.2%}")
