#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""彩票号码助手：排列三 + 大乐透 + 快乐8 统一桌面 GUI（算法独立分区）。"""

from __future__ import annotations

import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable, List, Optional

from pl3 import predict as pl3api
from dlt import predict as dltapi
from kl8 import predict as kl8api

APP_TITLE = "彩票号码助手"
WIN_SIZE = "1180x820"
WIN_MIN = (1020, 720)


class Theme:
    BG = "#F2F2F7"
    SURFACE = "#ffffff"
    BORDER = "#E5E5EA"
    HEADER = "#1C1C1E"
    HEADER_SUB = "#8E8E93"
    ACCENT = "#007AFF"
    ACCENT_HOVER = "#0066D6"
    SECONDARY = "#E5E5EA"
    SECONDARY_HOVER = "#D1D1D6"
    PL3 = "#007AFF"
    PL3_HOVER = "#0066D6"
    DLT = "#FF3B30"
    DLT_HOVER = "#E0342B"
    KL8 = "#FF9500"
    KL8_HOVER = "#E08600"
    TEXT = "#1C1C1E"
    MUTED = "#8E8E93"
    SELECT_PL3 = "#D6E8FF"
    SELECT_DLT = "#FFE0DD"
    SELECT_KL8 = "#FFE8C2"
    HIT_RED = "#E53935"
    OMIT_GRAY = "#9E9E9E"
    FONT = ("Microsoft YaHei UI", 10)
    FONT_BOLD = ("Microsoft YaHei UI", 10, "bold")
    FONT_TITLE = ("Microsoft YaHei UI", 17, "bold")
    FONT_SUB = ("Microsoft YaHei UI", 10)
    FONT_TAB = ("Microsoft YaHei UI", 12, "bold")
    FONT_NUM = ("Consolas", 12)
    FONT_HINT = ("Microsoft YaHei UI", 9)
    FONT_DIGIT = ("Microsoft YaHei UI", 14, "bold")
    FONT_BTN = ("Microsoft YaHei UI", 10, "bold")


def card_frame(parent, **kwargs) -> tk.Frame:
    f = tk.Frame(
        parent,
        bg=Theme.SURFACE,
        highlightbackground=Theme.BORDER,
        highlightthickness=1,
        bd=0,
        **kwargs,
    )
    return f


class BubbleButton(tk.Canvas):
    """iOS-style filled pill button."""

    def __init__(
        self,
        parent,
        text: str,
        command: Optional[Callable] = None,
        *,
        bg_color: str = Theme.PL3,
        fg: str = "#ffffff",
        hover_color: Optional[str] = None,
        width: int = 132,
        height: int = 36,
        canvas_bg: str = Theme.SURFACE,
        font=None,
        **kwargs,
    ) -> None:
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=canvas_bg,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
            **kwargs,
        )
        self._text = text
        self._command = command
        self._bg = bg_color
        self._fg = fg
        self._hover = hover_color or bg_color
        self._font = font or Theme.FONT_BTN
        self._enabled = True
        self._pill = None
        self._label = None
        self.bind("<Configure>", self._redraw)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
        self.after_idle(self._redraw)

    def _redraw(self, _event=None) -> None:
        self.delete("all")
        w = max(self.winfo_width(), int(self.cget("width")))
        h = max(self.winfo_height(), int(self.cget("height")))
        r = h / 2
        fill = self._bg if self._enabled else Theme.SECONDARY
        self._pill = self.create_oval(0, 0, h, h, fill=fill, outline="")
        self.create_oval(w - h, 0, w, h, fill=fill, outline="")
        self.create_rectangle(r, 0, w - r, h, fill=fill, outline="")
        self._label = self.create_text(
            w / 2,
            h / 2,
            text=self._text,
            fill=self._fg if self._enabled else Theme.MUTED,
            font=self._font,
        )

    def _on_enter(self, _event=None) -> None:
        if not self._enabled:
            return
        self._paint(self._hover)

    def _on_leave(self, _event=None) -> None:
        self._paint(self._bg if self._enabled else Theme.SECONDARY)

    def _paint(self, color: str) -> None:
        for item in self.find_all():
            if self.type(item) in ("oval", "rectangle"):
                self.itemconfigure(item, fill=color)

    def _on_click(self, _event=None) -> None:
        if self._enabled and self._command:
            self._command()

    def configure(self, cnf=None, **kwargs):  # type: ignore[override]
        if cnf:
            kwargs.update(cnf)
        if "text" in kwargs:
            self._text = kwargs.pop("text")
        if "command" in kwargs:
            self._command = kwargs.pop("command")
        if "state" in kwargs:
            state = kwargs.pop("state")
            self._enabled = state not in (tk.DISABLED, "disabled")
            kwargs["cursor"] = "hand2" if self._enabled else "arrow"
        result = super().configure(**kwargs) if kwargs else None
        self._redraw()
        return result

    config = configure


def _set_history_sash(
    paned: ttk.Panedwindow,
    tab: tk.Misc,
    form_height: int = 300,
    history_min: int = 220,
) -> None:
    """录入区保留完整高度，其余给历史列表（不压没表单）。"""

    def _apply(tries: int = 0) -> None:
        try:
            h = int(tab.winfo_height())
            if h < 120:
                if tries < 25:
                    tab.after(60, lambda: _apply(tries + 1))
                return
            # 尽量按表单所需高度；若窗口太矮则对半分，但表单不低于 240
            top = form_height
            if h - top < history_min:
                top = max(240, h - history_min)
            if top >= h - 120:
                top = max(240, h // 2)
            paned.sashpos(0, int(top))
        except tk.TclError:
            if tries < 15:
                tab.after(80, lambda: _apply(tries + 1))

    tab.after(40, lambda: _apply(0))


class SegmentedControl(tk.Canvas):
    """iOS 风格胶囊分段开关（排列三 / 大乐透）。"""

    def __init__(
        self,
        parent,
        labels: List[str],
        command: Optional[Callable[[int], None]] = None,
        *,
        width: int = 280,
        height: int = 40,
        **kwargs,
    ) -> None:
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=Theme.BG,
            highlightthickness=0,
            bd=0,
            **kwargs,
        )
        self._labels = labels
        self._command = command
        self._index = 0
        self._width = width
        self._height = height
        self.bind("<Button-1>", self._on_click)
        self.bind("<Configure>", lambda _e: self._redraw())
        self.after_idle(self._redraw)

    def get(self) -> int:
        return self._index

    def set(self, index: int, *, notify: bool = False) -> None:
        if index < 0 or index >= len(self._labels):
            return
        self._index = index
        self._redraw()
        if notify and self._command:
            self._command(self._index)

    def _on_click(self, event) -> None:
        n = max(len(self._labels), 1)
        w = max(self.winfo_width(), self._width)
        idx = int(event.x / (w / n))
        idx = max(0, min(n - 1, idx))
        if idx != self._index:
            self._index = idx
            self._redraw()
            if self._command:
                self._command(self._index)

    def _redraw(self, _event=None) -> None:
        self.delete("all")
        w = max(self.winfo_width(), self._width)
        h = max(self.winfo_height(), self._height)
        pad = 3
        r = (h - pad * 2) / 2
        # track
        self._round_rect(0, 0, w, h, h / 2, Theme.SECONDARY)
        n = len(self._labels)
        seg_w = (w - pad * 2) / n
        # selected bubble
        x0 = pad + self._index * seg_w
        x1 = x0 + seg_w
        self._round_rect(x0, pad, x1, h - pad, r, Theme.SURFACE)
        for i, label in enumerate(self._labels):
            cx = pad + seg_w * (i + 0.5)
            color = Theme.HEADER if i == self._index else Theme.MUTED
            font = Theme.FONT_TAB if i == self._index else ("Microsoft YaHei UI", 12)
            self.create_text(cx, h / 2, text=label, fill=color, font=font)

    def _round_rect(self, x0, y0, x1, y1, radius, fill: str) -> None:
        r = min(radius, (x1 - x0) / 2, (y1 - y0) / 2)
        self.create_oval(x0, y0, x0 + 2 * r, y0 + 2 * r, fill=fill, outline="")
        self.create_oval(x1 - 2 * r, y0, x1, y0 + 2 * r, fill=fill, outline="")
        self.create_oval(x0, y1 - 2 * r, x0 + 2 * r, y1, fill=fill, outline="")
        self.create_oval(x1 - 2 * r, y1 - 2 * r, x1, y1, fill=fill, outline="")
        self.create_rectangle(x0 + r, y0, x1 - r, y1, fill=fill, outline="")
        self.create_rectangle(x0, y0 + r, x1, y1 - r, fill=fill, outline="")


def apply_theme(root: tk.Tk) -> ttk.Style:
    root.configure(bg=Theme.BG)
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure(".", font=Theme.FONT, background=Theme.BG, foreground=Theme.TEXT)
    style.configure("TFrame", background=Theme.BG)
    style.configure("Card.TFrame", background=Theme.SURFACE)
    style.configure("TLabel", background=Theme.BG, foreground=Theme.TEXT, font=Theme.FONT)
    style.configure("Card.TLabel", background=Theme.SURFACE, foreground=Theme.TEXT, font=Theme.FONT)
    style.configure("Title.TLabel", background=Theme.BG, foreground=Theme.TEXT, font=Theme.FONT_TITLE)
    style.configure(
        "PanelTitle.TLabel",
        background=Theme.SURFACE,
        foreground=Theme.TEXT,
        font=("Microsoft YaHei UI", 13, "bold"),
    )
    style.configure("Hint.TLabel", background=Theme.BG, foreground=Theme.MUTED, font=Theme.FONT_HINT)
    style.configure("CardHint.TLabel", background=Theme.SURFACE, foreground=Theme.MUTED, font=Theme.FONT_HINT)
    style.configure("Status.TLabel", background=Theme.SURFACE, foreground=Theme.TEXT, font=Theme.FONT)
    style.configure("Pos.TLabel", background=Theme.SURFACE, foreground=Theme.TEXT, font=Theme.FONT_BOLD)
    style.configure("HeroTitle.TLabel", background=Theme.HEADER, foreground="#ffffff", font=Theme.FONT_TITLE)
    style.configure("HeroSub.TLabel", background=Theme.HEADER, foreground=Theme.HEADER_SUB, font=Theme.FONT_SUB)
    style.configure("HeroStatus.TLabel", background=Theme.HEADER, foreground="#F2F2F7", font=Theme.FONT)

    style.configure("TButton", font=Theme.FONT, padding=(16, 9))
    style.configure(
        "Primary.TButton",
        font=Theme.FONT_BOLD,
        padding=(18, 10),
        background=Theme.PL3,
        foreground="#ffffff",
        bordercolor=Theme.PL3,
        lightcolor=Theme.PL3,
        darkcolor=Theme.PL3_HOVER,
        focuscolor=Theme.PL3,
    )
    style.map(
        "Primary.TButton",
        background=[("active", Theme.PL3_HOVER), ("pressed", Theme.PL3_HOVER), ("disabled", "#A0CFFF")],
        foreground=[("disabled", "#f5f5f5")],
    )
    style.configure(
        "Ghost.TButton",
        font=Theme.FONT,
        padding=(16, 9),
        background=Theme.SECONDARY,
        foreground=Theme.TEXT,
        bordercolor=Theme.SECONDARY,
        lightcolor=Theme.SECONDARY,
        darkcolor=Theme.SECONDARY_HOVER,
    )
    style.map(
        "Ghost.TButton",
        background=[("active", Theme.SECONDARY_HOVER), ("pressed", Theme.SECONDARY_HOVER)],
    )
    style.configure(
        "Pl3.TButton",
        font=Theme.FONT_BOLD,
        padding=(18, 10),
        background=Theme.PL3,
        foreground="#ffffff",
        bordercolor=Theme.PL3,
        lightcolor=Theme.PL3,
        darkcolor=Theme.PL3_HOVER,
    )
    style.map("Pl3.TButton", background=[("active", Theme.PL3_HOVER), ("disabled", "#A0CFFF")])
    style.configure(
        "Dlt.TButton",
        font=Theme.FONT_BOLD,
        padding=(18, 10),
        background=Theme.DLT,
        foreground="#ffffff",
        bordercolor=Theme.DLT,
        lightcolor=Theme.DLT,
        darkcolor=Theme.DLT_HOVER,
    )
    style.map("Dlt.TButton", background=[("active", Theme.DLT_HOVER), ("disabled", "#FFB3AE")])
    style.configure(
        "Kl8.TButton",
        font=Theme.FONT_BOLD,
        padding=(18, 10),
        background=Theme.KL8,
        foreground="#ffffff",
        bordercolor=Theme.KL8,
        lightcolor=Theme.KL8,
        darkcolor=Theme.KL8_HOVER,
    )
    style.map("Kl8.TButton", background=[("active", Theme.KL8_HOVER), ("disabled", "#FFD59A")])

    style.configure("TNotebook", background=Theme.BG, borderwidth=0)
    style.configure(
        "Outer.TNotebook.Tab",
        font=Theme.FONT_TAB,
        padding=(32, 14),
        background=Theme.SECONDARY,
        foreground=Theme.TEXT,
    )
    style.map(
        "Outer.TNotebook.Tab",
        background=[("selected", Theme.SURFACE), ("active", "#EBEBF0")],
        foreground=[("selected", Theme.HEADER)],
    )
    style.configure(
        "Inner.TNotebook.Tab",
        font=("Microsoft YaHei UI", 11),
        padding=(18, 10),
        background=Theme.SECONDARY,
        foreground=Theme.TEXT,
    )
    style.map(
        "Inner.TNotebook.Tab",
        background=[("selected", Theme.SURFACE), ("active", "#EBEBF0")],
    )
    style.configure("TLabelframe", background=Theme.SURFACE, bordercolor=Theme.BORDER, relief="solid")
    style.configure("TLabelframe.Label", background=Theme.SURFACE, foreground=Theme.TEXT, font=Theme.FONT_BOLD)
    style.configure("TEntry", fieldbackground="#F9F9FB", bordercolor=Theme.BORDER, padding=8)
    style.configure("TSpinbox", fieldbackground="#F9F9FB", bordercolor=Theme.BORDER, padding=6)
    style.configure("Digit.TSpinbox", fieldbackground="#F9F9FB", bordercolor=Theme.BORDER, padding=10)
    style.configure(
        "Treeview",
        background=Theme.SURFACE,
        fieldbackground=Theme.SURFACE,
        foreground=Theme.TEXT,
        rowheight=28,
        font=Theme.FONT_NUM,
        bordercolor=Theme.BORDER,
    )
    style.configure("Treeview.Heading", font=Theme.FONT_BOLD, background="#F2F2F7", foreground=Theme.TEXT)
    style.map("Treeview", background=[("selected", Theme.SELECT_PL3)], foreground=[("selected", Theme.TEXT)])
    style.configure("TScrollbar", background=Theme.BG, troughcolor=Theme.SECONDARY, bordercolor=Theme.BORDER)
    style.configure("TPanedwindow", background=Theme.BG)
    return style


# ---------------------------------------------------------------------------
# 排列三
# ---------------------------------------------------------------------------


class Pl3Panel(ttk.Frame):
    def __init__(
        self,
        master,
        status_callback: Optional[Callable[[str], None]] = None,
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)
        self.status_callback = status_callback
        self._period_map: List[int] = []
        self.status_var = tk.StringVar(value="")
        self._build_ui()
        self._refresh_weight_label()
        self.refresh_status()

    def _build_ui(self) -> None:
        tip = card_frame(self)
        tip.pack(fill="x", padx=12, pady=(8, 4))
        inner = ttk.Frame(tip, style="Card.TFrame", padding=(16, 8))
        inner.pack(fill="x")
        accent = tk.Frame(inner, bg=Theme.PL3, width=4)
        accent.pack(side="left", fill="y", padx=(0, 12))
        txt = ttk.Frame(inner, style="Card.TFrame")
        txt.pack(side="left", fill="x", expand=True)
        ttk.Label(txt, text="排列三 · 直选预测", style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(txt, textvariable=self.status_var, style="Status.TLabel").pack(anchor="w", pady=(2, 0))

        self.nb = ttk.Notebook(self, style="Inner.TNotebook")
        self.nb.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.tab_predict = ttk.Frame(self.nb, style="Card.TFrame", padding=16)
        self.tab_input = ttk.Frame(self.nb, style="Card.TFrame", padding=10)
        self.nb.add(self.tab_predict, text="  预测  ")
        self.nb.add(self.tab_input, text="  录入 / 删除  ")
        self._input_paned: Optional[ttk.Panedwindow] = None
        self._form_max = 320
        self._build_predict_tab()
        self._build_input_tab()
        self.nb.bind("<<NotebookTabChanged>>", self._on_inner_tab)

        foot = ttk.Frame(self, padding=(12, 0, 12, 6))
        foot.pack(fill="x")
        ttk.Label(
            foot,
            text=f"数据目录：{pl3api.DATA_PATH}　｜　模型仅供统计参考",
            style="Hint.TLabel",
        ).pack(anchor="w")

    def _build_predict_tab(self) -> None:
        bar = ttk.Frame(self.tab_predict, style="Card.TFrame")
        bar.pack(fill="x")
        ttk.Label(bar, text="显示条数", style="Card.TLabel").pack(side="left")
        self.top_n_var = tk.StringVar(value="10")
        ttk.Spinbox(bar, from_=1, to=50, width=5, textvariable=self.top_n_var).pack(
            side="left", padx=(6, 12)
        )
        self.btn_predict = BubbleButton(
            bar,
            text="开始预测",
            command=self.run_predict,
            bg_color=Theme.PL3,
            hover_color=Theme.PL3_HOVER,
            width=110,
            height=36,
        )
        self.btn_predict.pack(side="left")
        ttk.Button(bar, text="刷新状态", style="Ghost.TButton", command=self.refresh_status).pack(
            side="left", padx=8
        )
        ttk.Button(
            bar, text="对比上次预测", style="Ghost.TButton", command=self.compare_latest_draw
        ).pack(side="left", padx=4)
        ttk.Button(
            bar, text="重置算法权重", style="Ghost.TButton", command=self.reset_weights
        ).pack(side="left", padx=4)
        self.next_period_var = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.next_period_var, style="Status.TLabel").pack(side="right")

        self.recent_var = tk.StringVar(value="")
        ttk.Label(self.tab_predict, textvariable=self.recent_var, style="CardHint.TLabel").pack(
            anchor="w", pady=(8, 2)
        )
        self.weight_var = tk.StringVar(value="")
        ttk.Label(self.tab_predict, textvariable=self.weight_var, style="CardHint.TLabel").pack(
            anchor="w", pady=(0, 6)
        )

        table_box = ttk.Frame(self.tab_predict, style="Card.TFrame")
        table_box.pack(fill="both", expand=True)
        cols = ("rank", "number", "prob", "blend", "markov", "pattern", "mc")
        self.tree = ttk.Treeview(table_box, columns=cols, show="headings", height=14)
        for c, (title, w) in {
            "rank": ("排名", 50),
            "number": ("1号位 2号位 3号位", 140),
            "prob": ("综合概率", 90),
            "blend": ("各位融合", 80),
            "markov": ("马尔可夫", 80),
            "pattern": ("形态", 70),
            "mc": ("蒙特卡洛", 80),
        }.items():
            self.tree.heading(c, text=title)
            self.tree.column(c, width=w, anchor="center")
        scroll = ttk.Scrollbar(table_box, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self.recommend_var = tk.StringVar(value="")
        ttk.Label(
            self.tab_predict, textvariable=self.recommend_var, style="PanelTitle.TLabel"
        ).pack(anchor="w", pady=(10, 0))

    def _build_input_tab(self) -> None:
        paned = ttk.Panedwindow(self.tab_input, orient=tk.VERTICAL)
        paned.pack(fill="both", expand=True)
        self._input_paned = paned
        self._form_max = 320

        form_wrap = ttk.Frame(paned, style="Card.TFrame")
        form = ttk.LabelFrame(
            form_wrap, text="录入最新中奖号码（1号位 / 2号位 / 3号位）", padding=(16, 10)
        )
        form.pack(fill="x", expand=False)
        ttk.Label(
            form,
            text="例如开奖 1 2 4：1号位填 1，2号位填 2，3号位填 4",
            style="CardHint.TLabel",
        ).pack(anchor="w", pady=(0, 6))

        row1 = ttk.Frame(form, style="Card.TFrame")
        row1.pack(fill="x", pady=2)
        ttk.Label(row1, text="期号", width=8, style="Card.TLabel").pack(side="left")
        self.period_var = tk.StringVar()
        ttk.Entry(row1, textvariable=self.period_var, width=14).pack(side="left")
        ttk.Button(
            row1, text="自动填下一期", style="Ghost.TButton", command=self.fill_next_period
        ).pack(side="left", padx=8)

        row2 = ttk.Frame(form, style="Card.TFrame")
        row2.pack(fill="x", pady=8)
        self.p1_var = tk.StringVar()
        self.p2_var = tk.StringVar()
        self.p3_var = tk.StringVar()
        for label, var, ex in (
            ("1号位", self.p1_var, "例:1"),
            ("2号位", self.p2_var, "例:2"),
            ("3号位", self.p3_var, "例:4"),
        ):
            box = ttk.Frame(row2, style="Card.TFrame")
            box.pack(side="left", padx=(0, 28))
            ttk.Label(box, text=label, style="Pos.TLabel").pack()
            ttk.Spinbox(
                box,
                from_=0,
                to=9,
                width=6,
                textvariable=var,
                justify="center",
                font=Theme.FONT_DIGIT,
                style="Digit.TSpinbox",
            ).pack(pady=6, ipady=4)
            ttk.Label(box, text=ex, style="CardHint.TLabel").pack()

        row3 = ttk.Frame(form, style="Card.TFrame")
        row3.pack(fill="x", pady=(4, 0))
        BubbleButton(
            row3,
            text="保存并重新预测",
            command=self.save_and_predict,
            bg_color=Theme.PL3,
            hover_color=Theme.PL3_HOVER,
            width=148,
            height=38,
        ).pack(side="left")
        BubbleButton(
            row3,
            text="仅保存",
            command=self.save_only,
            bg_color=Theme.PL3,
            hover_color=Theme.PL3_HOVER,
            width=96,
            height=38,
        ).pack(side="left", padx=8)
        ttk.Button(row3, text="清空输入", style="Ghost.TButton", command=self.clear_input).pack(
            side="left"
        )
        ttk.Button(
            row3, text="对比上次预测", style="Ghost.TButton", command=self.compare_latest_draw
        ).pack(side="left", padx=8)
        paned.add(form_wrap, weight=0)

        hist_wrap = ttk.Frame(paned, style="Card.TFrame")
        hist = ttk.LabelFrame(hist_wrap, text="历史开奖（可选中后删除）", padding=(14, 10))
        hist.pack(fill="both", expand=True)
        btn = ttk.Frame(hist, style="Card.TFrame")
        btn.pack(fill="x", pady=(0, 6))
        ttk.Button(btn, text="删除选中期号", style="Ghost.TButton", command=self.delete_selected).pack(
            side="left"
        )
        ttk.Button(
            btn, text="删除最新一期", style="Ghost.TButton", command=self.delete_latest_draw
        ).pack(side="left", padx=8)
        ttk.Button(btn, text="清空全部数据", style="Ghost.TButton", command=self.delete_all).pack(
            side="left"
        )
        ttk.Button(
            btn, text="刷新列表", style="Ghost.TButton", command=self.reload_history_list
        ).pack(side="right")

        list_box = ttk.Frame(hist, style="Card.TFrame")
        list_box.pack(fill="both", expand=True)
        self.hist = tk.Listbox(
            list_box,
            font=Theme.FONT_NUM,
            height=12,
            selectmode="extended",
            bg=Theme.SURFACE,
            fg=Theme.TEXT,
            highlightthickness=1,
            highlightbackground=Theme.BORDER,
            selectbackground=Theme.SELECT_PL3,
            selectforeground=Theme.TEXT,
            relief="flat",
            activestyle="none",
        )
        scroll = ttk.Scrollbar(list_box, orient="vertical", command=self.hist.yview)
        self.hist.configure(yscrollcommand=scroll.set)
        self.hist.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        hist_wrap.configure(height=280)
        paned.add(hist_wrap, weight=3)
        self.ensure_history_layout()

        self.fill_next_period()
        self.reload_history_list()

    def _on_inner_tab(self, _event=None) -> None:
        try:
            if self.nb.index(self.nb.select()) == 1:
                self.ensure_history_layout()
        except tk.TclError:
            pass

    def ensure_history_layout(self) -> None:
        if self._input_paned is not None:
            _set_history_sash(self._input_paned, self.tab_input, self._form_max)

    def _emit_status(self, text: str) -> None:
        self.status_var.set(text)
        if self.status_callback:
            self.status_callback(f"排列三 ｜ {text}")

    def refresh_status(self) -> None:
        draws = pl3api.load_history()
        if not draws:
            self._emit_status("暂无历史数据")
            self.next_period_var.set("")
            self.recent_var.set("")
            return
        last = draws[-1]
        d = last["digits"]
        self._emit_status(
            f"已录入 {len(draws)} 期 | 最新 {last['period']}期：1号位={d[0]} 2号位={d[1]} 3号位={d[2]}"
        )
        self.next_period_var.set(f"预测目标：第 {int(last['period']) + 1} 期")
        parts = [f"{x['period']}:{''.join(map(str, x['digits']))}" for x in draws[-5:]]
        self.recent_var.set("最近5期：" + "  |  ".join(parts))

    def _refresh_weight_label(self) -> None:
        w = pl3api.load_algo_weights()
        top = " ".join(
            f"{pl3api.ALGO_NAMES[k]}{w['digit'][k]:.0%}" for k in ("freq", "softmax", "gap", "bayes")
        )
        self.weight_var.set(
            f"当前权重(已修正{w.get('updates', 0)}次)：{top} …  "
            f"组合 digit_blend={w['combo']['digit_blend']:.0%} markov={w['combo']['markov']:.0%}"
        )

    def reset_weights(self) -> None:
        if not messagebox.askyesno("重置权重", "确定把排列三所有算法权重恢复为初始值？"):
            return
        pl3api.reset_algo_weights()
        self._refresh_weight_label()
        messagebox.showinfo("完成", "算法权重已重置。")

    def fill_next_period(self) -> None:
        draws = pl3api.load_history()
        self.period_var.set(str(int(draws[-1]["period"]) + 1) if draws else "")

    def reload_history_list(self) -> None:
        self.hist.delete(0, tk.END)
        self._period_map = []
        for d in reversed(pl3api.load_history()):
            self.hist.insert(
                tk.END,
                f"{d['period']}期    1号位={d['digits'][0]}  2号位={d['digits'][1]}  "
                f"3号位={d['digits'][2]}    ({pl3api.fmt_digits(d['digits'])})",
            )
            self._period_map.append(int(d["period"]))

    def clear_input(self) -> None:
        self.p1_var.set("")
        self.p2_var.set("")
        self.p3_var.set("")
        self.fill_next_period()

    def _parse_input(self) -> Optional[tuple]:
        try:
            period = int(self.period_var.get().strip())
            p1 = int(self.p1_var.get().strip())
            p2 = int(self.p2_var.get().strip())
            p3 = int(self.p3_var.get().strip())
        except ValueError:
            messagebox.showerror("输入错误", "期号和 1/2/3 号位都必须是整数。")
            return None
        if any(x < 0 or x > 9 for x in (p1, p2, p3)):
            messagebox.showerror("输入错误", "每位号码必须是 0-9。")
            return None
        if period <= 0:
            messagebox.showerror("输入错误", "期号必须为正整数。")
            return None
        return period, p1, p2, p3

    def save_only(self) -> None:
        parsed = self._parse_input()
        if not parsed:
            return
        period, p1, p2, p3 = parsed
        try:
            pl3api.add_draw(period, p1, p2, p3)
        except ValueError as e:
            messagebox.showerror("保存失败", str(e))
            return
        self.clear_input()
        self.reload_history_list()
        self.refresh_status()
        self.show_compare_for_draw(period, [p1, p2, p3], after_save=True)

    def save_and_predict(self) -> None:
        parsed = self._parse_input()
        if not parsed:
            return
        period, p1, p2, p3 = parsed
        try:
            pl3api.add_draw(period, p1, p2, p3)
        except ValueError as e:
            messagebox.showerror("保存失败", str(e))
            return
        self.clear_input()
        self.reload_history_list()
        self.refresh_status()
        self.show_compare_for_draw(period, [p1, p2, p3], after_save=True)
        self.nb.select(self.tab_predict)
        self.run_predict()

    def delete_selected(self) -> None:
        sel = list(self.hist.curselection())
        if not sel:
            messagebox.showwarning("提示", "请先选中要删除的期号。")
            return
        periods = [self._period_map[i] for i in sel]
        if not messagebox.askyesno("确认删除", f"确定删除期号？\n{periods}"):
            return
        for period in periods:
            try:
                pl3api.delete_draw(period)
            except ValueError as e:
                messagebox.showerror("删除失败", str(e))
                return
        self.reload_history_list()
        self.refresh_status()
        self.fill_next_period()
        messagebox.showinfo("完成", f"已删除 {len(periods)} 条。")

    def delete_latest_draw(self) -> None:
        draws = pl3api.load_history()
        if not draws:
            messagebox.showerror("删除失败", "没有可删除的数据")
            return
        last = draws[-1]
        d = last["digits"]
        if not messagebox.askyesno(
            "确认删除",
            f"确定删除最新一期 {last['period']}？\n1号位={d[0]} 2号位={d[1]} 3号位={d[2]}",
        ):
            return
        try:
            pl3api.delete_latest()
        except ValueError as e:
            messagebox.showerror("删除失败", str(e))
            return
        self.reload_history_list()
        self.refresh_status()
        self.fill_next_period()
        messagebox.showinfo("完成", f"已删除 {last['period']}期。")

    def delete_all(self) -> None:
        if not messagebox.askyesno("危险操作", "确定清空排列三全部历史数据？"):
            return
        if not messagebox.askyesno("再次确认", "真的要清空全部数据吗？"):
            return
        pl3api.clear_all_history()
        self.reload_history_list()
        self.refresh_status()
        self.fill_next_period()
        messagebox.showinfo("完成", "已清空全部数据。")

    def compare_latest_draw(self) -> None:
        draws = pl3api.load_history()
        if not draws:
            messagebox.showwarning("提示", "暂无开奖数据可对比。")
            return
        last = draws[-1]
        self.show_compare_for_draw(int(last["period"]), list(last["digits"]), after_save=False)

    def show_compare_for_draw(self, period: int, digits: list, after_save: bool = False) -> None:
        pred = pl3api.load_last_prediction()
        if not pred or not pred.get("top"):
            if after_save:
                messagebox.showinfo(
                    "录入成功",
                    f"已录入 {period}期。\n尚无上次预测记录，请先预测后再对比下一期。",
                )
            else:
                messagebox.showwarning("无法对比", "没有上次预测 Top10，请先点击「开始预测」。")
            return
        try:
            result = pl3api.compare_prediction_to_draw(
                digits, prediction=pred, actual_period=period
            )
        except ValueError as e:
            messagebox.showwarning("无法对比", str(e))
            return
        self._open_compare_window(result, after_save=after_save)

    def _open_compare_window(self, result: dict, after_save: bool = False) -> None:
        root = self.winfo_toplevel()
        win = tk.Toplevel(root)
        win.title(f"排列三预测对比 - {result.get('actual_period')}期")
        win.geometry("880x660")
        win.configure(bg=Theme.BG)
        win.transient(root)

        s = result["summary"]
        head = ttk.Frame(win, padding=10)
        head.pack(fill="x")
        title = (
            "录入成功：Top10 对比 + 按准确率修正算法权重"
            if after_save
            else "Top10 对比 + 按准确率修正算法权重"
        )
        ttk.Label(head, text=title, style="PanelTitle.TLabel").pack(anchor="w")
        ad = result["actual_digits"]
        ttk.Label(
            head,
            text=(
                f"开奖：1号位={ad[0]}  2号位={ad[1]}  3号位={ad[2]}"
                + ("" if result.get("period_match", True) else "  （期号与预测目标不一致）")
            ),
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(4, 0))
        ttk.Label(
            head,
            text=(
                f"最佳一注：第{s['best_rank']}名 位命中 {s['best_pos_hit']}/3  准确率 {s['best_accuracy']:.1%}"
                + ("  【整注命中】" if s["exact_hit"] else "")
                + f"    |    Top10平均准确率 {s['avg_accuracy']:.1%}"
                + f"    |    Top10整注命中 {s['exact_in_top10']} 注"
            ),
            style="Status.TLabel",
        ).pack(anchor="w", pady=(8, 0))

        nb = ttk.Notebook(win, style="Inner.TNotebook")
        nb.pack(fill="both", expand=True, padx=10, pady=6)
        tab_pred = ttk.Frame(nb, padding=6)
        tab_algo = ttk.Frame(nb, padding=6)
        nb.add(tab_pred, text=" Top10命中 ")
        nb.add(tab_algo, text=" 算法排名与权重修正 ")

        cols = ("rank", "number", "hit", "acc", "exact", "positions")
        tree = ttk.Treeview(tab_pred, columns=cols, show="headings", height=12)
        for c, (t, w) in {
            "rank": ("排名", 50),
            "number": ("预测号码", 120),
            "hit": ("位命中", 70),
            "acc": ("准确率", 70),
            "exact": ("整注", 70),
            "positions": ("命中号位", 120),
        }.items():
            tree.heading(c, text=t)
            tree.column(c, width=w, anchor="center")
        scroll = ttk.Scrollbar(tab_pred, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        for r in result["rows"]:
            tree.insert(
                "",
                "end",
                values=(
                    r["rank"],
                    pl3api.fmt_digits(r["digits"]),
                    f"{r['pos_hit']}/3",
                    f"{r['accuracy']:.1%}",
                    "是" if r["exact"] else "否",
                    ",".join(str(x) for x in r["hit_pos"]) or "-",
                ),
            )

        ttk.Label(
            tab_algo,
            text="准确率从高到低排序，提高排名靠前算法的权重（平滑更新）。",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(0, 6))
        acols = ("group", "rank", "name", "hit", "acc", "old_w", "new_w", "delta")
        atree = ttk.Treeview(tab_algo, columns=acols, show="headings", height=14)
        for c, (t, w) in {
            "group": ("类别", 80),
            "rank": ("准确率排名", 80),
            "name": ("算法", 110),
            "hit": ("命中", 70),
            "acc": ("准确率", 70),
            "old_w": ("原权重", 70),
            "new_w": ("新权重", 70),
            "delta": ("变化", 70),
        }.items():
            atree.heading(c, text=t)
            atree.column(c, width=w, anchor="center")
        ascroll = ttk.Scrollbar(tab_algo, orient="vertical", command=atree.yview)
        atree.configure(yscrollcommand=ascroll.set)
        atree.pack(side="left", fill="both", expand=True)
        ascroll.pack(side="right", fill="y")

        corr = result.get("correction")
        if corr:
            for row in corr["ranking_rows"]:
                sign = "+" if row["delta"] >= 0 else ""
                atree.insert(
                    "",
                    "end",
                    values=(
                        row["group"],
                        row["rank"],
                        row["name"],
                        f"{row['total_hit']}/3",
                        f"{row['accuracy']:.1%}",
                        f"{row['old_weight']:.1%}",
                        f"{row['new_weight']:.1%}",
                        f"{sign}{row['delta']:.1%}",
                    ),
                )
            ttk.Label(
                tab_algo,
                text=f"已写入权重文件，累计修正 {corr['updates']} 次。下次预测使用新权重。",
                style="Status.TLabel",
            ).pack(anchor="w", pady=6)
            nb.select(tab_algo)
        else:
            ttk.Label(tab_algo, text="本期未能完成算法修正。", style="Hint.TLabel").pack(anchor="w")

        btns = ttk.Frame(win, padding=10)
        btns.pack(fill="x")

        def _copy() -> None:
            root.clipboard_clear()
            root.clipboard_append(pl3api.format_compare_text(result))
            messagebox.showinfo("已复制", "对比结果已复制到剪贴板。", parent=win)

        ttk.Button(btns, text="复制对比文本", style="Primary.TButton", command=_copy).pack(
            side="left"
        )
        ttk.Button(btns, text="关闭", style="Ghost.TButton", command=win.destroy).pack(side="right")
        self._refresh_weight_label()

    def run_predict(self) -> None:
        draws = pl3api.load_history()
        if len(draws) < 8:
            messagebox.showwarning("数据不足", "至少需要 8 期历史数据才能预测。")
            return
        try:
            top_n = max(1, min(50, int(self.top_n_var.get())))
        except ValueError:
            top_n = 10
            self.top_n_var.set("10")

        self.btn_predict.configure(state="disabled")
        self.update_idletasks()
        try:
            model = pl3api.EnsemblePredictor()
            model.fit(draws)
            model.save()
            pred_n = max(top_n, 10)
            preds = model.predict(top_n=pred_n)
            target_period = int(draws[-1]["period"]) + 1
            pl3api.save_last_prediction(target_period, preds[:10], int(draws[-1]["period"]))
        except Exception as e:
            messagebox.showerror("预测失败", str(e))
            self.btn_predict.configure(state="normal")
            return

        for item in self.tree.get_children():
            self.tree.delete(item)
        for i, (key, prob, detail) in enumerate(preds[:top_n], 1):
            self.tree.insert(
                "",
                "end",
                values=(
                    i,
                    f"{key[0]}  {key[1]}  {key[2]}",
                    f"{prob:.2%}",
                    f"{detail['digit_blend']:.4f}",
                    f"{detail['markov']:.4f}",
                    f"{detail['pattern']:.4f}",
                    f"{detail['montecarlo']:.4f}",
                ),
            )
        if preds:
            k = preds[0][0]
            self.recommend_var.set(
                f"推荐直选 Top1：{k[0]} {k[1]} {k[2]}    "
                f"（已保存第 {target_period} 期 Top10，开奖录入后自动对比并修正算法）"
            )
        self._refresh_weight_label()
        self.refresh_status()
        self.btn_predict.configure(state="normal")


# ---------------------------------------------------------------------------
# 大乐透
# ---------------------------------------------------------------------------


class DltPanel(ttk.Frame):
    def __init__(
        self,
        master,
        status_callback: Optional[Callable[[str], None]] = None,
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)
        self.status_callback = status_callback
        self._period_map: List[int] = []
        self.status_var = tk.StringVar(value="")
        self._build_ui()
        self._refresh_weight_label()
        self.refresh_status()

    def _build_ui(self) -> None:
        tip = card_frame(self)
        tip.pack(fill="x", padx=12, pady=(8, 4))
        inner = ttk.Frame(tip, style="Card.TFrame", padding=(16, 8))
        inner.pack(fill="x")
        accent = tk.Frame(inner, bg=Theme.DLT, width=4)
        accent.pack(side="left", fill="y", padx=(0, 12))
        txt = ttk.Frame(inner, style="Card.TFrame")
        txt.pack(side="left", fill="x", expand=True)
        ttk.Label(txt, text="大乐透 · 前区 + 后区", style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(txt, textvariable=self.status_var, style="Status.TLabel").pack(anchor="w", pady=(2, 0))

        self.nb = ttk.Notebook(self, style="Inner.TNotebook")
        self.nb.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.tab_predict = ttk.Frame(self.nb, style="Card.TFrame", padding=16)
        self.tab_input = ttk.Frame(self.nb, style="Card.TFrame", padding=10)
        self.nb.add(self.tab_predict, text="  预测  ")
        self.nb.add(self.tab_input, text="  录入 / 删除  ")
        self._input_paned: Optional[ttk.Panedwindow] = None
        self._form_max = 290
        self._build_predict_tab()
        self._build_input_tab()
        self.nb.bind("<<NotebookTabChanged>>", self._on_inner_tab)

        foot = ttk.Frame(self, padding=(12, 0, 12, 6))
        foot.pack(fill="x")
        ttk.Label(
            foot,
            text=f"数据目录：{dltapi.DATA_PATH}　｜　模型仅供统计参考",
            style="Hint.TLabel",
        ).pack(anchor="w")

    def _build_predict_tab(self) -> None:
        bar = ttk.Frame(self.tab_predict, style="Card.TFrame")
        bar.pack(fill="x")
        ttk.Label(bar, text="显示注数", style="Card.TLabel").pack(side="left")
        self.top_n_var = tk.StringVar(value="10")
        ttk.Spinbox(bar, from_=1, to=30, width=5, textvariable=self.top_n_var).pack(
            side="left", padx=(6, 12)
        )
        self.btn_predict = BubbleButton(
            bar,
            text="开始预测",
            command=self.run_predict,
            bg_color=Theme.DLT,
            hover_color=Theme.DLT_HOVER,
            width=110,
            height=36,
        )
        self.btn_predict.pack(side="left")
        ttk.Button(bar, text="刷新状态", style="Ghost.TButton", command=self.refresh_status).pack(
            side="left", padx=8
        )
        ttk.Button(
            bar, text="对比上次预测", style="Ghost.TButton", command=self.compare_latest_draw
        ).pack(side="left", padx=4)
        ttk.Button(
            bar, text="重置算法权重", style="Ghost.TButton", command=self.reset_weights
        ).pack(side="left", padx=4)
        self.next_period_var = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.next_period_var, style="Status.TLabel").pack(side="right")

        self.recent_var = tk.StringVar(value="")
        ttk.Label(self.tab_predict, textvariable=self.recent_var, style="CardHint.TLabel").pack(
            anchor="w", pady=(8, 4)
        )
        self.hot_var = tk.StringVar(value="")
        ttk.Label(self.tab_predict, textvariable=self.hot_var, style="CardHint.TLabel").pack(
            anchor="w", pady=(0, 2)
        )
        self.weight_var = tk.StringVar(value="")
        ttk.Label(self.tab_predict, textvariable=self.weight_var, style="CardHint.TLabel").pack(
            anchor="w", pady=(0, 6)
        )

        table_box = ttk.Frame(self.tab_predict, style="Card.TFrame")
        table_box.pack(fill="both", expand=True)
        cols = ("rank", "front", "back", "prob", "pattern", "cooc", "gap")
        self.tree = ttk.Treeview(table_box, columns=cols, show="headings", height=14)
        heads = {
            "rank": ("排名", 50),
            "front": ("前区(5个)", 200),
            "back": ("后区(2个)", 90),
            "prob": ("综合概率", 80),
            "pattern": ("形态分", 70),
            "cooc": ("共现分", 70),
            "gap": ("冷热分", 70),
        }
        for c, (title, w) in heads.items():
            self.tree.heading(c, text=title)
            self.tree.column(c, width=w, anchor="center")
        scroll = ttk.Scrollbar(table_box, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self.recommend_var = tk.StringVar(value="")
        ttk.Label(
            self.tab_predict, textvariable=self.recommend_var, style="PanelTitle.TLabel"
        ).pack(anchor="w", pady=(10, 0))

    def _build_input_tab(self) -> None:
        paned = ttk.Panedwindow(self.tab_input, orient=tk.VERTICAL)
        paned.pack(fill="both", expand=True)
        self._input_paned = paned
        self._form_max = 290

        form_wrap = ttk.Frame(paned, style="Card.TFrame")
        form = ttk.LabelFrame(form_wrap, text="录入最新开奖号码", padding=(14, 10))
        form.pack(fill="x", expand=False)
        ttk.Label(
            form,
            text="前区 5 个（01-35）+ 后区 2 个（01-12），空格分隔",
            style="CardHint.TLabel",
        ).pack(anchor="w", pady=(0, 6))

        row1 = ttk.Frame(form, style="Card.TFrame")
        row1.pack(fill="x", pady=2)
        ttk.Label(row1, text="期号", width=8, style="Card.TLabel").pack(side="left")
        self.period_var = tk.StringVar()
        ttk.Entry(row1, textvariable=self.period_var, width=14).pack(side="left")
        ttk.Button(
            row1, text="自动填下一期", style="Ghost.TButton", command=self.fill_next_period
        ).pack(side="left", padx=8)
        ttk.Label(row1, text="日期(可选)", width=10, style="Card.TLabel").pack(
            side="left", padx=(12, 0)
        )
        self.date_var = tk.StringVar()
        ttk.Entry(row1, textvariable=self.date_var, width=14).pack(side="left")

        row2 = ttk.Frame(form, style="Card.TFrame")
        row2.pack(fill="x", pady=6)
        ttk.Label(row2, text="前区5个", style="Pos.TLabel", width=8).pack(side="left")
        self.front_var = tk.StringVar()
        ttk.Entry(row2, textvariable=self.front_var, width=36, font=Theme.FONT_DIGIT).pack(
            side="left", ipady=4
        )
        ttk.Label(row2, text="  例: 06 16 18 19 28", style="CardHint.TLabel").pack(side="left")

        row3 = ttk.Frame(form, style="Card.TFrame")
        row3.pack(fill="x", pady=4)
        ttk.Label(row3, text="后区2个", style="Pos.TLabel", width=8).pack(side="left")
        self.back_var = tk.StringVar()
        ttk.Entry(row3, textvariable=self.back_var, width=36, font=Theme.FONT_DIGIT).pack(
            side="left", ipady=4
        )
        ttk.Label(row3, text="  例: 07 11", style="CardHint.TLabel").pack(side="left")

        row4 = ttk.Frame(form, style="Card.TFrame")
        row4.pack(fill="x", pady=(6, 0))
        BubbleButton(
            row4,
            text="保存并重新预测",
            command=self.save_and_predict,
            bg_color=Theme.DLT,
            hover_color=Theme.DLT_HOVER,
            width=148,
            height=38,
        ).pack(side="left")
        BubbleButton(
            row4,
            text="仅保存",
            command=self.save_only,
            bg_color=Theme.DLT,
            hover_color=Theme.DLT_HOVER,
            width=96,
            height=38,
        ).pack(side="left", padx=8)
        ttk.Button(row4, text="清空输入", style="Ghost.TButton", command=self.clear_input).pack(
            side="left"
        )
        ttk.Button(
            row4, text="对比上次预测", style="Ghost.TButton", command=self.compare_latest_draw
        ).pack(side="left", padx=8)
        ttk.Button(
            row4, text="从Excel导入", style="Ghost.TButton", command=self.import_from_excel
        ).pack(side="right")
        paned.add(form_wrap, weight=0)

        hist_wrap = ttk.Frame(paned, style="Card.TFrame")
        hist = ttk.LabelFrame(hist_wrap, text="历史开奖（可选中后删除）", padding=(12, 8))
        hist.pack(fill="both", expand=True)
        btn = ttk.Frame(hist, style="Card.TFrame")
        btn.pack(fill="x", pady=(0, 4))
        ttk.Button(btn, text="删除选中", style="Ghost.TButton", command=self.delete_selected).pack(
            side="left"
        )
        ttk.Button(
            btn, text="删除最新", style="Ghost.TButton", command=self.delete_latest_draw
        ).pack(side="left", padx=6)
        ttk.Button(btn, text="清空全部", style="Ghost.TButton", command=self.delete_all).pack(
            side="left"
        )
        ttk.Button(
            btn, text="刷新列表", style="Ghost.TButton", command=self.reload_history_list
        ).pack(side="right")

        list_box = ttk.Frame(hist, style="Card.TFrame")
        list_box.pack(fill="both", expand=True)
        self.hist = tk.Listbox(
            list_box,
            font=Theme.FONT_NUM,
            height=14,
            selectmode="extended",
            bg=Theme.SURFACE,
            fg=Theme.TEXT,
            highlightthickness=1,
            highlightbackground=Theme.BORDER,
            selectbackground=Theme.SELECT_DLT,
            selectforeground=Theme.TEXT,
            relief="flat",
            activestyle="none",
        )
        scroll = ttk.Scrollbar(list_box, orient="vertical", command=self.hist.yview)
        self.hist.configure(yscrollcommand=scroll.set)
        self.hist.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        hist_wrap.configure(height=320)
        paned.add(hist_wrap, weight=4)
        self.ensure_history_layout()

        self.fill_next_period()
        self.reload_history_list()

    def _on_inner_tab(self, _event=None) -> None:
        try:
            if self.nb.index(self.nb.select()) == 1:
                self.after(40, self.ensure_history_layout)
                self.after(200, self.ensure_history_layout)
        except tk.TclError:
            pass

    def ensure_history_layout(self) -> None:
        if self._input_paned is not None:
            _set_history_sash(self._input_paned, self.tab_input, self._form_max)

    def _emit_status(self, text: str) -> None:
        self.status_var.set(text)
        if self.status_callback:
            self.status_callback(f"大乐透 ｜ {text}")

    def refresh_status(self) -> None:
        draws = dltapi.load_history()
        if not draws:
            self._emit_status("暂无历史数据")
            self.next_period_var.set("")
            self.recent_var.set("")
            return
        last = draws[-1]
        self._emit_status(
            f"已录入 {len(draws)} 期 | 最新 {last['period']}期：前区 {dltapi.fmt_nums(last['front'])}  "
            f"后区 {dltapi.fmt_nums(last['back'])}"
        )
        self.next_period_var.set(f"预测目标：第 {int(last['period']) + 1} 期")
        parts = [
            f"{d['period']}:{dltapi.fmt_nums(d['front']).replace(' ', '')}+"
            f"{dltapi.fmt_nums(d['back']).replace(' ', '')}"
            for d in draws[-4:]
        ]
        self.recent_var.set("最近开奖：" + "  |  ".join(parts))

    def fill_next_period(self) -> None:
        draws = dltapi.load_history()
        self.period_var.set(str(int(draws[-1]["period"]) + 1) if draws else "")

    def reload_history_list(self) -> None:
        self.hist.delete(0, tk.END)
        self._period_map = []
        for d in reversed(dltapi.load_history()):
            line = (
                f"{d['period']}期  "
                f"前区 {dltapi.fmt_nums(d['front'])}   "
                f"后区 {dltapi.fmt_nums(d['back'])}"
            )
            if d.get("date"):
                line += f"   [{d['date']}]"
            self.hist.insert(tk.END, line)
            self._period_map.append(int(d["period"]))

    def clear_input(self) -> None:
        self.front_var.set("")
        self.back_var.set("")
        self.date_var.set("")
        self.fill_next_period()

    @staticmethod
    def _parse_nums(text: str) -> List[int]:
        raw = text.replace("，", " ").replace(",", " ").replace("、", " ").replace("-", " ")
        parts = [p for p in raw.split() if p.strip()]
        return [int(p) for p in parts]

    def _parse_input(self) -> Optional[tuple]:
        try:
            period = int(self.period_var.get().strip())
            front = self._parse_nums(self.front_var.get())
            back = self._parse_nums(self.back_var.get())
        except ValueError:
            messagebox.showerror("输入错误", "期号和号码必须是整数。")
            return None
        if len(front) != dltapi.FRONT_PICK:
            messagebox.showerror(
                "输入错误",
                f"前区必须正好 {dltapi.FRONT_PICK} 个号码（01-{dltapi.FRONT_MAX:02d}）。",
            )
            return None
        if len(back) != dltapi.BACK_PICK:
            messagebox.showerror(
                "输入错误",
                f"后区必须正好 {dltapi.BACK_PICK} 个号码（01-{dltapi.BACK_MAX:02d}）。",
            )
            return None
        return period, front, back, self.date_var.get().strip()

    def save_only(self) -> None:
        parsed = self._parse_input()
        if not parsed:
            return
        period, front, back, date = parsed
        try:
            dltapi.add_draw(period, front, back, date)
        except ValueError as e:
            messagebox.showerror("保存失败", str(e))
            return
        self.clear_input()
        self.reload_history_list()
        self.refresh_status()
        self.show_compare_for_draw(period, front, back, after_save=True)

    def save_and_predict(self) -> None:
        parsed = self._parse_input()
        if not parsed:
            return
        period, front, back, date = parsed
        try:
            dltapi.add_draw(period, front, back, date)
        except ValueError as e:
            messagebox.showerror("保存失败", str(e))
            return
        self.clear_input()
        self.reload_history_list()
        self.refresh_status()
        self.show_compare_for_draw(period, front, back, after_save=True)
        self.nb.select(self.tab_predict)
        self.run_predict()

    def import_from_excel(self) -> None:
        path = filedialog.askopenfilename(
            title="选择大乐透开奖 Excel",
            filetypes=[("Excel", "*.xlsx *.xls"), ("全部", "*.*")],
        )
        if not path:
            return
        try:
            draws = dltapi.import_excel(path)
        except Exception as e:
            messagebox.showerror("导入失败", str(e))
            return
        self.reload_history_list()
        self.refresh_status()
        self.fill_next_period()
        messagebox.showinfo("导入成功", f"当前共 {len(draws)} 期历史数据。")

    def delete_selected(self) -> None:
        sel = list(self.hist.curselection())
        if not sel:
            messagebox.showwarning("提示", "请先选中要删除的期号。")
            return
        periods = [self._period_map[i] for i in sel]
        if not messagebox.askyesno("确认删除", f"确定删除期号？\n{periods}"):
            return
        for period in periods:
            try:
                dltapi.delete_draw(period)
            except ValueError as e:
                messagebox.showerror("删除失败", str(e))
                return
        self.reload_history_list()
        self.refresh_status()
        self.fill_next_period()
        messagebox.showinfo("完成", f"已删除 {len(periods)} 条。")

    def delete_latest_draw(self) -> None:
        draws = dltapi.load_history()
        if not draws:
            messagebox.showerror("删除失败", "没有可删除的数据")
            return
        last = draws[-1]
        if not messagebox.askyesno(
            "确认删除",
            f"确定删除最新一期 {last['period']}？\n"
            f"前区 {dltapi.fmt_nums(last['front'])}\n后区 {dltapi.fmt_nums(last['back'])}",
        ):
            return
        try:
            dltapi.delete_latest()
        except ValueError as e:
            messagebox.showerror("删除失败", str(e))
            return
        self.reload_history_list()
        self.refresh_status()
        self.fill_next_period()
        messagebox.showinfo("完成", f"已删除 {last['period']}期。")

    def delete_all(self) -> None:
        if not messagebox.askyesno("危险操作", "确定清空大乐透全部历史数据？"):
            return
        if not messagebox.askyesno("再次确认", "真的要清空全部数据吗？"):
            return
        dltapi.clear_all_history()
        self.reload_history_list()
        self.refresh_status()
        self.fill_next_period()
        messagebox.showinfo("完成", "已清空全部数据。")

    def compare_latest_draw(self) -> None:
        draws = dltapi.load_history()
        if not draws:
            messagebox.showwarning("提示", "暂无开奖数据可对比。")
            return
        last = draws[-1]
        self.show_compare_for_draw(
            int(last["period"]),
            list(last["front"]),
            list(last["back"]),
            after_save=False,
        )

    def show_compare_for_draw(
        self,
        period: int,
        front: list,
        back: list,
        after_save: bool = False,
    ) -> None:
        pred = dltapi.load_last_prediction()
        if not pred or not pred.get("top"):
            if after_save:
                messagebox.showinfo(
                    "录入成功",
                    f"已录入 {period}期。\n尚无上次预测记录，请先预测后再对比下一期。",
                )
            else:
                messagebox.showwarning("无法对比", "没有上次预测 Top10，请先点击「开始预测」。")
            return
        try:
            result = dltapi.compare_prediction_to_draw(
                front, back, prediction=pred, actual_period=period
            )
        except ValueError as e:
            messagebox.showwarning("无法对比", str(e))
            return
        self._open_compare_window(result, after_save=after_save)

    def reset_weights(self) -> None:
        if not messagebox.askyesno("重置权重", "确定把大乐透所有算法权重恢复为初始值？"):
            return
        dltapi.reset_algo_weights()
        self._refresh_weight_label()
        messagebox.showinfo("完成", "算法权重已重置。下次预测将使用初始权重。")

    def _refresh_weight_label(self) -> None:
        w = dltapi.load_algo_weights()
        num = " ".join(
            f"{dltapi.ALGO_NAMES[k]}{w['number'][k]:.0%}" for k in ("gap", "freq", "bayes", "cooc")
        )
        self.weight_var.set(
            f"当前权重(已修正{w.get('updates', 0)}次)：{num} …  "
            f"组合分 number={w['combo']['number']:.0%} pattern={w['combo']['pattern']:.0%}"
        )

    def _open_compare_window(self, result: dict, after_save: bool = False) -> None:
        root = self.winfo_toplevel()
        win = tk.Toplevel(root)
        win.title(f"大乐透预测对比 - {result.get('actual_period')}期")
        win.geometry("900x680")
        win.configure(bg=Theme.BG)
        win.transient(root)

        s = result["summary"]
        head = ttk.Frame(win, padding=10)
        head.pack(fill="x")
        title = (
            "录入成功：Top10 对比 + 按准确率修正算法权重"
            if after_save
            else "Top10 对比 + 按准确率修正算法权重"
        )
        ttk.Label(head, text=title, style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(
            head,
            text=(
                f"开奖：前区 {dltapi.fmt_nums(result['actual_front'])}  "
                f"后区 {dltapi.fmt_nums(result['actual_back'])}"
                + ("" if result.get("period_match", True) else "  （期号与预测目标不一致）")
            ),
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(4, 0))
        summary = (
            f"最佳一注：第{s['best_rank']}名 命中 {s['best_total_hit']}/7"
            f"（前区{result['best']['front_hit']}/5 + 后区{result['best']['back_hit']}/2）"
            f"  准确率 {s['best_accuracy']:.1%}    |    "
            f"Top10平均准确率 {s['avg_accuracy']:.1%}    |    "
            f"号码池覆盖 {s['cover_total']}/7（{s['cover_accuracy']:.1%}）"
        )
        ttk.Label(head, text=summary, style="Status.TLabel").pack(anchor="w", pady=(8, 0))

        nb = ttk.Notebook(win, style="Inner.TNotebook")
        nb.pack(fill="both", expand=True, padx=10, pady=6)
        tab_pred = ttk.Frame(nb, padding=6)
        tab_algo = ttk.Frame(nb, padding=6)
        nb.add(tab_pred, text=" Top10命中 ")
        nb.add(tab_algo, text=" 算法排名与权重修正 ")

        cols = ("rank", "front", "back", "fh", "bh", "total", "acc", "hit_nums")
        tree = ttk.Treeview(tab_pred, columns=cols, show="headings", height=12)
        for c, title_w in (
            ("rank", ("排名", 50)),
            ("front", ("预测前区", 180)),
            ("back", ("预测后区", 80)),
            ("fh", ("前区命中", 70)),
            ("bh", ("后区命中", 70)),
            ("total", ("合计", 60)),
            ("acc", ("准确率", 70)),
            ("hit_nums", ("命中号码", 160)),
        ):
            tree.heading(c, text=title_w[0])
            tree.column(c, width=title_w[1], anchor="center")
        scroll = ttk.Scrollbar(tab_pred, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        for r in result["rows"]:
            hit_txt = (
                f"前{dltapi.fmt_nums(r['front_hit_nums']) or '-'} "
                f"后{dltapi.fmt_nums(r['back_hit_nums']) or '-'}"
            )
            tree.insert(
                "",
                "end",
                values=(
                    r["rank"],
                    dltapi.fmt_nums(r["front"]),
                    dltapi.fmt_nums(r["back"]),
                    f"{r['front_hit']}/5",
                    f"{r['back_hit']}/2",
                    f"{r['total_hit']}/7",
                    f"{r['accuracy']:.1%}",
                    hit_txt,
                ),
            )

        corr = result.get("correction")
        ttk.Label(
            tab_algo,
            text="准确率从高到低排序，提高排名靠前算法的权重（平滑更新，避免单期剧烈波动）。",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(0, 6))

        acols = ("group", "rank", "name", "hit", "acc", "old_w", "new_w", "delta")
        atree = ttk.Treeview(tab_algo, columns=acols, show="headings", height=14)
        for c, title_w in (
            ("group", ("类别", 80)),
            ("rank", ("准确率排名", 80)),
            ("name", ("算法", 110)),
            ("hit", ("命中", 70)),
            ("acc", ("准确率", 70)),
            ("old_w", ("原权重", 70)),
            ("new_w", ("新权重", 70)),
            ("delta", ("变化", 70)),
        ):
            atree.heading(c, text=title_w[0])
            atree.column(c, width=title_w[1], anchor="center")
        ascroll = ttk.Scrollbar(tab_algo, orient="vertical", command=atree.yview)
        atree.configure(yscrollcommand=ascroll.set)
        atree.pack(side="left", fill="both", expand=True)
        ascroll.pack(side="right", fill="y")

        if corr:
            for row in corr["ranking_rows"]:
                sign = "+" if row["delta"] >= 0 else ""
                atree.insert(
                    "",
                    "end",
                    values=(
                        row["group"],
                        row["rank"],
                        row["name"],
                        f"{row['total_hit']}/7",
                        f"{row['accuracy']:.1%}",
                        f"{row['old_weight']:.1%}",
                        f"{row['new_weight']:.1%}",
                        f"{sign}{row['delta']:.1%}",
                    ),
                )
            ttk.Label(
                tab_algo,
                text=f"已写入权重文件，累计修正 {corr['updates']} 次。下次「开始预测」将使用新权重。",
                style="Status.TLabel",
            ).pack(anchor="w", pady=6)
            nb.select(tab_algo)
        else:
            ttk.Label(
                tab_algo,
                text="本期未能完成算法修正（历史期数不足或其他原因）。",
                style="Hint.TLabel",
            ).pack(anchor="w")

        btns = ttk.Frame(win, padding=10)
        btns.pack(fill="x")

        def _copy() -> None:
            root.clipboard_clear()
            root.clipboard_append(dltapi.format_compare_text(result))
            messagebox.showinfo("已复制", "对比结果已复制到剪贴板。", parent=win)

        ttk.Button(btns, text="复制对比文本", style="Primary.TButton", command=_copy).pack(
            side="left"
        )
        ttk.Button(btns, text="关闭", style="Ghost.TButton", command=win.destroy).pack(side="right")
        self._refresh_weight_label()

    def run_predict(self) -> None:
        draws = dltapi.load_history()
        if len(draws) < 10:
            messagebox.showwarning("数据不足", "至少需要 10 期历史数据才能预测。")
            return
        try:
            top_n = max(1, min(30, int(self.top_n_var.get())))
        except ValueError:
            top_n = 10
            self.top_n_var.set("10")

        self.btn_predict.configure(state="disabled")
        self.update_idletasks()
        try:
            model = dltapi.EnsemblePredictor()
            model.fit(draws)
            model.save()
            pred_n = max(top_n, 10)
            preds = model.predict(top_n=pred_n)
            boards = model.hot_cold_boards(10)
            target_period = int(draws[-1]["period"]) + 1
            dltapi.save_last_prediction(
                target_period=target_period,
                preds=preds[:10],
                based_on_period=int(draws[-1]["period"]),
            )
        except Exception as e:
            messagebox.showerror("预测失败", str(e))
            self.btn_predict.configure(state="normal")
            return

        hot_f = " ".join(f"{n:02d}" for n, _ in boards["front_hot"][:10])
        hot_b = " ".join(f"{n:02d}" for n, _ in boards["back_hot"][:6])
        self.hot_var.set(f"前区热号参考：{hot_f}    后区热号参考：{hot_b}")

        for item in self.tree.get_children():
            self.tree.delete(item)
        for i, (front, back, prob, detail) in enumerate(preds[:top_n], 1):
            self.tree.insert(
                "",
                "end",
                values=(
                    i,
                    dltapi.fmt_nums(front),
                    dltapi.fmt_nums(back),
                    f"{prob:.2%}",
                    f"{detail['pattern']:.4f}",
                    f"{detail['cooccur']:.4f}",
                    f"{detail['gap_avg']:.4f}",
                ),
            )
        if preds:
            f, b, _, _ = preds[0]
            self.recommend_var.set(
                f"推荐 Top1：前区 {dltapi.fmt_nums(f)}   后区 {dltapi.fmt_nums(b)}    "
                f"（已保存第 {target_period} 期 Top10，开奖录入后可自动对比并修正算法）"
            )
        self._refresh_weight_label()
        self.refresh_status()
        self.btn_predict.configure(state="normal")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 快乐8
# ---------------------------------------------------------------------------


class Kl8TrendCanvas(tk.Canvas):
    """快乐8 基本走势图：红圈命中 + 灰色遗漏值。"""

    def __init__(self, parent, **kwargs) -> None:
        super().__init__(parent, bg="#FAFAFA", highlightthickness=0, **kwargs)
        self._rows = []
        self._cell_w = 18
        self._cell_h = 20
        self._label_w = 72
        self._header_h = 22

    def set_rows(self, rows) -> None:
        self._rows = rows or []
        self._redraw()

    def _redraw(self) -> None:
        self.delete("all")
        rows = self._rows
        if not rows:
            self.create_text(
                20,
                20,
                anchor="nw",
                text="暂无走势数据，请先导入 CSV 或录入开奖号码。",
                fill=Theme.MUTED,
                font=Theme.FONT,
            )
            self.configure(scrollregion=(0, 0, 400, 200))
            return

        cw, ch = self._cell_w, self._cell_h
        lw, hh = self._label_w, self._header_h
        cols = 80
        width = lw + cols * cw + 8
        height = hh + len(rows) * ch + 8

        # 表头 01-80，分区色带
        zone_colors = ["#FFF3E0", "#E3F2FD", "#F3E5F5", "#E8F5E9"]
        for n in range(1, cols + 1):
            x0 = lw + (n - 1) * cw
            zi = (n - 1) // 20
            self.create_rectangle(x0, 0, x0 + cw, hh, fill=zone_colors[zi], outline="#EEEEEE")
            self.create_text(
                x0 + cw / 2,
                hh / 2,
                text=f"{n:02d}",
                fill=Theme.MUTED if n % 5 else Theme.TEXT,
                font=("Consolas", 7),
            )
        self.create_text(lw / 2, hh / 2, text="期号", fill=Theme.TEXT, font=Theme.FONT_HINT)

        for r_i, row in enumerate(rows):
            y0 = hh + r_i * ch
            bg = "#FFFFFF" if r_i % 2 == 0 else "#F7F7F8"
            self.create_rectangle(0, y0, width, y0 + ch, fill=bg, outline="")
            self.create_text(
                lw / 2,
                y0 + ch / 2,
                text=str(row["period"]),
                fill=Theme.TEXT,
                font=("Consolas", 8),
            )
            # 分区竖线背景
            for zi in range(4):
                x0 = lw + zi * 20 * cw
                self.create_rectangle(
                    x0, y0, x0 + 20 * cw, y0 + ch, fill=zone_colors[zi], outline=""
                )
            for cell in row["cells"]:
                n = int(cell["num"])
                x = lw + (n - 1) * cw + cw / 2
                y = y0 + ch / 2
                if cell["hit"]:
                    r = min(cw, ch) / 2 - 2
                    self.create_oval(
                        x - r, y - r, x + r, y + r, fill=Theme.HIT_RED, outline=""
                    )
                    self.create_text(
                        x, y, text=f"{n:02d}", fill="#ffffff", font=("Consolas", 7, "bold")
                    )
                else:
                    omit = int(cell["omit"])
                    color = Theme.OMIT_GRAY
                    if omit >= 25:
                        color = "#FB8C00"
                    self.create_text(
                        x, y, text=str(omit), fill=color, font=("Consolas", 7)
                    )

        # 分区分隔线
        for zi in range(1, 4):
            x = lw + zi * 20 * cw
            self.create_line(x, 0, x, height, fill="#BDBDBD")

        self.configure(scrollregion=(0, 0, width, height))


class Kl8Panel(ttk.Frame):
    def __init__(
        self,
        master,
        status_callback: Optional[Callable[[str], None]] = None,
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)
        self.status_callback = status_callback
        self._period_map: List[int] = []
        self.status_var = tk.StringVar(value="")
        self._last_result = None
        self._build_ui()
        self._refresh_weight_label()
        self.refresh_status()

    def _build_ui(self) -> None:
        tip = card_frame(self)
        tip.pack(fill="x", padx=12, pady=(8, 4))
        inner = ttk.Frame(tip, style="Card.TFrame", padding=(16, 8))
        inner.pack(fill="x")
        accent = tk.Frame(inner, bg=Theme.KL8, width=4)
        accent.pack(side="left", fill="y", padx=(0, 12))
        txt = ttk.Frame(inner, style="Card.TFrame")
        txt.pack(side="left", fill="x", expand=True)
        ttk.Label(txt, text="快乐8 · 走势图选十", style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(txt, textvariable=self.status_var, style="Status.TLabel").pack(
            anchor="w", pady=(2, 0)
        )

        self.nb = ttk.Notebook(self, style="Inner.TNotebook")
        self.nb.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.tab_predict = ttk.Frame(self.nb, style="Card.TFrame", padding=12)
        self.tab_trend = ttk.Frame(self.nb, style="Card.TFrame", padding=8)
        self.tab_input = ttk.Frame(self.nb, style="Card.TFrame", padding=10)
        self.nb.add(self.tab_predict, text="  选十预测  ")
        self.nb.add(self.tab_trend, text="  基本走势图  ")
        self.nb.add(self.tab_input, text="  录入 / 导入  ")
        self._input_paned: Optional[ttk.Panedwindow] = None
        self._form_max = 300
        self._build_predict_tab()
        self._build_trend_tab()
        self._build_input_tab()
        self.nb.bind("<<NotebookTabChanged>>", self._on_inner_tab)

        foot = ttk.Frame(self, padding=(12, 0, 12, 6))
        foot.pack(fill="x")
        ttk.Label(
            foot,
            text=f"数据目录：{kl8api.DATA_PATH}　｜　模型仅供统计参考，请理性购彩",
            style="Hint.TLabel",
        ).pack(anchor="w")

    def _build_predict_tab(self) -> None:
        bar = ttk.Frame(self.tab_predict, style="Card.TFrame")
        bar.pack(fill="x")
        self.btn_predict = BubbleButton(
            bar,
            text="生成选十/选五/复式六",
            command=self.run_predict,
            bg_color=Theme.KL8,
            hover_color=Theme.KL8_HOVER,
            width=170,
            height=36,
        )
        self.btn_predict.pack(side="left")
        ttk.Button(bar, text="刷新状态", style="Ghost.TButton", command=self.refresh_status).pack(
            side="left", padx=8
        )
        ttk.Button(
            bar, text="对比上次预测", style="Ghost.TButton", command=self.compare_latest_draw
        ).pack(side="left", padx=4)
        ttk.Button(
            bar, text="复制结果", style="Ghost.TButton", command=self.copy_result
        ).pack(side="left", padx=4)
        ttk.Button(
            bar, text="重置规则权重", style="Ghost.TButton", command=self.reset_weights
        ).pack(side="left", padx=4)
        self.next_period_var = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.next_period_var, style="Status.TLabel").pack(side="right")

        self.recent_var = tk.StringVar(value="")
        ttk.Label(self.tab_predict, textvariable=self.recent_var, style="CardHint.TLabel").pack(
            anchor="w", pady=(8, 2)
        )
        self.weight_var = tk.StringVar(value="")
        ttk.Label(self.tab_predict, textvariable=self.weight_var, style="CardHint.TLabel").pack(
            anchor="w", pady=(0, 6)
        )

        table_box = ttk.Frame(self.tab_predict, style="Card.TFrame")
        table_box.pack(fill="both", expand=True)
        cols = ("group", "focus", "nums", "score")
        self.tree = ttk.Treeview(table_box, columns=cols, show="headings", height=10)
        heads = {
            "group": ("组别", 120),
            "focus": ("规则侧重", 260),
            "nums": ("号码", 320),
            "score": ("综合分", 70),
        }
        for c, (title, w) in heads.items():
            self.tree.heading(c, text=title)
            self.tree.column(c, width=w, anchor="center" if c != "focus" else "w")
        scroll = ttk.Scrollbar(table_box, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self.dan_var = tk.StringVar(value="")
        self.tuo_var = tk.StringVar(value="")
        ttk.Label(self.tab_predict, textvariable=self.dan_var, style="PanelTitle.TLabel").pack(
            anchor="w", pady=(8, 2)
        )
        ttk.Label(self.tab_predict, textvariable=self.tuo_var, style="CardHint.TLabel").pack(
            anchor="w"
        )

        tip_box = ttk.Frame(self.tab_predict, style="Card.TFrame")
        tip_box.pack(fill="both", expand=True, pady=(8, 0))
        ttk.Label(tip_box, text="规则分析", style="Card.TLabel").pack(anchor="w")
        self.tips = tk.Text(
            tip_box,
            height=8,
            wrap="word",
            font=Theme.FONT_HINT,
            bg="#F9F9FB",
            fg=Theme.TEXT,
            relief="flat",
            highlightthickness=1,
            highlightbackground=Theme.BORDER,
        )
        self.tips.pack(fill="both", expand=True, pady=(4, 0))
        self.tips.configure(state="disabled")

    def _build_trend_tab(self) -> None:
        bar = ttk.Frame(self.tab_trend, style="Card.TFrame")
        bar.pack(fill="x", pady=(0, 6))
        ttk.Label(bar, text="显示期数", style="Card.TLabel").pack(side="left")
        self.trend_n_var = tk.StringVar(value="50")
        ttk.Spinbox(
            bar, from_=20, to=300, width=5, textvariable=self.trend_n_var, values=(30, 50, 100, 300)
        ).pack(side="left", padx=(6, 10))
        BubbleButton(
            bar,
            text="刷新走势图",
            command=self.refresh_trend,
            bg_color=Theme.KL8,
            hover_color=Theme.KL8_HOVER,
            width=120,
            height=34,
        ).pack(side="left")
        ttk.Label(
            bar,
            text="红圈=开出　灰字=遗漏　橙色=遗漏≥25",
            style="CardHint.TLabel",
        ).pack(side="right")

        wrap = ttk.Frame(self.tab_trend, style="Card.TFrame")
        wrap.pack(fill="both", expand=True)
        self.trend_canvas = Kl8TrendCanvas(wrap)
        yscroll = ttk.Scrollbar(wrap, orient="vertical", command=self.trend_canvas.yview)
        xscroll = ttk.Scrollbar(wrap, orient="horizontal", command=self.trend_canvas.xview)
        self.trend_canvas.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.trend_canvas.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)

        self.trend_summary = tk.StringVar(value="")
        ttk.Label(self.tab_trend, textvariable=self.trend_summary, style="CardHint.TLabel").pack(
            anchor="w", pady=(6, 0)
        )

    def _build_input_tab(self) -> None:
        paned = ttk.Panedwindow(self.tab_input, orient=tk.VERTICAL)
        paned.pack(fill="both", expand=True)
        self._input_paned = paned
        self._form_max = 300

        form_wrap = ttk.Frame(paned, style="Card.TFrame")
        form = ttk.LabelFrame(form_wrap, text="录入最新开奖号码", padding=(14, 10))
        form.pack(fill="x", expand=False)
        ttk.Label(
            form,
            text="快乐8：每期 20 个号码（01-80），空格/逗号分隔",
            style="CardHint.TLabel",
        ).pack(anchor="w")

        row1 = ttk.Frame(form, style="Card.TFrame")
        row1.pack(fill="x", pady=(8, 4))
        ttk.Label(row1, text="期号", style="Pos.TLabel").pack(side="left")
        self.period_var = tk.StringVar(value="")
        ttk.Entry(row1, textvariable=self.period_var, width=14).pack(side="left", padx=(8, 16))
        ttk.Label(row1, text="日期", style="Pos.TLabel").pack(side="left")
        self.date_var = tk.StringVar(value="")
        ttk.Entry(row1, textvariable=self.date_var, width=14).pack(side="left", padx=(8, 0))

        row2 = ttk.Frame(form, style="Card.TFrame")
        row2.pack(fill="x", pady=4)
        ttk.Label(row2, text="号码", style="Pos.TLabel").pack(side="left")
        self.nums_var = tk.StringVar(value="")
        ttk.Entry(row2, textvariable=self.nums_var).pack(
            side="left", fill="x", expand=True, padx=(8, 0)
        )

        btns = ttk.Frame(form, style="Card.TFrame")
        btns.pack(fill="x", pady=(10, 0))
        BubbleButton(
            btns,
            text="保存本期",
            command=self.save_draw,
            bg_color=Theme.KL8,
            hover_color=Theme.KL8_HOVER,
            width=110,
            height=34,
        ).pack(side="left")
        ttk.Button(btns, text="导入 CSV", style="Ghost.TButton", command=self.import_csv).pack(
            side="left", padx=8
        )
        ttk.Button(
            btns, text="删除选中期", style="Ghost.TButton", command=self.delete_selected
        ).pack(side="left", padx=4)
        ttk.Button(btns, text="删除最新期", style="Ghost.TButton", command=self.delete_latest).pack(
            side="left", padx=4
        )
        ttk.Button(btns, text="清空历史", style="Ghost.TButton", command=self.clear_history).pack(
            side="left", padx=4
        )

        hist_wrap = ttk.Frame(paned, style="Card.TFrame")
        hist = ttk.LabelFrame(hist_wrap, text="历史开奖", padding=(8, 6))
        hist.pack(fill="both", expand=True)
        cols = ("period", "date", "nums")
        self.hist = ttk.Treeview(hist, columns=cols, show="headings", height=12)
        self.hist.heading("period", text="期号")
        self.hist.heading("date", text="日期")
        self.hist.heading("nums", text="开奖号码（20）")
        self.hist.column("period", width=90, anchor="center")
        self.hist.column("date", width=100, anchor="center")
        self.hist.column("nums", width=620, anchor="w")
        hscroll = ttk.Scrollbar(hist, orient="vertical", command=self.hist.yview)
        self.hist.configure(yscrollcommand=hscroll.set)
        self.hist.pack(side="left", fill="both", expand=True)
        hscroll.pack(side="right", fill="y")

        paned.add(form_wrap, weight=0)
        paned.add(hist_wrap, weight=1)
        self.after(80, self.ensure_history_layout)

    def _on_inner_tab(self, _event=None) -> None:
        try:
            tab = self.nb.select()
            if tab == str(self.tab_input):
                self.after(40, self.ensure_history_layout)
            elif tab == str(self.tab_trend):
                self.after(40, self.refresh_trend)
        except tk.TclError:
            pass

    def ensure_history_layout(self) -> None:
        if self._input_paned is not None:
            _set_history_sash(self._input_paned, self.tab_input, self._form_max)

    def _notify(self, text: str) -> None:
        if self.status_callback:
            self.status_callback(text)

    def _refresh_weight_label(self) -> None:
        w = kl8api.load_algo_weights()
        parts = [
            f"{kl8api.ALGO_NAMES[k]}{w[k]:.0%}"
            for k in ("gap3", "gap4", "gap6", "hot_cold", "zone_sym")
        ]
        self.weight_var.set("规则权重：" + " · ".join(parts))

    def refresh_status(self) -> None:
        draws = kl8api.load_history()
        if not draws:
            self.status_var.set("尚未录入快乐8历史数据，可导入「快乐8_近100期开奖数据.csv」")
            self.recent_var.set("")
            self.next_period_var.set("")
            self._fill_history([])
            self._notify("快乐8：无历史数据")
            return
        last = draws[-1]
        self.status_var.set(
            f"已录入 {len(draws)} 期 | 最新 {last['period']}期：{kl8api.fmt_nums(last['nums'])}"
        )
        self.recent_var.set(
            "近5期："
            + " ｜ ".join(
                f"{d['period']}:{kl8api.fmt_nums(d['nums']).replace(' ', '')}"
                for d in draws[-5:]
            )
        )
        self.next_period_var.set(f"下一期建议：{int(last['period']) + 1}")
        self._fill_history(draws)
        self._suggest_period(draws)
        self._notify(f"快乐8：{len(draws)} 期 | 最新 {last['period']}")

    def _fill_history(self, draws) -> None:
        self.hist.delete(*self.hist.get_children())
        self._period_map = []
        for d in reversed(draws):
            self.hist.insert(
                "",
                "end",
                values=(d["period"], d.get("date", ""), kl8api.fmt_nums(d["nums"])),
            )
            self._period_map.append(int(d["period"]))

    def _suggest_period(self, draws) -> None:
        if not self.period_var.get().strip() and draws:
            self.period_var.set(str(int(draws[-1]["period"]) + 1))

    def _parse_nums(self, text: str) -> List[int]:
        return kl8api._parse_num_token(text)

    def save_draw(self) -> None:
        try:
            period = int("".join(c for c in self.period_var.get() if c.isdigit()))
            nums = self._parse_nums(self.nums_var.get())
            if len(set(nums)) != kl8api.DRAW_COUNT:
                raise ValueError(f"需要正好 {kl8api.DRAW_COUNT} 个不重复号码")
            kl8api.add_draw(period, nums, self.date_var.get().strip())
            self.nums_var.set("")
            self.refresh_status()
            self.refresh_trend()
            messagebox.showinfo("成功", f"已保存 {period} 期")
        except Exception as e:
            messagebox.showerror("录入失败", str(e))

    def import_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="导入快乐8 CSV",
            filetypes=[("CSV", "*.csv"), ("全部文件", "*.*")],
        )
        if not path:
            return
        try:
            draws = kl8api.import_csv(path)
            self.refresh_status()
            self.refresh_trend()
            messagebox.showinfo("导入完成", f"当前共 {len(draws)} 期历史数据")
        except Exception as e:
            messagebox.showerror("导入失败", str(e))

    def delete_selected(self) -> None:
        sel = self.hist.selection()
        if not sel:
            messagebox.showwarning("提示", "请先选中要删除的期号")
            return
        item = sel[0]
        period = int(self.hist.item(item, "values")[0])
        if not messagebox.askyesno("确认", f"删除期号 {period}？"):
            return
        try:
            kl8api.delete_draw(period)
            self.refresh_status()
            self.refresh_trend()
        except Exception as e:
            messagebox.showerror("删除失败", str(e))

    def delete_latest(self) -> None:
        draws = kl8api.load_history()
        if not draws:
            return
        last = draws[-1]
        if not messagebox.askyesno(
            "确认",
            f"删除最新期 {last['period']}？\n{kl8api.fmt_nums(last['nums'])}",
        ):
            return
        kl8api.delete_latest()
        self.refresh_status()
        self.refresh_trend()

    def clear_history(self) -> None:
        if not messagebox.askyesno("确认", "清空全部快乐8历史？此操作不可恢复"):
            return
        kl8api.clear_all_history()
        self.refresh_status()
        self.refresh_trend()

    def reset_weights(self) -> None:
        kl8api.reset_algo_weights()
        self._refresh_weight_label()
        messagebox.showinfo("完成", "规则权重已恢复默认")

    def refresh_trend(self) -> None:
        draws = kl8api.load_history()
        try:
            n = int(self.trend_n_var.get())
        except ValueError:
            n = 50
        rows = kl8api.build_trend_matrix(draws, recent=n)
        self.trend_canvas.set_rows(rows)
        if not rows:
            self.trend_summary.set("")
            return
        last = rows[-1]
        hits = [c["num"] for c in last["cells"] if c["hit"]]
        omit = kl8api.compute_omissions(draws)
        cold = sorted((o, n) for n, o in omit.items() if o >= 25)[-8:]
        cold_txt = " ".join(f"{n:02d}({o})" for o, n in cold) if cold else "-"
        gaps = kl8api.find_gaps(hits)
        gap_txt = " ".join(f"{a:02d}-{b:02d}" for a, b in gaps[:8])
        self.trend_summary.set(
            f"最新 {last['period']} | 开出 {kl8api.fmt_nums(hits)} | "
            f"空位 {gap_txt} | 遗漏≥25：{cold_txt}"
        )

    def run_predict(self) -> None:
        draws = kl8api.load_history()
        if len(draws) < 5:
            messagebox.showwarning("提示", "历史数据不足 5 期，请先导入 CSV")
            return
        try:
            model = kl8api.TrendPredictor()
            model.fit(draws)
            result = model.predict_groups()
            kl8api.save_last_prediction(result)
            self._last_result = result
            self._show_result(result)
            self._refresh_weight_label()
            self._notify(
                f"快乐8：已生成选十+选五+复式六 → {result['target_period']} 期"
            )
        except Exception as e:
            messagebox.showerror("预测失败", str(e))

    def _show_result(self, result: dict) -> None:
        self.tree.delete(*self.tree.get_children())
        self.next_period_var.set(f"目标期 {result['target_period']}")
        all_groups = list(result.get("groups") or [])
        all_groups.extend(result.get("groups_x5") or [])
        all_groups.extend(result.get("groups_x6") or [])
        for g in all_groups:
            self.tree.insert(
                "",
                "end",
                values=(
                    g["name"],
                    g["focus"],
                    kl8api.fmt_nums(g["nums"]),
                    f"{g.get('score', 0):.3f}",
                ),
            )
        self.dan_var.set(f"胆码（选五/选六定胆）：{kl8api.fmt_nums(result['dan'])}")
        self.tuo_var.set(
            f"拖码建议：{kl8api.fmt_nums(result.get('tuo', [])[:10])}　｜　"
            "复式六＝选五复式6注或选六；选七以上可用胆拖"
        )
        self.tips.configure(state="normal")
        self.tips.delete("1.0", "end")
        self.tips.insert("1.0", "\n".join(f"· {t}" for t in result.get("tips", [])))
        self.tips.configure(state="disabled")

    def copy_result(self) -> None:
        if not self._last_result:
            messagebox.showwarning("提示", "请先生成预测")
            return
        text = kl8api.format_prediction_text(self._last_result)
        self.clipboard_clear()
        self.clipboard_append(text)
        messagebox.showinfo("已复制", "预测结果已复制到剪贴板")

    def compare_latest_draw(self) -> None:
        draws = kl8api.load_history()
        if not draws:
            messagebox.showwarning("提示", "没有开奖数据可对比")
            return
        pred = kl8api.load_last_prediction()
        if not pred:
            messagebox.showwarning("提示", "没有上次预测记录，请先生成选十")
            return
        last = draws[-1]
        try:
            result = kl8api.compare_prediction_to_draw(
                last["nums"], period=int(last["period"])
            )
        except Exception as e:
            messagebox.showerror("对比失败", str(e))
            return
        self._refresh_weight_label()
        win = tk.Toplevel(self)
        win.title("快乐8 预测对比")
        win.geometry("720x420")
        win.configure(bg=Theme.BG)
        ttk.Label(
            win,
            text=f"开奖 {result['period']}：{kl8api.fmt_nums(result['actual'])}",
            style="PanelTitle.TLabel",
        ).pack(anchor="w", padx=12, pady=(12, 6))
        cols = ("name", "nums", "hit", "hit_nums")
        tree = ttk.Treeview(win, columns=cols, show="headings", height=6)
        for c, t, w in (
            ("name", "组别", 120),
            ("nums", "预测选十", 280),
            ("hit", "命中", 60),
            ("hit_nums", "命中号码", 200),
        ):
            tree.heading(c, text=t)
            tree.column(c, width=w, anchor="center")
        tree.pack(fill="both", expand=True, padx=12, pady=6)
        for g in result["groups"]:
            tree.insert(
                "",
                "end",
                values=(
                    g["name"],
                    kl8api.fmt_nums(g["nums"]),
                    f"{g['hit_count']}/10",
                    kl8api.fmt_nums(g["hit_nums"]) or "-",
                ),
            )
        ttk.Label(
            win,
            text=f"胆码命中：{kl8api.fmt_nums(result['dan_hit']) or '-'} / {kl8api.fmt_nums(result['dan'])}"
            + ("　｜　已按命中微调规则权重" if result.get("adjusted") else ""),
            style="CardHint.TLabel",
        ).pack(anchor="w", padx=12, pady=(0, 10))
        ttk.Button(
            win,
            text="复制对比",
            style="Ghost.TButton",
            command=lambda: (
                self.clipboard_clear(),
                self.clipboard_append(kl8api.format_compare_text(result)),
            ),
        ).pack(anchor="e", padx=12, pady=(0, 12))


# 主窗口
# ---------------------------------------------------------------------------


class LotteryApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        pl3api.ensure_data_files()
        dltapi.ensure_data_files()
        kl8api.ensure_data_files()
        self.title(APP_TITLE)
        self.geometry(WIN_SIZE)
        self.minsize(*WIN_MIN)
        apply_theme(self)
        self.hero_status = tk.StringVar(value="")
        self._active = 0
        self._panels: List[ttk.Frame] = []
        self._build_hero()
        self._build_body()
        self.after(150, self._startup_predict)

    def _build_hero(self) -> None:
        hero = tk.Frame(self, bg=Theme.HEADER, height=70)
        hero.pack(fill="x")
        hero.pack_propagate(False)
        left = tk.Frame(hero, bg=Theme.HEADER)
        left.pack(side="left", fill="y", padx=20, pady=12)
        ttk.Label(left, text=APP_TITLE, style="HeroTitle.TLabel").pack(anchor="w")
        ttk.Label(
            left, text="排列三 · 大乐透 · 快乐8｜算法独立分区", style="HeroSub.TLabel"
        ).pack(anchor="w")
        right = tk.Frame(hero, bg=Theme.HEADER)
        right.pack(side="right", fill="y", padx=20, pady=18)
        ttk.Label(right, textvariable=self.hero_status, style="HeroStatus.TLabel").pack(
            anchor="e"
        )

    def _set_hero_status(self, text: str) -> None:
        self.hero_status.set(text)

    def _build_body(self) -> None:
        wrap = ttk.Frame(self, padding=(14, 10, 14, 12))
        wrap.pack(fill="both", expand=True)

        switch_row = ttk.Frame(wrap)
        switch_row.pack(fill="x", pady=(0, 10))
        self.segment = SegmentedControl(
            switch_row,
            labels=["排列三", "大乐透", "快乐8"],
            command=self._switch_zone,
            width=420,
            height=42,
        )
        self.segment.pack(anchor="center")

        self.content = ttk.Frame(wrap)
        self.content.pack(fill="both", expand=True)

        self.pl3_panel = Pl3Panel(self.content, status_callback=self._set_hero_status)
        self.dlt_panel = DltPanel(self.content, status_callback=self._set_hero_status)
        self.kl8_panel = Kl8Panel(self.content, status_callback=self._set_hero_status)
        self._panels = [self.pl3_panel, self.dlt_panel, self.kl8_panel]
        self.pl3_panel.pack(fill="both", expand=True)
        self.pl3_panel.refresh_status()

    def _switch_zone(self, index: int) -> None:
        self._active = index
        for p in self._panels:
            p.pack_forget()
        panel = self._panels[index]
        panel.pack(fill="both", expand=True)
        panel.refresh_status()
        if hasattr(panel, "ensure_history_layout"):
            self.after(40, panel.ensure_history_layout)
            self.after(160, panel.ensure_history_layout)
        if index == 2 and hasattr(panel, "refresh_trend"):
            self.after(80, panel.refresh_trend)

    def _startup_predict(self) -> None:
        try:
            self.pl3_panel.run_predict()
        except Exception:
            pass


def main() -> None:
    # 保证从 lottery_suite 目录运行时可找到 pl3 / dlt / kl8
    if getattr(sys, "frozen", False):
        pass
    app = LotteryApp()
    app.mainloop()


if __name__ == "__main__":
    main()
