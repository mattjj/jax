"""Why transposing shard_map is subtle, and how tracking replication fixes it.

A short explainer in the style of 3Blue1Brown, built with Manim Community
Edition (tested with v0.21). No LaTeX: all text is Pango (Text / MarkupText).

Fonts: CMU Serif (Debian/Ubuntu package fonts-cmu) and JetBrains Mono
(fonts-jetbrains-mono).

Render every scene at 720p30:
    manim -qm --disable_caching shmap_transpose.py \
        S1Devices S2Psum S3WholePieces S4Pvary S5Toy S6OldSeam S7DesignSpace
"""

import re

import numpy as np
from manim import *

# ---------------------------------------------------------------- style ----

BG = "#0f1013"
C_TEXT = "#ECECEC"
C_DIM = "#8A8A92"
C_FAINT = "#3C3D44"
C_VAL = "#58C4DD"   # forward values
C_COT = "#F4D345"   # cotangents
C_COMM = "#B98EDC"  # communication
C_OK = "#83C167"    # correct / free
C_BAD = "#FC6255"   # wrong / wasteful

# One shade per source device, so stacked pieces show where they came from.
VAL_SHADES = ["#236B80", "#2F97B8", "#58C4DD", "#A6E1EF"]
COT_SHADES = ["#B8860B", "#E0B21C", "#F4D345", "#FFE99A"]

SERIF = "CMU Serif"
MONO = "JetBrains Mono"

N = 4
COLS = [-4.5, -1.5, 1.5, 4.5]  # x position of each device's column
BASE = -1.9                    # baseline for bars
CAP_Y = -3.35                  # caption position
CAP_SIZE = 30
CAP_W = 13.0


def tx(s, size=32, color=C_TEXT, font=SERIF, **kw):
    """Serif text with Pango markup (<i>, <b>, <sub>, <span>)."""
    return MarkupText(s, font=font, font_size=size, color=color, **kw)


def code(s, size=24, color=C_TEXT):
    """Monospace text; also accepts Pango markup."""
    return MarkupText(s, font=MONO, font_size=size, color=color)


def mono(s):
    """Inline code inside serif markup."""
    return f'<span font_family="{MONO}" size="smaller">{s}</span>'


def sym(base, sub=None):
    """An italic symbol with an optional subscript, e.g. sym('w', 'k')."""
    s = f"<i>{base}</i>"
    if sub is not None:
        sub = str(sub)
        s += f"<sub>{sub if sub.isdigit() else '<i>' + sub + '</i>'}</sub>"
    return s


def strip_markup(s):
    return re.sub(r"<[^>]+>", "", s)


def bar(h, x, base=BASE, w=0.8, color=C_VAL, op=0.85):
    h = max(h, 0.02)
    r = Rectangle(width=w, height=h, stroke_color=color, stroke_width=1.5,
                  fill_color=color, fill_opacity=op)
    return r.move_to([x, base + h / 2, 0])


def glyph(heights, cx, base, color=C_COT, w=0.24, gap=0.36):
    """A small 4-bar picture of one value (one bar per device)."""
    xs = [cx + (k - 1.5) * gap for k in range(N)]
    return VGroup(*[bar(h, x, base, w=w, color=color) for h, x in zip(heights, xs)])


def glyph_arcs(cx, y, gap=0.36):
    """Communication arcs between neighbouring bars of a glyph."""
    return VGroup(*[CurvedDoubleArrow([cx + (k - 1.5) * gap, y, 0],
                                      [cx + (k - 0.5) * gap, y, 0],
                                      angle=-PI / 2, color=C_COMM, stroke_width=2.5,
                                      tip_length=0.1) for k in range(N - 1)])


def device_axis(cols=COLS, base=BASE, half=0.62, size=22, dy=0.36):
    return VGroup(*[
        VGroup(Line([x - half, base, 0], [x + half, base, 0],
                    stroke_width=2, color=C_FAINT),
               tx(f"device {k}", size=size, color=C_DIM).move_to([x, base - dy, 0]))
        for k, x in enumerate(cols)])


def legend(*items):
    rows = VGroup()
    for color, label in items:
        sq = Square(0.24, stroke_width=0, fill_color=color, fill_opacity=0.9)
        rows.add(VGroup(sq, tx(label, size=24, color=C_DIM)).arrange(RIGHT, buff=0.15))
    rows.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
    return rows.to_corner(UR, buff=0.4)


def check_mark(color=C_OK, size=0.34):
    m = VMobject(stroke_color=color, stroke_width=7)
    m.set_points_as_corners([[-0.5, 0.05, 0], [-0.15, -0.35, 0], [0.5, 0.45, 0]])
    return m.scale(size)


def x_mark(color=C_BAD, size=0.28):
    return VGroup(Line([-0.5, -0.5, 0], [0.5, 0.5, 0]),
                  Line([-0.5, 0.5, 0], [0.5, -0.5, 0])).set_stroke(color, 7).scale(size)


def dashed_box(w, h, x, base, color=C_OK):
    r = Rectangle(width=w, height=h).move_to([x, base + h / 2, 0])
    return DashedVMobject(r, num_dashes=18).set_stroke(color, 2)


def keep_in_frame(mob, margin=0.3):
    xmax = config.frame_width / 2 - margin
    if mob.get_right()[0] > xmax:
        mob.shift((xmax - mob.get_right()[0]) * RIGHT)
    if mob.get_left()[0] < -xmax:
        mob.shift((-xmax - mob.get_left()[0]) * RIGHT)
    return mob


class Base(Scene):
    """Shared helpers: captions that stay up long enough to read, headers,
    and the all-reduce animation (every device receives every piece)."""

    def setup(self):
        self.camera.background_color = BG
        self.cap = None
        self.cap_t0 = 0.0
        self.cap_hold = 0.0
        self.head = None

    def now(self):
        return self.renderer.time

    def settle(self):
        """Wait until the current caption has been on screen long enough."""
        if self.cap is not None:
            left = self.cap_hold - (self.now() - self.cap_t0)
            if left > 0.04:
                self.wait(left)

    def say(self, *lines, anims=(), hold=None):
        """Replace the caption (1-2 lines), optionally alongside other animations.

        Give each extra animation its own run_time; the caption swap itself
        takes 0.7 s. The caption then stays up for about 3 s per line, longer
        for long lines.
        """
        self.settle()
        new = VGroup(*[tx(l, size=CAP_SIZE) for l in lines]).arrange(DOWN, buff=0.16)
        if new.width > CAP_W:
            new.scale_to_fit_width(CAP_W)
        new.move_to([0, CAP_Y, 0])
        if self.cap is None:
            swap = FadeIn(new, shift=0.1 * UP, run_time=0.5)
        else:
            swap = Succession(FadeOut(self.cap, run_time=0.3),
                              FadeIn(new, shift=0.1 * UP, run_time=0.4))
        t0 = self.now()
        self.play(swap, *anims)
        self.cap = new
        self.cap_t0 = t0 + 0.6
        n = sum(len(strip_markup(l)) for l in lines)
        self.cap_hold = hold if hold is not None else max(2.7 * len(lines), n / 16)

    def header(self, text):
        self.head = tx(text, size=24, color=C_DIM).to_corner(UL, buff=0.4)
        self.play(FadeIn(self.head), run_time=0.5)

    def chapter(self, text):
        big = tx(text, size=46)
        self.play(FadeIn(big, shift=0.2 * UP), run_time=0.7)
        self.wait(0.7)
        small = tx(text, size=24, color=C_DIM).to_corner(UL, buff=0.4)
        self.play(Transform(big, small), run_time=0.7)
        self.head = big

    def clear_stage(self, keep=(), run_time=0.7):
        keep = set(keep) | {self.cap, self.head}
        mobs = [m for m in self.mobjects if m not in keep]
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=run_time)

    def finish(self):
        self.settle()
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
        self.wait(0.3)

    def all_reduce(self, bars, heights, colors, cols=COLS, base=BASE, width=0.8,
                   run_time=2.4, lag=0.2, arc=0.8):
        """Copies of every device's bar fly to every device and stack up there.

        Returns, per destination column, the list of stacked pieces.
        """
        stacks = [[] for _ in cols]
        groups = []
        for s, b in enumerate(bars):
            anims = []
            for d, cx in enumerate(cols):
                off = sum(heights[:s])
                tgt = bar(heights[s], cx, base + off, w=width, color=colors[s])
                mob = b if d == s else b.copy()
                dx = cx - b.get_x()
                pa = 0 if abs(dx) < 1e-6 else -arc * np.sign(dx)
                anims.append(Transform(mob, tgt, path_arc=pa))
                stacks[d].append(mob)
            groups.append(AnimationGroup(*anims))
        self.play(LaggedStart(*groups, lag_ratio=lag), run_time=run_time)
        return stacks

    def merge(self, stacks, total, cols=COLS, base=BASE, width=0.8, color=C_COT,
              run_time=0.7):
        """Dissolve each stack into one solid bar."""
        news = [bar(total, cx, base, w=width, color=color) for cx in cols]
        self.play(*[FadeIn(n) for n in news],
                  *[FadeOut(m) for st in stacks for m in st], run_time=run_time)
        return news


# ------------------------------------------------------------ scene 1 -----

class S1Devices(Base):
    def construct(self):
        title = tx(f"Why transposing {mono('shard_map')} is subtle", size=50)
        sub = tx("and how tracking replication fixes it", size=30, color=C_DIM)
        VGroup(title, sub).arrange(DOWN, buff=0.4).move_to(0.2 * UP)
        self.play(Write(title), run_time=1.8)
        self.play(FadeIn(sub, shift=0.15 * UP), run_time=0.8)
        self.wait(1.8)
        self.play(FadeOut(title), FadeOut(sub), run_time=0.6)

        self.header("<i>N</i> devices")
        axis = device_axis()
        nlab = tx("<i>N</i> = 4", size=30).to_corner(UR, buff=0.45)
        self.play(LaggedStart(*[FadeIn(d, shift=0.1 * UP) for d in axis], lag_ratio=0.15),
                  FadeIn(nlab), run_time=1.4)

        hs = [1.3, 2.4, 0.8, 1.9]
        bars = [bar(h, x) for h, x in zip(hs, COLS)]
        labs = [tx(sym("x", k), size=30, color=C_VAL).next_to(b, UP, 0.15)
                for k, b in enumerate(bars)]
        self.say(f"Inside {mono('shard_map')}, every value is really <i>N</i> values.",
                 anims=[LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars],
                                    lag_ratio=0.12, run_time=1.4),
                        LaggedStart(*[FadeIn(l, shift=0.1 * UP) for l in labs],
                                    lag_ratio=0.12, run_time=1.4)])
        self.wait(0.8)

        tag = tx("varying", size=34, color=C_VAL).move_to([0, 2.55, 0])
        self.say("A value is <b>varying</b> when its entries differ across devices…",
                 anims=[FadeIn(tag, shift=0.1 * DOWN, run_time=0.8),
                        LaggedStart(*[Wiggle(b, scale_value=1.05) for b in bars],
                                    lag_ratio=0.1, run_time=1.6)])

        E = 1.6
        new = [bar(E, x) for x in COLS]
        tag2 = tx("invariant  (replicated)", size=34, color=C_VAL).move_to(tag)
        eq = DashedLine([COLS[0] - 0.9, BASE + E, 0], [COLS[-1] + 0.9, BASE + E, 0],
                        dash_length=0.1, stroke_width=2, color=C_VAL).set_opacity(0.6)
        self.say("…and <b>invariant</b>, or replicated, when all <i>N</i> entries are equal.",
                 anims=[*[Transform(b, nb, run_time=1.5) for b, nb in zip(bars, new)],
                        *[l.animate(run_time=1.5).next_to(nb, UP, 0.15)
                          for l, nb in zip(labs, new)],
                        FadeTransform(tag, tag2, run_time=1.0)])
        self.play(Create(eq), run_time=0.8)
        self.finish()


# ------------------------------------------------------------ scene 2 -----

class S2Psum(Base):
    def construct(self):
        self.chapter("psum, forward and backward")
        axis = device_axis()
        leg = legend((C_VAL, "value"), (C_COT, "cotangent"))
        self.play(*[FadeIn(d) for d in axis], FadeIn(leg), run_time=0.8)

        XV = [0.55, 1.0, 0.35, 0.8]
        Y = sum(XV)
        xb = [bar(h, x, color=VAL_SHADES[k]) for k, (h, x) in enumerate(zip(XV, COLS))]
        xl = [tx(sym("x", k), size=28, color=C_VAL).next_to(b, UP, 0.15)
              for k, b in enumerate(xb)]
        fx = code("y = psum(x)", size=30).move_to([0, 2.6, 0])
        self.say("psum, an all-reduce sum: every device ends up",
                 "holding the sum of all devices' inputs.",
                 anims=[LaggedStart(*[GrowFromEdge(b, DOWN) for b in xb],
                                    lag_ratio=0.1, run_time=1.2),
                        LaggedStart(*[FadeIn(l) for l in xl], lag_ratio=0.1, run_time=1.2),
                        FadeIn(fx, run_time=0.8)])
        self.play(*[FadeOut(l) for l in xl], run_time=0.4)
        stacks = self.all_reduce(xb, XV, VAL_SHADES, run_time=2.8)
        yb = self.merge(stacks, Y, color=C_VAL)
        yl = [tx("<i>y</i>", size=30, color=C_VAL).next_to(b, UP, 0.15) for b in yb]
        eqline = DashedLine([COLS[0] - 0.9, BASE + Y, 0], [COLS[-1] + 0.9, BASE + Y, 0],
                            dash_length=0.1, stroke_width=2, color=C_VAL).set_opacity(0.6)
        self.say("Its output is always invariant.",
                 anims=[*[FadeIn(l, shift=0.1 * UP) for l in yl],
                        Create(eqline, run_time=1.0)])

        yl2 = [tx(f"{sym('w', k)} · <i>y</i>", size=30).next_to(b, UP, 0.15)
               for k, b in enumerate(yb)]
        self.say("Then each device multiplies its copy of <i>y</i> by its own "
                 "<i>w</i><sub><i>k</i></sub>.",
                 anims=[*[FadeTransform(a, b, run_time=1.0) for a, b in zip(yl, yl2)]])

        # Backward pass.
        WV = [0.9, 0.35, 1.25, 0.6]
        W = sum(WV)
        yb_t = [bar(Y, x - 0.45, w=0.7, color=C_VAL) for x in COLS]
        yl3 = [tx("<i>y</i>", size=30, color=C_VAL).next_to(b, UP, 0.15) for b in yb_t]
        cb = [bar(h, x + 0.45, w=0.7, color=C_COT) for h, x in zip(WV, COLS)]
        cl = [tx(sym("w", k), size=28, color=C_COT).next_to(b, UP, 0.15)
              for k, b in enumerate(cb)]
        self.settle()
        self.play(*[Transform(a, b) for a, b in zip(yb, yb_t)],
                  *[FadeTransform(a, b) for a, b in zip(yl2, yl3)], run_time=1.0)
        self.say("Backward: the cotangent of <i>y</i> on device <i>k</i> is "
                 "<i>w</i><sub><i>k</i></sub>.",
                 anims=[LaggedStart(*[GrowFromEdge(b, DOWN) for b in cb],
                                    lag_ratio=0.12, run_time=1.2),
                        LaggedStart(*[FadeIn(l) for l in cl], lag_ratio=0.12, run_time=1.2)])
        self.say("A constant value can have a non-constant cotangent.",
                 anims=[*[Indicate(b, color=C_VAL, scale_factor=1.04, run_time=1.2)
                          for b in yb],
                        LaggedStart(*[Wiggle(b, scale_value=1.08) for b in cb],
                                    lag_ratio=0.1, run_time=2.0)])

        cross = tx(f"cotangent of {sym('x', 'j')} = Σ<sub><i>k</i></sub> {sym('w', 'k')}",
                   size=32, color=C_COT).move_to([0, 2.6, 0])
        self.say(f"Every {sym('x', 'j')} affects every device's <i>y</i>,",
                 f"so {sym('x', 'j')}'s cotangent is Σ<sub><i>k</i></sub> {sym('w', 'k')}: "
                 "a cross-device sum.",
                 anims=[*[FadeOut(m, run_time=0.8) for m in yb + yl3 + cl],
                        FadeOut(eqline, run_time=0.8), FadeOut(fx, run_time=0.8),
                        *[b.animate(run_time=0.8).set_color(COT_SHADES[k])
                          for k, b in enumerate(cb)]])
        stacks = self.all_reduce(cb, WV, COT_SHADES, run_time=2.8)
        sb = self.merge(stacks, W, color=C_COT)
        sl = [tx(f"Σ<sub><i>k</i></sub> {sym('w', 'k')}", size=28, color=C_COT)
              .next_to(b, UP, 0.15) for b in sb]
        self.play(*[FadeIn(l, shift=0.1 * UP) for l in sl], FadeIn(cross), run_time=0.8)
        self.say("With no extra information, psum's transpose must sum:",
                 "it is itself a psum.")
        self.finish()


# ------------------------------------------------------------ scene 3 -----

class S3WholePieces(Base):
    def construct(self):
        self.chapter("Whole vs. pieces")
        axis = device_axis()
        leg = legend((C_COT, "cotangent"), (C_COMM, "communication"))
        self.play(*[FadeIn(d) for d in axis], FadeIn(leg), run_time=0.8)

        G = 1.2
        GK = [0.2, 0.45, 0.25, 0.3]
        RX = -6.05

        def row_label(s):
            return tx(s, size=28).move_to([RX, BASE + 0.6, 0])

        def labels(texts, bars_):
            return [tx(t, size=30, color=C_COT).next_to(b, UP, 0.15)
                    for t, b in zip(texts, bars_)]

        bars = [bar(G, x, color=C_COT) for x in COLS]
        labs = labels(["<i>g</i>"] * N, bars)
        row = row_label("whole")
        self.say("Two ways to store the cotangent <i>g</i> of a replicated value.",
                 "<b>Whole</b>: every device holds the full cotangent <i>g</i>.",
                 anims=[LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars],
                                    lag_ratio=0.1, run_time=1.2),
                        LaggedStart(*[FadeIn(l) for l in labs], lag_ratio=0.1, run_time=1.2),
                        FadeIn(row, run_time=0.8)])

        pb = [bar(h, x, color=COT_SHADES[k]) for k, (h, x) in enumerate(zip(GK, COLS))]
        pl = labels([sym("g", k) for k in range(N)], pb)
        row2 = row_label("pieces")
        sumf = tx("<i>g</i> = " + " + ".join(sym("g", k) for k in range(N)),
                  size=34, color=C_COT).move_to([0, 2.4, 0])
        self.say("<b>Pieces</b>: device <i>k</i> holds a piece <i>g</i><sub><i>k</i></sub>,",
                 "and the cotangent is the sum of the pieces.",
                 anims=[*[Transform(a, b, run_time=1.2) for a, b in zip(bars, pb)],
                        *[FadeTransform(a, b, run_time=1.2) for a, b in zip(labs, pl)],
                        FadeTransform(row, row2, run_time=1.0),
                        FadeIn(sumf, shift=0.1 * DOWN, run_time=1.0)])

        # Pieces -> whole: a psum, with communication.
        ya = BASE + G + 0.75
        arrows = [CurvedDoubleArrow([COLS[k] + 0.45, ya, 0], [COLS[k + 1] - 0.45, ya, 0],
                                    angle=-PI / 3, color=C_COMM, stroke_width=3,
                                    tip_length=0.16) for k in range(N - 1)]
        comm = tx("psum: communication", size=30, color=C_COMM).move_to([0, 2.4, 0])
        self.say("Pieces → whole is a psum: every device needs every piece.",
                 anims=[FadeOut(sumf, run_time=0.6), FadeIn(comm, run_time=0.8),
                        *[FadeOut(l, run_time=0.5) for l in pl],
                        LaggedStart(*[Create(a) for a in arrows], lag_ratio=0.2,
                                    run_time=1.2)])
        stacks = self.all_reduce(bars, GK, COT_SHADES, run_time=2.4)
        wb = self.merge(stacks, G, color=C_COT)
        wl = labels(["<i>g</i>"] * N, wb)
        row3 = row_label("whole")
        self.play(*[FadeIn(l) for l in wl], FadeTransform(row2, row3),
                  *[FadeOut(a) for a in arrows], run_time=0.8)

        # Whole -> pieces: a local split, no communication.
        qb = [bar(G / 4, x, color=C_COT) for x in COLS]
        ql = labels(["<i>g</i>/4"] * N, qb)
        local = tx("local split: no communication", size=30, color=C_OK).move_to([0, 2.4, 0])
        row4 = row_label("pieces")
        self.say("Whole → pieces is a local split: <i>g</i>/4 on each device…",
                 anims=[FadeOut(comm, run_time=0.6), FadeIn(local, run_time=0.8),
                        *[Transform(a, b, run_time=1.2) for a, b in zip(wb, qb)],
                        *[FadeTransform(a, b, run_time=1.2) for a, b in zip(wl, ql)],
                        FadeTransform(row3, row4, run_time=1.0)])
        ab = [bar(G if k == 0 else 0, x, color=C_COT) for k, x in enumerate(COLS)]
        al = labels(["<i>g</i>", "0", "0", "0"], ab)
        self.say("…or <i>g</i> on one device and 0 on the others. No arrows either way.",
                 anims=[*[Transform(a, b, run_time=1.2) for a, b in zip(wb, ab)],
                        *[FadeTransform(a, b, run_time=1.2) for a, b in zip(ql, al)]])
        self.say("Pieces to whole is the only step that communicates.")

        # Reading whole values as if they were pieces: off by a factor of N.
        wb2 = [bar(G, x, color=C_COT) for x in COLS]
        wl2 = labels(["<i>g</i>"] * N, wb2)
        row5 = row_label("whole")
        self.settle()
        self.play(*[Transform(a, b) for a, b in zip(wb, wb2)],
                  *[FadeTransform(a, b) for a, b in zip(al, wl2)],
                  FadeTransform(row4, row5), FadeOut(local), run_time=1.0)
        self.say("Reading whole-stored values as if they were pieces",
                 "is off by a factor of <i>N</i>.",
                 anims=[*[FadeOut(m, run_time=0.6) for m in list(axis) + wl2 + [row5]]])
        SX, RX2 = -1.6, 1.3
        tg = [bar(G, SX, BASE + k * G, color=COT_SHADES[k]) for k in range(N)]
        floor = Line([SX - 0.8, BASE, 0], [RX2 + 0.8, BASE, 0], stroke_width=2, color=C_FAINT)
        self.play(FadeIn(floor, run_time=0.5),
                  LaggedStart(*[Transform(b, t, path_arc=-0.5 * np.sign(SX - b.get_x()))
                                for b, t in zip(wb, tg)], lag_ratio=0.15, run_time=2.0))
        ref = dashed_box(0.8, G, RX2, BASE, color=C_COT)
        lab4 = tx("sum = 4<i>g</i>", size=32, color=C_BAD).next_to(tg[-1], LEFT, 0.4)
        labg = tx("true cotangent: <i>g</i>", size=28, color=C_COT).next_to(ref, UP, 0.2)
        xm = x_mark().next_to(lab4, DOWN, 0.25)
        self.play(Create(ref), FadeIn(labg), FadeIn(lab4), Create(xm), run_time=1.0)
        self.finish()


# ------------------------------------------------------------ scene 4 -----

class S4Pvary(Base):
    def construct(self):
        self.chapter("pvary: the fan-out")
        leg = legend((C_VAL, "value"), (C_COT, "cotangent"), (C_COMM, "communication"))

        wnode = Circle(radius=0.42, color=C_VAL, stroke_width=3).move_to([0, 2.45, 0])
        wtx = tx("<i>w</i>", size=38, color=C_VAL).move_to(wnode)
        wtag = tx("replicated", size=22, color=C_DIM).next_to(wnode, RIGHT, 0.25)
        pv = RoundedRectangle(width=1.7, height=0.66, corner_radius=0.14,
                              stroke_color=C_TEXT, stroke_width=2.5).move_to([0, 1.15, 0])
        pvt = code("pvary", size=26).move_to(pv)
        pvg = VGroup(pv, pvt)
        e0 = Arrow(wnode.get_bottom(), pv.get_top(), buff=0.06, stroke_width=3,
                   color=C_DIM, tip_length=0.18)
        UY = -0.95
        uses = [RoundedRectangle(width=1.8, height=0.66, corner_radius=0.14,
                                 stroke_color=C_DIM, stroke_width=2).move_to([x, UY, 0])
                for x in COLS]
        ut = [tx(f"<i>w</i> · {sym('x', k)}", size=28).move_to(u) for k, u in enumerate(uses)]
        edges = [Arrow(pv.get_bottom(), u.get_top(), buff=0.06, stroke_width=3,
                       color=C_DIM, tip_length=0.18) for u in uses]
        dls = [tx(f"device {k}", size=22, color=C_DIM).next_to(u, DOWN, 0.2)
               for k, u in enumerate(uses)]
        types = tx("invariant → varying", size=24, color=C_DIM).next_to(pv, RIGHT, 0.4)

        self.play(FadeIn(leg), GrowFromCenter(wnode), FadeIn(wtx), FadeIn(wtag),
                  run_time=1.0)
        self.say("pvary marks where a replicated value fans out into per-device uses.",
                 anims=[GrowArrow(e0, run_time=0.6), FadeIn(pvg, run_time=0.6),
                        FadeIn(types, run_time=0.8),
                        LaggedStart(*[GrowArrow(e) for e in edges], lag_ratio=0.1,
                                    run_time=1.4),
                        LaggedStart(*[FadeIn(VGroup(u, t)) for u, t in zip(uses, ut)],
                                    lag_ratio=0.1, run_time=1.6),
                        LaggedStart(*[FadeIn(d) for d in dls], lag_ratio=0.1,
                                    run_time=1.6)])

        wcopies = [bar(0.5, x - 1.2, UY - 0.33, w=0.26, color=C_VAL) for x in COLS]
        noop = tx("forward: no-op", size=26, color=C_OK).move_to([-3.4, 1.15, 0])
        self.say("Forward, it's a no-op: every device already holds <i>w</i>. Nothing moves.",
                 anims=[FadeIn(noop, run_time=0.8),
                        *[FadeIn(b, run_time=0.8) for b in wcopies],
                        Indicate(pvg, color=C_OK, scale_factor=1.1, run_time=1.2)])

        GK = [0.25, 0.45, 0.2, 0.35]
        pieces = [bar(h, x + 1.2, UY - 0.33, w=0.26, color=COT_SHADES[k])
                  for k, (h, x) in enumerate(zip(GK, COLS))]
        plabs = [tx(sym("g", k), size=24, color=C_COT).next_to(b, UP, 0.1)
                 for k, b in enumerate(pieces)]
        self.say("Backward, each use sends back its own piece of <i>w</i>'s cotangent…",
                 anims=[FadeOut(noop, run_time=0.6),
                        *[FadeOut(b, run_time=0.6) for b in wcopies],
                        LaggedStart(*[GrowFromEdge(b, DOWN) for b in pieces],
                                    lag_ratio=0.1, run_time=1.0),
                        LaggedStart(*[FadeIn(l) for l in plabs], lag_ratio=0.1,
                                    run_time=1.0)])
        SX, SB = -1.5, pv.get_bottom()[1]
        tg = [bar(GK[k], SX, SB + sum(GK[:k]), w=0.4, color=COT_SHADES[k])
              for k in range(N)]
        psl = tx("backward: psum", size=26, color=C_COMM).move_to([-3.45, 1.15, 0])
        # Arc each piece below the straight path so it doesn't cross the pvary box.
        moves = [Transform(p, t, path_arc=0.5 if t.get_x() > p.get_x() else -0.5)
                 for p, t in zip(pieces, tg)]
        self.say("…and pvary sums them: a psum.",
                 anims=[*[FadeOut(l, run_time=0.5) for l in plabs],
                        LaggedStart(*moves, lag_ratio=0.12, run_time=2.0),
                        FadeIn(psl, run_time=0.8)])
        merged = bar(sum(GK), SX, SB, w=0.4, color=C_COT)
        gl = tx("<i>g</i>", size=28, color=C_COT).next_to(merged, UP, 0.12)
        self.play(FadeIn(merged), *[FadeOut(p) for p in pieces], FadeIn(gl),
                  Indicate(pvg, color=C_COMM, scale_factor=1.1), run_time=1.0)
        self.say("pvary: a no-op forward, a psum backward.")

        # The dual: psum with types.
        self.settle()
        self.clear_stage(keep=[leg])
        xs = [-4.9, -1.6, 1.9, 4.7]
        ys = [1.7, 0.65, -0.45]
        hdr = [tx(s, size=24, color=C_DIM) for s in ["type", "forward", "backward"]]
        r1 = [code("pvary", size=28), tx("invariant → varying", size=28),
              tx("no-op", size=28, color=C_OK), tx("psum", size=28, color=C_COMM)]
        r2 = [code("psum", size=28), tx("varying → invariant", size=28),
              tx("all-reduce", size=28, color=C_COMM), tx("free", size=28, color=C_OK)]
        for j, h in enumerate(hdr):
            h.move_to([xs[j + 1], ys[0], 0])
        for j, c in enumerate(r1):
            c.move_to([xs[j], ys[1], 0])
        for j, c in enumerate(r2):
            c.move_to([xs[j], ys[2], 0])
        rule = Line([-6.2, 1.25, 0], [6.2, 1.25, 0], stroke_width=1.5, color=C_FAINT)
        self.play(*[FadeIn(h) for h in hdr], Create(rule),
                  LaggedStart(*[FadeIn(c, shift=0.1 * UP) for c in r1], lag_ratio=0.15),
                  run_time=1.2)
        self.say("Its partner, psum with types, goes varying → invariant:",
                 "an all-reduce forward, and free backward.",
                 anims=[LaggedStart(*[FadeIn(c, shift=0.1 * UP) for c in r2],
                                    lag_ratio=0.15, run_time=1.2)])
        box = SurroundingRectangle(r2[3], color=C_OK, buff=0.15, corner_radius=0.08)
        self.say("Its output's cotangent is known to be whole,",
                 "so it's just handed to every device.",
                 anims=[Create(box, run_time=0.8)])
        self.finish()


# ------------------------------------------------------------ scene 5 -----

class S5Toy(Base):
    def construct(self):
        self.chapter("A toy example")
        S = 0.3   # bar height per unit

        dim = f'<span foreground="{C_DIM}">'
        src = VGroup(
            code(f"w = 2.0              {dim}# replicated: same on every device</span>", size=22),
            code(f"x = [1., 2., 3., 4.] {dim}# device k holds x[k]</span>", size=22),
            code(f"loss = sum over devices of w * x[k]   {dim}# = 20</span>", size=22),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to([0, 2.3, 0])
        axis = device_axis()
        xb = [bar((k + 1) * S, x, color=C_VAL) for k, x in enumerate(COLS)]
        xl = [tx(str(k + 1), size=26, color=C_VAL).next_to(b, UP, 0.12)
              for k, b in enumerate(xb)]
        fwd = tx("loss = 2·1 + 2·2 + 2·3 + 2·4 = 20", size=30).move_to([0, 0.7, 0])
        self.play(FadeIn(src), *[FadeIn(d) for d in axis], run_time=1.0)
        self.say("<i>w</i> = 2 is replicated, device <i>k</i> holds <i>x</i>[<i>k</i>],"
                 " and the loss is 20.",
                 anims=[LaggedStart(*[GrowFromEdge(b, DOWN) for b in xb], lag_ratio=0.1,
                                    run_time=1.2),
                        LaggedStart(*[FadeIn(l) for l in xl], lag_ratio=0.1, run_time=1.2),
                        FadeIn(fwd, run_time=1.0)])

        # Backward: each device only has its own piece, x[k].
        pb = [bar((k + 1) * S, x, color=COT_SHADES[k]) for k, x in enumerate(COLS)]
        self.say("Backward: device <i>k</i> can only compute its own piece, <i>x</i>[<i>k</i>].",
                 anims=[FadeOut(src, run_time=0.8), FadeOut(fwd, run_time=0.8),
                        *[Transform(a, b, run_time=1.0) for a, b in zip(xb, pb)],
                        *[l.animate(run_time=1.0).set_color(C_COT) for l in xl]])

        # Slide the devices to the left half to make room for a scoreboard.
        C5 = [-5.9, -4.35, -2.8, -1.25]
        BW = 0.6
        axis5 = device_axis(cols=C5, half=0.5, size=20)
        pb5 = [bar((k + 1) * S, x, w=BW, color=COT_SHADES[k]) for k, x in enumerate(C5)]
        region = tx("grad <i>w</i>", size=30, color=C_COT).move_to([-3.6, 2.45, 0])
        self.say("The true gradient is sum(<i>x</i>) = 10: it needs a cross-device sum.",
                 anims=[*[Transform(a, b, run_time=1.2) for a, b in zip(axis, axis5)],
                        *[Transform(a, b, run_time=1.2) for a, b in zip(xb, pb5)],
                        *[l.animate(run_time=1.2).next_to(b, UP, 0.12)
                          for l, b in zip(xl, pb5)],
                        FadeIn(region, run_time=1.0)])

        def score_row(mech, result, ok, y, note=None):
            m = code(mech, size=20, color=C_DIM)
            parts = [tx(result, size=28), check_mark() if ok else x_mark()]
            if note:
                parts.append(tx(note, size=22, color=C_DIM))
            line2 = VGroup(*parts).arrange(RIGHT, buff=0.25)
            g = VGroup(m, line2).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
            if g.width > 6.4:
                g.scale_to_fit_width(6.4)
            return g.move_to([0.35, y, 0], aligned_edge=LEFT)

        def show_row(row):
            self.play(FadeIn(row[0]), FadeIn(row[1][0]), run_time=0.6)
            self.play(Create(row[1][1]), *[FadeIn(p) for p in row[1][2:]], run_time=0.5)

        heights = [(k + 1) * S for k in range(N)]

        def reset(old_bars, old_labels):
            nb = [bar((k + 1) * S, x, w=BW, color=COT_SHADES[k]) for k, x in enumerate(C5)]
            nl = [tx(str(k + 1), size=26, color=C_COT).next_to(b, UP, 0.12)
                  for k, b in enumerate(nb)]
            self.play(*[FadeOut(m) for m in old_bars + old_labels],
                      *[FadeIn(m) for m in nb + nl], run_time=0.7)
            return nb, nl

        def sum_up(bars_, run_time):
            stacks = self.all_reduce(bars_, heights, COT_SHADES, cols=C5, width=BW,
                                     run_time=run_time)
            tb = self.merge(stacks, 10 * S, cols=C5, width=BW, color=C_COT)
            tl = [tx("10", size=26, color=C_COT).next_to(b, UP, 0.12) for b in tb]
            self.play(*[FadeIn(l) for l in tl], run_time=0.5)
            return tb, tl

        # Case 1: check_vma=True.
        self.say(f"With {mono('check_vma=True')}, JAX inserts pvary(<i>w</i>) where <i>w</i>"
                 " meets <i>x</i>,", "so the backward pass sums the pieces: 10.",
                 anims=[*[FadeOut(l, run_time=0.5) for l in xl]])
        tb, tl = sum_up(xb, 2.2)
        show_row(score_row("check_vma=True: auto pvary(w)", "grad <i>w</i> = 10", True, 2.45))

        # Case 2: untyped transpose, no pvary.
        self.settle()
        pb, pl = reset(tb, tl)
        box = SurroundingRectangle(VGroup(pb[0], pl[0]), color=C_BAD, buff=0.12)
        self.say("An untyped transpose, with no pvary, returns device 0's piece: 1.",
                 anims=[*[m.animate(run_time=0.8).set_opacity(0.25) for m in pb[1:] + pl[1:]],
                        Create(box, run_time=0.8)])
        show_row(score_row("untyped: no conversions, no pvary", "grad <i>w</i> = 1", False,
                           1.25, note="device 0's piece"))

        # Case 3: mark the fan-out explicitly.
        self.settle()
        self.play(FadeOut(box), *[m.animate.set_opacity(1) for m in pb[1:]],
                  *[m.animate.set_opacity(1) for m in pl[1:]], run_time=0.5)
        # set_opacity(1) also turns the fill fully opaque; restore the bar look.
        for b in pb:
            b.set_fill(opacity=0.85)
        self.say("Marking the fan-out by hand, pcast(<i>w</i>, 'i', to='varying'), "
                 "also gives 10.",
                 anims=[*[FadeOut(l, run_time=0.5) for l in pl]])
        tb, tl = sum_up(pb, 1.8)
        show_row(score_row("pcast(w, 'i', to='varying')", "grad <i>w</i> = 10", True, 0.05))

        # Case 4: also mark x, which was already per-device.
        self.settle()
        gx = [bar(2 * S, x, w=BW, color=COT_SHADES[k]) for k, x in enumerate(C5)]
        gxl = [tx("2", size=26, color=C_COT).next_to(b, UP, 0.12) for b in gx]
        region2 = tx("grad <i>x</i>", size=30, color=C_COT).move_to(region)
        self.say("Now also mark <i>x</i>, which was already per-device…",
                 anims=[*[FadeOut(m, run_time=0.8) for m in tb + tl],
                        *[FadeIn(m, run_time=0.8) for m in gx + gxl],
                        FadeTransform(region, region2, run_time=0.8)])
        self.play(*[FadeOut(l) for l in gxl], run_time=0.4)
        stacks = self.all_reduce(gx, [2 * S] * N, COT_SHADES, cols=C5, width=BW, run_time=2.0)
        eb = self.merge(stacks, 8 * S, cols=C5, width=BW, color=C_COT)
        el = [tx("8", size=26, color=C_BAD).next_to(b, UP, 0.12) for b in eb]
        self.play(*[FadeIn(l) for l in el], run_time=0.5)
        show_row(score_row("also mark x (already per-device)", "grad <i>x</i> = [8 8 8 8]",
                           False, -1.15, note="should be [2 2 2 2]"))
        self.say("…and the extra backward psum sums four devices' separate gradients.")
        self.say("The backward psums have to go exactly where the fan-outs are.")
        self.finish()


# ------------------------------------------------------------ scene 6 -----

class S6OldSeam(Base):
    def construct(self):
        self.chapter(f"The old seam: {mono('check_vma=False')}")
        leg = legend((C_COT, "cotangent"), (C_COMM, "communication"))
        box = RoundedRectangle(width=5.4, height=3.0, corner_radius=0.3,
                               stroke_color=C_DIM, stroke_width=2.5).move_to([0, 0.45, 0])
        btitle = tx(f"{mono('shard_map')} body: no types inside", size=24,
                    color=C_DIM).move_to([0, 1.62, 0])
        inl = VGroup(tx("replicated input", size=26), code("in_specs=P()", size=20, color=C_DIM)
                     ).arrange(DOWN, buff=0.12).move_to([-4.9, 1.85, 0])
        outl = VGroup(tx("replicated output", size=26),
                      code("out_specs=P()", size=20, color=C_DIM)
                      ).arrange(DOWN, buff=0.12).move_to([4.9, 1.85, 0])
        GB = -0.55
        G = 1.2
        back = Arrow([5.9, -1.6, 0], [-5.9, -1.6, 0], color=C_COT, stroke_width=3, buff=0,
                     tip_length=0.2, max_tip_length_to_length_ratio=0.05)
        backl = tx("backward pass", size=22, color=C_COT).next_to(back, DOWN, 0.1)

        self.play(Create(box), FadeIn(btitle), FadeIn(inl), FadeIn(outl), FadeIn(leg),
                  run_time=1.2)
        self.say(f"With {mono('check_vma=False')}, replication is known only at the boundary.")

        g = glyph([G] * N, 4.9, GB)
        lab = tx("whole", size=22, color=C_COT).next_to(g, DOWN, 0.12)
        self.say("Inside the body, every cotangent is stored as pieces.",
                 anims=[GrowArrow(back, run_time=1.0), FadeIn(backl, run_time=1.0),
                        LaggedStart(*[GrowFromEdge(b, DOWN) for b in g], lag_ratio=0.1,
                                    run_time=1.0),
                        FadeIn(lab, run_time=1.0)])

        def boundary_label(text, color, x):
            t = tx(text, size=30, color=color)
            bgr = BackgroundRectangle(t, color=BG, fill_opacity=1, buff=0.1)
            return VGroup(bgr, t).move_to([x, 0.95, 0])

        divn = boundary_label("÷ <i>N</i>", C_COT, 2.7)
        g_in = glyph([G / N] * N, 1.3, GB)
        lab_in = tx("pieces", size=22, color=C_COT).next_to(g_in, DOWN, 0.12)
        self.say("A replicated output's cotangent is divided by <i>N</i> on the way in…",
                 anims=[FadeIn(divn, run_time=0.8), Transform(g, g_in, run_time=1.8),
                        FadeTransform(lab, lab_in, run_time=1.8)])
        g_mid = glyph([G / N] * N, -1.3, GB)
        lab_mid = tx("pieces", size=22, color=C_COT).next_to(g_mid, DOWN, 0.12)
        self.play(Transform(g, g_mid), Transform(lab_in, lab_mid), run_time=1.4)

        psl = boundary_label("psum", C_COMM, -2.7)
        g_out = glyph([G] * N, -4.9, GB)
        lab_out = tx("whole", size=22, color=C_COT).next_to(g_out, DOWN, 0.12)
        arcs = glyph_arcs(-4.9, GB + G + 0.15)
        self.say("…and a replicated input gets a defensive psum on the way out.",
                 anims=[FadeIn(psl, run_time=0.8), Transform(g, g_out, run_time=1.8),
                        FadeTransform(lab_in, lab_out, run_time=1.8),
                        Succession(Wait(1.0), Create(arcs, run_time=0.8))])
        self.say("psum in the body transposes to psum.")

        # The cursed identity.
        self.settle()
        self.clear_stage()
        name = tx("the “cursed identity”", size=26, color=C_DIM).move_to([0, 2.85, 0])
        ci = code("shard_map(lambda x: x, in_specs=P(), out_specs=P())", size=22)
        ci.move_to([0, 2.3, 0])
        keep_in_frame(ci)
        LX, CX = -1.5, -1.1

        def row(label, text, y, color=C_DIM):
            l = tx(label, size=26, color=color).move_to([LX, y, 0], aligned_edge=RIGHT)
            c = code(text, size=22).move_to([CX, y, 0], aligned_edge=LEFT)
            return l, c

        l1, c1 = row("transpose", "psum(x / N)", 1.25)
        l2, c2 = row("transpose again", "psum(psum(x / N / N))", 0.4)
        dots = code("…", size=22).move_to([CX, -0.35, 0], aligned_edge=LEFT)
        l3, c3 = row("transpose, with vma types", "lambda x: x", -1.35, color=C_OK)
        ck = check_mark().next_to(c3, RIGHT, 0.35)
        self.say("The “cursed identity” just returns its replicated input.",
                 anims=[FadeIn(name, run_time=0.8), FadeIn(ci, run_time=0.8)])
        self.say("Transposing it gives psum(<i>x</i> / <i>N</i>)…",
                 anims=[FadeIn(l1, run_time=0.8), FadeIn(c1, shift=0.1 * DOWN, run_time=0.8)])
        self.say("…and again, psum(psum(<i>x</i> / <i>N</i> / <i>N</i>)). It keeps growing.",
                 anims=[FadeIn(l2, run_time=0.8), FadeIn(c2, shift=0.1 * DOWN, run_time=0.8),
                        Succession(Wait(1.0), FadeIn(dots, run_time=0.6))])
        self.say("With vma types, it transposes to the identity.",
                 anims=[FadeIn(l3, run_time=0.8), FadeIn(c3, run_time=0.8),
                        Create(ck, run_time=0.8)])
        self.say("It's always correct, but it communicates whether or not that's needed,",
                 "and the divide-by-<i>N</i> hurts numerics.")
        self.say("Always correct, rarely efficient.")
        self.finish()


# ------------------------------------------------------------ scene 7 -----

class S7DesignSpace(Base):
    def construct(self):
        self.chapter("The design space")
        q = VGroup(tx("For each replicated value:", size=32),
                   tx("is its cotangent stored <b>whole</b>, or as <b>pieces</b>?", size=32)
                   ).arrange(DOWN, buff=0.15).move_to([0, 2.65, 0])
        self.say("One question organizes the design space.",
                 anims=[FadeIn(q, shift=0.1 * DOWN, run_time=1.0)])

        AY = 0.85
        L, R = -4.6, 4.6
        axis = Line([L, AY, 0], [R, AY, 0], stroke_width=3, color=C_DIM)
        ticks = VGroup(*[Line([x, AY - 0.12, 0], [x, AY + 0.12, 0], stroke_width=3,
                              color=C_DIM) for x in (L, 0, R)])
        left = VGroup(tx("always pieces", size=28),
                      tx("classic pmap / all-mapped", size=22, color=C_DIM)
                      ).arrange(DOWN, buff=0.1).move_to([L, AY - 0.62, 0])
        right = VGroup(tx("always whole", size=28),
                       tx("global view", size=22, color=C_DIM),
                       tx("(sharding-in-types without unreduced)", size=20, color=C_DIM)
                       ).arrange(DOWN, buff=0.08)
        right.move_to([R, AY - 0.1, 0], aligned_edge=UP)
        keep_in_frame(right)

        # The waste at each end.
        wl_g = glyph([0.45] * N, L - 0.9, -1.95)
        wl_a = glyph_arcs(L - 0.9, -1.35)
        wl_t = VGroup(tx("unneeded psum", size=24, color=C_BAD),
                      tx("(cotangent already whole)", size=20, color=C_DIM)
                      ).arrange(DOWN, buff=0.08).next_to(wl_g, RIGHT, 0.4)
        waste_l = VGroup(wl_g, wl_a, wl_t)
        keep_in_frame(waste_l)
        waste_r = VGroup(tx("FSDP: all-reduce + slice", size=24, color=C_BAD),
                         tx(f'instead of <span foreground="{C_OK}">reduce-scatter</span>',
                            size=22, color=C_DIM)
                         ).arrange(DOWN, buff=0.1).move_to([R, -1.6, 0])
        keep_in_frame(waste_r)

        self.play(Create(axis), FadeIn(ticks), run_time=1.0)
        self.say("Always pieces, as in classic pmap: simple, but wasteful when",
                 "the cotangent is already complete on every device, like the loss.",
                 anims=[FadeIn(left, shift=0.1 * UP, run_time=0.8),
                        FadeIn(wl_g, run_time=1.0), FadeIn(wl_t, run_time=1.0),
                        Succession(Wait(0.6), Create(wl_a, run_time=0.8))])
        self.say("Always whole, the global view: simple, but wasteful for FSDP,",
                 "where an all-reduce plus a slice happens instead of a reduce-scatter.",
                 anims=[FadeIn(right, shift=0.1 * UP, run_time=0.8),
                        FadeIn(waste_r, run_time=1.2)])

        mid = tx("per-value choice, tracked by types", size=28, color=C_OK).move_to(
            [0, AY + 0.95, 0])
        d1 = Dot([L, AY, 0], radius=0.11, color=C_TEXT)
        d2 = Dot([R, AY, 0], radius=0.11, color=C_TEXT)
        t1 = tx(mono("shard_map"), size=24).next_to(d1, UP, 0.18)
        t2 = tx("sharding-in-types", size=24).next_to(d2, UP, 0.18)
        t1.add_updater(lambda m: m.next_to(d1, UP, 0.18))
        t2.add_updater(lambda m: m.next_to(d2, UP, 0.18))
        n1 = tx(f"vma in {mono('shard_map')}", size=22)
        n2 = tx("reduced / unreduced in sharding-in-types", size=22)
        # Below the end labels and above the waste notes, so nothing overlaps.
        notes = VGroup(n1, n2).arrange(DOWN, buff=0.12).move_to([0, AY - 1.5, 0])
        self.say("In between: a per-value choice, tracked by types.",
                 anims=[FadeIn(mid, shift=0.1 * DOWN, run_time=1.0)])
        self.play(FadeIn(d1), FadeIn(d2), FadeIn(t1), FadeIn(t2), run_time=0.6)
        self.say(f"vma in {mono('shard_map')}, reduced / unreduced in sharding-in-types:",
                 "two systems pushed to the same place from opposite ends.",
                 anims=[d1.animate(run_time=2.4).move_to([-0.12, AY, 0]),
                        d2.animate(run_time=2.4).move_to([0.12, AY, 0]),
                        Succession(Wait(1.8), AnimationGroup(FadeOut(t1, run_time=0.6),
                                                             FadeOut(t2, run_time=0.6))),
                        Succession(Wait(2.0), FadeIn(notes, run_time=1.0))])
        t1.clear_updaters()
        t2.clear_updaters()
        self.settle()
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.0)
        self.cap = None

        close = VGroup(tx("In an untyped system,", size=36),
                       tx("the efficient backward pass isn't derivable by autodiff,", size=36),
                       tx("and the hand-written one isn't checkable.", size=36)
                       ).arrange(DOWN, buff=0.28)
        self.play(FadeIn(close[0], shift=0.1 * UP), run_time=1.0)
        self.play(FadeIn(close[1], shift=0.1 * UP), run_time=1.0)
        self.play(FadeIn(close[2], shift=0.1 * UP), run_time=1.0)
        self.wait(5.0)
        self.play(FadeOut(close), run_time=1.2)
        self.wait(0.5)
