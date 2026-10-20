"""Method figure, drawn in the style of Figure 1 of MAAC (Iqbal & Sha, 2019).

    python make_method_figure.py --out figs/method

(a) The critic for agent i as in MAAC: the query comes from s_i = g^s(o_i), keys and values from e_j = g_j(o_j, a_j),
    and a router with no parameters turns positions into the attention set S_i.
(b) One routed head. Each group is pooled into one summary (mean key and value, + log n_g on its score; agent i's own
    group without i). With Z-MAAC the nearby agents also keep their own key and value. One softmax over S_i.
(c) C-MAAC on 48 agents: k-means on positions with c = ceil(sqrt N) = 7 clusters, one summary per cluster.
(d) Z-MAAC on the same agents with b = 4: sort along the Z-order curve, cut into blocks, attend to the own and the two
    neighbouring blocks agent by agent and to every other block as a summary.
Blue marks per-type modules and yellow shared ones, as in MAAC; green marks what routing adds.
Writes fig_method.svg, and fig_method.pdf/.png if Playwright's Chromium is installed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

W, H = 1110, 1150
INK = "#000000"
BLUE, YELLOW, GREEN = "#B4C7E7", "#FFE699", "#C5E0B4"
GREEN_D, GREY, GREY_L = "#548235", "#A6A6A6", "#D9D9D9"
SANS = "Carlito, Calibri, 'Liberation Sans', sans-serif"
MATH = "'LMRomanTT', 'Latin Modern Roman', 'CMU Serif', 'DejaVu Serif', serif"
SCRIPT = "'Latin Modern Math', 'STIX Two Math', serif"
LW_BOX, LW_ARROW = 3.2, 1.4

out: list[str] = []


def add(s):
    out.append(s)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def math(s):
    """Italic math with tiny markup: x_{i} subscript, ^{T} superscript, {bar k} overbar, {up log} upright,
    {cal N} script letter."""
    res, i = [], 0
    while i < len(s):
        if s.startswith("_{", i) or s.startswith("^{", i):
            j = s.index("}", i)
            dy = 5 if s[i] == "_" else -8
            res.append(f'<tspan dy="{dy}" font-size="70%">{math(s[i + 2:j])}</tspan><tspan dy="{-dy}">​</tspan>')
            i = j + 1
        elif s.startswith("{bar ", i):
            j = s.index("}", i)
            res.append(f'<tspan>{esc(s[i + 5:j])}̄</tspan>')
            i = j + 1
        elif s.startswith("{up ", i):
            j = s.index("}", i)
            res.append(f'<tspan font-style="normal">{esc(s[i + 4:j])}</tspan>')
            i = j + 1
        elif s.startswith("{cal ", i):
            j = s.index("}", i)
            cal = {"D": "\U0001D49F", "N": "\U0001D4A9", "G": "\U0001D4A2", "M": "\U0001D4DC"}[s[i + 5:j]]
            res.append(f'<tspan font-family="{SCRIPT}" font-style="normal">{cal}</tspan>')
            i = j + 1
        else:
            j = i
            while j < len(s) and not (s.startswith("_{", j) or s.startswith("^{", j) or s.startswith("{", j)):
                j += 1
            chunk = s[i:j]
            for ch in chunk:   # operators and digits upright, letters italic
                if ch.isalpha():
                    res.append(esc(ch))
                else:
                    res.append(f'<tspan font-style="normal">{esc(ch)}</tspan>')
            i = j
    return "".join(res)


def mtext(x, y, s, size=22, anchor="middle", fill=INK):
    add(f'<text x="{x}" y="{y}" font-family="{MATH}" font-style="italic" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}">{math(s)}</text>')


def stext(x, y, s, size=19, anchor="middle", fill=INK, weight="normal", italic=False):
    st = ' font-style="italic"' if italic else ""
    add(f'<text x="{x}" y="{y}" font-family="{SANS}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}"{st}>{esc(s)}</text>')


def mixed(x, y, s, size=19, anchor="middle", fill=INK, italic=False):
    """Sans text with $math$ segments."""
    parts = s.split("$")
    body = "".join(esc(p) if k % 2 == 0 else
                   f'<tspan font-family="{MATH}" font-style="italic" font-size="{size * 1.12:.1f}">{math(p)}</tspan>'
                   for k, p in enumerate(parts))
    st = ' font-style="italic"' if italic else ""
    add(f'<text x="{x}" y="{y}" font-family="{SANS}" font-size="{size}" fill="{fill}" text-anchor="{anchor}"{st}>'
        f'{body}</text>')


def rbox(x, y, w, h, fill="white", rx=9, sw=LW_BOX, stroke=INK, dash=None, opacity=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" fill-opacity="{opacity}" '
        f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def arrow(*pts, sw=LW_ARROW, dash=None, head=True, color=INK):
    d = " ".join(("M" if k == 0 else "L") + f"{x:.1f},{y:.1f}" for k, (x, y) in enumerate(pts))
    da = f' stroke-dasharray="{dash}"' if dash else ""
    mk = ' marker-end="url(#ah)"' if head else ""
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{da}{mk}/>')


def house(cx, tip, base, label):
    """MAAC's upward block arrow (shared projection W_q, W_k, V)."""
    hw, bw, mid = 37, 23, tip + 32
    pts = [(cx, tip), (cx + hw, mid), (cx + bw, mid), (cx + bw, base), (cx - bw, base), (cx - bw, mid), (cx - hw, mid)]
    add('<polygon points="' + " ".join(f"{x},{y}" for x, y in pts) + f'" fill="{YELLOW}" stroke="{INK}" '
        f'stroke-width="2.2" stroke-linejoin="miter"/>')
    mtext(cx, base - 12, label, 22)


def morton(xy, bits=5):
    g = np.clip((xy * (1 << bits)).astype(int), 0, (1 << bits) - 1)
    code = np.zeros(len(xy), dtype=np.int64)
    for b in range(bits):
        code |= ((g[:, 0] >> b) & 1) << (2 * b)
        code |= ((g[:, 1] >> b) & 1) << (2 * b + 1)
    return code


def panel_a():
    stext(18, 36, "(a)", 21, anchor="start", weight="bold")
    # agent i's own column (per agent type)
    cx = 100
    arrow((cx, 704), (cx, 524))                                   # o_i from the minibatch
    mtext(cx + 10, 640, "o_{i}", anchor="start")
    rbox(55, 480, 90, 44, BLUE)
    stext(cx, 508, "MLP")
    mtext(170, 470, "s_{i}", anchor="start")
    arrow((145, 486), (219, 414))                                 # query s_i into the shared attention
    arrow((cx, 480), (cx, 238))                                   # s_i also feeds the Q head
    rbox(55, 194, 90, 44, BLUE)
    stext(cx, 222, "MLP")
    arrow((cx, 194), (cx, 140))
    mtext(cx + 6, 126, "Q_{i}(o, (a_{i}, a_{−i}))", 21)
    mixed(cx + 6, 98, "one value for every own action $a_{i}$", 14, italic=True)
    arrow((219, 262), (147, 222))                                 # x_i back to agent i
    mtext(176, 262, "x_{i}", anchor="middle")

    # shared attention block
    rbox(219, 52, 330, 470, YELLOW, rx=36, sw=3.6)
    mtext(384, 86, "x_{1} … x_{N}")
    rbox(300, 120, 168, 54, "white", rx=9)
    stext(384, 143, "Concatenate", 18)
    stext(384, 165, "heads per agent", 18)
    for dx in (-24, 0, 24):
        arrow((384, 120), (384 + dx, 97))
    for k in (2, 1, 0):                                           # stacked heads (back to front)
        rbox(246 + 22 * k, 238 - 22 * k, 170, 148, "white", rx=22, sw=3.2)
    for xk, y0 in ((360, 238), (384, 216), (408, 194)):
        arrow((xk, y0), (xk, 176))
    stext(331, 296, "Routed", 19)
    stext(331, 318, "Attention", 19)
    stext(331, 340, "Head", 19)
    # router (green: added, no parameters)
    rbox(236, 436, 150, 56, GREEN, rx=9)
    stext(311, 460, "Router", 19)
    stext(311, 481, "action-free, no gradient", 14, italic=True)
    arrow((311, 436), (311, 388))
    mtext(320, 420, "S_{1} … S_{N}", 19, anchor="start")
    # inputs from below: o_j feeds the encoder and, through its position slice, the router; a_j only the encoder
    arrow((340, 704), (340, 632), (311, 632), (311, 494))
    add(f'<circle cx="340" cy="632" r="3.2" fill="{INK}"/>')
    mtext(302, 552, "p_{j}", 21, anchor="end")
    stext(302, 574, "position", 15, anchor="end", italic=True)
    mixed(302, 593, "part of $o_{j}$", 15, anchor="end", italic=True)
    arrow((340, 632), (385, 632), (385, 614))
    arrow((440, 704), (440, 614))
    for k in (2, 1, 0):
        rbox(360 + 7 * k, 572 - 7 * k, 84, 40, BLUE, rx=8, sw=2.8)
    stext(402, 598, "MLP", 18)
    mtext(350, 676, "o_{j}", 21, anchor="start")
    mtext(450, 676, "a_{j}", 21, anchor="start")
    stext(470, 600, "every agent j", 14, anchor="start", italic=True)
    panel_data()
    arrow((400, 557), (400, 389))
    mtext(410, 546, "e_{1} … e_{N}", 20, anchor="start")
    # zoom lines to (b)
    arrow((416, 244), (606, 60), dash="5,4", head=False, sw=1.1)
    arrow((416, 384), (606, 548), dash="5,4", head=False, sw=1.1)


def cylinder(x, y, w, h, ry=9, fill="white"):
    """Database drum (replay buffer)."""
    add(f'<path d="M{x},{y} L{x},{y + h} A{w / 2},{ry} 0 0 0 {x + w},{y + h} L{x + w},{y} Z" fill="{fill}" '
        f'stroke="{INK}" stroke-width="2.6"/>')
    add(f'<ellipse cx="{x + w / 2}" cy="{y}" rx="{w / 2}" ry="{ry}" fill="{fill}" stroke="{INK}" stroke-width="2.6"/>')


def panel_data():
    """Where the critic's inputs come from: decentralized rollouts fill the replay buffer D; each update draws a
    minibatch (o, a, r, o') from D. (o, a) feed the critic drawn above; r and o' feed the TD target (Algorithm 1)."""
    x0, x1, y0 = 40, 520, 714
    cylinder(x0, y0, x1 - x0, 46)
    mixed((x0 + x1) / 2, y0 + 26, "Replay buffer ${cal D}$: minibatch of joint transitions $(o,\u2009a,\u2009r,\u2009o')$", 17)
    mixed((x0 + x1) / 2, y0 + 47, "$(o,\u2009a)$ feed the critic above; $r$ and $o'$ the TD target", 14, italic=True)
    # decentralized rollouts that fill D
    rbox(812, 604, 200, 46, BLUE)
    mixed(912, 626, "Policies \u03c0$_{j}(a_{j}\u2009|\u2009o_{j})$", 18)
    stext(912, 644, "own observation only", 13, italic=True)
    rbox(812, 700, 200, 46, "white")
    stext(912, 722, "Environments", 18)
    stext(912, 740, "E in parallel", 13, italic=True)
    arrow((850, 700), (850, 652))
    mtext(842, 680, "o_{j}", 19, anchor="end")
    arrow((974, 650), (974, 698))
    mtext(982, 680, "a_{j}", 19, anchor="start")
    arrow((812, 734), (524, 734))
    mixed(668, 724, "store $(o,\u2009a,\u2009r,\u2009o')$, all $N$ agents", 15)


def panel_b():
    x0, x1, y0, y1 = 606, 1094, 52, 560
    stext(x0 - 4, 36, "(b)", 21, anchor="start", weight="bold")
    rbox(x0, y0, x1 - x0, y1 - y0, "white", rx=36, sw=3.6)
    tip, base = 446, 522
    house(662, tip, base, "W_{q}")
    house(848, tip, base, "W_{k}")
    house(1010, tip, base, "V")
    mtext(662, 549, "s_{i}", 21)
    mtext(929, 549, "e_{j}, j ≠ i", 21)
    # select and pool (green, parameter-free); S_i says which entries agent i uses
    rbox(772, 346, 306, 82, GREEN, rx=9)
    mixed(925, 368, "Select and pool by $S_{i}$", 17)
    mixed(925, 393, "group $g$: means ${bar k}_{g}, {bar v}_{g}$, size $n_{g}$", 16)
    mixed(925, 417, "Z-MAAC only, near agent $j$: own $k_{j}, v_{j}$", 16)
    arrow((848, tip), (848, 430))
    arrow((1010, tip), (1010, 430))
    arrow((738, 387), (770, 387))
    mtext(732, 394, "S_{i}", 20, anchor="end")
    # scores
    rbox(630, 282, 236, 46, "white")
    stext(748, 311, "Scaled Dot Product", 18)
    arrow((662, tip), (662, 330))
    arrow((848, 346), (848, 330))
    rbox(668, 212, 160, 48, GREEN)
    mtext(748, 237, "+ {up log} n_{g}", 20)
    stext(748, 253, "summaries only", 12, italic=True)
    arrow((748, 282), (748, 262))
    rbox(668, 148, 160, 44, "white")
    mixed(748, 176, "Softmax over $S_{i}$", 18)
    arrow((748, 212), (748, 194))
    rbox(770, 86, 270, 44, "white")
    stext(905, 114, "Dot Product", 18)
    arrow((748, 148), (800, 132))
    arrow((1010, 346), (1010, 132))
    arrow((905, 86), (905, 32))
    mtext(918, 72, "x_{i}", 21, anchor="start")


def hull(pts):
    pts = sorted(map(tuple, pts))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def blob(pts, fill, stroke, pad=11):
    """Rounded hull around a group of agents: hull edges stroked thick with round joins."""
    h = hull(pts)
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in h) + " Z"
    add(f'<path d="{d}" fill="{stroke}" stroke="{stroke}" stroke-width="{2 * pad + 2.6}" stroke-linejoin="round"/>')
    add(f'<path d="{d}" fill="{fill}" stroke="{fill}" stroke-width="{2 * pad}" stroke-linejoin="round"/>')


def diamond(cx, cy, r=6):
    add(f'<rect x="{cx - r:.1f}" y="{cy - r:.1f}" width="{2 * r}" height="{2 * r}" fill="white" stroke="{INK}" '
        f'stroke-width="1.7" transform="rotate(45 {cx:.1f} {cy:.1f})"/>')


def _layout(seed=3, cols=8, rows=6):
    """The 48-agent toy layout shared by panels (c) and (d): jittered grid in [0,1]^2 and the Z-order rank."""
    rng = np.random.default_rng(seed)
    N = cols * rows
    cell = np.array([(c, r) for r in range(rows) for c in range(cols)])
    xy = (cell + 0.5 + rng.uniform(-0.28, 0.28, size=cell.shape)) / np.array([cols, rows])
    code = np.zeros(N, dtype=np.int64)                      # Z-order (Morton) code of each agent's grid cell
    for bit in range(3):
        code |= ((cell[:, 0] >> bit) & 1) << (2 * bit)
        code |= ((cell[:, 1] >> bit) & 1) << (2 * bit + 1)
    order = np.argsort(code, kind="stable")
    return xy, order


def _kmeans(xy, c, iters=50, seed=0):
    """Plain Lloyd's algorithm with k-means++ seeding (deterministic), as C-MAAC's router does on positions."""
    rng = np.random.default_rng(seed)
    cent = [xy[rng.integers(len(xy))]]
    for _ in range(c - 1):
        d2 = np.min(((xy[:, None] - np.array(cent)[None]) ** 2).sum(-1), 1)
        cent.append(xy[rng.choice(len(xy), p=d2 / d2.sum())])
    cent = np.array(cent)
    for _ in range(iters):
        assign = ((xy[:, None] - cent[None]) ** 2).sum(-1).argmin(1)
        cent = np.array([xy[assign == g].mean(0) if (assign == g).any() else cent[g] for g in range(c)])
    return assign


def _space(X0, Y0, SW, SH, xy):
    return np.c_[X0 + xy[:, 0] * SW, Y0 + (1 - xy[:, 1]) * SH]


def _agent(P, j, fill):
    add(f'<circle cx="{P[j, 0]:.1f}" cy="{P[j, 1]:.1f}" r="4.6" fill="{fill}" stroke="{INK}" stroke-width="0.7"/>')


def _me(P, i):
    add(f'<circle cx="{P[i, 0]:.1f}" cy="{P[i, 1]:.1f}" r="7.5" fill="{YELLOW}" stroke="{INK}" stroke-width="2.2"/>')
    mtext(P[i, 0] + 10, P[i, 1] - 9, "i", 19, anchor="start")


TOP, PX, PY, PW, PH = 822, 0, 0, 262, 200     # bottom row: picture size of panels (c) and (d)


def _focal(order, b=4, me_blk=5):
    return int(order[me_blk * b + 3])


def panel_c():
    """C-MAAC: k-means clusters of positions, one summary per cluster, the own cluster without i."""
    top, x0 = TOP, 18
    stext(x0, top - 14, "(c)", 21, anchor="start", weight="bold")
    stext(x0 + 38, top - 14, "C-MAAC: centroid router", 19, anchor="start")
    xy, order = _layout()
    N, c = len(xy), int(np.ceil(np.sqrt(len(xy))))
    i = _focal(order)
    assign = _kmeans(xy, c)
    P = _space(x0 + 22, top + 14, PW, PH, xy)
    for g in range(c):
        blob(P[assign == g], "#F2F2F2", GREY)
    for j in range(N):
        if j != i:
            _agent(P, j, GREY)
    for g in range(c):
        m = (assign == g) & (np.arange(N) != i)            # own cluster: mean without i
        cm = P[m].mean(0)
        diamond(cm[0], cm[1], 6)
    own = P[(assign == assign[i]) & (np.arange(N) != i)].mean(0)
    _me(P, i)
    tx = x0 + 22 + PW + 22
    lines = ["k-means on positions,", "$c ≈ √N$ clusters", "", "every cluster: one summary,",
             "score $+ {up log} n_{g}$", "own cluster: mean without $i$", "",
             "no agent attended one by one"]
    for k, ln in enumerate(lines):
        if ln:
            mixed(tx, top + 40 + 22 * k, ln, 16, anchor="start")
    mixed(tx, top + 40 + 22 * 9 + 4, "$N$ = 48, $c$ = 7:", 16, anchor="start", italic=True)
    mixed(tx, top + 40 + 22 * 10 + 4, "7 entries instead of 47", 16, anchor="start", italic=True)
    return own


def panel_d():
    """Z-MAAC: block near-far routing along the Z-order curve."""
    top, x0 = TOP, 566
    stext(x0, top - 14, "(d)", 21, anchor="start", weight="bold")
    stext(x0 + 38, top - 14, "Z-MAAC: block near-far router", 19, anchor="start")
    xy, order = _layout()
    N, b, me_blk = len(xy), 4, 5
    nb = N // b
    rank = np.empty(N, int); rank[order] = np.arange(N)
    blk = rank // b
    i = _focal(order, b, me_blk)
    near_blk = {me_blk - 1, me_blk, me_blk + 1}
    near = np.isin(blk, list(near_blk))
    P = _space(x0 + 10, top + 14, PW, PH, xy)
    for g in range(nb):
        m = blk == g
        blob(P[m], "#E2F0D9", GREEN_D) if g in near_blk else blob(P[m], "#F2F2F2", GREY)
    d = " ".join(("M" if k == 0 else "L") + f"{P[j, 0]:.1f},{P[j, 1]:.1f}" for k, j in enumerate(order))
    add(f'<path d="{d}" fill="none" stroke="{GREY}" stroke-width="1.1" stroke-dasharray="3,2"/>')
    for j in range(N):
        if j != i:
            _agent(P, j, GREEN_D if near[j] else GREY)
    for g in range(nb):
        if g not in near_blk:
            cm = P[blk == g].mean(0)
            diamond(cm[0], cm[1], 5)
    _me(P, i)
    tx = x0 + 10 + PW + 22
    lines = ["sort along the Z-order curve", "(dashed), blocks of $b$", "",
             "own block and both", "neighbours: one by one", "every other block:",
             "one summary, $+ {up log} n_{g}$", "$|S_{i}| ≈ 3b + N/b$,", "least at $b ≈ √(N/3)$"]
    for k, ln in enumerate(lines):
        if ln:
            mixed(tx, top + 40 + 22 * k, ln, 16, anchor="start")
    mixed(tx, top + 40 + 22 * 9 + 4, "$N$ = 48, $b$ = 4: 11 agents", 16, anchor="start", italic=True)
    mixed(tx, top + 40 + 22 * 10 + 4, "+ 9 summaries = 20 entries", 16, anchor="start", italic=True)


def legend():
    y = 1132
    items = [(BLUE, "= unique to each agent (type)"), (YELLOW, "= shared among agents"),
             (GREEN, "= added by routing (no parameters)")]
    x = 40
    for fill, lab in items:
        rbox(x, y - 22, 56, 30, fill, rx=6, sw=2.8)
        stext(x + 68, y, lab, 18, anchor="start")
        x += 350


def build():
    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    add('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#000"/></marker></defs>')
    add(f'<rect width="{W}" height="{H}" fill="white"/>')
    panel_a()
    panel_b()
    panel_c()
    panel_d()
    legend()
    add("</svg>")
    return "\n".join(out)


LM_OTF = {("normal", "/usr/share/texmf/fonts/opentype/public/lm/lmroman10-regular.otf"),
          ("italic", "/usr/share/texmf/fonts/opentype/public/lm/lmroman10-italic.otf")}


def otf_to_ttf(src, dst):
    """CFF (OpenType) -> TrueType outlines, so Chromium embeds the math font as TrueType rather than Type 3
    (some submission checkers reject Type 3 fonts)."""
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.ttLib import TTFont, newTable
    font = TTFont(src)
    order, gs = font.getGlyphOrder(), font.getGlyphSet()
    glyf = newTable("glyf")
    glyf.glyphOrder, glyf.glyphs = order, {}
    for name in order:
        pen = TTGlyphPen(None)
        gs[name].draw(Cu2QuPen(pen, 1.0, reverse_direction=True))
        glyf.glyphs[name] = pen.glyph()
    font["glyf"], font["loca"] = glyf, newTable("loca")
    del font["CFF "]
    for t in ("VORG",):
        if t in font:
            del font[t]
    font["head"].glyphDataFormat = 0
    maxp = font["maxp"]
    maxp.tableVersion = 0x00010000
    for a in ("maxZones", "maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs",
              "maxStackElements", "maxSizeOfInstructions", "maxComponentElements", "maxComponentDepth"):
        setattr(maxp, a, 1 if a == "maxZones" else 0)
    post = font["post"]
    post.formatType, post.extraNames, post.mapping = 2.0, [], {}
    post.glyphOrder = order
    font.sfntVersion = "\x00\x01\x00\x00"
    font.save(dst)


def font_faces(cache):
    """@font-face rules with TrueType copies of Latin Modern Roman, or '' if unavailable."""
    import base64
    try:
        import fontTools  # noqa: F401
    except ImportError:
        return ""
    rules = []
    cache.mkdir(parents=True, exist_ok=True)
    for style, src in sorted(LM_OTF):
        if not Path(src).exists():
            return ""
        dst = cache / (Path(src).stem + ".ttf")
        if not dst.exists():
            otf_to_ttf(src, dst)
        b64 = base64.b64encode(dst.read_bytes()).decode()
        rules.append(f"@font-face{{font-family:'LMRomanTT';font-style:{style};"
                     f"src:url(data:font/ttf;base64,{b64}) format('truetype');}}")
    return "<style>" + "".join(rules) + "</style>"


def render(svg_path, out_dir):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed: wrote the SVG only")
        return
    faces = font_faces(Path(out_dir) / ".fontcache")
    html = f"<html><head>{faces}</head><body style='margin:0'>{svg_path.read_text()}</body></html>"
    with sync_playwright() as pw:
        br = pw.chromium.launch()
        pg = br.new_page(viewport={"width": W, "height": H}, device_scale_factor=3)
        pg.set_content(html)
        pg.evaluate("document.fonts.ready")
        pg.wait_for_timeout(300)
        pg.screenshot(path=str(out_dir / "fig_method.png"), clip={"x": 0, "y": 0, "width": W, "height": H})
        pg.pdf(path=str(out_dir / "fig_method.pdf"), width=f"{W}px", height=f"{H}px", print_background=True,
               page_ranges="1")
        br.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs/method")
    a = ap.parse_args()
    d = Path(a.out); d.mkdir(parents=True, exist_ok=True)
    svg = d / "fig_method.svg"
    svg.write_text(build())
    render(svg, d)
    print("wrote", sorted(x.name for x in d.iterdir()))


if __name__ == "__main__":
    main()
