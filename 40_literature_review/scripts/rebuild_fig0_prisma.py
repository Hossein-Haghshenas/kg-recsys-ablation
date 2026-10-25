#!/usr/bin/env python3
"""Rebuild fig0_prisma.

The shipped figure had one line running 9.7pt past the right edge of its box
("...does not license."). Every text block here is wrapped to the measured inner
width of its own box, so a line can no longer exceed the box that holds it.
Geometry, colours and font sizes reproduce the original figure.
"""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 3
matplotlib.rcParams["font.family"] = "DejaVu Serif"
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

PW, PH = 427.32, 491.184           # points, as the original
INK      = "#1a1a1a"
EDGE     = "#3a3a3a"
BAND     = "#e8eef4"
FILL_MAIN, FILL_SIDE, FILL_NOTE = "#ffffff", "#f5f5f5", "#fbfbfb"
PAD      = 8.8                     # inner left/right padding, from the original
LEAD     = 1.30                    # line pitch multiplier

fig = plt.figure(figsize=(PW / 72, PH / 72))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, PW); ax.set_ylim(PH, 0); ax.axis("off")

_fp = {}
def fp(size, italic=False):
    k = (size, italic)
    if k not in _fp:
        _fp[k] = FontProperties(family="DejaVu Serif", size=size,
                                style="italic" if italic else "normal")
    return _fp[k]

def width_of(s, size):
    if not s:
        return 0.0
    return TextPath((0, 0), s, prop=fp(size)).get_extents().width

def _greedy(text, size, avail):
    out, line = [], ""
    for word in text.split():
        trial = word if not line else line + " " + word
        if width_of(trial, size) <= avail or not line:
            line = trial
        else:
            out.append(line); line = word
    if line:
        out.append(line)
    return out

def wrap(text, size, avail):
    """Wrap to the measured width, then even out a stub last line.

    Greedy wrapping can leave a one-word final line. Re-wrapping against a
    slightly narrower target keeps the same line count while pushing words down,
    which fills the last line without ever exceeding `avail`.
    """
    best = _greedy(text, size, avail)
    if len(best) < 2:
        return best
    if width_of(best[-1], size) >= 0.45 * avail:
        return best
    for f in (0.97, 0.94, 0.91, 0.88, 0.85, 0.82):
        cand = _greedy(text, size, avail * f)
        if len(cand) == len(best) and width_of(cand[-1], size) >= 0.45 * avail:
            return cand
    return best

def band(x0, y0, x1, y1, label):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0,rounding_size=2.5",
        fc=BAND, ec=BAND, lw=0.7, zorder=1))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, label, rotation=90,
            ha="center", va="center", fontproperties=fp(7.4), color=INK, zorder=2)

def box(x0, y0, x1, y1, blocks, fill=FILL_MAIN, lw=0.8):
    """blocks: list of (text, size). Each is wrapped to the inner width."""
    avail = (x1 - x0) - 2 * PAD
    lines = []
    for text, size in blocks:
        for ln in wrap(text, size, avail):
            lines.append((ln, size))
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0,rounding_size=2.2",
        fc=fill, ec=EDGE, lw=lw, zorder=3))
    total = sum(s * LEAD for _, s in lines)
    y = (y0 + y1) / 2 - total / 2
    for ln, size in lines:
        y += size * LEAD
        ax.text(x0 + PAD, y - size * 0.32, ln, ha="left", va="baseline",
                fontproperties=fp(size), color=INK, zorder=4)
    return lines

def vgap(x, y0, y1):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1),
        arrowstyle="-|>", mutation_scale=6.5, lw=0.9,
        color=EDGE, shrinkA=0, shrinkB=0, zorder=2))

def hgap(y, x0, x1):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y),
        arrowstyle="-|>", mutation_scale=6.5, lw=0.9,
        color=EDGE, shrinkA=0, shrinkB=0, zorder=2))

# ---------------- lane bands ----------------
band(8.4, 12.9, 33.2, 95.6,  "Identification")
band(8.4, 98.2, 33.2, 314.3, "Screening")
band(8.4, 316.9, 33.2, 392.0, "Retrieval")

L0, L1 = 40.9, 250.2          # main column
R0, R1 = 261.8, 413.3         # side column

# ---------------- main flow ----------------
box(L0, 14.4, L1, 61.2, [
    ("Records identified from databases (n = 13,764)", 7.0),
    ("OpenAlex  O1 5,520 · O2 788 · O3 1,950 · O4 1,203", 7.0),
    ("SpringerLink 3,088 · IEEE Xplore 1,215", 7.0)])

box(L0, 69.1, L1, 94.0, [("Records entering deduplication (n = 12,549)", 7.0)])

box(L0, 104.1, L1, 129.0, [("Records screened on title/abstract (n = 7,898)", 7.0)])

box(L0, 137.3, L1, 172.8, [
    ("Stage 1 provenance", 6.9),
    ("6,475 original structured decisions", 6.9),
    ("1,423 re-screened after a recorded tool failure", 6.9)])

box(L0, 182.8, L1, 207.8, [("Records passed to Stage 1b (n = 4,669)", 7.0)])

box(L0, 218.7, L1, 255.9, [
    ("Stage 1b relevance gate, applied to all records", 6.9),
    ("requires a recommendation signal and a knowledge "
     "signal in the retained title/abstract metadata", 6.9)])

box(L0, 266.4, L1, 297.4, [
    ("Carried forward on insufficient metadata (n = 38)", 6.9),
    ("title of three words or fewer, and no abstract", 6.9)])

box(L0, 313.6, L1, 341.2, [("Reports sought for retrieval (n = 1,912)", 7.4)], lw=1.2)

# the box that overflowed: same text, wrapped to the box
box(L0, 348.6, L1, 389.0, [
    ("Full-text screening was not completed for this set. "
     "The verification search is reported to this stage; the accompanying "
     "text states what it does and does not license.", 6.9)])

# ---------------- side notes ----------------
box(R0, 14.4, R1, 61.2, [
    ("Not carried into deduplication:", 6.8),
    ("IEEE Xplore counted only — export requires sign-in (n = 1,215)", 6.8),
    ("ACM DL and ScienceDirect blocked; no count on this date", 6.8)],
    fill=FILL_SIDE)

box(R0, 68.7, R1, 98.0, [
    ("Duplicates removed (n = 4,651)", 6.8),
    ("4,638 exact · 13 fuzzy, confirmed", 6.8)], fill=FILL_SIDE)

box(R0, 129.0, R1, 177.6, [
    ("Excluded at Stage 1 (n = 3,229)", 6.8),
    ("E1  KG post-hoc only  12", 6.8),
    ("E2  not a recommendation task  2,266", 6.8),
    ("E3  no empirical evaluation  231", 6.8),
    ("E4  not peer-reviewed / English  720", 6.8)], fill=FILL_SIDE)

box(R0, 209.5, R1, 259.8, [
    ("Excluded at Stage 1b (n = 2,757)", 6.8),
    ("F1  no recommendation signal  1,552", 6.8),
    ("F2  no knowledge signal  1,162", 6.8),
    ("F3  not a peer-reviewed report  40", 6.8),
    ("F4  not in English  3", 6.8)], fill=FILL_SIDE)

# ---------------- reliability note ----------------
box(L0, 410.3, 398.9, 481.1, [
    ("Screening reliability", 7.5),
    ("Stage 1 used a single automated screening tool. A second screener, "
     "independently configured, re-screened a random sample of 150 of the "
     "6,475 originally decided records: raw agreement 72.7 per cent, "
     "Cohen's κ = 0.47 (95 per cent CI 0.33 to 0.61). Disagreement was "
     "one-directional — 37 records passed forward that the second screener "
     "excluded, against 4 in the reverse direction — and 92 per cent of "
     "Stage-1 exclusions were confirmed. Agreement tracked the recorded "
     "confidence: 0.88 high, 0.31 low.", 6.4)], fill=FILL_NOTE)

# ---------------- connectors ----------------
for y0, y1 in [(61.2, 69.1), (94.0, 104.1), (129.0, 137.3), (172.8, 182.8),
               (207.8, 218.7), (255.9, 266.4), (297.4, 313.6), (341.2, 348.6)]:
    vgap(145.5, y0, y1)
for y in (37.8, 83.4, 153.3, 234.8):
    hgap(y, L1, R0)

fig.savefig("fig0_prisma.pdf", format="pdf")
print("wrote fig0_prisma.pdf")
