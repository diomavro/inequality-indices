#!/usr/bin/env python3
"""
Audit matplotlib figures for overlapping text/legend boxes and
elements that leak outside their axes.

Strategy: monkey-patch plt.savefig so that just before every saved
figure, we walk the current figure's Text/Legend/Annotation artists,
force a draw, and compare their renderer-reported bounding boxes.

Run from the figures/ directory:
    python3 audit_figures.py
"""

import os
import runpy
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_orig_savefig = plt.savefig
_findings = []


def rect_overlap_area(a, b):
    x0 = max(a.x0, b.x0)
    x1 = min(a.x1, b.x1)
    y0 = max(a.y0, b.y0)
    y1 = min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def audit_current_figure(filename):
    fig = plt.gcf()
    try:
        fig.canvas.draw()
    except Exception as e:
        _findings.append((filename, 0, [], [], f"draw failed: {e}"))
        return
    renderer = fig.canvas.get_renderer()

    boxes = []  # (label, bbox, ax_bbox_or_None)

    for ax_i, ax in enumerate(fig.axes):
        try:
            ax_bbox = ax.get_window_extent(renderer)
        except Exception:
            ax_bbox = None

        # Things we should NOT flag: axis title, x/y labels, tick labels
        skip_ids = set()
        skip_ids.add(id(ax.title))
        skip_ids.add(id(ax.xaxis.label))
        skip_ids.add(id(ax.yaxis.label))
        for t in ax.xaxis.get_ticklabels():
            skip_ids.add(id(t))
        for t in ax.yaxis.get_ticklabels():
            skip_ids.add(id(t))

        # User-added Text / Annotation objects
        for txt in ax.texts:
            if id(txt) in skip_ids:
                continue
            s = txt.get_text()
            if not s or not s.strip():
                continue
            try:
                bb = txt.get_window_extent(renderer)
            except Exception:
                continue
            if bb.width <= 0 or bb.height <= 0:
                continue
            snippet = s.replace("\n", " ")[:40]
            boxes.append((f"ax{ax_i}:text '{snippet}'", bb, ax_bbox))

        leg = ax.get_legend()
        if leg is not None:
            try:
                bb = leg.get_window_extent(renderer)
                if bb.width > 0 and bb.height > 0:
                    boxes.append((f"ax{ax_i}:legend", bb, None))
            except Exception:
                pass

    # Figure-level texts (fig.text source captions, suptitles) — tracked for
    # overlap checks against each other, but never flagged as axes leaks
    # (ax_bbox=None excludes them from the leak pass below).
    for txt in fig.texts:
        s = txt.get_text()
        if not s or not s.strip():
            continue
        try:
            bb = txt.get_window_extent(renderer)
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        snippet = s.replace("\n", " ")[:40]
        boxes.append((f"fig:text '{snippet}'", bb, None))

    # Figure-level legends (e.g., fig.legend(...))
    for i, leg in enumerate(fig.legends or []):
        try:
            bb = leg.get_window_extent(renderer)
            if bb.width > 0 and bb.height > 0:
                boxes.append((f"fig:legend{i}", bb, None))
        except Exception:
            pass

    # Pairwise overlap test
    overlaps = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            area = rect_overlap_area(boxes[i][1], boxes[j][1])
            if area > 4.0:  # ignore sub-4 px² anti-aliasing touches
                overlaps.append((boxes[i][0], boxes[j][0], area))

    # Out-of-axes leak test (only for user texts, not legends)
    leaks = []
    PAD = 2.0
    for label, bb, ax_bb in boxes:
        if ax_bb is None:
            continue
        if "legend" in label:
            continue
        if (
            bb.x0 < ax_bb.x0 - PAD
            or bb.x1 > ax_bb.x1 + PAD
            or bb.y0 < ax_bb.y0 - PAD
            or bb.y1 > ax_bb.y1 + PAD
        ):
            # how much does it poke out
            dx = max(0, ax_bb.x0 - bb.x0) + max(0, bb.x1 - ax_bb.x1)
            dy = max(0, ax_bb.y0 - bb.y0) + max(0, bb.y1 - ax_bb.y1)
            leaks.append((label, int(dx), int(dy)))

    _findings.append((filename, len(boxes), overlaps, leaks, None))


def patched_savefig(fname, *args, **kwargs):
    try:
        audit_current_figure(str(fname))
    except Exception as e:
        _findings.append((str(fname), 0, [], [], f"audit error: {e}"))
    return _orig_savefig(fname, *args, **kwargs)


plt.savefig = patched_savefig

# Also patch the Figure.savefig method so generators that call fig.savefig(...)
# (rather than plt.savefig(...)) are audited too.
from matplotlib.figure import Figure  # noqa: E402

_orig_fig_savefig = Figure.savefig


def patched_fig_savefig(self, fname, *args, **kwargs):
    try:
        plt.figure(self.number)
        audit_current_figure(str(fname))
    except Exception as e:
        _findings.append((str(fname), 0, [], [], f"audit error: {e}"))
    return _orig_fig_savefig(self, fname, *args, **kwargs)


Figure.savefig = patched_fig_savefig

# Run any generation scripts present in the figures/ directory
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code")
os.chdir(HERE)
sys.path.insert(0, HERE)  # generators use bare imports

_ran = 0
for script in ("generate_figures.py", "make_pip_figures.py"):
    path = os.path.join(HERE, script)
    if not os.path.exists(path):
        continue  # skip generators this project does not use
    print(f"\n[audit] running {script} ...")
    runpy.run_path(path, run_name="__main__")
    plt.close("all")
    _ran += 1
if _ran == 0:
    raise SystemExit(
        "[audit] no make_figures.py found in figures/ -- create one that "
        "regenerates the paper's figures so they can be audited."
    )

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("FIGURE OVERLAP AUDIT — RESULTS")
print("=" * 72)

total_overlaps = 0
total_leaks = 0
dirty_figs = 0

for fname, n_boxes, overlaps, leaks, err in _findings:
    has_issue = bool(overlaps or leaks or err)
    if has_issue:
        dirty_figs += 1
        print(f"\n* {fname}   ({n_boxes} text/legend boxes inspected)")
        if err:
            print(f"    ERROR: {err}")
        for a, b, area in overlaps:
            print(f"    OVERLAP  {a}")
            print(f"          x  {b}")
            print(f"             area = {int(area)} px^2")
            total_overlaps += 1
        for label, dx, dy in leaks:
            print(f"    LEAK     {label}  (dx={dx}, dy={dy})")
            total_leaks += 1

clean = len(_findings) - dirty_figs
print("\n" + "-" * 72)
print(
    f"Audited {len(_findings)} figures:  "
    f"{clean} clean,  {dirty_figs} with issues  "
    f"({total_overlaps} overlaps, {total_leaks} out-of-axes leaks)"
)
print("=" * 72)

# Non-zero exit so CI / make can gate on the audit.
raise SystemExit(1 if dirty_figs else 0)
