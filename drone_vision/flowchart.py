"""
Generate Phase 2 system flowchart as a PNG image.
Usage: python drone_vision/flowchart.py
Output: drone_vision/phase2_flowchart.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.lines as mlines

# ── colour palette ────────────────────────────────────────────────────────────
C = {
    "start_end":  "#1a1a2e",
    "config":     "#1b2a4a",
    "init":       "#0f3460",
    "capture":    "#4a2580",
    "stereo":     "#c0392b",
    "detect":     "#d68910",
    "track":      "#1e8449",
    "motion":     "#1a5276",
    "alert":      "#922b21",
    "viz":        "#6c3483",
    "decision":   "#784212",
    "loop":       "#212f3d",
    "text":       "#ffffff",
    "subtext":    "#cccccc",
    "arrow":      "#888888",
    "bg":         "#0a0a14",
    "band":       "#ffffff",
}

FIG_W = 18
FIG_H = 44

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# centre x and layout constants
CX   = FIG_W / 2       # 9.0
BW   = 9.5             # box width
BH   = 0.90            # box height
GAP  = 1.30            # vertical spacing between node centres
FONT = "DejaVu Sans"


# ── drawing helpers ───────────────────────────────────────────────────────────

def rbox(x, y, w, h, title, color, sub="", fs=11):
    patch = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.06,rounding_size=0.30",
        linewidth=1.4, edgecolor="#ffffff25",
        facecolor=color, zorder=3,
    )
    ax.add_patch(patch)
    ty = y + (0.15 if sub else 0)
    ax.text(x, ty, title, ha="center", va="center",
            fontsize=fs, fontweight="bold",
            color=C["text"], fontfamily=FONT, zorder=4)
    if sub:
        ax.text(x, y - 0.26, sub, ha="center", va="center",
                fontsize=8.5, color=C["subtext"],
                fontfamily=FONT, style="italic", zorder=4)


def diamond(x, y, w, h, label, color, fs=10.5):
    hw, hh = w/2, h/2
    pts = [[x, y+hh], [x+hw, y], [x, y-hh], [x-hw, y]]
    p = plt.Polygon(pts, closed=True,
                    facecolor=color, edgecolor="#ffffff30",
                    linewidth=1.4, zorder=3)
    ax.add_patch(p)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fs, fontweight="bold",
            color=C["text"], fontfamily=FONT, zorder=4)


def pill(x, y, w, h, label, color, fs=13):
    patch = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.06,rounding_size=0.45",
        linewidth=2, edgecolor="#ffffff50",
        facecolor=color, zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fs, fontweight="bold",
            color=C["text"], fontfamily=FONT, zorder=4)


def arr(x1, y1, x2, y2, color=C["arrow"], lw=2.0):
    ax.annotate("",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=lw, mutation_scale=18),
        zorder=2)


def arr_path(pts, color=C["arrow"], lw=1.8):
    """Draw a polyline arrow (last segment gets arrowhead)."""
    for i in range(len(pts) - 2):
        x1, y1 = pts[i];  x2, y2 = pts[i+1]
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, zorder=2)
    x1, y1 = pts[-2];  x2, y2 = pts[-1]
    ax.annotate("",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=lw, mutation_scale=18),
        zorder=2)


def tag(x, y, text, color="#ffdd57", fs=9):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fs, fontweight="bold",
            color=color, fontfamily=FONT, zorder=5)


def side_label(y, label, color):
    """Section label on the left margin."""
    ax.text(0.30, y, label, ha="center", va="center",
            fontsize=8, color=color, fontfamily=FONT,
            fontweight="bold", rotation=90, zorder=5, alpha=0.6)


def callout(x, y, w, h, title, lines, border_color):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.25",
        linewidth=1.8, edgecolor=border_color,
        facecolor="#12122a", alpha=0.92, zorder=4,
    )
    ax.add_patch(patch)
    ax.text(x + w/2, y + h - 0.27, title,
            ha="center", va="center", fontsize=9.5,
            fontweight="bold", color=border_color,
            fontfamily=FONT, zorder=5)
    for i, line in enumerate(lines):
        ax.text(x + w/2, y + h - 0.62 - i*0.38, line,
                ha="center", va="center", fontsize=8.5,
                color="#dddddd", fontfamily=FONT,
                style="italic", zorder=5)


# ─────────────────────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────────────────────
Y = FIG_H - 0.9
ax.text(CX, Y, "Drone Vision — Phase 2 System Flowchart",
        ha="center", va="center", fontsize=18, fontweight="bold",
        color="#ffffff", fontfamily=FONT)
ax.text(CX, Y - 0.58,
        "Raspberry Pi 5  ·  Dual USB Cameras  ·  YOLOv8  ·  Stereo Depth  ·  Kalman Tracking",
        ha="center", va="center", fontsize=10,
        color="#aaaaaa", fontfamily=FONT, style="italic")
Y -= 1.4

# ─────────────────────────────────────────────────────────────────────────────
# SECTION: INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────
sec_top = Y + 0.55
pill(CX, Y, 5.0, BH, "START", C["start_end"], fs=13)
Y -= GAP
arr(CX, Y + GAP - BH/2, CX, Y + BH/2)

rbox(CX, Y, BW, BH, "Load Configuration", C["config"],
     "cameras.yaml  ·  system.yaml")
Y -= GAP
arr(CX, Y + GAP - BH/2, CX, Y + BH/2)

rbox(CX, Y, BW, BH, "Initialise Modules", C["init"],
     "StereoDepth  ·  Detector  ·  TrackManager  ·  MotionAnalyzer  ·  AlertManager")
Y -= GAP
arr(CX, Y + GAP - BH/2, CX, Y + BH/2)

rbox(CX, Y, BW, BH, "Stereo Calibration Available?", C["stereo"],
     "Looks for config/stereo_calib.npz")
side_label((Y + sec_top) / 2, "INIT", C["init"])

# callout box for depth formula — placed to the right
callout(CX + 5.2, Y - 0.6, 3.55, 1.9,
        "Depth Formula",
        ["Z = (baseline × fₓ) / d   [stereo]",
         "Z = (real_h × fₓ) / bbox_h  [fallback]",
         "baseline = 0.06 m"],
        C["stereo"])
ax.annotate("",
    xy=(CX + 5.2, Y + 0.1),
    xytext=(CX + BW/2, Y),
    arrowprops=dict(arrowstyle="-|>", color=C["stereo"],
                    lw=1.5, mutation_scale=14,
                    connectionstyle="arc3,rad=-0.25"),
    zorder=2)

Y -= GAP
arr(CX, Y + GAP - BH/2, CX, Y + BH/2)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION: CAPTURE
# ─────────────────────────────────────────────────────────────────────────────
sec_top_cap = Y + 0.55
rbox(CX, Y, BW, BH, "Start CaptureThread", C["capture"],
     "queue.Queue(maxsize=2)  —  Left USB + Right USB cameras  (CAP_V4L2)")
Y -= GAP
arr(CX, Y + GAP - BH/2, CX, Y + BH/2)

rbox(CX, Y, BW, BH, "Get Frame Pair from Queue", C["capture"],
     "(left_frame, right_frame, timestamp)")
LOOP_Y = Y   # back-arrow target
side_label((LOOP_Y + sec_top_cap) / 2, "CAPTURE", C["capture"])

Y -= GAP
arr(CX, Y + GAP - BH/2, CX, Y + BH/2)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION: STEREO
# ─────────────────────────────────────────────────────────────────────────────
sec_top_stereo = Y + 0.55
rbox(CX, Y, BW, BH, "Stereo Rectify", C["stereo"],
     "cv2.remap( map1_l/r, map2_l/r )  →  left_rect, right_rect")
Y -= GAP
arr(CX, Y + GAP - BH/2, CX, Y + BH/2)

rbox(CX, Y, BW, BH, "Compute Disparity Map", C["stereo"],
     "StereoSGBM.compute( gray_l, gray_r )  →  float32 disparity  (NaN = invalid)")
side_label((Y + sec_top_stereo) / 2, "STEREO", C["stereo"])

Y -= GAP
arr(CX, Y + GAP - BH/2, CX, Y + BH/2)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION: DETECTION + TRACKING
# ─────────────────────────────────────────────────────────────────────────────
sec_top_det = Y + 0.55
rbox(CX, Y, BW, BH, "YOLOv8 Detection + Tracking", C["detect"],
     "model.track( left_rect, persist=True )  →  list[ Detection(id, x, y, w, h, conf) ]")
Y -= GAP
arr(CX, Y + GAP - BH/2, CX, Y + BH/2 + 0.55)

diamond(CX, Y, 5.5, 1.1, "Object(s) Detected?", C["decision"])
DEC_Y = Y
Y -= 0.55 + BH/2 + 0.2

# "No" left branch — loops back to GetFramePair
NO_X = CX - 6.5
arr_path([
    (CX - 2.75, DEC_Y),
    (NO_X,      DEC_Y),
    (NO_X,      LOOP_Y),
    (CX - BW/2, LOOP_Y),
], color="#d68910")
tag(CX - 4.0, DEC_Y + 0.25, "No  →  next frame", "#d68910")

# "Yes" arrow down
arr(CX, DEC_Y - 0.55, CX, Y + BH/2)
tag(CX + 0.5, DEC_Y - 0.85, "Yes", "#7ed321")

rbox(CX, Y, BW, BH, "Resolve Depth Z per Detection", C["track"],
     "disparity.depth_at(x, y)  or  height-ratio fallback")
Y -= GAP
arr(CX, Y + GAP - BH/2, CX, Y + BH/2)

rbox(CX, Y, BW, BH, "Kalman Tracker — Predict + Update", C["track"],
     "state = [x, y, z, vx, vy, vz]  ·  constant-velocity model  ·  expire after N frames")
side_label((Y + sec_top_det) / 2, "DETECTION & TRACKING", C["detect"])

Y -= GAP
arr(CX, Y + GAP - BH/2, CX, Y + BH/2)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION: MOTION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
sec_top_mot = Y + 0.55
rbox(CX, Y, BW, BH, "Motion Analyzer", C["motion"],
     "speed (m/s) from Kalman velocity  ·  direction angle + label  ·  predict next (x, y, z)")
Y -= GAP
arr(CX, Y + GAP - BH/2, CX, Y + BH/2)

rbox(CX, Y, BW, BH, "Alert Manager — Check Thresholds", C["alert"],
     "Triggers: object detected  ·  speed > threshold  ·  Z < z_min  ·  cooldown per track-ID")
side_label((Y + sec_top_mot) / 2, "MOTION", C["motion"])

Y -= GAP
arr(CX, Y + GAP - BH/2, CX, Y + BH/2)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION: VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
sec_top_viz = Y + 0.55
rbox(CX, Y, BW, BH, "Draw Bounding Box + Label", C["viz"],
     "ID  ·  Z (m)  ·  speed (m/s)  ·  direction label")
Y -= GAP
arr(CX, Y + GAP - BH/2, CX, Y + BH/2)

rbox(CX, Y, BW, BH, "Draw Trajectory Path", C["viz"],
     "Fading green polyline  —  last 30 positions")
Y -= GAP
arr(CX, Y + GAP - BH/2, CX, Y + BH/2)

rbox(CX, Y, BW, BH, "Draw Prediction Arrow + Point", C["viz"],
     "Yellow arrow →  Red dot  (dt seconds ahead)")
Y -= GAP
arr(CX, Y + GAP - BH/2, CX, Y + BH/2)

rbox(CX, Y, BW, BH, "Draw Alerts + HUD Overlay", C["viz"],
     "Red alert text  ·  FPS counter  ·  object count  ·  Stereo ON/OFF")
side_label((Y + sec_top_viz) / 2, "VISUALISATION", C["viz"])

Y -= GAP
arr(CX, Y + GAP - BH/2, CX, Y + BH/2)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION: OUTPUT + LOOP
# ─────────────────────────────────────────────────────────────────────────────
sec_top_out = Y + 0.55
rbox(CX, Y, BW, BH, "Display Output Frame", C["loop"],
     "cv2.imshow  ·  log alerts to file  ·  FPS warning if < 10")
Y -= GAP
arr(CX, Y + GAP - BH/2, CX, Y + BH/2 + 0.55)

diamond(CX, Y, 4.8, 1.1, "ESC Pressed?", C["loop"])
ESC_Y = Y
Y -= 0.55 + BH/2 + 0.2

# "No" right branch — loops back to GetFramePair
YES_X = CX + 6.8
arr_path([
    (CX + 2.4,  ESC_Y),
    (YES_X,     ESC_Y),
    (YES_X,     LOOP_Y),
    (CX + BW/2, LOOP_Y),
], color=C["arrow"])
tag(CX + 4.9, ESC_Y + 0.28, "No  →  loop", "#aaaaaa")

# "Yes" arrow down
arr(CX, ESC_Y - 0.55, CX, Y + BH/2)
tag(CX + 0.5, ESC_Y - 0.85, "Yes", "#e74c3c")

rbox(CX, Y, BW, BH, "Shutdown", C["start_end"],
     "CaptureThread.stop()  ·  join()  ·  cv2.destroyAllWindows()")
Y -= GAP
arr(CX, Y + GAP - BH/2, CX, Y + BH/2)

pill(CX, Y, 5.0, BH, "END", C["start_end"], fs=13)
side_label((Y + sec_top_out) / 2, "OUTPUT", C["loop"])

# ─────────────────────────────────────────────────────────────────────────────
# LEGEND
# ─────────────────────────────────────────────────────────────────────────────
LX, LY = 0.55, 3.8
leg_bg = FancyBboxPatch(
    (LX - 0.2, LY - 2.85), 7.5, 3.25,
    boxstyle="round,pad=0.05,rounding_size=0.25",
    linewidth=1.2, edgecolor="#ffffff20",
    facecolor="#12122a", alpha=0.9, zorder=4,
)
ax.add_patch(leg_bg)
ax.text(LX + 3.55, LY + 0.17, "Module Colour Legend",
        ha="center", fontsize=10, fontweight="bold",
        color="#ffffff", fontfamily=FONT, zorder=5)

items = [
    ("Initialisation",        C["init"]),
    ("Capture Thread",        C["capture"]),
    ("Stereo Vision",         C["stereo"]),
    ("Detection / YOLO",      C["detect"]),
    ("Kalman Tracking",       C["track"]),
    ("Motion Analysis",       C["motion"]),
    ("Alert Manager",         C["alert"]),
    ("Visualisation",         C["viz"]),
]
cols = 2
col_w = 3.5
for i, (lbl, col) in enumerate(items):
    row = i // cols
    c   = i % cols
    lx  = LX + 0.2 + c * col_w
    ly  = LY - 0.35 - row * 0.60
    sq = FancyBboxPatch(
        (lx, ly - 0.17), 0.32, 0.34,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=col, edgecolor="none", zorder=5,
    )
    ax.add_patch(sq)
    ax.text(lx + 0.45, ly, lbl, fontsize=9,
            color="#cccccc", fontfamily=FONT,
            va="center", zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase2_flowchart.png")
plt.tight_layout(pad=0)
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=C["bg"])
plt.close()
print(f"Saved: {out}")
