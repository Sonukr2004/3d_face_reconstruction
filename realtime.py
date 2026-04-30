"""
realtime.py
Real-time 3D Face Reconstruction using webcam — pure OpenCV.

Controls:
  [M]  Cycle visual mode
  [S]  Save current snapshot
  [Q]  Quit

Modes:
  0 – NATURAL   camera feed + transparent mesh overlay
  1 – NEON      dark background + neon cyan mesh (Tron style)
  2 – HEATMAP   depth-coloured points + wireframe
  3 – HOLOGRAM  blue tint + scanlines + futuristic HUD
"""

import cv2
import numpy as np
import time
import os
from scipy.spatial import Delaunay


# ── Visual mode constants ───────────────────────────────────────────────────
MODE_NATURAL  = 0
MODE_NEON     = 1
MODE_HEATMAP  = 2
MODE_HOLOGRAM = 3
MODE_NAMES    = ["NATURAL", "NEON", "HEATMAP", "HOLOGRAM"]


# ── Depth model (same hemisphere formula) ───────────────────────────────────
def depth_model(nx: float, ny: float) -> float:
    cx, cy, rx, ry = 0.50, 0.46, 0.46, 0.50
    dx = (nx - cx) / rx
    dy = (ny - cy) / ry
    r2 = min(dx**2 + dy**2, 1.0)
    base = np.sqrt(1.0 - r2) * 0.15

    nose_r2 = ((nx - 0.50) / 0.07)**2 + ((ny - 0.60) / 0.07)**2
    nose     = 0.05 * np.exp(-nose_r2)

    l_eye_r2 = ((nx - 0.34) / 0.07)**2 + ((ny - 0.40) / 0.05)**2
    r_eye_r2 = ((nx - 0.66) / 0.07)**2 + ((ny - 0.40) / 0.05)**2
    eyes     = -0.02 * (np.exp(-l_eye_r2) + np.exp(-r_eye_r2))

    lip_r2   = ((nx - 0.50) / 0.10)**2 + ((ny - 0.75) / 0.04)**2
    lips     = 0.02 * np.exp(-lip_r2)

    return float(np.clip(base + nose + eyes + lips, 0.0, 0.22))


# ── Dense face mesh generator ────────────────────────────────────────────────
def build_face_mesh(fx, fy, fw, fh, cols=22, rows=26):
    """Return (N,3) landmarks and precomputed Delaunay triangles."""
    cx, cy, rx, ry = 0.50, 0.49, 0.46, 0.50
    pts = []
    for row in range(rows):
        for col in range(cols):
            nx = col / (cols - 1)
            ny = row / (rows - 1)
            if ((nx - cx) / rx)**2 + ((ny - cy) / ry)**2 > 1.0:
                continue
            px = fx + nx * fw
            py = fy + ny * fh
            z  = depth_model(nx, ny) * fw
            pts.append([px, py, z])

    lm = np.array(pts, dtype=np.float32)
    tri = Delaunay(lm[:, :2]).simplices if len(lm) >= 3 else np.empty((0, 3), int)
    return lm, tri


# ── Colour helpers ───────────────────────────────────────────────────────────
def depth_to_bgr_heat(z_norm: float):
    """Jet colourmap: blue (shallow) → green → red (deep)."""
    v = z_norm
    r = int(np.clip(1.5 - abs(4 * v - 3), 0, 1) * 255)
    g = int(np.clip(1.5 - abs(4 * v - 2), 0, 1) * 255)
    b = int(np.clip(1.5 - abs(4 * v - 1), 0, 1) * 255)
    return (b, g, r)


def depth_to_neon(z_norm: float):
    """Cyan→magenta neon gradient."""
    r = int(z_norm * 200)
    g = int(220 - z_norm * 120)
    b = int(220 + z_norm * 35)
    return (int(np.clip(b, 0, 255)), int(np.clip(g, 0, 255)), int(np.clip(r, 0, 255)))


# ── Drawing functions ─────────────────────────────────────────────────────────
def draw_natural(frame, lm, tri):
    """Transparent coloured triangle fill."""
    z   = lm[:, 2]
    z_n = (z - z.min()) / (z.max() - z.min() + 1e-6)
    overlay = frame.copy()
    for t in tri:
        p1 = (int(lm[t[0], 0]), int(lm[t[0], 1]))
        p2 = (int(lm[t[1], 0]), int(lm[t[1], 1]))
        p3 = (int(lm[t[2], 0]), int(lm[t[2], 1]))
        z_avg = (z_n[t[0]] + z_n[t[1]] + z_n[t[2]]) / 3.0
        color = depth_to_neon(z_avg)
        cv2.fillConvexPoly(overlay, np.array([p1, p2, p3]), color)
    cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)
    # Wireframe edges
    for t in tri:
        pts_i = [(int(lm[t[j], 0]), int(lm[t[j], 1])) for j in range(3)]
        for a, b in [(0,1),(1,2),(2,0)]:
            cv2.line(frame, pts_i[a], pts_i[b], (0, 220, 255), 1, cv2.LINE_AA)
    return frame


def draw_neon(frame, lm, tri):
    """Black background + bright neon wireframe."""
    black = np.zeros_like(frame)
    z   = lm[:, 2]
    z_n = (z - z.min()) / (z.max() - z.min() + 1e-6)
    for t in tri:
        pts_i = [(int(lm[t[j], 0]), int(lm[t[j], 1])) for j in range(3)]
        z_avg = (z_n[t[0]] + z_n[t[1]] + z_n[t[2]]) / 3.0
        color = depth_to_neon(z_avg)
        for a, b in [(0,1),(1,2),(2,0)]:
            cv2.line(black, pts_i[a], pts_i[b], color, 1, cv2.LINE_AA)
    # Glow: blur neon layer and blend
    glow = cv2.GaussianBlur(black, (7, 7), 0)
    result = cv2.addWeighted(frame, 0.25, black, 0.75, 0)
    result = cv2.add(result, (glow * 0.4).astype(np.uint8))
    return result


def draw_heatmap(frame, lm, tri):
    """Depth heatmap triangles + points."""
    z   = lm[:, 2]
    z_n = (z - z.min()) / (z.max() - z.min() + 1e-6)
    overlay = frame.copy()
    for t in tri:
        p1 = (int(lm[t[0], 0]), int(lm[t[0], 1]))
        p2 = (int(lm[t[1], 0]), int(lm[t[1], 1]))
        p3 = (int(lm[t[2], 0]), int(lm[t[2], 1]))
        z_avg = (z_n[t[0]] + z_n[t[1]] + z_n[t[2]]) / 3.0
        color = depth_to_bgr_heat(z_avg)
        cv2.fillConvexPoly(overlay, np.array([p1, p2, p3]), color)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, pt in enumerate(lm):
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 2,
                   depth_to_bgr_heat(z_n[i]), -1, cv2.LINE_AA)
    return frame


def draw_hologram(frame, lm, tri, frame_idx):
    """Blue hologram style with animated scan line."""
    # Blue tint overlay
    blue_tint = np.zeros_like(frame)
    blue_tint[:, :, 0] = 40   # B
    blue_tint[:, :, 1] = 15   # G
    frame = cv2.addWeighted(frame, 0.6, blue_tint, 0.4, 0)

    z   = lm[:, 2]
    z_n = (z - z.min()) / (z.max() - z.min() + 1e-6)
    for t in tri:
        pts_i = [(int(lm[t[j], 0]), int(lm[t[j], 1])) for j in range(3)]
        z_avg = (z_n[t[0]] + z_n[t[1]] + z_n[t[2]]) / 3.0
        alpha = 0.4 + 0.6 * z_avg
        color = (int(180 * alpha), int(240 * alpha), int(255 * alpha))
        for a, b in [(0,1),(1,2),(2,0)]:
            cv2.line(frame, pts_i[a], pts_i[b], color, 1, cv2.LINE_AA)

    # Animated horizontal scan line
    h = frame.shape[0]
    scan_y = int((frame_idx * 3) % h)
    cv2.line(frame, (0, scan_y), (frame.shape[1], scan_y), (100, 255, 255), 1, cv2.LINE_AA)

    # Scanline texture (every other row slightly dimmed)
    scanlines = np.ones_like(frame, dtype=np.float32)
    scanlines[::2] = 0.85
    frame = (frame * scanlines).astype(np.uint8)

    return frame


# ── HUD overlay ───────────────────────────────────────────────────────────────
def draw_hud(frame, mode, fps, n_pts, face_found):
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (w, 36), (0, 200, 255), 1)

    title = f"3D FACE RECONSTRUCTION  |  MODE: {MODE_NAMES[mode]}"
    cv2.putText(frame, title, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)

    fps_txt = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_txt, (w - 110, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 1, cv2.LINE_AA)

    # Bottom bar
    cv2.rectangle(frame, (0, h - 34), (w, h), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, h - 34), (w, h), (0, 200, 255), 1)

    status = f"Points: {n_pts}  |  Face: {'DETECTED' if face_found else 'SEARCHING...'}"
    cv2.putText(frame, status, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 255), 1, cv2.LINE_AA)

    controls = "[M] Mode  [S] Save  [Q] Quit"
    cv2.putText(frame, controls, (w - 260, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (120, 120, 200), 1, cv2.LINE_AA)

    return frame


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    cap = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    vcap = cv2.VideoCapture(0)

    if not vcap.isOpened():
        print("❌ Cannot open webcam. Check camera permissions.")
        return

    vcap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    vcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("✅ Webcam opened")
    print("   [M] Change visual mode")
    print("   [S] Save snapshot")
    print("   [Q] Quit")

    mode       = MODE_NATURAL
    face_bbox  = None
    lm         = None
    tri        = None
    frame_idx  = 0
    detect_interval = 6   # run face detection every N frames

    fps       = 0.0
    fps_t     = time.time()
    fps_count = 0

    os.makedirs("snapshots", exist_ok=True)

    while True:
        ret, frame = vcap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)   # mirror so it feels natural

        # ── Face detection (every N frames) ─────────────────────────────────
        if frame_idx % detect_interval == 0:
            gray   = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            faces  = cap.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
            if len(faces) > 0:
                face_bbox = tuple(int(v) for v in max(faces, key=lambda f: f[2]*f[3]))
                fx, fy, fw, fh = face_bbox
                lm, tri = build_face_mesh(fx, fy, fw, fh)
            else:
                face_bbox = None
                lm = tri = None

        # ── Draw visual effects ──────────────────────────────────────────────
        if lm is not None and len(tri) > 0:
            if   mode == MODE_NATURAL:  frame = draw_natural(frame, lm, tri)
            elif mode == MODE_NEON:     frame = draw_neon(frame, lm, tri)
            elif mode == MODE_HEATMAP:  frame = draw_heatmap(frame, lm, tri)
            elif mode == MODE_HOLOGRAM: frame = draw_hologram(frame, lm, tri, frame_idx)
        else:
            # No face: animated "scanning" text
            pulse = int(128 + 127 * np.sin(frame_idx * 0.08))
            cv2.putText(frame, "Scanning for face...", (frame.shape[1]//2 - 130, frame.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, pulse, 255), 2, cv2.LINE_AA)

        # ── HUD ──────────────────────────────────────────────────────────────
        frame = draw_hud(frame, mode, fps, len(lm) if lm is not None else 0, face_bbox is not None)

        # ── FPS ──────────────────────────────────────────────────────────────
        fps_count += 1
        elapsed = time.time() - fps_t
        if elapsed >= 0.5:
            fps = fps_count / elapsed
            fps_count = 0
            fps_t = time.time()

        cv2.imshow("3D Face Reconstruction — Real Time", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            mode = (mode + 1) % 4
            print(f"Mode: {MODE_NAMES[mode]}")
        elif key == ord('s'):
            fname = f"snapshots/snapshot_{int(time.time())}_mode{mode}.png"
            cv2.imwrite(fname, frame)
            print(f"✅ Saved: {fname}")

        frame_idx += 1

    vcap.release()
    cv2.destroyAllWindows()
    print("👋 Done")


if __name__ == "__main__":
    main()
