"""
landmark_extractor.py
Generates a dense face mesh (~700 points) using:
  - OpenCV Haar Cascade for face bbox detection
  - Dense ellipse grid sampled inside the face oval
  - Anatomically correct 3D depth: hemisphere base + subtle nose/eye features

No mediapipe, no internet, no model downloads.
"""

import cv2
import numpy as np


class LandmarkExtractor:
    """
    Dense face point cloud with physically correct depth proportions.
    A real human face protrudes only ~15-20% of its width in depth.
    """

    def __init__(self, grid_cols: int = 30, grid_rows: int = 36):
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows

    # ── public API ─────────────────────────────────────────────────────────────
    def extract(self, image: np.ndarray, face_bbox: tuple = None):
        h, w = image.shape[:2]

        if face_bbox is None:
            face_bbox = self._detect_face(image)
            if face_bbox is None:
                return None, None

        fx, fy, fw, fh = face_bbox

        # Gentle padding
        pad_x = int(fw * 0.05)
        pad_y = int(fh * 0.04)
        fx  = max(0, fx - pad_x)
        fy  = max(0, fy - pad_y)
        fw  = min(w - fx, fw + 2 * pad_x)
        fh  = min(h - fy, fh + 2 * pad_y)

        points = self._dense_face_grid(fx, fy, fw, fh)
        if len(points) < 10:
            return None, None

        return np.array(points, dtype=np.float32), None

    def draw_landmarks(self, image: np.ndarray, face_bbox: tuple = None) -> np.ndarray:
        landmarks_3d, _ = self.extract(image, face_bbox)
        annotated = image.copy()
        if landmarks_3d is None:
            return annotated

        z     = landmarks_3d[:, 2]
        z_min = z.min()
        z_max = z.max()
        z_rng = z_max - z_min + 1e-6

        for lm in landmarks_3d:
            px, py = int(lm[0]), int(lm[1])
            t = (lm[2] - z_min) / z_rng          # 0 = flat edge, 1 = nose tip
            color = (
                int(30  + 20  * t),               # B — stays dark
                int(100 + 155 * t),               # G — brightens toward nose
                int(200 - 100 * t),               # R — fades from cool to warm
            )
            cv2.circle(annotated, (px, py), 2, color, -1)

        if face_bbox:
            x, y, bw, bh = face_bbox
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 255, 100), 2)

        return annotated

    def close(self):
        pass

    # ── internals ──────────────────────────────────────────────────────────────
    def _dense_face_grid(self, fx, fy, fw, fh):
        """
        Sample grid inside a face ellipse with hemisphere-based Z depth.
        Depth is capped at 18% of face width — anatomically realistic.
        """
        # Face ellipse centre and semi-axes (normalised [0,1])
        cx, cy = 0.50, 0.49
        rx, ry = 0.46, 0.50

        points = []
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                nx = col / (self.grid_cols - 1)
                ny = row / (self.grid_rows - 1)

                # Reject points outside face ellipse
                ellipse_d = ((nx - cx) / rx) ** 2 + ((ny - cy) / ry) ** 2
                if ellipse_d > 1.0:
                    continue

                px = fx + nx * fw
                py = fy + ny * fh
                z  = self._depth_model(nx, ny) * fw   # depth in pixels

                points.append([px, py, z])

        return points

    def _depth_model(self, nx: float, ny: float) -> float:
        """
        Anatomically correct depth model.

        Layers (all in normalised face space):
          1. Hemisphere base  — overall face curvature (dominant)
          2. Nose protrusion  — +5% of face width at nose tip
          3. Eye recessions   — −2% at each eye socket
          4. Lip protrusion   — +2% at lips
          5. Chin taper       — gentle reduction at chin

        Return value is a fraction of face width (max ≈ 0.18).
        """

        # ── 1. Hemisphere base (half-ellipsoid) ─────────────────────────────
        # Gives a smooth, rounded face shape with zero depth at the edges.
        cx_b, cy_b = 0.50, 0.46          # ellipsoid centre (slightly above mid)
        rx_b, ry_b = 0.46, 0.50          # semi-axes matching face ellipse
        dx = (nx - cx_b) / rx_b
        dy = (ny - cy_b) / ry_b
        r2 = dx ** 2 + dy ** 2
        r2 = min(r2, 1.0)
        base = np.sqrt(1.0 - r2)         # hemisphere: 1 at centre, 0 at edge
        base *= 0.14                     # scale: max depth = 14% of face width

        # ── 2. Nose protrusion (small, realistic) ───────────────────────────
        # Nose tip is only ~5% of face width beyond the surrounding cheek.
        nose_cx, nose_cy = 0.50, 0.60
        nose_r2 = (((nx - nose_cx) / 0.07) ** 2
                 + ((ny - nose_cy) / 0.07) ** 2)
        nose = 0.05 * np.exp(-nose_r2)

        # ── 3. Eye socket recessions ─────────────────────────────────────────
        l_eye_r2 = (((nx - 0.34) / 0.07) ** 2
                  + ((ny - 0.40) / 0.05) ** 2)
        r_eye_r2 = (((nx - 0.66) / 0.07) ** 2
                  + ((ny - 0.40) / 0.05) ** 2)
        l_eye = -0.02 * np.exp(-l_eye_r2)
        r_eye = -0.02 * np.exp(-r_eye_r2)

        # ── 4. Lip slight protrusion ─────────────────────────────────────────
        lip_r2 = (((nx - 0.50) / 0.10) ** 2
                + ((ny - 0.75) / 0.04) ** 2)
        lips = 0.02 * np.exp(-lip_r2)

        # ── 5. Chin taper (flatten lower face slightly) ──────────────────────
        chin_r2 = (((nx - 0.50) / 0.15) ** 2
                 + ((ny - 0.90) / 0.08) ** 2)
        chin = -0.015 * np.exp(-chin_r2)

        depth = base + nose + l_eye + r_eye + lips + chin
        return float(np.clip(depth, 0.0, 0.22))   # hard cap at 22%

    def _detect_face(self, image: np.ndarray):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        gray  = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        best = max(faces, key=lambda f: f[2] * f[3])
        return tuple(int(v) for v in best)
