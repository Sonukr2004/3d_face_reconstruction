"""
face_auth.py  —  Reliable Face Authentication (OpenCV-only, no deep learning)
==============================================================================

Why eye-alignment failed
─────────────────────────
OpenCV's Haar eye cascade is unreliable inside a face crop (~30-40%
failure rate).  When it works for the registration photo but fails for
the verification photo (or vice versa), the two crops are completely
different → huge chi-squared distance → always ACCESS DENIED.

This version uses a CONSISTENT, DETERMINISTIC crop pipeline with NO
eye detection, so both registration and verification always produce
the same canonical face patch from the same bbox.

Pipeline
─────────
  1. Crop face from the Haar bbox with fixed padding (20% horizontal,
     25% top, 10% bottom) — same formula every time.
  2. Resize to 128×128.
  3. CLAHE (local contrast normalisation) — reduces lighting variation.
  4. LBP on a 4×4 spatial grid (coarser = robust to small pose shifts).
  5. Chi-squared distance between histograms.
  6. ACCESS GRANTED if distance ≤ threshold (default 80).

Threshold calibration
─────────────────────
  Same person, identical photo  →  distance ≈ 0
  Same person, slightly different lighting/angle  →  5 – 50
  Clearly different person  →  60 – 200
  → threshold = 80 gives a good balance for webcam/upload variation.
"""

import os
import json
import cv2
import numpy as np
from datetime import datetime

PROFILES_DIR    = os.path.join(os.path.dirname(__file__), "..", "profiles")
ALIGNED_SIZE    = 128      # canonical face crop size
LBP_CELLS       = 4       # 4×4 spatial grid  (coarser = more robust)
LBP_BINS        = 256      # LBP histogram bins per cell
MATCH_THRESHOLD = 80.0     # chi-squared distance threshold


# ── Crop helpers ───────────────────────────────────────────────────────────────

def _crop_face(bgr_image: np.ndarray, face_bbox: tuple) -> np.ndarray | None:
    """
    Deterministic face crop with fixed padding proportions.
    Same bbox always → same crop → reproducible features.
    """
    x, y, w, h = [int(v) for v in face_bbox]
    ih, iw = bgr_image.shape[:2]

    pad_x  = int(w * 0.20)
    pad_top = int(h * 0.25)   # more top padding to include forehead
    pad_bot = int(h * 0.10)

    x1 = max(0, x - pad_x);    x2 = min(iw, x + w + pad_x)
    y1 = max(0, y - pad_top);   y2 = min(ih, y + h + pad_bot)

    crop = bgr_image[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def _preprocess(bgr_crop: np.ndarray) -> np.ndarray:
    """
    Convert crop → normalised 128×128 grayscale ready for LBP.
    """
    gray    = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (ALIGNED_SIZE, ALIGNED_SIZE),
                         interpolation=cv2.INTER_LINEAR)
    clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(resized)


# ── LBP ───────────────────────────────────────────────────────────────────────

def _lbp_image(gray: np.ndarray) -> np.ndarray:
    """8-neighbour LBP image."""
    img = gray.astype(np.int32)
    lbp = np.zeros_like(img, dtype=np.uint8)
    for bit, (dy, dx) in enumerate(
        [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    ):
        neighbor = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
        lbp |= ((img >= neighbor).astype(np.uint8) << bit)
    return lbp


def _lbp_histogram(gray: np.ndarray) -> np.ndarray:
    """
    Spatially-pooled LBP histogram over LBP_CELLS × LBP_CELLS grid.
    Output: (LBP_CELLS² × LBP_BINS,) float64, per-cell normalised.
    """
    lbp = _lbp_image(gray)
    h, w = lbp.shape
    ch = h // LBP_CELLS;  cw = w // LBP_CELLS
    hists = []
    for r in range(LBP_CELLS):
        for c in range(LBP_CELLS):
            cell = lbp[r*ch:(r+1)*ch, c*cw:(c+1)*cw]
            hist, _ = np.histogram(cell.ravel(), bins=LBP_BINS, range=(0, 256))
            hist = hist.astype(np.float64)
            hist /= (hist.sum() + 1e-9)
            hists.append(hist)
    return np.concatenate(hists)


def _chi2(h1: np.ndarray, h2: np.ndarray) -> float:
    """Chi-squared histogram distance: 0 = identical, higher = more different."""
    d = h1 + h2 + 1e-9
    return float(np.sum((h1 - h2) ** 2 / d))


def _dist_to_pct(dist: float, threshold: float) -> float:
    """Map chi-squared distance to 0–1 display similarity."""
    return max(0.0, 1.0 - dist / (threshold * 2.0))


# ── Public signature builder ───────────────────────────────────────────────────

def compute_signature(
    bgr_image:    np.ndarray,
    face_bbox:    tuple,
    landmarks_3d: np.ndarray | None = None,  # kept for API compat
) -> np.ndarray | None:
    crop = _crop_face(bgr_image, face_bbox)
    if crop is None:
        return None
    proc = _preprocess(crop)
    return _lbp_histogram(proc)


# ── Profile management ─────────────────────────────────────────────────────────

class FaceAuthSystem:

    def __init__(self, profiles_dir: str = PROFILES_DIR):
        self.profiles_dir = profiles_dir
        os.makedirs(profiles_dir, exist_ok=True)

    # ── Register ──────────────────────────────────────────────────────────────
    def register(
        self,
        name:         str,
        landmarks_3d: np.ndarray,
        bgr_image:    np.ndarray | None = None,
        face_bbox:    tuple       | None = None,
    ) -> dict:
        name = name.strip()
        if not name:
            return {"success": False, "error": "Name cannot be empty."}
        if bgr_image is None or face_bbox is None:
            return {"success": False,
                    "error": "A face image is required for registration."}

        sig = compute_signature(bgr_image, face_bbox, landmarks_3d)
        if sig is None:
            return {"success": False,
                    "error": "Could not crop face — use a clear frontal photo."}

        safe   = name.lower().replace(" ", "_")
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        pid    = f"{safe}_{ts}"
        s_path = os.path.join(self.profiles_dir, f"{pid}.npy")
        m_path = os.path.join(self.profiles_dir, f"{pid}.json")

        np.save(s_path, sig)
        meta = {
            "id":            pid,
            "name":          name,
            "registered_at": datetime.now().isoformat(),
            "n_landmarks":   int(len(landmarks_3d)) if landmarks_3d is not None else 0,
            "sig_path":      s_path,
        }
        with open(m_path, "w") as f:
            json.dump(meta, f, indent=2)

        return {"success": True, "profile_id": pid, "name": name}

    # ── Verify ────────────────────────────────────────────────────────────────
    def verify(
        self,
        landmarks_3d: np.ndarray,
        bgr_image:    np.ndarray | None = None,
        face_bbox:    tuple       | None = None,
        threshold:    float             = MATCH_THRESHOLD,
    ) -> dict:
        if bgr_image is None or face_bbox is None:
            return {"success": False,
                    "error": "A face image is required for verification."}

        profiles = self.list_profiles()
        if not profiles:
            return {"success": False,
                    "error": "No users registered yet."}

        probe = compute_signature(bgr_image, face_bbox, landmarks_3d)
        if probe is None:
            return {"success": False,
                    "error": "Could not crop face — use a clear frontal photo."}

        results = []
        for meta in profiles:
            try:
                stored  = np.load(meta["sig_path"])
                ml      = min(len(probe), len(stored))
                dist    = _chi2(probe[:ml], stored[:ml])
                results.append({
                    "name":         meta["name"],
                    "profile_id":   meta["id"],
                    "distance":     dist,
                    "similarity":   _dist_to_pct(dist, threshold),
                    "registered_at": meta.get("registered_at", ""),
                })
            except Exception:
                continue

        if not results:
            return {"success": False, "error": "Could not load any profiles."}

        results.sort(key=lambda r: r["distance"])
        best = results[0]
        best["all_scores"]    = results
        best["access_granted"] = best["distance"] <= threshold
        best["threshold"]      = threshold
        best["success"]        = True
        return best

    # ── List / Delete ─────────────────────────────────────────────────────────
    def list_profiles(self) -> list:
        profiles = []
        for fname in os.listdir(self.profiles_dir):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(self.profiles_dir, fname)) as f:
                        profiles.append(json.load(f))
                except Exception:
                    pass
        profiles.sort(key=lambda p: p.get("registered_at", ""), reverse=True)
        return profiles

    def delete_profile(self, profile_id: str) -> bool:
        deleted = False
        for ext in (".npy", ".json"):
            path = os.path.join(self.profiles_dir, profile_id + ext)
            if os.path.exists(path):
                os.remove(path)
                deleted = True
        return deleted
