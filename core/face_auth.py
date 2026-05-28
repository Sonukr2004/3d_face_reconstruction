"""
face_auth.py — Face Authentication using OpenCV DNN (SFace embeddings)
=======================================================================

Uses OpenCV's built-in deep learning face recognition:
  • YuNet   — DNN face detector (returns bbox + 5 landmarks)
  • SFace   — 128-dim face embedding (cosine similarity for matching)

These models give >99% accuracy on LFW benchmark.
SFace cosine threshold: 0.363 (OpenCV recommended for same-person match).

Pipeline
────────
  1. YuNet detects face + 5 key landmarks inside the image
  2. SFace uses those landmarks to align the face internally
     and produces a 128-D embedding vector
  3. Registration saves the embedding
  4. Verification compares probe embedding vs all stored embeddings
     using cosine similarity (built into SFace)
"""

import os
import json
from typing import Optional
import cv2
import numpy as np
from datetime import datetime

PROFILES_DIR = os.path.join(os.path.dirname(__file__), "..", "profiles")
MODELS_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")

YUNET_MODEL  = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_MODEL  = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

# SFace cosine similarity threshold (OpenCV recommended = 0.363)
# We use 0.40 for slightly stricter security
MATCH_THRESHOLD = 0.40


# ── Lazy-loaded singletons ────────────────────────────────────────────────────

_yunet = None
_sface = None


def _get_yunet(w: int, h: int):
    """Get or create YuNet detector sized for the input image."""
    global _yunet
    _yunet = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (w, h),
                                        score_threshold=0.7,
                                        nms_threshold=0.3,
                                        top_k=10)
    return _yunet


def _get_sface():
    """Get or create SFace recogniser."""
    global _sface
    if _sface is None:
        _sface = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")
    return _sface


# ── Core: extract 128-D embedding ─────────────────────────────────────────────

def extract_embedding(bgr_image: np.ndarray) -> tuple:
    """
    Detect face with YuNet, extract 128-D SFace embedding.

    Returns (embedding, face_info) or (None, None) on failure.
    face_info is a 1-D array: [x,y,w,h, x_re,y_re, x_le,y_le,
                                x_nt,y_nt, x_rcm,y_rcm, x_lcm,y_lcm, score]
    """
    h, w = bgr_image.shape[:2]
    yunet = _get_yunet(w, h)
    _, faces = yunet.detect(bgr_image)

    if faces is None or len(faces) == 0:
        return None, None

    # Pick largest face by area
    best_idx = int(np.argmax(faces[:, 2] * faces[:, 3]))
    face = faces[best_idx]

    sface = _get_sface()
    aligned = sface.alignCrop(bgr_image, face)
    embedding = sface.feature(aligned)  # (1, 128) float32

    return embedding.flatten(), face


def compare_embeddings(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Cosine similarity between two 128-D SFace embeddings. Range [-1, 1]."""
    sface = _get_sface()
    return float(sface.match(
        emb1.reshape(1, -1).astype(np.float32),
        emb2.reshape(1, -1).astype(np.float32),
        cv2.FaceRecognizerSF_FR_COSINE
    ))


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
        bgr_image:    Optional[np.ndarray] = None,
        face_bbox:    Optional[tuple] = None,
    ) -> dict:
        name = name.strip()
        if not name:
            return {"success": False, "error": "Name cannot be empty."}
        if bgr_image is None:
            return {"success": False,
                    "error": "A face image is required for registration."}

        emb, face_info = extract_embedding(bgr_image)
        if emb is None:
            return {"success": False,
                    "error": "DNN could not detect a face. Use a clear frontal photo."}

        safe   = name.lower().replace(" ", "_")
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        pid    = f"{safe}_{ts}"
        e_path = os.path.join(self.profiles_dir, f"{pid}.npy")
        m_path = os.path.join(self.profiles_dir, f"{pid}.json")

        np.save(e_path, emb)
        meta = {
            "id":            pid,
            "name":          name,
            "registered_at": datetime.now().isoformat(),
            "n_landmarks":   int(len(landmarks_3d)) if landmarks_3d is not None else 0,
            "sig_path":      e_path,
        }
        with open(m_path, "w") as f:
            json.dump(meta, f, indent=2)

        return {"success": True, "profile_id": pid, "name": name}

    # ── Verify ────────────────────────────────────────────────────────────────
    def verify(
        self,
        landmarks_3d: np.ndarray,
        bgr_image:    Optional[np.ndarray] = None,
        face_bbox:    Optional[tuple] = None,
        threshold:    float             = MATCH_THRESHOLD,
    ) -> dict:
        if bgr_image is None:
            return {"success": False,
                    "error": "A face image is required for verification."}

        profiles = self.list_profiles()
        if not profiles:
            return {"success": False,
                    "error": "No users registered yet."}

        probe_emb, _ = extract_embedding(bgr_image)
        if probe_emb is None:
            return {"success": False,
                    "error": "DNN could not detect a face. Use a clear frontal photo."}

        results = []
        for meta in profiles:
            try:
                stored_emb = np.load(meta["sig_path"])
                sim = compare_embeddings(probe_emb, stored_emb)
                results.append({
                    "name":         meta["name"],
                    "profile_id":   meta["id"],
                    "similarity":   sim,
                    "registered_at": meta.get("registered_at", ""),
                })
            except Exception:
                continue

        if not results:
            return {"success": False, "error": "Could not load any profiles."}

        results.sort(key=lambda r: r["similarity"], reverse=True)
        best = results[0]
        best["all_scores"]    = results
        best["access_granted"] = best["similarity"] >= threshold
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
