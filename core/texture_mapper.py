"""
texture_mapper.py
Maps RGB pixel colors from the original 2D face image onto 3D mesh vertices.
Produces per-vertex colors for texture rendering in Plotly/Open3D.
"""

import cv2
import numpy as np


class TextureMapper:
    """Samples pixel colors from an image at each landmark location."""

    def __init__(self):
        pass

    def extract_vertex_colors(
        self,
        image: np.ndarray,
        landmarks_3d: np.ndarray,
    ) -> np.ndarray:
        """
        For every landmark (x, y), sample the pixel color from the image.

        Args:
            image        : BGR numpy array (original face image)
            landmarks_3d : (N, 3) array of landmark coords in pixel space

        Returns:
            colors : (N, 3) float array, values in [0, 1] — RGB
        """
        h, w = image.shape[:2]
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        colors = []
        for lm in landmarks_3d:
            px = int(np.clip(lm[0], 0, w - 1))
            py = int(np.clip(lm[1], 0, h - 1))
            r, g, b = rgb_image[py, px]
            colors.append([r / 255.0, g / 255.0, b / 255.0])

        return np.array(colors, dtype=np.float32)

    def colors_to_hex(self, colors: np.ndarray) -> list:
        """Convert float RGB array → list of '#RRGGBB' hex strings."""
        hex_colors = []
        for c in colors:
            r = int(np.clip(c[0] * 255, 0, 255))
            g = int(np.clip(c[1] * 255, 0, 255))
            b = int(np.clip(c[2] * 255, 0, 255))
            hex_colors.append(f"#{r:02x}{g:02x}{b:02x}")
        return hex_colors

    def build_uv_image(
        self,
        image: np.ndarray,
        landmarks_3d: np.ndarray,
        output_size: int = 512,
    ) -> np.ndarray:
        """
        Create a simple UV texture map by warping the face into a
        square canvas. Useful for .obj export.

        Returns:
            uv_image : (output_size, output_size, 3) uint8 RGB image
        """
        h, w = image.shape[:2]
        pts = landmarks_3d[:, :2].astype(np.float32)

        # Normalize to [0, output_size]
        pts_norm = pts.copy()
        pts_norm[:, 0] = (pts[:, 0] / w) * output_size
        pts_norm[:, 1] = (pts[:, 1] / h) * output_size

        # Simple: just resize the face crop to output_size
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        uv_img = cv2.resize(rgb, (output_size, output_size))
        return uv_img
