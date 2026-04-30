"""
mesh_builder.py
Builds a 3D triangle mesh from the 468 facial landmarks.
Uses MediaPipe's built-in FACEMESH_TESSELATION for connectivity.
"""

import numpy as np


class MeshBuilder:
    """Constructs a 3D face mesh from landmarks using Delaunay triangulation."""

    def __init__(self, depth_scale: float = 3.0):
        """
        Args:
            depth_scale: Multiplier applied to the Z axis to exaggerate depth
                         for better visual effect (MediaPipe Z is subtle).
        """
        self.depth_scale = depth_scale

    def build(self, landmarks_3d: np.ndarray):
        """
        Build mesh from (468, 3) landmark array.

        Returns:
            vertices  : np.ndarray (N, 3)  — x, y, z
            triangles : np.ndarray (M, 3)  — triangle index triplets
        """
        vertices = landmarks_3d.copy()

        # Scale Z for better depth perception
        vertices[:, 2] *= self.depth_scale

        # Center the mesh around origin
        vertices[:, 0] -= vertices[:, 0].mean()
        vertices[:, 1] -= vertices[:, 1].mean()
        vertices[:, 2] -= vertices[:, 2].mean()

        # Flip Y so "up" is positive (image coords have Y increasing downward)
        vertices[:, 1] *= -1

        # Build triangle index list from MediaPipe connections
        triangles = self._build_triangles(vertices)

        return vertices, triangles

    def _build_triangles(self, vertices: np.ndarray) -> np.ndarray:
        """
        MediaPipe FACEMESH_TESSELATION is a set of (i, j) edges.
        We convert it to triangles using a greedy fan approach.
        For a robust mesh we use the known MediaPipe triangle list.
        """
        # MediaPipe connections give us edges; we derive triangles
        # by treating every unique triplet from the tesselation edges.
        # A reliable approach: use scipy Delaunay on 2D then project to 3D.
        from scipy.spatial import Delaunay

        pts2d = vertices[:, :2]  # Use x, y for triangulation
        tri = Delaunay(pts2d)
        return tri.simplices.astype(np.int32)

    def compute_vertex_normals(self, vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
        """Compute per-vertex normals by averaging adjacent face normals."""
        normals = np.zeros_like(vertices)

        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]

        face_normals = np.cross(v1 - v0, v2 - v0)  # (M, 3)

        for i, tri in enumerate(triangles):
            normals[tri[0]] += face_normals[i]
            normals[tri[1]] += face_normals[i]
            normals[tri[2]] += face_normals[i]

        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normals /= norms

        return normals
