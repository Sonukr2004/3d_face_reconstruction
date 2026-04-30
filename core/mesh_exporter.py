"""
mesh_exporter.py
Exports the reconstructed 3D face mesh to standard formats:
  - .obj  (Wavefront OBJ — opens in Blender, Windows 3D Viewer, etc.)
  - .ply  (ASCII PLY — opens in MeshLab, CloudCompare, etc.)
"""

import os
import numpy as np


class MeshExporter:
    """Exports 3D face mesh to .obj and .ply files."""

    def export_obj(
        self,
        filepath: str,
        vertices: np.ndarray,
        triangles: np.ndarray,
        colors: np.ndarray = None,
    ) -> str:
        """
        Export mesh as Wavefront .obj file.

        Args:
            filepath  : output path (e.g. 'exports/face.obj')
            vertices  : (N, 3) float array
            triangles : (M, 3) int array (0-indexed)
            colors    : (N, 3) float array in [0,1], optional

        Returns:
            Absolute filepath of the saved file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        with open(filepath, "w") as f:
            f.write("# 3D Face Reconstruction\n")
            f.write(f"# Vertices: {len(vertices)}\n")
            f.write(f"# Triangles: {len(triangles)}\n\n")

            # Write vertices (and optional vertex colors as comments or as 'v r g b')
            for i, v in enumerate(vertices):
                if colors is not None:
                    r, g, b = colors[i]
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {r:.4f} {g:.4f} {b:.4f}\n")
                else:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            f.write("\n")

            # Write faces (OBJ uses 1-indexed)
            for tri in triangles:
                f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

        return os.path.abspath(filepath)

    def export_ply(
        self,
        filepath: str,
        vertices: np.ndarray,
        triangles: np.ndarray,
        colors: np.ndarray = None,
    ) -> str:
        """
        Export mesh as ASCII PLY file with optional vertex colors.

        Args:
            filepath  : output path (e.g. 'exports/face.ply')
            vertices  : (N, 3) float array
            triangles : (M, 3) int array (0-indexed)
            colors    : (N, 3) float array in [0,1], optional

        Returns:
            Absolute filepath of the saved file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        has_colors = colors is not None
        n_verts = len(vertices)
        n_faces = len(triangles)

        with open(filepath, "w") as f:
            # PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write("comment 3D Face Reconstruction\n")
            f.write(f"element vertex {n_verts}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            if has_colors:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            f.write(f"element face {n_faces}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")

            # Vertices
            for i, v in enumerate(vertices):
                if has_colors:
                    r = int(np.clip(colors[i][0] * 255, 0, 255))
                    g = int(np.clip(colors[i][1] * 255, 0, 255))
                    b = int(np.clip(colors[i][2] * 255, 0, 255))
                    f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {r} {g} {b}\n")
                else:
                    f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Faces
            for tri in triangles:
                f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")

        return os.path.abspath(filepath)
