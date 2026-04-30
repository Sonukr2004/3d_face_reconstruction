"""
plotly_viewer.py
Creates an interactive 3D face mesh using Plotly.
Features: texture colors, wireframe mode, auto-rotation animation, premium dark theme.
"""

import numpy as np
import plotly.graph_objects as go


def build_3d_figure(
    vertices: np.ndarray,
    triangles: np.ndarray,
    colors: np.ndarray = None,
    wireframe: bool = False,
    title: str = "",
    auto_rotate: bool = False,
) -> go.Figure:
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    i = triangles[:, 0]
    j = triangles[:, 1]
    k = triangles[:, 2]

    # Build vertex colors
    if colors is not None:
        vertex_colors = [
            f"rgb({int(np.clip(c[0]*255,0,255))},"
            f"{int(np.clip(c[1]*255,0,255))},"
            f"{int(np.clip(c[2]*255,0,255))})"
            for c in colors
        ]
    else:
        z_norm = (z - z.min()) / ((z.max() - z.min()) + 1e-6)
        vertex_colors = [
            f"rgb({int(200*v+55)},{int(150*v+50)},{int(100*v+100)})"
            for v in z_norm
        ]

    mesh_trace = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        vertexcolor=vertex_colors,
        opacity=1.0 if not wireframe else 0.05,
        flatshading=False,
        lighting=dict(
            ambient=0.45,
            diffuse=0.85,
            specular=0.40,
            roughness=0.45,
            fresnel=0.30,
        ),
        lightposition=dict(x=200, y=-300, z=400),
        name="Face Mesh",
        showscale=False,
        hoverinfo="none",
    )

    traces = [mesh_trace]

    # Wireframe overlay
    if wireframe:
        wire_x, wire_y, wire_z = [], [], []
        for tri in triangles:
            for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
                wire_x += [vertices[a, 0], vertices[b, 0], None]
                wire_y += [vertices[a, 1], vertices[b, 1], None]
                wire_z += [vertices[a, 2], vertices[b, 2], None]
        traces.append(go.Scatter3d(
            x=wire_x, y=wire_y, z=wire_z,
            mode="lines",
            line=dict(color="rgba(0,220,180,0.6)", width=1),
            name="Wireframe",
            hoverinfo="none",
        ))

    fig = go.Figure(data=traces)

    # Auto-rotation frames
    if auto_rotate:
        frames = []
        for deg in range(0, 360, 4):
            rad = np.deg2rad(deg)
            r = 1.8
            frames.append(go.Frame(
                layout=dict(scene_camera=dict(
                    eye=dict(x=r*np.sin(rad), y=-0.25, z=r*np.cos(rad)),
                    up=dict(x=0, y=1, z=0),
                ))
            ))
        fig.frames = frames
        updatemenus = [dict(
            type="buttons",
            showactive=False,
            y=0.05, x=0.5, xanchor="center",
            buttons=[dict(
                label="▶ Rotate",
                method="animate",
                args=[None, dict(
                    frame=dict(duration=40, redraw=True),
                    fromcurrent=True,
                    transition=dict(duration=0),
                    mode="immediate",
                )]
            ), dict(
                label="⏸ Stop",
                method="animate",
                args=[[None], dict(
                    frame=dict(duration=0, redraw=False),
                    mode="immediate",
                )]
            )],
        )]
    else:
        updatemenus = []

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="#e0e0e0"), x=0.5) if title else None,
        scene=dict(
            xaxis=dict(visible=False, showgrid=False),
            yaxis=dict(visible=False, showgrid=False),
            zaxis=dict(visible=False, showgrid=False),
            bgcolor="rgb(8,8,18)",
            camera=dict(
                eye=dict(x=0.7, y=-0.25, z=1.3),
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
            ),
            aspectmode="data",
        ),
        paper_bgcolor="rgb(8,8,18)",
        plot_bgcolor="rgb(8,8,18)",
        margin=dict(l=0, r=0, t=10, b=0),
        font=dict(color="#e0e0e0"),
        height=600,
        updatemenus=updatemenus,
    )

    return fig


def build_landmarks_figure(vertices: np.ndarray) -> go.Figure:
    """Render landmarks as a depth-coloured point cloud."""
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    z_norm = (z - z.min()) / ((z.max() - z.min()) + 1e-6)

    fig = go.Figure(go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(
            size=3,
            color=z_norm,
            colorscale="Turbo",
            opacity=0.9,
            colorbar=dict(
                title="Depth",
                thickness=10,
                tickfont=dict(color="#aaa", size=9),
            ),
        ),
        hoverinfo="none",
        name="Landmarks",
    ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="rgb(8,8,18)",
            camera=dict(eye=dict(x=0.7, y=-0.25, z=1.3)),
            aspectmode="data",
        ),
        paper_bgcolor="rgb(8,8,18)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=380,
    )
    return fig
