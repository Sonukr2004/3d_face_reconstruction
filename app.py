"""
app.py
3D Face Reconstruction System — Streamlit Web Application

Run:  streamlit run app.py
Uses pure OpenCV (no mediapipe, no internet required).
"""

import os
import time
import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ── local imports ──────────────────────────────────────────────────────────────
from core.face_detector       import FaceDetector
from core.landmark_extractor  import LandmarkExtractor
from core.mesh_builder        import MeshBuilder
from core.texture_mapper      import TextureMapper
from core.mesh_exporter       import MeshExporter
from core.face_auth           import FaceAuthSystem
from core.custom_model        import CustomFaceClassifier
from visualization.plotly_viewer import build_3d_figure, build_landmarks_figure
from utils.image_utils        import pil_to_cv2, cv2_to_pil, resize_keep_aspect, auto_orient

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="3D Face Reconstruction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 40%, #1a0a2e 100%);
    color: #e0e0e0;
  }

  [data-testid="stSidebar"] {
    background: rgba(10,10,30,0.95) !important;
    border-right: 1px solid rgba(100,100,255,0.15);
  }

  .hero-title {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
    background: linear-gradient(90deg, #6c63ff, #00d4ff, #ff6b9d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: -1px;
  }

  .hero-subtitle {
    text-align: center;
    color: rgba(200,200,220,0.65);
    font-size: 1rem;
    margin-bottom: 2rem;
    font-weight: 300;
  }

  .stat-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(100,100,255,0.15);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s, border-color 0.2s;
    margin-bottom: 0.5rem;
  }
  .stat-card:hover { transform: translateY(-2px); border-color: rgba(108,99,255,0.4); }
  .stat-value {
    font-size: 1.8rem; font-weight: 700;
    background: linear-gradient(90deg,#6c63ff,#00d4ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .stat-label {
    font-size: 0.75rem; color: rgba(180,180,210,0.6);
    text-transform: uppercase; letter-spacing: 1px; margin-top: 0.2rem;
  }

  .section-header {
    font-size: 1.1rem; font-weight: 600; color: #a0a8ff;
    text-transform: uppercase; letter-spacing: 1.5px;
    margin: 1.5rem 0 0.75rem;
    border-bottom: 1px solid rgba(100,100,255,0.2);
    padding-bottom: 0.4rem;
  }

  [data-testid="stFileUploaderDropzone"] {
    background: rgba(108,99,255,0.05) !important;
    border: 2px dashed rgba(108,99,255,0.3) !important;
    border-radius: 12px !important;
  }

  .stButton > button {
    background: linear-gradient(135deg,#6c63ff,#00d4ff) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important; width: 100%;
    transition: opacity 0.2s, transform 0.2s !important;
  }
  .stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }

  .stDownloadButton > button {
    background: linear-gradient(135deg,#ff6b9d,#ff8e53) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important; width: 100%;
  }

  .stProgress > div > div {
    background: linear-gradient(90deg,#6c63ff,#00d4ff) !important;
  }
</style>
""", unsafe_allow_html=True)


# ── cached models ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return (
        FaceDetector(),
        LandmarkExtractor(),
        MeshBuilder(depth_scale=3.5),
        TextureMapper(),
        MeshExporter(),
    )


# ── pipeline ───────────────────────────────────────────────────────────────────
def process_image(
    bgr: np.ndarray,
    face_detector, lm_extractor, mesh_builder,
    texture_mapper, mesh_exporter,
    depth_scale: float = 1.0,
    wireframe: bool = False,
    auto_rotate: bool = True,
    export_dir: str = "exports",
):
    result = {"success": False}

    img = resize_keep_aspect(bgr, max_side=720)

    # Step 1 – face detection
    detections = face_detector.detect(img)
    if not detections:
        result["error"] = (
            "❌ No face detected. Please use a clear frontal face photo "
            "with good lighting."
        )
        return result

    best = max(detections, key=lambda d: d["bbox"][2] * d["bbox"][3])
    face_bbox = best["bbox"]

    # Step 2 – 3D landmarks
    mesh_builder.depth_scale = depth_scale
    landmarks_3d, _ = lm_extractor.extract(img, face_bbox)
    if landmarks_3d is None:
        result["error"] = "❌ Could not extract landmarks."
        return result

    result["n_landmarks"] = len(landmarks_3d)

    # Annotated 2D image
    annotated_bgr = lm_extractor.draw_landmarks(img, face_bbox)
    result["landmark_img"] = cv2_to_pil(annotated_bgr)

    # Step 3 – mesh
    vertices, triangles = mesh_builder.build(landmarks_3d)
    result.update(vertices=vertices, triangles=triangles, n_triangles=len(triangles))

    # Step 4 – texture
    colors = texture_mapper.extract_vertex_colors(img, landmarks_3d)
    result["colors"] = colors

    # Step 5 – Plotly figures
    result["mesh_fig"] = build_3d_figure(
        vertices, triangles, colors,
        wireframe=wireframe, title="",
        auto_rotate=auto_rotate,
    )
    result["landmarks_fig"] = build_landmarks_figure(vertices)

    # Step 6 – export
    os.makedirs(export_dir, exist_ok=True)
    obj_path = os.path.join(export_dir, "face_3d.obj")
    ply_path = os.path.join(export_dir, "face_3d.ply")
    mesh_exporter.export_obj(obj_path, vertices, triangles, colors)
    mesh_exporter.export_ply(ply_path, vertices, triangles, colors)
    result.update(obj_path=obj_path, ply_path=ply_path)

    result["success"] = True
    return result


# ── sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="section-header">⚙️ Settings</div>', unsafe_allow_html=True)

        depth_scale = st.slider(
            "Depth Scale", 0.5, 3.0, 1.0, 0.1,
            help="Fine-tune depth exaggeration. 1.0 = anatomically realistic."
        )
        wireframe = st.checkbox("Wireframe Mode", False,
                                help="Show mesh edges instead of solid surface")
        auto_rotate = st.checkbox("🔄 Auto-Rotate 3D Model", True,
                                  help="Spin the 3D model automatically")

        st.markdown('<div class="section-header">📹 Real-Time Mode</div>', unsafe_allow_html=True)
        st.markdown("""
<div style='color:rgba(180,180,220,0.7);font-size:0.80rem;margin-bottom:0.6rem'>
Run live webcam with 4 visual modes (Neon, Heatmap, Hologram).
Press <b>[M]</b> to switch mode, <b>[S]</b> to save, <b>[Q]</b> to quit.
</div>""", unsafe_allow_html=True)
        if st.button("🎥 Launch Real-Time Webcam", use_container_width=True):
            import subprocess, sys
            subprocess.Popen([sys.executable, "realtime.py"],
                             cwd=os.path.dirname(__file__))
            st.success("✅ Webcam window opened! Check your taskbar.")

        st.markdown('<div class="section-header">🌍 Real-World Applications</div>', unsafe_allow_html=True)
        st.markdown("""
<div style='color:rgba(180,180,220,0.80);font-size:0.80rem;line-height:1.9'>
🔐 <b>Face ID / Biometrics</b><br>
&nbsp;&nbsp;Apple Face ID, Windows Hello<br>
🎬 <b>VFX & Film</b><br>
&nbsp;&nbsp;Digital actors in movies<br>
🏥 <b>Medical / Surgery</b><br>
&nbsp;&nbsp;Pre-surgery facial planning<br>
🎮 <b>Gaming / Avatars</b><br>
&nbsp;&nbsp;Personalized game characters<br>
🕵️ <b>Forensics</b><br>
&nbsp;&nbsp;Facial reconstruction of victims<br>
🥽 <b>AR / VR</b><br>
&nbsp;&nbsp;Real-time face filters & masks
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">📦 Export Formats</div>', unsafe_allow_html=True)
        st.markdown("""
<div style='color:rgba(180,180,220,0.75);font-size:0.82rem;line-height:1.8'>
🔵 <b>.OBJ</b> — Blender, Windows 3D Viewer<br>
🟣 <b>.PLY</b> — MeshLab, CloudCompare<br>
🟡 <b>.NPY</b> — Raw landmark numpy array
</div>""", unsafe_allow_html=True)

    return depth_scale, wireframe, auto_rotate


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    st.markdown('<h1 class="hero-title">🧠 3D Face Reconstruction</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Computer Vision · 3D Reconstruction · Face Authentication Security</p>',
        unsafe_allow_html=True,
    )

    with st.spinner("Loading models..."):
        face_detector, lm_extractor, mesh_builder, texture_mapper, mesh_exporter = load_models()

    depth_scale, wireframe, auto_rotate = render_sidebar()

    # ── Top-level page tabs ────────────────────────────────────────────────────
    page_3d, page_auth, page_train = st.tabs(["🌐 3D Reconstruction", "🔐 Face Authentication", "🧠 Model Training"])

    with page_auth:
        render_auth_page(face_detector, lm_extractor)

    with page_train:
        render_training_page()

    with page_3d:
        # ── input ─────────────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">📷 Input</div>', unsafe_allow_html=True)

        bgr_image = None
        upload_tab, webcam_tab = st.tabs(["📁 Upload Image", "📸 Webcam Capture"])

        with upload_tab:
            uploaded = st.file_uploader(
                "Drop a face image here (JPG, PNG, BMP, WEBP)",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                label_visibility="collapsed",
            )
            if uploaded:
                pil_img = auto_orient(Image.open(uploaded))
                bgr_image = pil_to_cv2(pil_img)

        with webcam_tab:
            st.info("📸 Allow camera access, then click the button below to capture your face.")
            webcam_img = st.camera_input("Take a photo", label_visibility="collapsed")
            if webcam_img:
                bgr_image = pil_to_cv2(Image.open(webcam_img))

        # ── preview + generate button ──────────────────────────────────────────────
        if bgr_image is not None:
            prev_col, btn_col = st.columns([2, 1])
            with prev_col:
                st.image(
                    cv2_to_pil(resize_keep_aspect(bgr_image, 400)),
                    caption="✅ Image loaded — ready to reconstruct",
                    use_container_width=False,
                    width=320,
                )
            with btn_col:
                st.markdown("<br><br>", unsafe_allow_html=True)
                generate = st.button("🚀 Generate 3D Model", use_container_width=True)
        else:
            generate = False

        # ── process ───────────────────────────────────────────────────────────────
        if bgr_image is not None and generate:
            export_dir = os.path.join(os.path.dirname(__file__), "exports")

            with st.spinner("🔬 Reconstructing 3D face..."):
                prog = st.progress(0, text="Detecting face…")
                time.sleep(0.05)
                prog.progress(30, text="Extracting landmarks…")
                result = process_image(
                    bgr_image,
                    face_detector, lm_extractor, mesh_builder,
                    texture_mapper, mesh_exporter,
                    depth_scale=depth_scale,
                    wireframe=wireframe,
                    auto_rotate=auto_rotate,
                    export_dir=export_dir,
                )
                prog.progress(80, text="Building 3-D mesh…")
                time.sleep(0.05)
                prog.progress(100, text="Done ✅")
                time.sleep(0.3)
                prog.empty()

            if not result["success"]:
                st.error(result.get("error", "Unknown error."))
                return

            # ── stats row ─────────────────────────────────────────────────────────
            c1, c2, c3, c4 = st.columns(4)
            for col, val, lbl in [
                (c1, str(result["n_landmarks"]),  "3D Landmarks"),
                (c2, str(result["n_triangles"]),  "Triangles"),
                (c3, f"{depth_scale}×",            "Depth Scale"),
                (c4, "✓ Ready",                    "Export"),
            ]:
                with col:
                    st.markdown(
                        f'<div class="stat-card">'
                        f'<div class="stat-value">{val}</div>'
                        f'<div class="stat-label">{lbl}</div></div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("")

            # ── side-by-side viewer ───────────────────────────────────────────────
            left, right = st.columns([1, 1.6])

            with left:
                st.markdown('<div class="section-header">🗺️ 2D Landmarks</div>', unsafe_allow_html=True)
                st.image(result["landmark_img"], use_container_width=True,
                         caption=f"Dense face mesh — {result['n_landmarks']} points")
                with st.expander("🔵 3D Point Cloud"):
                    st.plotly_chart(result["landmarks_fig"], use_container_width=True)

            with right:
                st.markdown('<div class="section-header">🌐 Interactive 3D Model</div>', unsafe_allow_html=True)
                st.plotly_chart(result["mesh_fig"], use_container_width=True)
                st.caption("🖱️ Drag to rotate · Scroll to zoom · Right-click to pan")

            # ── export ────────────────────────────────────────────────────────────
            st.markdown('<div class="section-header">💾 Export 3D Model</div>', unsafe_allow_html=True)
            e1, e2, e3 = st.columns(3)

            with e1:
                if os.path.exists(result.get("obj_path", "")):
                    st.download_button(
                        "⬇️ Download .OBJ (Blender)",
                        open(result["obj_path"], "rb").read(),
                        "face_3d.obj", "application/octet-stream",
                        use_container_width=True,
                    )

            with e2:
                if os.path.exists(result.get("ply_path", "")):
                    st.download_button(
                        "⬇️ Download .PLY (MeshLab)",
                        open(result["ply_path"], "rb").read(),
                        "face_3d.ply", "application/octet-stream",
                        use_container_width=True,
                    )

            with e3:
                buf = io.BytesIO()
                np.save(buf, result["vertices"])
                buf.seek(0)
                st.download_button(
                    "⬇️ Download Landmarks .npy",
                    buf, "landmarks_3d.npy",
                    "application/octet-stream",
                    use_container_width=True,
                )

            st.success(
                f"✅ Done! {result['n_landmarks']} landmarks · "
                f"{result['n_triangles']} triangles · Files saved in `exports/`"
            )

        else:
            # ── empty state ───────────────────────────────────────────────────────
            st.markdown("""
<div style='margin:3rem auto;max-width:520px;text-align:center;padding:3rem 2rem;
background:rgba(108,99,255,0.06);border:1px dashed rgba(108,99,255,0.25);border-radius:20px'>
  <div style='font-size:4rem;margin-bottom:1rem'>🧠</div>
  <h3 style='color:#a0a8ff;margin-bottom:0.5rem'>No face loaded yet</h3>
  <p style='color:rgba(180,180,220,0.6);font-size:0.9rem'>
    Upload a photo or use your webcam above.<br>
    Works best with a clear, front-facing photo.
  </p>
  <div style='margin-top:1.5rem;color:rgba(150,150,200,0.5);font-size:0.8rem'>
    💡 Good lighting = better reconstruction
  </div>
</div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-header">✨ Features</div>', unsafe_allow_html=True)
            cols = st.columns(4)
            features = [
                ("🎯", "Dense Face Mesh", "700+ points for smooth 3D surface"),
                ("🌐", "Real-Time Mesh", "Delaunay triangulated 3D mesh"),
                ("🎨", "Texture Mapping", "Actual skin colours from your photo"),
                ("🔐", "Face Auth", "Register & verify identity with 3D signatures"),
            ]
            for col, (icon, title, desc) in zip(cols, features):
                with col:
                    st.markdown(
                        f'<div class="stat-card" style="text-align:left;padding:1.2rem">'
                        f'<div style="font-size:1.8rem">{icon}</div>'
                        f'<div style="font-weight:600;color:#c0c8ff;margin:0.4rem 0 0.3rem">{title}</div>'
                        f'<div style="font-size:0.78rem;color:rgba(170,170,210,0.65)">{desc}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


# ── Model Training Page ──────────────────────────────────────────────────────
def render_training_page():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.markdown("""
<div style='background:linear-gradient(135deg,rgba(108,99,255,0.12),rgba(255,107,157,0.08));
border:1px solid rgba(108,99,255,0.25);border-radius:16px;padding:1.5rem;margin-bottom:1.5rem'>
<h3 style='color:#a0a8ff;margin:0 0 0.5rem'>🧠 Custom Face Recognition Model — Trained From Scratch</h3>
<p style='color:rgba(180,180,220,0.75);font-size:0.88rem;margin:0'>
This page trains a <b>Multi-Layer Perceptron (MLP)</b> entirely from scratch using pure NumPy —
no pretrained weights, no external deep learning frameworks. The model learns directly on
the 128-D face embeddings extracted from your registered users.
</p>
</div>""", unsafe_allow_html=True)

    auth = FaceAuthSystem()
    profiles = auth.list_profiles()

    # ── Architecture Diagram ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">🏗️ Model Architecture</div>', unsafe_allow_html=True)

    arch_layers = [
        {"name": "Input Layer",   "neurons": 128, "activation": "Raw 128-D SFace Embeddings", "color": "#6c63ff"},
        {"name": "PCA Layer",     "neurons": 32,  "activation": "Linear Projection (32-D)",   "color": "#00d4ff"},
        {"name": "Hidden Layer 1","neurons": 64,  "activation": "ReLU + Dropout (20%)",       "color": "#ff6b9d"},
        {"name": "Hidden Layer 2","neurons": 32,  "activation": "ReLU + Dropout (20%)",       "color": "#ff8e53"},
        {"name": "Output Layer",  "neurons": "N", "activation": "Softmax (1 per user)",       "color": "#00e87a"},
    ]
    arch_cols = st.columns(len(arch_layers))
    for col, layer in zip(arch_cols, arch_layers):
        with col:
            st.markdown(
                f'<div class="stat-card" style="border-color:{layer["color"]}33">'
                f'<div style="font-size:1.6rem;font-weight:700;color:{layer["color"]}">{layer["neurons"]}</div>'
                f'<div style="font-weight:600;color:#c0c8ff;font-size:0.85rem;margin:0.3rem 0">{layer["name"]}</div>'
                f'<div style="font-size:0.70rem;color:rgba(170,170,210,0.65)">{layer["activation"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("""
<div style='text-align:center;color:rgba(150,150,200,0.5);font-size:0.78rem;margin:0.3rem 0 1.2rem'>
⚡ Training: Mini-batch SGD · Momentum=0.9 · LR Decay (×0.97 per 10 epochs) · L2 Regularisation (λ=1e-4)
</div>""", unsafe_allow_html=True)

    # ── Check registered users ────────────────────────────────────────────────
    if len(profiles) < 2:
        st.warning("""
⚠️ **Need at least 2 registered users to train!**
Go to the **🔐 Face Authentication → Register User** tab and register at least 2 people, then come back here.
        """)
        return

    # ── Training Config ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⚙️ Training Configuration</div>', unsafe_allow_html=True)
    cfg1, cfg2, cfg3, cfg4 = st.columns(4)
    with cfg1:
        n_epochs = st.slider("Epochs", 20, 150, 60, 5, key="train_epochs")
    with cfg2:
        aug_factor = st.slider("Data Augmentation ×", 5, 30, 15, 1, key="train_aug",
                               help="Multiply training samples via Gaussian noise augmentation")
    with cfg3:
        pca_k = st.slider("PCA Components", 8, 32, 24, 2, key="train_pca")
    with cfg4:
        lr_val = st.select_slider("Learning Rate", [0.01, 0.02, 0.05, 0.1, 0.2], value=0.05, key="train_lr")

    # ── Load embeddings from disk ─────────────────────────────────────────────
    embeddings, labels = [], []
    for p in profiles:
        sig_path = p.get("sig_path", "")
        if os.path.exists(sig_path):
            emb = np.load(sig_path)
            embeddings.append(emb.flatten()[:128])
            labels.append(p["name"])

    if len(set(labels)) < 2:
        st.error("❌ Could not load embeddings for 2+ distinct users. Re-register users.")
        return

    n_users  = len(set(labels))
    raw_embs = np.array(embeddings, dtype=np.float32)
    raw_lbls = np.array(labels)

    # ── Info row ─────────────────────────────────────────────────────────────
    info1, info2, info3, info4 = st.columns(4)
    for col, val, lbl in [
        (info1, str(len(raw_embs)),                   "Raw Samples"),
        (info2, str(len(raw_embs) * aug_factor),      "Augmented Samples"),
        (info3, str(n_users),                         "Classes (Users)"),
        (info4, str(pca_k),                           "PCA Dimensions"),
    ]:
        with col:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-value">{val}</div>'
                f'<div class="stat-label">{lbl}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("")

    # ── Train Button ──────────────────────────────────────────────────────────
    if not st.button("🚀 Start Training Custom Model", use_container_width=True, key="btn_train"):
        st.markdown("""
<div style='text-align:center;padding:2rem;color:rgba(180,180,220,0.5);font-size:0.9rem'>
👆 Click the button above to begin training your custom model from scratch.
</div>""", unsafe_allow_html=True)
        return

    # ── Data Augmentation ─────────────────────────────────────────────────────
    rng_aug = np.random.default_rng(7)
    aug_embs, aug_lbls = list(raw_embs), list(raw_lbls)
    for emb, lbl in zip(raw_embs, raw_lbls):
        for _ in range(aug_factor - 1):
            noise = rng_aug.normal(0, 0.012, emb.shape).astype(np.float32)
            aug_embs.append(emb + noise)
            aug_lbls.append(lbl)
    aug_embs = np.array(aug_embs, dtype=np.float32)
    aug_lbls = np.array(aug_lbls)

    # Shuffle
    perm = rng_aug.permutation(len(aug_embs))
    aug_embs, aug_lbls = aug_embs[perm], aug_lbls[perm]

    # ── PCA ───────────────────────────────────────────────────────────────────
    clf = CustomFaceClassifier(pca_components=pca_k)
    clf.mlp = None
    X_pca, y = clf.prepare_data(list(aug_embs), list(aug_lbls))
    mlp = clf.build_mlp(n_classes=n_users)
    mlp.lr = lr_val

    # ── PCA Variance chart ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 PCA Explained Variance</div>', unsafe_allow_html=True)
    var = clf.pca.explained_variance_ratio_
    cum_var = np.cumsum(var)
    fig_pca = go.Figure()
    fig_pca.add_bar(
        x=list(range(1, len(var)+1)), y=var * 100,
        name="Per Component",
        marker_color="#6c63ff", opacity=0.7,
    )
    fig_pca.add_scatter(
        x=list(range(1, len(cum_var)+1)), y=cum_var * 100,
        name="Cumulative", mode="lines+markers",
        line={"color": "#00d4ff", "width": 2},
        marker={"size": 5},
    )
    fig_pca.add_hline(y=90, line_dash="dash", line_color="#ff6b9d",
                      annotation_text="90% threshold", annotation_font_color="#ff6b9d")
    fig_pca.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.2)",
        height=260,
        margin={"t": 20, "b": 40, "l": 50, "r": 20},
        legend={"orientation": "h", "y": 1.05},
        xaxis_title="Principal Component",
        yaxis_title="Variance Explained (%)",
    )
    st.plotly_chart(fig_pca, use_container_width=True)

    # ── Live Training Charts ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Live Training Progress</div>', unsafe_allow_html=True)

    prog_bar  = st.progress(0, text="Epoch 0 / " + str(n_epochs))
    chart_ph  = st.empty()
    stats_ph  = st.empty()

    loss_hist, acc_hist, epochs_hist, lr_hist = [], [], [], []

    UPDATE_EVERY = max(1, n_epochs // 40)   # refresh chart ~40 times

    for metrics in mlp.fit_epoch_by_epoch(X_pca, y, n_epochs=n_epochs, batch_size=16):
        ep    = metrics["epoch"]
        loss_hist.append(metrics["loss"])
        acc_hist.append(metrics["accuracy"] * 100)
        epochs_hist.append(ep)
        lr_hist.append(metrics["lr"])

        prog_bar.progress(ep / n_epochs, text=f"Epoch {ep} / {n_epochs} — Loss: {metrics['loss']:.4f}  Acc: {metrics['accuracy']*100:.1f}%")

        if ep % UPDATE_EVERY == 0 or ep == n_epochs:
            fig_train = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Cross-Entropy Loss", "Training Accuracy (%)"),
            )
            fig_train.add_scatter(
                x=epochs_hist, y=loss_hist, mode="lines",
                line={"color": "#ff6b9d", "width": 2},
                name="Loss", row=1, col=1,
            )
            fig_train.add_scatter(
                x=epochs_hist, y=acc_hist, mode="lines",
                line={"color": "#00d4ff", "width": 2},
                fill="tozeroy", fillcolor="rgba(0,212,255,0.08)",
                name="Accuracy", row=1, col=2,
            )
            fig_train.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0.2)",
                height=300,
                margin={"t": 40, "b": 40, "l": 50, "r": 20},
                showlegend=False,
            )
            fig_train.update_xaxes(title_text="Epoch")
            chart_ph.plotly_chart(fig_train, use_container_width=True)

    prog_bar.progress(1.0, text="✅ Training Complete!")

    final_acc  = acc_hist[-1]
    final_loss = loss_hist[-1]

    # ── Final stats ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🏆 Training Results</div>', unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns(4)
    for col, val, lbl in [
        (r1, f"{final_acc:.1f}%",  "Final Accuracy"),
        (r2, f"{final_loss:.4f}",  "Final Loss"),
        (r3, f"{n_epochs}",         "Epochs Trained"),
        (r4, f"{len(aug_embs)}",    "Training Samples"),
    ]:
        with col:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-value">{val}</div>'
                f'<div class="stat-label">{lbl}</div></div>',
                unsafe_allow_html=True,
            )

    # ── Weight Heatmap ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔥 Learned Weight Heatmaps</div>', unsafe_allow_html=True)
    winfo = mlp.get_weights_info()
    wc1, wc2, wc3 = st.columns(3)
    for wc, wkey, wtitle in [(wc1,"W1","W1: PCA→Hidden 1 (32×64)"),(wc2,"W2","W2: H1→H2 (64×32)"),(wc3,"W3",f"W3: H2→Output (32×{n_users})")]:
        with wc:
            wmat = winfo[wkey]
            fig_w = go.Figure(go.Heatmap(
                z=wmat[:32, :32] if wmat.shape[0] > 32 else wmat,
                colorscale="RdBu", zmid=0,
                showscale=False,
            ))
            fig_w.update_layout(
                title={"text": wtitle, "font": {"size": 11, "color": "#a0a8ff"}},
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=200,
                margin={"t": 30, "b": 5, "l": 5, "r": 5},
            )
            st.plotly_chart(fig_w, use_container_width=True)

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🗂️ Confusion Matrix (Training Set)</div>', unsafe_allow_html=True)
    last_metrics_preds = mlp.predict(X_pca)
    label_list = sorted(set(labels))
    label_map_inv = {i: lbl for i, lbl in enumerate(label_list)}
    n_cls = len(label_list)
    cm = np.zeros((n_cls, n_cls), dtype=int)
    y_idx_arr = np.array([label_list.index(lbl) for lbl in aug_lbls])
    for true, pred in zip(y_idx_arr, last_metrics_preds):
        cm[true][pred] += 1
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig_cm = go.Figure(go.Heatmap(
        z=cm_norm,
        x=[f"Pred: {l}" for l in label_list],
        y=[f"True: {l}" for l in label_list],
        text=cm,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=True,
    ))
    fig_cm.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.2)",
        height=max(280, n_cls * 80),
        margin={"t": 20, "b": 60, "l": 100, "r": 20},
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    st.success(
        f"✅ Custom MLP trained successfully! "
        f"{n_epochs} epochs · {len(aug_embs)} samples · "
        f"Final accuracy **{final_acc:.1f}%** · Loss **{final_loss:.4f}**"
    )
    st.markdown("""
<div style='background:rgba(0,212,255,0.06);border:1px solid rgba(0,212,255,0.2);
border-radius:12px;padding:1rem 1.2rem;font-size:0.82rem;color:rgba(180,180,220,0.75);margin-top:0.5rem'>
<b style='color:#00d4ff'>How this model was trained:</b><br>
1. <b>Feature Extraction</b> — 128-D SFace embeddings from each enrolled face<br>
2. <b>Data Augmentation</b> — Gaussian noise copies to simulate real-world variation<br>
3. <b>PCA</b> — Dimensionality reduction from 128-D → 32-D (from scratch using eigen-decomposition)<br>
4. <b>MLP Forward Pass</b> — ReLU activations, dropout regularisation, Softmax output<br>
5. <b>Backpropagation</b> — Chain-rule gradient computation, fully implemented in NumPy<br>
6. <b>Optimiser</b> — SGD with momentum (0.9), learning rate decay, L2 regularisation
</div>""", unsafe_allow_html=True)


# ── Face Authentication Page ───────────────────────────────────────────────────
def render_auth_page(face_detector, lm_extractor):
    auth = FaceAuthSystem()

    st.markdown("""
<div style='background:linear-gradient(135deg,rgba(108,99,255,0.12),rgba(0,212,255,0.08));
border:1px solid rgba(108,99,255,0.25);border-radius:16px;padding:1.5rem;margin-bottom:1.5rem'>
<h3 style='color:#a0a8ff;margin:0 0 0.5rem'>🔐 Face Authentication System</h3>
<p style='color:rgba(180,180,220,0.75);font-size:0.88rem;margin:0'>
Simulates how <b>Apple Face ID</b> and <b>Windows Hello</b> work —
register your face as a 3D geometric profile, then verify your identity.
The 3D depth signature cannot be spoofed by a flat photo.
</p>
</div>""", unsafe_allow_html=True)

    reg_tab, ver_tab, mgr_tab = st.tabs(
        ["📝 Register User", "✅ Verify Identity", "👥 Manage Users"]
    )

    # ── REGISTER ────────────────────────────────────────────────────────────────
    with reg_tab:
        st.markdown('<div class="section-header">📝 Register a New User</div>',
                    unsafe_allow_html=True)

        name = st.text_input("Full Name", placeholder="e.g. Sonu Kumar",
                             key="reg_name")

        reg_src_upload, reg_src_cam = st.tabs(["📁 Upload Photo", "📸 Use Webcam"])
        reg_bgr = None

        with reg_src_upload:
            reg_upload = st.file_uploader("Upload face photo",
                                          type=["jpg","jpeg","png","bmp","webp"],
                                          key="reg_upload")
            if reg_upload:
                reg_bgr = pil_to_cv2(auto_orient(Image.open(reg_upload)))

        with reg_src_cam:
            st.info("📸 Click below to take your photo for registration.")
            reg_cam = st.camera_input("Take registration photo",
                                      label_visibility="collapsed", key="reg_cam")
            if reg_cam:
                reg_bgr = pil_to_cv2(Image.open(reg_cam))

        if reg_bgr is not None:
            reg_bgr = resize_keep_aspect(reg_bgr, 720)
            st.image(cv2_to_pil(reg_bgr), width=280, caption="Preview")

            if not name:
                st.warning("⚠️ Please enter a name above before registering.")
            elif st.button("📥 Register Face", use_container_width=True, key="btn_register"):
                with st.spinner("Extracting 3D face signature..."):
                    det = face_detector.detect(reg_bgr)
                    if not det:
                        st.error("❌ No face detected. Use a clear frontal photo.")
                    else:
                        bbox = max(det, key=lambda d: d["bbox"][2]*d["bbox"][3])["bbox"]
                        lm, _ = lm_extractor.extract(reg_bgr, bbox)
                        if lm is None:
                            st.error("❌ Could not extract landmarks.")
                        else:
                            result = auth.register(name, lm,
                                                   bgr_image=reg_bgr,
                                                   face_bbox=bbox)
                            if result["success"]:
                                st.success(
                                    f"✅ **{name}** registered successfully!\n\n"
                                    f"Profile ID: `{result['profile_id']}`"
                                )
                            else:
                                st.error(result.get("error", "Registration failed."))

    # ── VERIFY ──────────────────────────────────────────────────────────────────
    with ver_tab:
        st.markdown('<div class="section-header">✅ Verify Identity</div>',
                    unsafe_allow_html=True)

        ver_src_upload, ver_src_cam = st.tabs(["📁 Upload Photo", "📸 Use Webcam"])
        ver_bgr = None

        with ver_src_upload:
            ver_upload = st.file_uploader("Upload face photo to verify",
                                          type=["jpg","jpeg","png","bmp","webp"],
                                          key="ver_upload")
            if ver_upload:
                ver_bgr = pil_to_cv2(auto_orient(Image.open(ver_upload)))

        with ver_src_cam:
            st.info("📸 Snap a live photo to verify your identity.")
            ver_cam = st.camera_input("Take verification photo",
                                      label_visibility="collapsed", key="ver_cam")
            if ver_cam:
                ver_bgr = pil_to_cv2(Image.open(ver_cam))

        if ver_bgr is not None:
            ver_bgr = resize_keep_aspect(ver_bgr, 720)
            st.image(cv2_to_pil(ver_bgr), width=280, caption="Photo to verify")

            if st.button("🔍 Verify Identity", use_container_width=True, key="btn_verify"):
                with st.spinner("Comparing 3D face signatures..."):
                    det = face_detector.detect(ver_bgr)
                    if not det:
                        st.error("❌ No face detected.")
                    else:
                        bbox = max(det, key=lambda d: d["bbox"][2]*d["bbox"][3])["bbox"]
                        lm, _ = lm_extractor.extract(ver_bgr, bbox)
                        if lm is None:
                            st.error("❌ Could not extract landmarks.")
                        else:
                            res = auth.verify(lm,
                                             bgr_image=ver_bgr,
                                             face_bbox=bbox)
                            if not res["success"]:
                                st.error(res.get("error","Verification failed."))
                            else:
                                sim_pct = res["similarity"] * 100

                                # Big access result banner
                                if res["access_granted"]:
                                    st.markdown(f"""
<div style='background:linear-gradient(135deg,rgba(0,200,100,0.15),rgba(0,255,150,0.08));
border:2px solid #00c864;border-radius:16px;padding:2rem;text-align:center;margin:1rem 0'>
<div style='font-size:3rem'>🟢</div>
<div style='font-size:1.8rem;font-weight:700;color:#00e87a;margin:0.5rem 0'>ACCESS GRANTED</div>
<div style='font-size:1.1rem;color:#80ffb0'>Welcome, <b>{res["name"]}</b></div>
<div style='font-size:0.9rem;color:rgba(150,255,180,0.7);margin-top:0.5rem'>
Match Score: <b>{sim_pct:.1f}%</b> (threshold: {res["threshold"]*100:.0f}%)
</div>
</div>""", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
<div style='background:linear-gradient(135deg,rgba(255,50,50,0.15),rgba(255,0,0,0.08));
border:2px solid #ff3333;border-radius:16px;padding:2rem;text-align:center;margin:1rem 0'>
<div style='font-size:3rem'>🔴</div>
<div style='font-size:1.8rem;font-weight:700;color:#ff4444;margin:0.5rem 0'>ACCESS DENIED</div>
<div style='font-size:0.9rem;color:rgba(255,150,150,0.8);margin-top:0.5rem'>
Best match: <b>{res["name"]}</b> at <b>{sim_pct:.1f}%</b>
(need {res["threshold"]*100:.0f}%)
</div>
</div>""", unsafe_allow_html=True)

                                # Scores table
                                st.markdown('<div class="section-header">📊 All Match Scores</div>',
                                            unsafe_allow_html=True)
                                for r in res["all_scores"]:
                                    pct = r["similarity"] * 100
                                    bar_color = "#00c864" if r["similarity"] >= res["threshold"] else "#6c63ff"
                                    st.markdown(f"""
<div style='margin:0.4rem 0;padding:0.6rem 1rem;
background:rgba(255,255,255,0.03);border-radius:8px;
border:1px solid rgba(100,100,255,0.15)'>
<div style='display:flex;justify-content:space-between;margin-bottom:0.3rem'>
  <span style='color:#c0c8ff;font-weight:600'>{r["name"]}</span>
  <span style='color:{bar_color};font-weight:700'>{pct:.1f}%</span>
</div>
<div style='background:rgba(255,255,255,0.08);border-radius:4px;height:6px'>
  <div style='width:{min(pct,100):.1f}%;background:{bar_color};height:6px;border-radius:4px'></div>
</div>
</div>""", unsafe_allow_html=True)

    # ── MANAGE USERS ────────────────────────────────────────────────────────────
    with mgr_tab:
        st.markdown('<div class="section-header">👥 Registered Users</div>',
                    unsafe_allow_html=True)
        profiles = auth.list_profiles()

        if not profiles:
            st.markdown("""
<div style='text-align:center;padding:2rem;color:rgba(180,180,220,0.5)'>
No users registered yet. Go to <b>Register User</b> tab to add the first user.
</div>""", unsafe_allow_html=True)
        else:
            st.success(f"✅ {len(profiles)} user(s) registered")
            for p in profiles:
                col_info, col_del = st.columns([4, 1])
                with col_info:
                    reg_time = p.get("registered_at","")[:16].replace("T"," ")
                    st.markdown(f"""
<div class='stat-card' style='text-align:left;padding:0.8rem 1rem;margin:0.3rem 0'>
<span style='font-size:1.2rem'>👤</span>
<span style='font-weight:600;color:#c0c8ff;margin-left:0.5rem'>{p["name"]}</span>
<span style='color:rgba(150,150,200,0.5);font-size:0.75rem;margin-left:1rem'>
Registered: {reg_time}</span>
</div>""", unsafe_allow_html=True)
                with col_del:
                    if st.button("🗑️", key=f"del_{p['id']}",
                                 help=f"Delete {p['name']}"):
                        auth.delete_profile(p["id"])
                        st.rerun()


if __name__ == "__main__":
    main()
