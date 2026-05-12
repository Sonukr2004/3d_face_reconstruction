# 3D Face Reconstruction System

**End-to-end system for reconstructing 3D faces from single images with real-time web interface and face authentication**

## Project Overview

This project combines multiple face reconstruction techniques into a unified system:
- **Deep Learning Model** (Microsoft Deep3DFaceReconstruction) for robust 3D shape inference
- **Computer Vision Pipeline** for real-time processing and mesh refinement  
- **Web Application** for easy interaction and visualization
- **Face Authentication** system for biometric verification

## Architecture

```
Input Image
    ↓
┌─ Deep Learning Path ─────────────────────┐
│  DLModel/: Deep3DFaceReconstruction      │
│  ├─ 3DMM parameter prediction            │
│  ├─ Texture synthesis                    │
│  └─ Lighting estimation                  │
└──────────────────────────────────────────┘
                ↓
┌─ Post-processing & Refinement ────────────┐
│  core/: Face processing modules           │
│  ├─ Face detection (Haar Cascade)         │
│  ├─ Landmark extraction                   │
│  ├─ Mesh generation (Delaunay)            │
│  ├─ Texture mapping                       │
│  └─ Authentication (LBP histogram)        │
└──────────────────────────────────────────┘
                ↓
       3D Face Model (OBJ/PLY/GLTF)
                ↓
┌─ Visualization & Output ──────────────────┐
│  ├─ Interactive 3D viewer (Plotly)        │
│  ├─ Model export                          │
│  ├─ Face authentication verification      │
│  └─ Streamlit web UI                      │
└──────────────────────────────────────────┘
```

## Directory Structure

```
3d_face_reconstruction/
├── README.md                          # This file
├── app.py                             # Streamlit web application
├── realtime.py                        # Real-time video processing
├── requirements.txt                   # Python dependencies
│
├── DLModel/                           # Deep Learning Model (Microsoft Deep3D)
│   ├── readme.md                      # Model documentation
│   ├── train.py                       # Training script
│   ├── demo.py                        # Inference demo
│   ├── reconstruction_model.py        # Main model architecture
│   ├── face_decoder.py                # 3DMM decoder
│   ├── networks.py                    # Neural network layers
│   ├── renderer/                      # 3D mesh renderer
│   ├── BFM/                           # Basel Face Model data
│   ├── input/                         # Test inputs
│   ├── images/                        # Output renders
│   └── tf_mesh_renderer/              # TensorFlow mesh rendering
│
├── core/                              # Computer Vision Post-processing
│   ├── face_detector.py               # Haar Cascade face detection
│   ├── landmark_extractor.py          # Dense landmark extraction
│   ├── mesh_builder.py                # 3D mesh generation
│   ├── texture_mapper.py              # Texture synthesis
│   ├── mesh_exporter.py               # Export to OBJ/PLY/GLTF
│   └── face_auth.py                   # Facial authentication system
│
├── visualization/                     # 3D Visualization
│   └── plotly_viewer.py               # Interactive 3D viewer
│
├── utils/                             # Utility Functions
│   └── image_utils.py                 # Image processing helpers
│
├── exports/                           # Output directory for 3D models
├── profiles/                          # Stored face profiles & embeddings
└── input/                             # Input test images
```

## Key Features

✅ **Deep Learning 3D Reconstruction** - CNN-based 3DMM coefficient prediction  
✅ **Real-time Processing** - GPU-accelerated inference with CUDA support  
✅ **High-fidelity Mesh** - Delaunay triangulation + normal computation  
✅ **Texture Synthesis** - Photorealistic texture generation  
✅ **Face Authentication** - Biometric face verification system  
✅ **Web Interface** - Streamlit app for easy access  
✅ **Multiple Export Formats** - OBJ, PLY, GLTF support  
✅ **Interactive Visualization** - 3D viewer with Plotly  

## Installation

### Prerequisites
- Python 3.6+ (DLModel requires 3.6 for TensorFlow 1.12 compatibility)
- OpenCV
- TensorFlow 1.12 (for DLModel)
- CUDA/cuDNN (optional, for GPU acceleration)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/3d_face_reconstruction.git
cd 3d_face_reconstruction

# Install dependencies
pip install -r requirements.txt

# For DLModel (optional)
cd DLModel
# Follow DLModel/readme.md for TensorFlow 1.12 setup
```

## Usage

### Web Application (Recommended)
```bash
streamlit run app.py
```
Access at `http://localhost:8501`

### Real-time Video Processing
```bash
python realtime.py
```

### Command Line Inference
```python
from core.face_detector import FaceDetector
from core.landmark_extractor import LandmarkExtractor
from core.mesh_builder import MeshBuilder
import cv2

# Load image
image = cv2.imread("face.jpg")

# Process
detector = FaceDetector()
faces = detector.detect(image)

if faces:
    landmark_extractor = LandmarkExtractor()
    landmarks_3d, _ = landmark_extractor.extract(image, faces[0]['bbox'])
    
    mesh_builder = MeshBuilder()
    vertices, triangles = mesh_builder.build(landmarks_3d)
    
    # Export
    from core.mesh_exporter import MeshExporter
    exporter = MeshExporter()
    exporter.export_obj("output.obj", vertices, triangles)
```

### Deep Learning Model Inference

```bash
cd DLModel
python demo.py --img_path ../input/face.jpg
```

## Model Details

### DLModel: Deep3DFaceReconstruction
**Source**: Microsoft Research  
**Paper**: [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878)  
**Framework**: TensorFlow 1.12  
**Basis**: Basel Face Model 2009 (BFM09)

**Output**:
- 3DMM shape parameters (80-dim)
- Expression coefficients (64-dim)
- Texture coefficients (80-dim)
- Illumination (spherical harmonics)
- 3D face mesh (with triangulation)

### Post-processing Pipeline
- **Face Detection**: Haar Cascade (OpenCV) + optional Retinaface CNN
- **Mesh Refinement**: Delaunay triangulation on 2D projection
- **Texture Mapping**: U-Net style synthesis network
- **Normal Computation**: Per-vertex surface normals

## Performance

| Task | Time (ms) | Notes |
|------|-----------|-------|
| Deep Learning Inference | 100-200 | Depends on image size |
| Face Detection | 30-50 | Haar Cascade |
| Mesh Generation | 20-40 | Delaunay |
| Texture Synthesis | 50-100 | Optional |
| **Total** | **200-400** | Real-time capable |

## Face Authentication

Built-in facial biometric authentication system:
- LBP (Local Binary Patterns) histogram matching
- Threshold-based verification (~80% accuracy on controlled conditions)
- Registration and verification modes
- Stores feature vectors (profiles/)


## References

1. Microsoft Deep3DFaceReconstruction: https://github.com/microsoft/Deep3DFaceReconstruction
2. Basel Face Model: https://faces.dmi.unibas.ch/
3. OpenCV Cascade Classifiers: https://docs.opencv.org/master/db/d28/tutorial_cascade_classifier.html
4. TensorFlow Mesh Renderer: https://github.com/google/tf_mesh_renderer
