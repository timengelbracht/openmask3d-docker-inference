# OpenMask3D – Docker Inference + CLIP Querying

Minimal wrapper for **OpenMask3D**.  
The server runs inside Docker; the client sends a zipped scene folder and optionally performs CLIP-based mask ranking.

## Requirements
- Docker
- NVIDIA GPU + nvidia-container-toolkit
- Python 3.9+

## Structure
```
openmask3d-docker-inference/
├── run_server.sh
├── client/
│   ├── run_openmask3d.py
│   └── standalone_visualize_masks.py
└── data/
    ├── input/scene/
    │   ├── scene.ply
    │   ├── color/*.jpg
    │   ├── depth/*.png
    │   └── intrinsic/intrinsic_color.txt
    └── output/
```

## Usage

### 1️⃣ Start the server
```bash
./run_server.sh
```

### 2️⃣ Run inference + CLIP query
```bash
python -m client.run_openmask3d     --input-dir data/input/scene     --output-root data/output     --query "cabinet"
```

### 3️⃣ Visualize masks without recomputing
```bash
python client/standalone_visualize_masks.py     --pcd data/input/scene/scene.ply     --features data/output/scene/clip_features_comp.npy     --masks data/output/scene/scene_MASKS_comp.npy     --query "cabinet"     --top-k 5
```
