# OpenMask3D – Docker Inference + CLIP Querying

Minimal wrapper to run **OpenMask3D** inside Docker, convert scenes into the expected format, and perform CLIP‑based text queries over returned masks.

---

## Requirements
- Docker + NVIDIA GPU + nvidia-container-toolkit
- Python 3.9+
- A virtual environment (`venv/`) is recommended

Create venv:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Repository Structure
```
openmask3d-docker-inference/
├── run_server.sh
├── client/
│   ├── run_openmask3d.py
│   ├── docker_communication.py
│   ├── standalone_visualize_masks.py
│   └── __init__.py
├── convert_to_openmask3d.py
├── data/
│   ├── input/<scene_name>/
│   │   ├── scene.ply
│   │   ├── color/*.jpg
│   │   ├── depth/*.png
│   │   ├── intrinsic/intrinsic_color.txt
│   │   └── pose/*.txt
│   └── output/
├── requirements.txt
└── .gitignore
```

---

## 1️⃣ Prepare Input Scene

Your scene folder must follow:

```
scene/
├── scene.ply
├── color/*.jpg
├── depth/*.png
├── intrinsic/intrinsic_color.txt
└── pose/*.txt
```

If your dataset differs, convert it:

```bash
python convert_to_openmask3d.py     --input <raw_scene_dir>     --output data/input/<scene_name>
```

---

## 2️⃣ Start OpenMask3D Docker Server

```bash
./run_server.sh
```

This launches:
```
craiden/openmask:v1.0
```

Server endpoint:
```
http://localhost:5001/openmask/save_and_predict
```

---

## 3️⃣ Run Inference + CLIP Query

```bash
python -m client.run_openmask3d     --input-dir data/input/<scene_name>     --output-root data/output     --query "cabinet"     --visualize
```

This will:
- Zip the scene
- Send it to Docker server
- Receive masks + CLIP features
- Deduplicate features
- Visualize or export segmented point clouds

---

## 4️⃣ Visualize Precomputed Masks (No Server Needed)

```bash
python client/standalone_visualize_masks.py     --pcd data/input/<scene_name>/scene.ply     --features data/output/<scene_name>/clip_features_comp.npy     --masks data/output/<scene_name>/scene_MASKS_comp.npy     --query "drawer handle"     --top-k 5
```

This is useful when you want to test different text prompts without recomputing masks.

---

## Notes
- Use `venv/` to avoid polluting system Python.
- OpenMask3D is memory heavy → consider reducing the number of input frames.
- Everything heavy runs in Docker — your client stays lightweight.
