# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile

from typing import Tuple

import numpy as np
import torch
import clip
import open3d as o3d

from .docker_communication import send_request

# CLIP model (client side, for text → mask scoring)
DEVICE = "cpu"  # change to "cuda" if you want
MODEL, PREPROCESS = clip.load("ViT-L/14@336px", device=DEVICE)


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def zip_point_cloud(input_dir: str, tmp_dir: str) -> str:
    """
    Zip a whole directory (containing scene.ply and other files)
    into a single .zip file in tmp_dir, and return its path.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    scene_name = os.path.basename(os.path.normpath(input_dir))
    zip_path = os.path.join(tmp_dir, f"{scene_name}.zip")

    if os.path.exists(zip_path):
        os.remove(zip_path)

    with zipfile.ZipFile(zip_path, "w") as zipf:
        for foldername, subfolders, filenames in os.walk(input_dir):
            for filename in filenames:
                if filename.endswith(".zip"):
                    continue
                file_path = os.path.join(foldername, filename)
                arcname = os.path.relpath(file_path, input_dir)
                zipf.write(file_path, arcname)
    return zip_path


def run_openmask_server_inference(
    input_dir: str,
    output_dir: str,
    server_url: str,
    scene_name: str,
    intrinsic_resolution: str,
    timeout: int = 900,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zip point cloud folder, send to OpenMask3D server, save returned features & masks.
    Returns (features, masks) as numpy arrays.
    """
    os.makedirs(output_dir, exist_ok=True)

    tmp_dir = os.path.join(output_dir, "_tmp")
    zip_path = zip_point_cloud(input_dir, tmp_dir)

    params = {
        "name": ("str", scene_name),
        "overwrite": ("bool", True),
        "scene_intrinsic_resolution": ("str", intrinsic_resolution),
    }

    # send_request will unzip server response into output_dir/scene_name_raw
    server_save_path = os.path.join(output_dir, f"{scene_name}_raw")
    contents = send_request(
        server_address=server_url,
        paths_dict={"scene": zip_path},
        params=params,
        timeout=timeout,
        save_path=server_save_path,
    )

    # We expect keys "clip_features" and "scene_MASKS"
    features = contents["clip_features"]
    masks = contents["scene_MASKS"]

    # save original
    np.save(os.path.join(output_dir, "clip_features.npy"), features)
    np.save(os.path.join(output_dir, "scene_MASKS.npy"), masks)

    # make unique (same logic as your code)
    features, feat_idx = np.unique(features, axis=0, return_index=True)
    masks = masks[:, feat_idx]
    masks, mask_idx = np.unique(masks, axis=1, return_index=True)
    features = features[mask_idx]

    np.save(os.path.join(output_dir, "clip_features_comp.npy"), features)
    np.save(os.path.join(output_dir, "scene_MASKS_comp.npy"), masks)

    # clean tmp
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return features, masks


def get_mask_points_for_query(
    query: str,
    features: np.ndarray,
    masks: np.ndarray,
    pcd_path: str,
    idx: int = 0,
    visualize: bool = False,
) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """
    Given text query and (features, masks) from OpenMask3D,
    returns (pcd_in, pcd_out) for the idx-th best match.
    """
    query = query.lower()
    text = clip.tokenize([query]).to(DEVICE)

    with torch.no_grad():
        text_features = MODEL.encode_text(text)

    cos_sim = torch.nn.functional.cosine_similarity(
        torch.tensor(features), text_features, dim=1
    )
    values, indices = torch.topk(cos_sim, idx + 1)
    most_sim_feat_idx = indices[-1].item()
    print(f"Best feature idx={most_sim_feat_idx}, similarity={values[-1].item():.4f}")

    mask = masks[:, most_sim_feat_idx].astype(bool)

    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd_in = pcd.select_by_index(np.where(mask)[0])
    pcd_out = pcd.select_by_index(np.where(~mask)[0])

    if visualize:
        pcd_in.paint_uniform_color([1, 0, 1])
        o3d.visualization.draw_geometries([pcd_in, pcd_out])

    return pcd_in, pcd_out

def visualize_top_k_masks_for_query(
    query: str,
    features: np.ndarray,
    masks: np.ndarray,
    pcd_path: str,
    top_k: int = 5,
) -> None:
    """
    Visualize the top_k best matching masks for a text query.

    For each rank r (0..top_k-1), opens an Open3D window with:
      - pcd_in: points inside the r-th best mask (magenta)
      - pcd_out: remaining points
    """
    top_k = max(1, min(top_k, features.shape[0]))

    query = query.lower()
    text = clip.tokenize([query]).to(DEVICE)

    with torch.no_grad():
        text_features = MODEL.encode_text(text)

    cos_sim = torch.nn.functional.cosine_similarity(
        torch.tensor(features, device=DEVICE),
        text_features,
        dim=1,
    )

    values, indices = torch.topk(cos_sim, top_k)

    # Load point cloud once
    pcd = o3d.io.read_point_cloud(pcd_path)

    for rank in range(top_k):
        feat_idx = indices[rank].item()
        score = values[rank].item()
        print(f"[TOP-{rank}] feat_idx={feat_idx}, similarity={score:.4f}")

        mask = masks[:, feat_idx].astype(bool)
        pcd_in = pcd.select_by_index(np.where(mask)[0])
        pcd_out = pcd.select_by_index(np.where(~mask)[0])

        pcd_in.paint_uniform_color([1, 0, 1])  # magenta
        o3d.visualization.draw_geometries([pcd_in, pcd_out])


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Client for running OpenMask3D via Docker + CLIP text querying."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/input/scene",
        help="Folder containing point cloud files (e.g. scene.ply, etc.)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/output",
        help="Root folder where outputs are stored.",
    )
    parser.add_argument(
        "--scene-name",
        type=str,
        default=None,
        help="Name of the scene (default: basename of input-dir).",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:5001/openmask/save_and_predict",
        help="OpenMask3D server URL.",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="[1440,1920]",
        help='Scene intrinsic resolution string, e.g. "[1440,1920]".',
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Request timeout (seconds).",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help='Optional text query, e.g. "cabinet, shelf".',
    )
    parser.add_argument(
        "--query-idx",
        type=int,
        default=0,
        help="If multiple matches, which index to pick (0 = best).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="If set, show Open3D visualization of in/out clouds for query.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="If set, visualize the top-K masks for the query instead of a single idx.",
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scene_name = (
        args.scene_name if args.scene_name is not None else os.path.basename(
            os.path.normpath(args.input_dir)
        )
    )
    scene_out_dir = os.path.join(args.output_root, scene_name)
    os.makedirs(scene_out_dir, exist_ok=True)

    print(f"[INFO] Running OpenMask3D for scene '{scene_name}'")
    print(f"[INFO] Input dir:  {args.input_dir}")
    print(f"[INFO] Output dir: {scene_out_dir}")
    print(f"[INFO] Server URL: {args.server_url}")

    # 1) Call server, get features + masks, save them
    features, masks = run_openmask_server_inference(
        input_dir=args.input_dir,
        output_dir=scene_out_dir,
        server_url=args.server_url,
        scene_name=scene_name,
        intrinsic_resolution=args.resolution,
        timeout=args.timeout,
    )

    print("[INFO] Saved:")
    print(f"  {os.path.join(scene_out_dir, 'clip_features.npy')}")
    print(f"  {os.path.join(scene_out_dir, 'scene_MASKS.npy')}")
    print(f"  {os.path.join(scene_out_dir, 'clip_features_comp.npy')}")
    print(f"  {os.path.join(scene_out_dir, 'scene_MASKS_comp.npy')}")

    # 2) Optional text query + segmentation / visualization
    if args.query is not None:
        print(f"[INFO] Running CLIP text query: '{args.query}'")
        pcd_path = os.path.join(args.input_dir, "scene.ply")  # adapt if needed

        if args.top_k is not None and args.top_k > 1:
            # Show the first N best masks one after another
            visualize_top_k_masks_for_query(
                query=args.query,
                features=features,
                masks=masks,
                pcd_path=pcd_path,
                top_k=args.top_k,
            )
        else:
            # Original single-mask behavior
            pcd_in, pcd_out = get_mask_points_for_query(
                query=args.query,
                features=features,
                masks=masks,
                pcd_path=pcd_path,
                idx=args.query_idx,
                visualize=args.visualize,
            )

            # save segmented point clouds
            safe_query = args.query.replace(" ", "_").replace(",", "_")
            in_path = os.path.join(scene_out_dir, f"{safe_query}_in.ply")
            out_path = os.path.join(scene_out_dir, f"{safe_query}_out.ply")
            o3d.io.write_point_cloud(in_path, pcd_in)
            o3d.io.write_point_cloud(out_path, pcd_out)

            print("[INFO] Saved segmented point clouds:")
            print(f"  IN:  {in_path}")
            print(f"  OUT: {out_path}")

if __name__ == "__main__":
    main()
