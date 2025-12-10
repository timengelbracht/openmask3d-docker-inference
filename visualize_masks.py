#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import numpy as np
import open3d as o3d
import torch
import clip


DEVICE = "cpu"     # change to "cuda" if desired
MODEL, _ = clip.load("ViT-L/14@336px", device=DEVICE)


def load_features_and_masks(features_path: str, masks_path: str):
    features = np.load(features_path)     # [N_masks, D]
    masks = np.load(masks_path)           # [N_points, N_masks]
    assert masks.shape[1] == features.shape[0], \
        f"Mismatch: masks={masks.shape}, feats={features.shape}"
    return features, masks


def visualize_top_k(
    query: str,
    features: np.ndarray,
    masks: np.ndarray,
    pcd_path: str,
    top_k: int = 5,
):
    """
    Visualize top_k masks for the given CLIP text query.
    """
    query = query.lower()
    text = clip.tokenize([query]).to(DEVICE)

    with torch.no_grad():
        text_feat = MODEL.encode_text(text)

    # Compute similarity: [N_masks]
    cos_sim = torch.nn.functional.cosine_similarity(
        torch.tensor(features, device=DEVICE), text_feat, dim=1
    )

    values, indices = torch.topk(cos_sim, top_k)
    pcd = o3d.io.read_point_cloud(pcd_path)

    for rank in range(top_k):
        feat_idx = indices[rank].item()
        score = values[rank].item()

        print(f"[TOP-{rank}] mask_idx={feat_idx}, similarity={score:.4f}")

        mask = masks[:, feat_idx].astype(bool)
        pcd_in = pcd.select_by_index(np.where(mask)[0])
        pcd_out = pcd.select_by_index(np.where(~mask)[0])

        pcd_in.paint_uniform_color([1, 0, 1])  # magenta
        o3d.visualization.draw_geometries([pcd_in, pcd_out])


def parse_args():
    parser = argparse.ArgumentParser("Visualize precomputed OpenMask3D masks using CLIP.")
    parser.add_argument("--pcd", required=True, help="Path to scene.ply")
    parser.add_argument("--features", required=True, help="clip_features_comp.npy OR clip_features.npy")
    parser.add_argument("--masks", required=True, help="scene_MASKS_comp.npy OR scene_MASKS.npy")
    parser.add_argument("--query", required=True, help='Text query, e.g. "cabinet, shelf"')
    parser.add_argument("--top-k", type=int, default=5, help="Number of masks to visualize")
    return parser.parse_args()


def main():
    args = parse_args()
    features, masks = load_features_and_masks(args.features, args.masks)

    visualize_top_k(
        query=args.query,
        features=features,
        masks=masks,
        pcd_path=args.pcd,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()

