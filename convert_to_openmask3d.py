#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image


# ------------------------- helper data classes -------------------------


@dataclass
class FrameFile:
    path: Path
    timestamp_ns: int  # nanoseconds


@dataclass
class OdomSample:
    timestamp: float       # seconds
    position: np.ndarray   # (3,)
    quat: np.ndarray       # (4,) (qx, qy, qz, qw)


# ------------------------- small math utils ----------------------------


def quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert quaternion (x,y,z,w) to 3x3 rotation matrix."""
    x, y, z, w = qx, qy, qz, qw
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        raise ValueError("Zero-norm quaternion")
    x /= norm
    y /= norm
    z /= norm
    w /= norm

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.array(
        [
            [1.0 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1.0 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1.0 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    return R


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 4x4 homogeneous transform from R(3x3), t(3,)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def parse_timestamp_from_name_ns(path: Path, prefix: str) -> int:
    """
    Extract <timestamp_ns> from filenames like:
      rgb_image_<timestamp_ns>.png
      depth_image_<timestamp_ns>.png

    Returns an integer nanosecond timestamp.
    """
    name = path.name
    stem = name.split(".")[0]
    # Example: rgb_image_1714440084123456789
    try:
        ts_str = stem.split(prefix)[1]
    except IndexError:
        ts_str = stem.split("_")[-1]
    return int(ts_str)


def nearest_index(sorted_ts: np.ndarray, target: float) -> int:
    """Return index in sorted_ts whose value is closest to target."""
    idx = np.searchsorted(sorted_ts, target)
    if idx == 0:
        return 0
    if idx >= len(sorted_ts):
        return len(sorted_ts) - 1
    before = sorted_ts[idx - 1]
    after = sorted_ts[idx]
    if abs(before - target) <= abs(after - target):
        return idx - 1
    else:
        return idx


# --------------------- parsing camera info / tf ------------------------


def parse_camera_info(camera_info_path: Path) -> Tuple[float, float, float, float, int, int]:
    """
    Parse camera_info.txt with fields like:
        width: 1280
        height: 720
        ...
        K: (fx, 0, cx, 0, fy, cy, 0, 0, 1)

    Returns (fx, fy, cx, cy, width, height).
    """
    fx = fy = cx = cy = None
    width = height = None

    float_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

    with camera_info_path.open("r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("width:"):
                width = int(line.split(":", 1)[1].strip())
            elif line.startswith("height:"):
                height = int(line.split(":", 1)[1].strip())
            elif line.startswith("K:"):
                nums = float_re.findall(line)
                vals = list(map(float, nums))
                if len(vals) != 9:
                    raise ValueError(f"Expected 9 values in K, got {len(vals)}: {vals}")
                fx = vals[0]
                cx = vals[2]
                fy = vals[4]
                cy = vals[5]

    if None in (fx, fy, cx, cy, width, height):
        raise ValueError(
            f"Failed to parse camera_info.txt: "
            f"fx={fx}, fy={fy}, cx={cx}, cy={cy}, width={width}, height={height}"
        )

    return fx, fy, cx, cy, width, height


def transform_from_azure(tf_entry: Dict) -> np.ndarray:
    """
    Build 4x4 transform from an azure_tf.json entry with:
      {
        "translation": {"x": ..., "y": ..., "z": ...},
        "rotation": {"x": ..., "y": ..., "z": ..., "w": ...}
      }
    Interpreted as parent_T_child.
    """
    t = tf_entry["translation"]
    r = tf_entry["rotation"]
    t_vec = np.array([t["x"], t["y"], t["z"]], dtype=np.float64)
    R = quat_to_rotmat(r["x"], r["y"], r["z"], r["w"])
    return make_transform(R, t_vec)


# --------------------- load odometry samples ---------------------------


def load_odom_csv(csv_path: Path) -> List[OdomSample]:
    """
    Load odometry CSV with columns:
      timestamp, x, y, z, qx, qy, qz, qw

    timestamp can be seconds or nanoseconds. If it's > 1e12, treat as ns
    and convert to seconds.
    """
    samples: List[OdomSample] = []

    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            raise ValueError(f"No odometry samples found in {csv_path}")

        first_ts_raw = float(rows[0]["timestamp"])
        is_ns = first_ts_raw > 1e12  # seconds since epoch ~1e9

        for row in rows:
            ts_raw = float(row["timestamp"])
            ts = ts_raw * 1e-9 if is_ns else ts_raw  # store in seconds
            pos = np.array(
                [float(row["x"]), float(row["y"]), float(row["z"])],
                dtype=np.float64,
            )
            quat = np.array(
                [float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])],
                dtype=np.float64,
            )
            samples.append(OdomSample(ts, pos, quat))

    samples.sort(key=lambda s: s.timestamp)
    return samples


def build_T_world_body(odom_sample: OdomSample) -> np.ndarray:
    """
    Build T_world_body from one odometry row, assuming:
      - position is in world frame
      - quat describes rotation body->world
    """
    R = quat_to_rotmat(*odom_sample.quat)
    return make_transform(R, odom_sample.position)


# --------------------- main conversion logic ---------------------------


def convert_scene(
    input_root: Path,
    output_root: Path,
    azure_tf_path: Optional[Path] = None,
    use_mesh_instead_of_pointcloud: bool = False,
    max_time_diff_rgb_depth: float = 0.05,
    max_time_diff_rgb_odom: float = 0.1,
    frame_stride: int = 1,
) -> None:
    """
    Convert one Azure-style scene into OpenMask3D format.

    Input layout (input_root):
        compressed_point_cloud.ply
        compressed_mesh.ply
        depth/
          camera_info.txt
          depth_image_<timestamp_ns>.png
        rgb/
          rgb_image_<timestamp_ns>.png or .jpg
        odom/
          <something>.csv
        azure_tf.json   (if azure_tf_path is None, expected here)

    Output layout (output_root):
        scene.ply
        color/0.jpg,1.jpg,...
        depth/0.png,1.png,...
        pose/0.txt,1.txt,...
        intrinsic/intrinsic_color.txt

    frame_stride: use every N-th RGB frame (time-ordered) to thin out views.
    """
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")

    input_root = input_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Converting scene from {input_root}")
    print(f"[INFO] Output OpenMask3D scene dir: {output_root}")
    print(f"[INFO] Frame stride: {frame_stride}")

    # 1) scene.ply
    if use_mesh_instead_of_pointcloud:
        src_scene_ply = input_root / "compressed_mesh.ply"
    else:
        src_scene_ply = input_root / "compressed_point_cloud.ply"

    if not src_scene_ply.exists():
        raise FileNotFoundError(f"Scene PLY not found: {src_scene_ply}")
    dst_scene_ply = output_root / "scene.ply"
    print(f"[INFO] Copying scene PLY:\n  {src_scene_ply} -> {dst_scene_ply}")
    shutil.copyfile(src_scene_ply, dst_scene_ply)

    # 2) intrinsics from depth/camera_info.txt
    camera_info_path = input_root / "depth" / "camera_info.txt"
    if not camera_info_path.exists():
        raise FileNotFoundError(f"camera_info.txt not found: {camera_info_path}")
    fx, fy, cx, cy, width, height = parse_camera_info(camera_info_path)
    print(
        f"[INFO] Parsed intrinsics: fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}, "
        f"width={width}, height={height}"
    )

    intrinsic_dir = output_root / "intrinsic"
    intrinsic_dir.mkdir(exist_ok=True)
    intrinsic_path = intrinsic_dir / "intrinsic_color.txt"
    K4 = np.eye(4, dtype=np.float64)
    K4[0, 0] = fx
    K4[1, 1] = fy
    K4[0, 2] = cx
    K4[1, 2] = cy
    np.savetxt(intrinsic_path, K4, fmt="%.8f")
    print(f"[INFO] Wrote intrinsics to {intrinsic_path}")
    print(f"[INFO] When calling Docker, use scene_intrinsic_resolution=\"[{height},{width}]\"")

    # 3) load RGB and depth file lists (nanosecond timestamps)
    rgb_dir = input_root / "rgb"
    depth_dir = input_root / "depth"

    rgb_files: List[FrameFile] = []

    # Accept png/jpg/jpeg
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for p in sorted(rgb_dir.glob(ext)):
            if not p.name.startswith("rgb_image_"):
                continue
            ts_ns = parse_timestamp_from_name_ns(p, prefix="rgb_image_")
            rgb_files.append(FrameFile(p, ts_ns))

    if not rgb_files:
        # fallback: any file starting with rgb_image_
        for p in sorted(rgb_dir.glob("rgb_image_*")):
            if p.is_file():
                ts_ns = parse_timestamp_from_name_ns(p, prefix="rgb_image_")
                rgb_files.append(FrameFile(p, ts_ns))

    if not rgb_files:
        raise FileNotFoundError(f"No rgb_image_* files found in {rgb_dir}")

    depth_files: List[FrameFile] = []
    for p in sorted(depth_dir.glob("depth_image_*.png")):
        ts_ns = parse_timestamp_from_name_ns(p, prefix="depth_image_")
        depth_files.append(FrameFile(p, ts_ns))

    if not depth_files:
        raise FileNotFoundError(f"No depth_image_*.png files found in {depth_dir}")

    # sort by timestamp_ns
    rgb_files.sort(key=lambda f: f.timestamp_ns)
    depth_files.sort(key=lambda f: f.timestamp_ns)

    print(f"[INFO] Found {len(rgb_files)} RGB frames, {len(depth_files)} depth frames")
    if frame_stride > 1:
        approx = (len(rgb_files) + frame_stride - 1) // frame_stride
        print(f"[INFO] With frame_stride={frame_stride}, expect ~{approx} frames after thinning.")

    # Build depth timestamps in seconds for nearest-match
    depth_ts_sec = np.array([f.timestamp_ns for f in depth_files], dtype=np.float64) * 1e-9

    # 4) load odometry + azure_tf
    odom_dir = input_root / "odom"
    csv_candidates = list(odom_dir.glob("*.csv"))
    if len(csv_candidates) != 1:
        raise RuntimeError(f"Expected exactly one CSV in {odom_dir}, got {csv_candidates}")
    odom_csv_path = csv_candidates[0]
    odom_samples = load_odom_csv(odom_csv_path)
    odom_ts = np.array([s.timestamp for s in odom_samples], dtype=np.float64)
    print(f"[INFO] Loaded {len(odom_samples)} odometry samples from {odom_csv_path}")

    if azure_tf_path is None:
        azure_tf_path = input_root / "azure_tf.json"
    azure_tf_path = azure_tf_path.resolve()
    if not azure_tf_path.exists():
        raise FileNotFoundError(f"azure_tf.json not found: {azure_tf_path}")
    with azure_tf_path.open("r") as f:
        tf_data = json.load(f)

    # These keys are from your example azure_tf.json
    T_base_depth = transform_from_azure(tf_data["base_T_depth"])
    T_base_body = transform_from_azure(tf_data["base_T_body"])
    T_depth_rgb = transform_from_azure(tf_data["depth_T_rgb"])

    T_body_base = np.linalg.inv(T_base_body)

    # 5) create output subdirs
    color_dir = output_root / "color"
    depth_out_dir = output_root / "depth"
    pose_dir = output_root / "pose"

    color_dir.mkdir(exist_ok=True)
    depth_out_dir.mkdir(exist_ok=True)
    pose_dir.mkdir(exist_ok=True)

    # 6) iterate frames with stride, match depth (nearest) + odom (nearest), write everything
    n_written = 0

    for rgb_idx, rgb_frame in enumerate(rgb_files):
        # frame thinning
        if rgb_idx % frame_stride != 0:
            continue

        ts_ns = rgb_frame.timestamp_ns
        ts_sec = ts_ns * 1e-9

        # depth: nearest in time
        d_idx = nearest_index(depth_ts_sec, ts_sec)
        d_frame = depth_files[d_idx]
        dt_depth = abs(depth_ts_sec[d_idx] - ts_sec)
        if dt_depth > max_time_diff_rgb_depth:
            print(
                f"[WARN] RGB {rgb_frame.path.name} and nearest depth "
                f"differ by {dt_depth:.6f}s > {max_time_diff_rgb_depth:.3f}s — skipping"
            )
            continue

        # odom: nearest in time (seconds)
        o_idx = nearest_index(odom_ts, ts_sec)
        o_sample = odom_samples[o_idx]
        dt_odom = abs(o_sample.timestamp - ts_sec)
        if dt_odom > max_time_diff_rgb_odom:
            print(
                f"[WARN] RGB {rgb_frame.path.name} and nearest odom differ by "
                f"{dt_odom:.3f}s > {max_time_diff_rgb_odom:.3f}s — skipping"
            )
            continue

        # compute T_world_body
        T_world_body = build_T_world_body(o_sample)
        T_world_base = T_world_body @ T_body_base
        T_world_depth = T_world_base @ T_base_depth
        T_world_rgb = T_world_depth @ T_depth_rgb

        # save pose/i.txt
        pose_path = pose_dir / f"{n_written}.txt"
        np.savetxt(pose_path, T_world_rgb, fmt="%.8f")

        # save color/i.jpg
        img_rgb = Image.open(rgb_frame.path).convert("RGB")
        rgb_out_path = color_dir / f"{n_written}.jpg"
        img_rgb.save(rgb_out_path, format="JPEG", quality=95)

        # save depth/i.png (uint16, mm)
        depth_img = Image.open(d_frame.path)
        depth_arr = np.array(depth_img)
        if depth_arr.dtype == np.float32 or depth_arr.dtype == np.float64:
            # interpret as meters → convert to mm
            depth_mm = (depth_arr * 1000.0).astype(np.uint16)
        elif np.issubdtype(depth_arr.dtype, np.integer):
            # assume already mm, but clip to uint16
            depth_mm = np.clip(depth_arr, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        else:
            raise ValueError(f"Unsupported depth dtype {depth_arr.dtype} in {d_frame.path}")

        depth_out_path = depth_out_dir / f"{n_written}.png"
        Image.fromarray(depth_mm).save(depth_out_path)

        n_written += 1

    print(f"[INFO] Wrote {n_written} synchronized frames to:")
    print(f"       color/: {color_dir}")
    print(f"       depth/: {depth_out_dir}")
    print(f"       pose/:  {pose_dir}")
    print(f"[INFO] Intrinsics at: {intrinsic_path}")
    print(f"[INFO] scene.ply at:  {dst_scene_ply}")
    print("[INFO] Conversion complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Azure-style scene folder to OpenMask3D single-scene format."
    )
    parser.add_argument(
        "--input-root",
        type=str,
        required=True,
        help=(
            "Path to your scene_name folder "
            "(with compressed_point_cloud.ply, rgb/, depth/, odom/, azure_tf.json)."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help=(
            "Path to output OpenMask3D scene folder "
            "(will contain scene.ply, color/, depth/, pose/, intrinsic/)."
        ),
    )
    parser.add_argument(
        "--azure-tf",
        type=str,
        default=None,
        help="Optional path to azure_tf.json. If not set, uses <input-root>/azure_tf.json.",
    )
    parser.add_argument(
        "--use-mesh-instead-of-pointcloud",
        action="store_true",
        help="If set, uses compressed_mesh.ply instead of compressed_point_cloud.ply as scene.ply.",
    )
    parser.add_argument(
        "--max-time-diff-rgb-depth",
        type=float,
        default=0.05,
        help="Max allowed time difference (seconds) between RGB and depth frames.",
    )
    parser.add_argument(
        "--max-time-diff-rgb-odom",
        type=float,
        default=0.1,
        help="Max allowed time difference (seconds) between RGB frames and odom samples.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Use every N-th RGB frame to thin out the sequence (default: 1 = use all).",
    )

    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    azure_tf_path = Path(args.azure_tf) if args.azure_tf is not None else None

    convert_scene(
        input_root=input_root,
        output_root=output_root,
        azure_tf_path=azure_tf_path,
        use_mesh_instead_of_pointcloud=args.use_mesh_instead_of_pointcloud,
        max_time_diff_rgb_depth=args.max_time_diff_rgb_depth,
        max_time_diff_rgb_odom=args.max_time_diff_rgb_odom,
        frame_stride=args.frame_stride,
    )


if __name__ == "__main__":
    main()

