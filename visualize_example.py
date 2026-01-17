#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import imageio
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def parse_ply_header(file_obj):
    vertex_count = None
    face_count = None
    data_format = None
    while True:
        line = file_obj.readline()
        if not line:
            raise ValueError("PLY header is incomplete.")
        line_str = line.decode("ascii", errors="replace").strip()
        if line_str.startswith("format "):
            data_format = line_str.split()[1]
        elif line_str.startswith("element vertex "):
            vertex_count = int(line_str.split()[-1])
        elif line_str.startswith("element face "):
            face_count = int(line_str.split()[-1])
        elif line_str == "end_header":
            break
    return data_format, vertex_count, face_count


def read_ply_vertices(file_obj, vertex_count):
    dtype = np.dtype(
        [
            ("x", "<f8"),
            ("y", "<f8"),
            ("z", "<f8"),
            ("nx", "<f8"),
            ("ny", "<f8"),
            ("nz", "<f8"),
            ("r", "u1"),
            ("g", "u1"),
            ("b", "u1"),
        ]
    )
    data = np.fromfile(file_obj, dtype=dtype, count=vertex_count)
    points = np.stack([data["x"], data["y"], data["z"]], axis=1)
    colors = np.stack([data["r"], data["g"], data["b"]], axis=1) / 255.0
    return points, colors


def read_ply_points(path):
    with path.open("rb") as f:
        data_format, vertex_count, _ = parse_ply_header(f)
        if data_format != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format: {data_format}")
        if vertex_count is None:
            raise ValueError("PLY header missing vertex count.")
        return read_ply_vertices(f, vertex_count)


def read_ply_mesh_sample(path, max_faces=20000, seed=13):
    with path.open("rb") as f:
        data_format, vertex_count, face_count = parse_ply_header(f)
        if data_format != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format: {data_format}")
        if vertex_count is None or face_count is None:
            raise ValueError("PLY header missing vertex or face count.")

        points, colors = read_ply_vertices(f, vertex_count)

        face_start = f.tell()
        sample_checks = min(face_count, 32)
        counts_ok = True
        for _ in range(sample_checks):
            count_bytes = f.read(1)
            if not count_bytes:
                counts_ok = False
                break
            count = int.from_bytes(count_bytes, "little")
            if count != 3:
                counts_ok = False
                break
            f.seek(4 * count, 1)
        f.seek(face_start)

        if not counts_ok:
            raise ValueError("Mesh faces are not triangle-only; cannot fast-read.")

        face_dtype = np.dtype(
            [("count", "u1"), ("i0", "<u4"), ("i1", "<u4"), ("i2", "<u4")]
        )
        faces_raw = np.fromfile(f, dtype=face_dtype, count=face_count)
        faces = np.stack(
            [faces_raw["i0"], faces_raw["i1"], faces_raw["i2"]], axis=1
        )

    sample_count = min(max_faces, faces.shape[0])
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(faces.shape[0], size=sample_count, replace=False)
    return points, colors, faces[sample_idx]


def select_mid_index(files, key=None):
    if not files:
        raise ValueError("No files found for selection.")
    files = sorted(files, key=key)
    mid = len(files) // 2
    return files[mid]


def extract_index(path):
    stem = path.stem
    parts = stem.split("_")
    return int(parts[-1])


def overlay_autoseg(frame_img, mask_img):
    frame = np.array(frame_img)
    mask = np.array(mask_img)
    h, w, _ = frame.shape
    if mask.shape[0] != h:
        raise ValueError("Mask height does not match frame height.")
    if mask.shape[1] % w != 0:
        raise ValueError("Mask width is not a multiple of frame width.")

    slice_w = w
    num_slices = mask.shape[1] // slice_w
    merged = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(num_slices):
        sl = mask[:, i * slice_w : (i + 1) * slice_w, :]
        merged = np.maximum(merged, sl)

    alpha = 0.6
    mask_alpha = (merged.sum(axis=2) > 0).astype(np.float32) * alpha
    overlay = frame * (1 - mask_alpha[..., None]) + merged * mask_alpha[..., None]
    return overlay.astype(np.uint8), merged


def overlay_monst3r(frame_img, mask_img):
    frame = np.array(frame_img).astype(np.float32)
    mask = np.array(mask_img).astype(np.float32) / 255.0
    color = np.zeros_like(frame)
    color[..., 0] = 255.0
    alpha = 0.7 * mask
    overlay = frame * (1 - alpha[..., None]) + color * alpha[..., None]
    return overlay.astype(np.uint8)


def render_point_cloud(points, colors, out_path):
    points = points - points.mean(axis=0, keepdims=True)
    fig = plt.figure(figsize=(5, 5), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        s=0.7,
        linewidths=0,
    )
    ax.set_axis_off()
    ax.view_init(elev=20, azim=35)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def render_mesh(points, colors, faces, out_path):
    points = points - points.mean(axis=0, keepdims=True)
    triangles = points[faces]
    face_colors = colors[faces].mean(axis=1)

    fig = plt.figure(figsize=(6, 6), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    collection = Poly3DCollection(triangles, facecolors=face_colors, linewidths=0)
    collection.set_edgecolor("none")
    ax.add_collection3d(collection)

    flat = triangles.reshape(-1, 3)
    min_vals = flat.min(axis=0)
    max_vals = flat.max(axis=0)
    center = (min_vals + max_vals) / 2.0
    max_range = (max_vals - min_vals).max() / 2.0
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    ax.set_axis_off()
    ax.view_init(elev=18, azim=35)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def build_autoseg_gif(
    video_dir,
    mask_dir,
    out_path,
    stride=2,
    max_frames=24,
    fps=8,
    scale=1.0,
):
    video_files = sorted(video_dir.glob("*.jpg"))
    frame_count = len(video_files)
    mask_files = sorted(mask_dir.glob("mask_*.png"), key=extract_index)
    if not video_files or not mask_files:
        raise ValueError("Missing video frames or AutoSeg masks for GIF.")

    with imageio.get_writer(out_path, mode="I", fps=fps) as writer:
        written = 0
        for mask_file in mask_files[::stride]:
            if max_frames and written >= max_frames:
                break
            mask_idx = extract_index(mask_file)
            original_idx = (frame_count - 1) - mask_idx
            frame_path = video_dir / f"{original_idx:06d}.jpg"
            if not frame_path.exists():
                continue
            frame_img = Image.open(frame_path).convert("RGB")
            mask_img = Image.open(mask_file).convert("RGB")
            overlay, _ = overlay_autoseg(frame_img, mask_img)
            if scale != 1.0:
                new_size = (
                    int(overlay.shape[1] * scale),
                    int(overlay.shape[0] * scale),
                )
                overlay = np.array(
                    Image.fromarray(overlay).resize(new_size, resample=Image.BILINEAR)
                )
            writer.append_data(overlay)
            written += 1

    return written


def main():
    parser = argparse.ArgumentParser(
        description="Visualize example results for the book demo."
    )
    parser.add_argument(
        "--example_dir",
        type=Path,
        default=Path("example/book"),
        help="Path to the example data folder.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("example/book/results/visualization"),
        help="Output directory for visualization images.",
    )
    parser.add_argument(
        "--mask_idx",
        type=int,
        default=None,
        help="AutoSeg mask index to visualize.",
    )
    parser.add_argument(
        "--monst3r_idx",
        type=int,
        default=None,
        help="MonST3R sample index to visualize.",
    )
    parser.add_argument(
        "--gif_stride",
        type=int,
        default=2,
        help="Stride for sampling frames in the AutoSeg GIF.",
    )
    parser.add_argument(
        "--gif_max_frames",
        type=int,
        default=24,
        help="Maximum frames to include in the AutoSeg GIF.",
    )
    parser.add_argument(
        "--gif_fps",
        type=int,
        default=8,
        help="Frames per second for the AutoSeg GIF.",
    )
    parser.add_argument(
        "--gif_scale",
        type=float,
        default=1.0,
        help="Scale factor for GIF frames.",
    )
    parser.add_argument(
        "--mesh_faces",
        type=int,
        default=20000,
        help="Number of mesh faces to sample for rendering.",
    )
    args = parser.parse_args()

    example_dir = args.example_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    video_dir = example_dir / "video_rgb"
    video_files = sorted(video_dir.glob("*.jpg"))
    if not video_files:
        raise FileNotFoundError(f"No video frames found in {video_dir}")
    frame_count = len(video_files)

    autoseg_dir = (
        example_dir
        / "results"
        / "preprocessing"
        / "video_segment_reverse"
        / "small"
        / "final-output"
    )
    if args.mask_idx is None:
        mask_file = select_mid_index(
            list(autoseg_dir.glob("mask_*.png")), key=extract_index
        )
        mask_idx = extract_index(mask_file)
    else:
        mask_idx = args.mask_idx
        mask_file = autoseg_dir / f"mask_{mask_idx:03d}.png"
    if not mask_file.exists():
        raise FileNotFoundError(f"AutoSeg mask not found: {mask_file}")

    original_idx = (frame_count - 1) - mask_idx
    if original_idx < 0 or original_idx >= frame_count:
        raise ValueError("Computed frame index is out of range.")
    frame_path = video_dir / f"{original_idx:06d}.jpg"

    monst3r_dir = example_dir / "results" / "preprocessing" / "monst3r"
    if args.monst3r_idx is None:
        dyn_mask_file = select_mid_index(
            list(monst3r_dir.glob("dynamic_mask_*.png")), key=extract_index
        )
        monst3r_idx = extract_index(dyn_mask_file)
    else:
        monst3r_idx = args.monst3r_idx
        dyn_mask_file = monst3r_dir / f"dynamic_mask_{monst3r_idx}.png"
    frame_m_file = monst3r_dir / f"frame_{monst3r_idx:04d}.png"
    if not dyn_mask_file.exists() or not frame_m_file.exists():
        raise FileNotFoundError("MonST3R frame or mask is missing.")

    pcd_candidates = [
        example_dir / "results" / "mesh" / "surface_pcd.ply",
        example_dir
        / "results"
        / "prediction"
        / "refinement"
        / "monst3r"
        / "chamfer"
        / "0"
        / "surface_pcd.ply",
    ]
    pcd_path = None
    for candidate in pcd_candidates:
        if candidate.exists():
            pcd_path = candidate
            break
    if pcd_path is None:
        raise FileNotFoundError("No surface_pcd.ply found in example results.")

    mesh_candidates = [
        example_dir / "results" / "mesh" / "surface_mesh.ply",
        example_dir
        / "results"
        / "prediction"
        / "refinement"
        / "monst3r"
        / "chamfer"
        / "0"
        / "surface_mesh.ply",
    ]
    mesh_path = None
    for candidate in mesh_candidates:
        if candidate.exists():
            mesh_path = candidate
            break
    if mesh_path is None:
        raise FileNotFoundError("No surface_mesh.ply found in example results.")

    frame_img = Image.open(frame_path).convert("RGB")
    mask_img = Image.open(mask_file).convert("RGB")
    autoseg_overlay, autoseg_merged = overlay_autoseg(frame_img, mask_img)

    monst3r_frame_img = Image.open(frame_m_file).convert("RGB")
    monst3r_mask_img = Image.open(dyn_mask_file).convert("L")
    monst3r_overlay = overlay_monst3r(monst3r_frame_img, monst3r_mask_img)

    points, colors = read_ply_points(pcd_path)

    autoseg_out = out_dir / f"autoseg_overlay_{mask_idx:03d}.png"
    Image.fromarray(autoseg_overlay).save(autoseg_out)

    monst3r_out = out_dir / f"monst3r_overlay_{monst3r_idx:02d}.png"
    Image.fromarray(monst3r_overlay).save(monst3r_out)

    pcd_out = out_dir / "surface_pcd.png"
    render_point_cloud(points, colors, pcd_out)

    mesh_points, mesh_colors, mesh_faces = read_ply_mesh_sample(
        mesh_path, max_faces=args.mesh_faces
    )
    mesh_out = out_dir / "surface_mesh.png"
    render_mesh(mesh_points, mesh_colors, mesh_faces, mesh_out)

    gif_out = out_dir / "autoseg_overlay.gif"
    gif_frames = build_autoseg_gif(
        video_dir,
        autoseg_dir,
        gif_out,
        stride=args.gif_stride,
        max_frames=args.gif_max_frames,
        fps=args.gif_fps,
        scale=args.gif_scale,
    )

    overview_out = out_dir / "example_overview.png"
    fig = plt.figure(figsize=(12, 9), dpi=160)
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(frame_img)
    ax1.set_title(f"Input RGB (frame {original_idx})")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(autoseg_overlay)
    ax2.set_title(f"AutoSeg mask overlay (idx {mask_idx})")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(monst3r_overlay)
    ax3.set_title(f"MonST3R motion mask (sample {monst3r_idx})")
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[1, 1])
    mesh_img = Image.open(mesh_out).convert("RGB")
    ax4.imshow(mesh_img)
    ax4.set_title(f"Surface mesh (sample {args.mesh_faces} faces)")
    ax4.axis("off")

    fig.tight_layout()
    fig.savefig(overview_out, bbox_inches="tight")
    plt.close(fig)

    print("Saved visualizations:")
    print(f"  {autoseg_out}")
    print(f"  {monst3r_out}")
    print(f"  {pcd_out}")
    print(f"  {mesh_out}")
    print(f"  {gif_out} ({gif_frames} frames)")
    print(f"  {overview_out}")


if __name__ == "__main__":
    main()
