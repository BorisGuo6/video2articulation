#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


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


def read_ply_faces(file_obj, face_count):
    face_start = file_obj.tell()
    sample_checks = min(face_count, 32)
    counts_ok = True
    for _ in range(sample_checks):
        count_bytes = file_obj.read(1)
        if not count_bytes:
            counts_ok = False
            break
        count = int.from_bytes(count_bytes, "little")
        if count != 3:
            counts_ok = False
            break
        file_obj.seek(4 * count, 1)
    file_obj.seek(face_start)

    if not counts_ok:
        raise ValueError("Mesh faces are not triangle-only; cannot fast-read.")

    face_dtype = np.dtype(
        [("count", "u1"), ("i0", "<u4"), ("i1", "<u4"), ("i2", "<u4")]
    )
    faces_raw = np.fromfile(file_obj, dtype=face_dtype, count=face_count)
    faces = np.stack([faces_raw["i0"], faces_raw["i1"], faces_raw["i2"]], axis=1)
    return faces


def sample_faces(faces, max_faces, seed):
    if max_faces is None or max_faces >= faces.shape[0]:
        return faces
    rng = np.random.default_rng(seed)
    idx = rng.choice(faces.shape[0], size=max_faces, replace=False)
    return faces[idx]


def remap_mesh(points, colors, faces):
    unique_idx, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    new_faces = inverse.reshape(-1, 3)
    new_points = points[unique_idx]
    new_colors = colors[unique_idx]
    return new_points, new_colors, new_faces


def write_obj(path, points, colors, faces):
    with path.open("w", encoding="ascii") as f:
        for (x, y, z), (r, g, b) in zip(points, colors):
            f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")
        for i0, i1, i2 in faces:
            f.write(f"f {i0 + 1} {i1 + 1} {i2 + 1}\n")


def load_joint_params(base_dir, joint_type):
    jdir = base_dir / joint_type
    pos = np.load(jdir / "joint_pos.npy").astype(float)
    axis = np.load(jdir / "joint_axis.npy").astype(float)
    vals = np.load(jdir / "joint_value.npy").astype(float)
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0:
        raise ValueError("Joint axis has zero length.")
    axis = axis / axis_norm
    lower = float(vals.min())
    upper = float(vals.max())
    return pos, axis, lower, upper


def write_urdf(
    path, mesh_filename, joint_type, joint_pos, joint_axis, lower, upper
):
    effort = 1.0
    velocity = 1.0
    urdf = f"""<robot name="book_predicted_{joint_type}">
  <link name="base_link"/>
  <link name="moving_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_filename}" scale="1 1 1"/>
      </geometry>
    </visual>
  </link>
  <joint name="predicted_joint" type="{joint_type}">
    <parent link="base_link"/>
    <child link="moving_link"/>
    <origin xyz="{joint_pos[0]:.6f} {joint_pos[1]:.6f} {joint_pos[2]:.6f}" rpy="0 0 0"/>
    <axis xyz="{joint_axis[0]:.6f} {joint_axis[1]:.6f} {joint_axis[2]:.6f}"/>
    <limit lower="{lower:.6f}" upper="{upper:.6f}" effort="{effort:.6f}" velocity="{velocity:.6f}"/>
  </joint>
</robot>
"""
    path.write_text(urdf, encoding="ascii")


def main():
    parser = argparse.ArgumentParser(
        description="Export predicted results to OBJ + URDF for viewing."
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
        help="Output directory for URDF and mesh.",
    )
    parser.add_argument(
        "--mesh_faces",
        type=int,
        default=100000,
        help="Number of mesh faces to sample for OBJ export.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for face sampling.",
    )
    args = parser.parse_args()

    example_dir = args.example_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

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

    pred_dir = (
        example_dir / "results" / "prediction" / "coarse_prediction" / "monst3r" / "0"
    )
    if not pred_dir.exists():
        raise FileNotFoundError("Prediction directory not found.")

    with mesh_path.open("rb") as f:
        data_format, vertex_count, face_count = parse_ply_header(f)
        if data_format != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format: {data_format}")
        if vertex_count is None or face_count is None:
            raise ValueError("PLY header missing vertex or face count.")

        points, colors = read_ply_vertices(f, vertex_count)
        faces = read_ply_faces(f, face_count)

    faces = sample_faces(faces, args.mesh_faces, args.seed)
    points, colors, faces = remap_mesh(points, colors, faces)

    obj_path = out_dir / "book_mesh.obj"
    write_obj(obj_path, points, colors, faces)

    for joint_type in ["revolute", "prismatic"]:
        if not (pred_dir / joint_type).exists():
            continue
        pos, axis, lower, upper = load_joint_params(pred_dir, joint_type)
        urdf_path = out_dir / f"book_predicted_{joint_type}.urdf"
        write_urdf(urdf_path, obj_path.name, joint_type, pos, axis, lower, upper)

    print("Exported files:")
    print(f"  {obj_path}")
    for joint_type in ["revolute", "prismatic"]:
        urdf_path = out_dir / f"book_predicted_{joint_type}.urdf"
        if urdf_path.exists():
            print(f"  {urdf_path}")


if __name__ == "__main__":
    main()
