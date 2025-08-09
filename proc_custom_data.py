import numpy as np
import igl
import trimesh
import argparse
import os

def main(args):
    """
    Preprocesses a single 3D mesh file (e.g., .obj, .ply) to compute LSCM UV coordinates
    and saves the result to a .npz file.
    """
    print(f"Loading mesh from {args.in_file}...")
    try:
        mesh = trimesh.load_mesh(args.in_file)
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return

    v = np.array(mesh.vertices)
    f = np.array(mesh.faces)

    print("Calculating LSCM UV coordinates...")
    try:
        v_igl = igl.eigen.MatrixXd(v)
        f_igl = igl.eigen.MatrixXi(f)
        bnd = igl.boundary_loop(f_igl)
        bnd_uv = igl.map_vertices_to_circle(f_igl, bnd)
        uv = igl.lscm(f_igl, bnd, bnd_uv)[1]
    except Exception as e:
        print(f"Error calculating LSCM: {e}")
        return

    if args.out_file is None:
        base_name = os.path.splitext(os.path.basename(args.in_file))[0]
        out_file = f"{base_name}_preprocessed.npz"
    else:
        out_file = args.out_file

    print(f"Saving preprocessed data to {out_file}...")
    np.savez(out_file, v=v, f=f, uv=uv)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess a custom 3D model for pose transfer.")
    parser.add_argument("--in_file", type=str, required=True, help="Path to the input mesh file (e.g., .obj, .ply).")
    parser.add_argument("--out_file", type=str, help="Path to the output .npz file. Defaults to [in_file]_preprocessed.npz.")
    args = parser.parse_args()
    main(args)
