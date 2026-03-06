import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import pyvista as pv
import vtk
import matplotlib.pyplot as plt
import matplotlib.colorbar as cbar

print("### RUNNING FILE:", __file__)

vtk.vtkObject.GlobalWarningDisplayOff()
pv.global_theme.jupyter_backend = "trame"
pv.global_theme.allow_empty_mesh = True

from main import (
    SineDenseLayer,
    FieldDecoder,
    ShapeDecoder,
)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = "12"


# -------------------------
# Clip ranges
# -------------------------
def get_clip_values(direction):
    if direction == "ver":
        return {
            0: (-0.068, 0.473),   # ux
            1: (-0.093, 0.073),   # uy
            2: (-0.003, 0.824),   # uz
            3: (0.0, 232.19),     # vm
            4: (0.0, 1.0),        # occ_prob
        }
    if direction == "hor":
        return {
            0: (-0.421, 0.008),
            1: (-0.024, 0.029),
            2: (-0.388, 0.109),
            3: (0.0, 227.78),
            4: (0.0, 1.0),
        }
    if direction == "dia":
        return {
            0: (-0.079, 0.016),
            1: (-0.057, 0.056),
            2: (-0.006, 0.214),
            3: (0.0, 172.19),
            4: (0.0, 1.0),
        }
    raise ValueError(direction)


# -------------------------
# Basic helpers
# -------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_bbox_grid(n: int, xyz_min, xyz_max):
    """
    EXACT same style as generate_and_visualize_occ.py
    """
    xyz_min = np.asarray(xyz_min, dtype=np.float32)
    xyz_max = np.asarray(xyz_max, dtype=np.float32)

    grid = pv.ImageData()
    grid.dimensions = (n, n, n)
    grid.origin = (float(xyz_min[0]), float(xyz_min[1]), float(xyz_min[2]))
    spacing = (xyz_max - xyz_min) / (n - 1)
    grid.spacing = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
    pts = grid.points.astype(np.float32)
    return grid, pts


def apply_trunk_scaler_to_xyz(pts_xyz, trunk_scaler):
    """
    trunk_scaler is fit on xyzd, so pad d=0 then transform, then keep xyz only.
    """
    N = pts_xyz.shape[0]
    tmp = np.zeros((N, trunk_scaler.n_features_in_), dtype=np.float32)
    tmp[:, :3] = pts_xyz
    tmp_scaled = trunk_scaler.transform(tmp)
    return tmp_scaled[:, :3].astype(np.float32)


def inv_field(field_scaled, output_scaler):
    shp = field_scaled.shape
    flat = field_scaled.reshape(-1, shp[-1])
    inv = output_scaler.inverse_transform(flat)
    return inv.reshape(shp)


def normalize_latent(snapshot):
    if "latent" not in snapshot:
        raise KeyError("Trajectory snapshot must contain 'latent'.")

    latent = np.asarray(snapshot["latent"], dtype=np.float32)
    if latent.ndim == 1:
        latent = latent[None, :]
    elif latent.ndim == 3 and latent.shape[0] == 1:
        latent = latent.reshape(1, -1)
    elif latent.ndim != 2:
        raise ValueError(f"Unexpected latent shape: {latent.shape}")
    return latent


# -------------------------
# case_key -> branch condition
# mlc only: [m, l, cx, cy, cz] = a[idx, 0, 4:9]
# -------------------------
def load_condition_from_case_key_xyzdmlc(xyzdmlc_path, case_key, branch_scaler):
    """
    Assumes xyzdmlc.npz contains:
      - a: (Ncase, Npt, 9) where [:, :, 4:9] == [m, l, cx, cy, cz]
      - c: (Ncase,) case keys

    Returns scaled branch condition of shape (1,5)
    """
    data = np.load(xyzdmlc_path, allow_pickle=True)[case_key]
    cond_raw = data[0][4:9].reshape(1,-1)
    cond_scaled = branch_scaler.transform(cond_raw).astype(np.float32)
    return cond_scaled


# -------------------------
# Trajectory helpers
# -------------------------
def find_snapshot_by_iter(traj, target_iter):
    exact = [x for x in traj if int(x["iter"]) == int(target_iter)]
    if exact:
        return exact[0]

    le = [x for x in traj if int(x["iter"]) <= int(target_iter)]
    if le:
        return max(le, key=lambda x: int(x["iter"]))

    return min(traj, key=lambda x: int(x["iter"]))


def pick_snapshots(traj, requested_iters):
    picked = []
    used = set()

    for t in requested_iters:
        snap = find_snapshot_by_iter(traj, t)
        it = int(snap["iter"])
        if it not in used:
            picked.append(snap)
            used.add(it)

    final_snap = max(traj, key=lambda x: int(x["iter"]))
    final_it = int(final_snap["iter"])
    if final_it not in used:
        picked.append(final_snap)

    return sorted(picked, key=lambda x: int(x["iter"]))


# -------------------------
# Visualization helpers
# -------------------------
def save_global_colorbar(a_min, a_max, label, out_path):
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    cmap = plt.get_cmap("jet")
    norm = plt.Normalize(vmin=a_min, vmax=a_max)
    cb1 = cbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation="horizontal")
    cb1.set_label(label)

    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Colorbar saved: {out_path}")


def get_scalar_name(component):
    return {
        0: "ux",
        1: "uy",
        2: "uz",
        3: "vm",
        4: "occ_prob",
    }[component]


def visualize_trajectory_surfaces(
    surface_infos,
    case_key,
    out_dir,
    direction,
    component=3,
    opacity=1.0,
    cmap="jet",
    use_clip=True,
    output_name=""
):
    os.makedirs(out_dir, exist_ok=True)

    try:
        pv.start_xvfb()
    except Exception:
        pass

    num_cols = len(surface_infos)
    per_subplot_pixel = 500
    window_width = num_cols * per_subplot_pixel
    window_height = per_subplot_pixel

    plotter = pv.Plotter(
        shape=(1, num_cols),
        window_size=(window_width, window_height),
        off_screen=True,
    )

    scalar_name = get_scalar_name(component)
    clim = None
    if use_clip:
        clim = get_clip_values(direction)[component]

    for j, info in enumerate(surface_infos):
        plotter.subplot(0, j)

        surf = pv.read(info["surf_path"])
        if surf.n_points == 0:
            plotter.add_text(f"EMPTY\niter={info['iter']}", font_size=10)
            continue

        if scalar_name not in surf.point_data:
            plotter.add_text(
                f"Missing '{scalar_name}'\nHave: {list(surf.point_data.keys())}",
                font_size=9,
            )
            plotter.add_mesh(surf, color="lightgray", opacity=opacity)
            continue

        plotter.add_mesh(
            surf,
            scalars=scalar_name,
            cmap=cmap,
            clim=clim,
            show_scalar_bar=(j == num_cols - 1),
            opacity=opacity,
            smooth_shading=True
        )

        txt = f"iter={info['iter']}"
        stats = info.get("stats", {})
        if "vm_max_hard_inside" in stats:
            txt += f"\nvm_max={stats['vm_max_hard_inside']:.3f}"
        if "vol_frac_soft" in stats:
            txt += f"\nvol={stats['vol_frac_soft']:.3f}"

        plotter.add_text(txt, font_size=10, position="upper_left", shadow=True)
        plotter.add_axes()
        plotter.camera_position = "iso"

    save_path = os.path.join(out_dir, f"0_traj_{case_key}_{scalar_name}_{output_name}.png")
    plotter.render()
    plotter.screenshot(save_path)
    plotter.close()

    print(f"[OK] Visualization saved: {save_path}")
    return save_path


# -------------------------
# Rebuild required modules
# -------------------------
def rebuild_modules(experiment_dir, latent_dim, branch_dim, device):
    """
    Rebuild only the modules needed for rendering:
      trunk_encoding, field_decoder, shape_decoder
    """
    state = torch.load(os.path.join(experiment_dir, "model_final.pth"), map_location=device)

    act = nn.SiLU()

    trunk_encoding = nn.Sequential(
        SineDenseLayer(3, latent_dim, w0=10.0, is_first=True),
        SineDenseLayer(latent_dim, latent_dim * 2, w0=10.0),
        SineDenseLayer(latent_dim * 2, latent_dim, w0=10.0),
    ).to(device).eval()

    field_decoder = FieldDecoder(
        branch_condition_input_dim=branch_dim,
        latent_dim=latent_dim,
        branch_hidden_dim=latent_dim,
        trunk_encoding_output_dim=latent_dim,
        trunk_hidden_dim=latent_dim,
        fc_hidden_dim=latent_dim,
        num_output_components=4,
        activation=act,
    ).to(device).eval()

    shape_decoder = ShapeDecoder(
        latent_dim=latent_dim,
        trunk_encoding_output_dim=latent_dim,
        trunk_hidden_dim=latent_dim,
        fc_hidden_dim=latent_dim,
        activation=act,
    ).to(device).eval()

    trunk_sd = {
        k.replace("trunk_encoding.", ""): v
        for k, v in state.items() if k.startswith("trunk_encoding.")
    }
    field_sd = {
        k.replace("field_decoder.", ""): v
        for k, v in state.items() if k.startswith("field_decoder.")
    }
    shape_sd = {
        k.replace("shape_decoder.", ""): v
        for k, v in state.items() if k.startswith("shape_decoder.")
    }

    trunk_encoding.load_state_dict(trunk_sd, strict=True)
    field_decoder.load_state_dict(field_sd, strict=True)
    shape_decoder.load_state_dict(shape_sd, strict=True)

    return trunk_encoding, field_decoder, shape_decoder


# -------------------------
# Render one snapshot from latent
# This is the key part:
# latent -> decode again on the SAME bbox pts ordering
# -------------------------
def render_snapshot_from_latent(
    snapshot,
    grid_template,
    trunk_enc,
    branch_cond_t,
    field_decoder,
    shape_decoder,
    output_scaler,
    out_dir,
    case_key,
    occ_thr=0.5,
    device="cuda",
):
    latent_np = normalize_latent(snapshot)
    latent_t = torch.from_numpy(latent_np).to(device)

    with torch.no_grad():
        field_scaled = field_decoder(branch_cond_t, latent_t, trunk_enc)   # (1, Ng, 4)
        occ_logit = shape_decoder(latent_t, trunk_enc).squeeze(-1)         # (1, Ng)

        field_scaled_np = field_scaled.squeeze(0).cpu().numpy()
        occ_prob_np = sigmoid(occ_logit.squeeze(0).cpu().numpy()).astype(np.float32)

    field_phys_np = inv_field(field_scaled_np[None, :, :], output_scaler).squeeze(0)

    grid_i = grid_template.copy(deep=True)
    grid_i.point_data["occ_prob"] = occ_prob_np
    grid_i.point_data["ux"] = field_phys_np[:, 0].astype(np.float32)
    grid_i.point_data["uy"] = field_phys_np[:, 1].astype(np.float32)
    grid_i.point_data["uz"] = field_phys_np[:, 2].astype(np.float32)
    grid_i.point_data["vm"] = field_phys_np[:, 3].astype(np.float32)

    it = int(snapshot["iter"])
    base = f"{case_key}_iter_{it:04d}"

    grid_path = os.path.join(out_dir, f"{base}_grid.vtk")
    grid_i.save(grid_path)

    inside = grid_i.threshold(value=float(occ_thr), scalars="occ_prob")
    surf = inside.extract_surface()

    surf_path = os.path.join(out_dir, f"{base}_surface.vtk")
    surf.save(surf_path)

    stats = snapshot.get("stats", {})
    print(
        f"[OK] iter={it} saved | "
        f"occ min={occ_prob_np.min():.4f}, max={occ_prob_np.max():.4f} | "
        f"surface points={surf.n_points}"
    )

    return {
        "iter": it,
        "grid_path": grid_path,
        "surf_path": surf_path,
        "stats": stats,
    }


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--trajectory_pkl", type=str, required=True)
    ap.add_argument("--xyz_npz", type=str, required=True,
                    help="inverse result npz containing xyz_phys")
    ap.add_argument("--xyzdmlc_path", type=str, required=True,
                    help="path to xyzdmlc.npz")
    ap.add_argument("--experiment_dir", type=str, required=True)

    ap.add_argument("--case_key", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="trajectory_render")
    ap.add_argument("--grid_n", type=int, required=True)
    ap.add_argument("--occ_thr", type=float, default=0.5)

    ap.add_argument("--latent_dim", type=int, default=128)

    ap.add_argument("--viz_component", type=int, default=3,
                    help="0 ux, 1 uy, 2 uz, 3 vm, 4 occ_prob")
    ap.add_argument("--viz_opacity", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda:0")

    ap.add_argument("--iters", type=int, nargs="*", default=[0, 50, 100, 200, 500],
                    help="requested iters; final is always added")
    ap.add_argument("--out_name", type=str, default="")
    

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("[INFO] device =", device)

    direction = args.case_key.split("_")[0]

    with open(args.trajectory_pkl, "rb") as f:
        traj = pickle.load(f)

    xyz_data = np.load(args.xyz_npz, allow_pickle=True)
    if "xyz_phys" not in xyz_data.files:
        raise KeyError(f"{args.xyz_npz} must contain 'xyz_phys'. Found: {xyz_data.files}")
    xyz_phys = np.asarray(xyz_data["xyz_phys"], dtype=np.float32)
    if xyz_phys.ndim != 2 or xyz_phys.shape[1] != 3:
        raise ValueError(f"xyz_phys must be (N,3). Got {xyz_phys.shape}")

    branch_scaler = pickle.load(open(os.path.join(args.experiment_dir, "branch_scaler.pkl"), "rb"))
    trunk_scaler = pickle.load(open(os.path.join(args.experiment_dir, "trunk_scaler.pkl"), "rb"))
    output_scaler = pickle.load(open(os.path.join(args.experiment_dir, "output_scaler.pkl"), "rb"))

    # case_key -> scaled condition
    cond_scaled = load_condition_from_case_key_xyzdmlc(
        args.xyzdmlc_path,
        args.case_key,
        branch_scaler,
    )
    branch_cond_t = torch.from_numpy(cond_scaled).to(device)

    # rebuild modules
    trunk_encoding, field_decoder, shape_decoder = rebuild_modules(
        experiment_dir=args.experiment_dir,
        latent_dim=args.latent_dim,
        branch_dim=cond_scaled.shape[1],
        device=device,
    )

    # bbox from xyz_npz, but ordering comes from freshly created ImageData grid
    xyz_min = xyz_phys.min(axis=0)
    xyz_max = xyz_phys.max(axis=0)

    grid_template, pts = make_bbox_grid(args.grid_n, xyz_min, xyz_max)
    pts_scaled_xyz = apply_trunk_scaler_to_xyz(pts, trunk_scaler)
    pts_scaled_xyz_t = torch.from_numpy(pts_scaled_xyz).to(device)

    # precompute trunk encoding once on the SAME pts ordering
    with torch.no_grad():
        trunk_enc = trunk_encoding(pts_scaled_xyz_t).unsqueeze(0)   # (1, Ng, H)

    chosen = pick_snapshots(traj, args.iters)
    print("[INFO] chosen iterations:", [int(x["iter"]) for x in chosen])

    surface_infos = []
    for snap in chosen:
        info = render_snapshot_from_latent(
            snapshot=snap,
            grid_template=grid_template,
            trunk_enc=trunk_enc,
            branch_cond_t=branch_cond_t,
            field_decoder=field_decoder,
            shape_decoder=shape_decoder,
            output_scaler=output_scaler,
            out_dir=args.out_dir,
            case_key=args.case_key,
            occ_thr=args.occ_thr,
            device=device,
        )
        surface_infos.append(info)

    fig_path = visualize_trajectory_surfaces(
        surface_infos=surface_infos,
        case_key=args.case_key,
        out_dir=args.out_dir,
        direction=direction,
        component=args.viz_component,
        opacity=args.viz_opacity,
        cmap="jet",
        use_clip=True,
        output_name=args.out_name
    )

    meta_path = os.path.join(args.out_dir, f"{args.case_key}_trajectory_render_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(surface_infos, f)

    print(f"[OK] meta saved: {meta_path}")
    print(f"[OK] figure saved: {fig_path}")


if __name__ == "__main__":
    main()