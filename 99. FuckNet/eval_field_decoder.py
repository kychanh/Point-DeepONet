import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import pyvista as pv
import vtk

vtk.vtkObject.GlobalWarningDisplayOff()
pv.global_theme.allow_empty_mesh = True


# =========================================================
# import from main.py
# =========================================================
from main import (
    set_random_seed,
    get_device,
    Encoder,
    get_clipping_ranges_for_direction,
    calculate_r2,
)


# =========================================================
# component names
# =========================================================
COMPONENT_NAMES = {
    0: "ux",
    1: "uy",
    2: "uz",
    3: "vm",
}


# =========================================================
# generate_and_visualize_occ.py style model blocks
# =========================================================
class SineDenseLayer(nn.Module):
    def __init__(self, in_features, out_features, w0=1.0, is_first=False):
        super().__init__()
        self.in_features = in_features
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights(is_first)

    def init_weights(self, is_first: bool):
        if is_first:
            nn.init.uniform_(self.linear.weight, -1 / self.in_features, 1 / self.in_features)
        else:
            nn.init.uniform_(
                self.linear.weight,
                -np.sqrt(6 / self.in_features) / self.w0,
                np.sqrt(6 / self.in_features) / self.w0,
            )

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))


class FieldDecoder(nn.Module):
    def __init__(self, branch_condition_input_dim, latent_dim,
                 trunk_hidden_dim, fc_hidden_dim,
                 num_output_components, activation):
        super().__init__()
        self.activation = activation
        self.tanh = nn.Tanh()
        self.b = nn.Parameter(torch.zeros(num_output_components))

        assert trunk_hidden_dim == latent_dim
        assert fc_hidden_dim == latent_dim

        self.branch_net_global = nn.Sequential(
            nn.Linear(branch_condition_input_dim, latent_dim),
            self.activation,
            nn.Linear(latent_dim, latent_dim * 2),
            self.activation,
            nn.Linear(latent_dim * 2, latent_dim),
            self.activation,
        )

        self.trunk_net = nn.Sequential(
            nn.Linear(latent_dim, trunk_hidden_dim),
            self.activation,
            nn.Linear(trunk_hidden_dim, trunk_hidden_dim * 2),
            self.activation,
            nn.Linear(trunk_hidden_dim * 2, trunk_hidden_dim * num_output_components),
            self.activation,
        )

        self.combined_branch = nn.Sequential(
            nn.Linear(latent_dim, fc_hidden_dim),
            self.activation,
            nn.Linear(fc_hidden_dim, fc_hidden_dim * 2),
            self.activation,
            nn.Linear(fc_hidden_dim * 2, fc_hidden_dim),
            self.activation,
        )

    def forward(self, branch_condition_input, latent, trunk_xyz_encoded):
        branch_output = self.branch_net_global(branch_condition_input)   # (B,D)
        combined_output = branch_output + latent                         # (B,D)

        mix = combined_output.unsqueeze(1) * trunk_xyz_encoded           # (B,N,D)
        mix_reduced = mix.mean(dim=1)                                    # (B,D)
        combined_output2 = self.combined_branch(mix_reduced)             # (B,D)

        B, N, D = trunk_xyz_encoded.shape
        x_trunk = self.trunk_net(trunk_xyz_encoded.reshape(B * N, D)).view(B, N, -1)
        x_trunk = x_trunk.view(B, N, D, -1)                              # (B,N,D,4)

        out = torch.einsum("bd,bndc->bnc", combined_output2, x_trunk) + self.b
        return self.tanh(out)  # scaled field in [-1, 1]


# =========================================================
# helpers
# =========================================================
def inv_field(y_scaled, output_scaler):
    """
    y_scaled: (1,N,4) or (N,4)
    returns same shape in original scale
    """
    if y_scaled.ndim == 3:
        B, N, C = y_scaled.shape
        flat = y_scaled.reshape(-1, C)
        inv = output_scaler.inverse_transform(flat)
        return inv.reshape(B, N, C)
    elif y_scaled.ndim == 2:
        return output_scaler.inverse_transform(y_scaled)
    else:
        raise ValueError(f"Unexpected shape for y_scaled: {y_scaled.shape}")


def find_vtk_file(volume_mesh_dir, filename):
    candidates = [
        os.path.join(volume_mesh_dir, f"{filename}.vtk"),
        os.path.join(volume_mesh_dir, "overall_vtks", f"{filename}.vtk"),
        os.path.join(volume_mesh_dir, "VolumeMesh", f"{filename}.vtk"),
        os.path.join(volume_mesh_dir, "VolumeMesh", "overall_vtks", f"{filename}.vtk"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"VTK file not found for '{filename}'. Tried:\n" + "\n".join(candidates)
    )


def choose_case_keys(all_case_keys, num_cases, mode="first", seed=2024):
    num_cases = min(num_cases, len(all_case_keys))
    if mode == "first":
        return list(all_case_keys[:num_cases])
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(all_case_keys), size=num_cases, replace=False))
    return [all_case_keys[i] for i in idx]


def pad_to_mesh_points(arr, mesh_n_points):
    n, c = arr.shape
    if n == mesh_n_points:
        return arr.astype(np.float32)
    if n > mesh_n_points:
        return arr[:mesh_n_points].astype(np.float32)

    pad = np.zeros((mesh_n_points - n, c), dtype=np.float32)
    return np.vstack([arr, pad]).astype(np.float32)


def load_direct_modules(args, device, cond_dim=5):
    """
    generate_and_visualize_occ.py처럼 trunk_encoding + encoder + field_decoder를
    joint checkpoint에서 직접 분리 로드
    """
    state = torch.load(os.path.join(args.experiment_dir, "model_final.pth"), map_location=device)
    act = nn.SiLU()

    trunk_encoding = nn.Sequential(
        SineDenseLayer(3, args.latent_dim, w0=10.0, is_first=True),
        SineDenseLayer(args.latent_dim, args.latent_dim * 2, w0=10.0, is_first=False),
        SineDenseLayer(args.latent_dim * 2, args.latent_dim, w0=10.0, is_first=False),
    ).to(device).eval()

    encoder = Encoder(
        pointnet_input_dim=3,
        latent_dim=args.latent_dim,
        activation=act,
    ).to(device).eval()

    field_dec = FieldDecoder(
        branch_condition_input_dim=cond_dim,
        latent_dim=args.latent_dim,
        trunk_hidden_dim=args.latent_dim,
        fc_hidden_dim=args.latent_dim,
        num_output_components=4,
        activation=act,
    ).to(device).eval()

    trunk_encoding.load_state_dict(
        {k.replace("trunk_encoding.", ""): v for k, v in state.items() if k.startswith("trunk_encoding.")},
        strict=True,
    )
    encoder.load_state_dict(
        {k.replace("encoder.", ""): v for k, v in state.items() if k.startswith("encoder.")},
        strict=True,
    )
    field_dec.load_state_dict(
        {k.replace("field_decoder.", ""): v for k, v in state.items() if k.startswith("field_decoder.")},
        strict=True,
    )

    return trunk_encoding, encoder, field_dec


def prepare_case_data_direct(
    case_key,
    xyzd_full,
    targets_full,
    branch_scaler,
    pointnet_scaler,
    trunk_scaler,
    output_scaler,
    trunk_encoding,
    encoder,
    field_dec,
    device,
    n_pt_sample=5000,
):
    """
    full mesh points 전체에 대해:
      trunk_encoding(xyz_full_scaled)
      latent = encoder(pointnet_sampled_scaled)
      field_dec(cond_scaled, latent, trunk_enc)
    """
    Xcase = xyzd_full[case_key].astype(np.float32)       # (N, >=9)
    outputs = targets_full[case_key].astype(np.float32)  # (N, 4)

    branch_condition = Xcase[0, 4:9].astype(np.float32)  # (5,)
    branch_pointnet = Xcase[:, :3].astype(np.float32)    # (N,3)
    trunk_input_case = Xcase[:, :4].astype(np.float32)   # (N,4) = xyzd

    total_points = branch_pointnet.shape[0]
    if total_points < n_pt_sample:
        sampled_indices = np.random.choice(total_points, n_pt_sample, replace=True)
    else:
        sampled_indices = np.random.choice(total_points, n_pt_sample, replace=False)

    branch_pointnet_sampled = branch_pointnet[sampled_indices]

    # scalers
    branch_condition_scaled = branch_scaler.transform(branch_condition[np.newaxis, :]).astype(np.float32)  # (1,5)
    branch_pointnet_scaled = pointnet_scaler.transform(branch_pointnet_sampled).astype(np.float32)[np.newaxis, :, :]  # (1,Np,3)
    trunk_input_scaled = trunk_scaler.transform(trunk_input_case).astype(np.float32)  # (N,4)

    # trunk_encoding은 xyz only
    trunk_xyz_scaled = trunk_input_scaled[:, :3]  # (N,3)

    cond_t = torch.from_numpy(branch_condition_scaled).to(device)     # (1,5)
    pointnet_t = torch.from_numpy(branch_pointnet_scaled).to(device)  # (1,Np,3)
    trunk_xyz_t = torch.from_numpy(trunk_xyz_scaled).to(device)       # (N,3)

    with torch.no_grad():
        latent = encoder(pointnet_t)                       # (1,D)
        trunk_enc = trunk_encoding(trunk_xyz_t).unsqueeze(0)  # (1,N,D)
        field_scaled = field_dec(cond_t, latent, trunk_enc)   # (1,N,4)

    u_pred = inv_field(field_scaled.cpu().numpy(), output_scaler).squeeze(0)  # (N,4)

    direction = case_key.split("_")[0]
    clipping_ranges_gt = get_clipping_ranges_for_direction(direction)

    outputs_clip = outputs.copy()
    for i in range(outputs_clip.shape[1]):
        min_val, max_val = clipping_ranges_gt.get(i, (-np.inf, np.inf))
        outputs_clip[:, i] = np.clip(outputs_clip[:, i], min_val, max_val)

    metrics = {}
    for i, comp in enumerate(["ux", "uy", "uz", "vm"]):
        mae = np.mean(np.abs(u_pred[:, i] - outputs_clip[:, i]))
        rmse = np.sqrt(np.mean((u_pred[:, i] - outputs_clip[:, i]) ** 2))
        r2 = calculate_r2(outputs_clip[:, i], u_pred[:, i])
        metrics[comp] = {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}

    return outputs_clip, u_pred, metrics, direction


def render_gt_pred_field_2xN(
    gt_meshes,
    pred_meshes,
    case_keys,
    metrics_list,
    component_idx,
    out_path,
    opacity=1.0,
    cmap="jet",
):
    try:
        pv.start_xvfb()
    except Exception:
        pass

    num_cols = len(case_keys)
    per_subplot_pixel = 500
    window_width = num_cols * per_subplot_pixel
    window_height = 2 * per_subplot_pixel

    scalar_name_gt = f"gt_{COMPONENT_NAMES[component_idx]}"
    scalar_name_pred = f"pred_{COMPONENT_NAMES[component_idx]}"
    title_name = COMPONENT_NAMES[component_idx]

    plotter = pv.Plotter(
        shape=(2, num_cols),
        window_size=(window_width, window_height),
        off_screen=True,
    )

    vals = []
    for m in gt_meshes:
        if scalar_name_gt in m.point_data:
            vals.append(np.asarray(m.point_data[scalar_name_gt]))
    for m in pred_meshes:
        if scalar_name_pred in m.point_data:
            vals.append(np.asarray(m.point_data[scalar_name_pred]))

    clim = None if len(vals) == 0 else [float(np.min(np.concatenate(vals))), float(np.max(np.concatenate(vals)))]

    for j in range(num_cols):
        case_key = case_keys[j]
        metric = metrics_list[j][title_name]

        # GT
        plotter.subplot(0, j)
        gt_mesh = gt_meshes[j]
        plotter.add_mesh(
            gt_mesh,
            scalars=scalar_name_gt,
            cmap=cmap,
            clim=clim,
            show_scalar_bar=(j == num_cols - 1),
            opacity=opacity,
            smooth_shading=True,
        )
        plotter.add_text(
            f"GT\n{case_key}\n{title_name}",
            font_size=10,
            position="upper_left",
            shadow=True,
        )
        plotter.add_axes()
        plotter.camera_position = "iso"

        # Pred
        plotter.subplot(1, j)
        pred_mesh = pred_meshes[j]
        plotter.add_mesh(
            pred_mesh,
            scalars=scalar_name_pred,
            cmap=cmap,
            clim=clim,
            show_scalar_bar=False,
            opacity=opacity,
            smooth_shading=True,
        )
        plotter.add_text(
            f"PRED\n{case_key}\n{title_name}\n"
            f"MAE={metric['mae']:.4f}\n"
            f"RMSE={metric['rmse']:.4f}\n"
            f"R2={metric['r2']:.4f}",
            font_size=10,
            position="upper_left",
            shadow=True,
        )
        plotter.add_axes()
        plotter.camera_position = "iso"

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plotter.render()
    plotter.screenshot(out_path)
    plotter.close()
    print(f"[OK] saved: {out_path}")


# =========================================================
# main
# =========================================================
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--experiment_dir", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True, help="contains npy/xyzdmlc.npz and npy/targets.npz")
    ap.add_argument("--volume_mesh_dir", type=str, required=True)

    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num_cases", type=int, default=5)
    ap.add_argument("--case_mode", type=str, choices=["first", "random"], default="first")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--n_pt_sample", type=int, default=5000)

    ap.add_argument("--component", type=int, default=3, choices=[0, 1, 2, 3], help="0 ux, 1 uy, 2 uz, 3 vm")
    ap.add_argument("--out_path", type=str, default="./field_eval_compare.png")
    ap.add_argument("--opacity", type=float, default=1.0)
    ap.add_argument("--cmap", type=str, default="jet")

    # must match checkpoint / generate_and_visualize_occ.py
    ap.add_argument("--latent_dim", type=int, default=128)
    ap.add_argument("--nf_num_flows", type=int, default=4)   # unused here, kept for consistency
    ap.add_argument("--nf_hidden_dim", type=int, default=64) # unused here, kept for consistency
    ap.add_argument("--nf_activation", type=str, default="Tanh") # unused here

    return ap.parse_args()


def main():
    args = parse_args()
    set_random_seed()
    device = get_device(args.gpu)

    # scalers
    branch_scaler = pickle.load(open(os.path.join(args.experiment_dir, "branch_scaler.pkl"), "rb"))
    pointnet_scaler = pickle.load(open(os.path.join(args.experiment_dir, "pointnet_scaler.pkl"), "rb"))
    trunk_scaler = pickle.load(open(os.path.join(args.experiment_dir, "trunk_scaler.pkl"), "rb"))
    output_scaler = pickle.load(open(os.path.join(args.experiment_dir, "output_scaler.pkl"), "rb"))

    # raw full data
    xyzd_full = np.load(os.path.join(args.data_root, "npy", "xyzdmlc.npz"), allow_pickle=True)
    targets_full = np.load(os.path.join(args.data_root, "npy", "targets.npz"), allow_pickle=True)

    all_case_keys = sorted(list(xyzd_full.files))
    case_keys = choose_case_keys(all_case_keys, args.num_cases, mode=args.case_mode, seed=args.seed)

    trunk_encoding, encoder, field_dec = load_direct_modules(args, device, cond_dim=5)

    gt_meshes = []
    pred_meshes = []
    metrics_list = []

    for case_key in case_keys:
        outputs_clip, u_pred, metrics, direction = prepare_case_data_direct(
            case_key=case_key,
            xyzd_full=xyzd_full,
            targets_full=targets_full,
            branch_scaler=branch_scaler,
            pointnet_scaler=pointnet_scaler,
            trunk_scaler=trunk_scaler,
            output_scaler=output_scaler,
            trunk_encoding=trunk_encoding,
            encoder=encoder,
            field_dec=field_dec,
            device=device,
            n_pt_sample=args.n_pt_sample,
        )

        if "_" in case_key:
            filename = case_key.split("_", 1)[1]
        else:
            filename = case_key

        vtk_filepath = find_vtk_file(args.volume_mesh_dir, filename)
        base_mesh = pv.read(vtk_filepath)

        mesh_n = base_mesh.n_points
        outputs_pad = pad_to_mesh_points(outputs_clip, mesh_n)
        u_pred_pad = pad_to_mesh_points(u_pred, mesh_n)

        gt_mesh = base_mesh.copy(deep=True)
        pred_mesh = base_mesh.copy(deep=True)

        gt_mesh["gt_ux"] = outputs_pad[:, 0]
        gt_mesh["gt_uy"] = outputs_pad[:, 1]
        gt_mesh["gt_uz"] = outputs_pad[:, 2]
        gt_mesh["gt_vm"] = outputs_pad[:, 3]

        pred_mesh["pred_ux"] = u_pred_pad[:, 0]
        pred_mesh["pred_uy"] = u_pred_pad[:, 1]
        pred_mesh["pred_uz"] = u_pred_pad[:, 2]
        pred_mesh["pred_vm"] = u_pred_pad[:, 3]

        gt_meshes.append(gt_mesh)
        pred_meshes.append(pred_mesh)
        metrics_list.append(metrics)

        print(f"[OK] case={case_key}")
        for comp in ["ux", "uy", "uz", "vm"]:
            print(
                f"  {comp}: "
                f"MAE={metrics[comp]['mae']:.6f}, "
                f"RMSE={metrics[comp]['rmse']:.6f}, "
                f"R2={metrics[comp]['r2']:.6f}"
            )

    render_gt_pred_field_2xN(
        gt_meshes=gt_meshes,
        pred_meshes=pred_meshes,
        case_keys=case_keys,
        metrics_list=metrics_list,
        component_idx=args.component,
        out_path=args.out_path,
        opacity=args.opacity,
        cmap=args.cmap,
    )


if __name__ == "__main__":
    main()