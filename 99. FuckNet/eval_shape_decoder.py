import os
import argparse
import pickle
import numpy as np
import torch
import pyvista as pv
import vtk

print("### RUNNING FILE:", __file__)

vtk.vtkObject.GlobalWarningDisplayOff()
pv.global_theme.jupyter_backend = "trame"
pv.global_theme.allow_empty_mesh = True

from main import (
    set_random_seed,
    get_device,
    load_and_preprocess_data,
    DesignGenPointDeepONet,
)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_bbox_grid(n: int, xyz_min, xyz_max):
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
    trunk_scaler는 xyzd에 fit되어 있으므로 d=0 padding 후 transform
    """
    N = pts_xyz.shape[0]
    tmp = np.zeros((N, trunk_scaler.n_features_in_), dtype=np.float32)
    tmp[:, :3] = pts_xyz
    tmp_scaled = trunk_scaler.transform(tmp)
    return tmp_scaled[:, :3].astype(np.float32)


def find_gt_vtk(volume_mesh_dir: str, case_key: str):
    filename = case_key.split('_', 1)[1]

    return os.path.join(volume_mesh_dir, f"{filename}.vtk")


def choose_case_indices(num_test, num_cases, mode="first", seed=2024):
    num_cases = min(num_cases, num_test)
    if mode == "first":
        return np.arange(num_cases)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_test, size=num_cases, replace=False))


def build_model(args, device, x_test, N_field):
    branch_condition_input_dim = x_test[0].shape[1]
    pointnet_input_dim = x_test[1].shape[-1]
    trunk_input_dim = x_test[2].shape[-1]

    net = DesignGenPointDeepONet(
        branch_condition_input_dim=branch_condition_input_dim,
        pointnet_input_dim=pointnet_input_dim,
        trunk_input_dim=trunk_input_dim,
        branch_hidden_dim=args.branch_hidden_dim,
        trunk_hidden_dim=args.trunk_hidden_dim,
        fc_hidden_dim=args.fc_hidden_dim,
        trunk_encoding_hidden_dim=args.trunk_encoding_hidden_dim,
        num_output_components=4,
        nf_num_flows=args.nf_num_flows,
        nf_hidden_dim=args.nf_hidden_dim,
        nf_activation=args.nf_activation,
        device_str=str(device),
    ).to(device)

    net.N_field = int(N_field)

    ckpt_path = os.path.join(args.experiment_dir, "model_final.pth")
    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state, strict=True)
    net.eval()
    return net


def render_gt_pred_2xN(
    gt_paths,
    pred_paths,
    case_keys,
    out_dir,
    out_name="0_eval_shape_compare.png",
    opacity=1.0,
):
    try:
        pv.start_xvfb()
    except Exception:
        pass

    num_cols = len(case_keys)
    per_subplot_pixel = 500
    window_width = num_cols * per_subplot_pixel
    window_height = 2 * per_subplot_pixel

    plotter = pv.Plotter(
        shape=(2, num_cols),
        window_size=(window_width, window_height),
        off_screen=True
    )

    for j in range(num_cols):
        case_key = case_keys[j]

        # GT
        plotter.subplot(0, j)
        gt_mesh = pv.read(gt_paths[j])
        if gt_mesh.n_points == 0:
            plotter.add_text(f"GT EMPTY\n{case_key}", font_size=10)
        else:
            plotter.add_mesh(gt_mesh, color="lightgray", opacity=opacity, smooth_shading=True)
            plotter.add_text(f"GT\n{case_key}", font_size=10, position="upper_left", shadow=True)
            plotter.add_axes()
            plotter.camera_position = "iso"

        # Pred
        plotter.subplot(1, j)
        pred_mesh = pv.read(pred_paths[j])
        if pred_mesh.n_points == 0:
            plotter.add_text(f"PRED EMPTY\n{case_key}", font_size=10)
        else:
            plotter.add_mesh(pred_mesh, color="lightgray", opacity=opacity, smooth_shading=True)
            plotter.add_text(f"PRED\n{case_key}", font_size=10, position="upper_left", shadow=True)
            plotter.add_axes()
            plotter.camera_position = "iso"

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, out_name)
    plotter.render()
    plotter.screenshot(save_path)
    plotter.close()

    print(f"[OK] Visualization saved: {save_path}")
    return save_path


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--experiment_dir", type=str, required=True)
    ap.add_argument("--dir_base_load_data", type=str, required=True)
    ap.add_argument("--volume_mesh_dir", type=str, required=True)
    ap.add_argument("--grid_path", type=str, required=True)

    ap.add_argument("--RUN", type=int, default=0)
    ap.add_argument("--N_pt", type=int, default=5000)
    ap.add_argument("--N_samples", type=int, default=3000)
    ap.add_argument("--split_method", type=str, choices=["mass", "random"], default="random")

    ap.add_argument("--branch_condition_input_components", type=str, default="mlc")
    ap.add_argument("--trunk_input_components", type=str, default="xyzd")
    ap.add_argument("--output_components", type=str, default="xyzs")

    ap.add_argument("--branch_hidden_dim", type=int, default=128)
    ap.add_argument("--trunk_hidden_dim", type=int, default=128)
    ap.add_argument("--trunk_encoding_hidden_dim", type=int, default=128)
    ap.add_argument("--fc_hidden_dim", type=int, default=128)
    ap.add_argument("--nf_num_flows", type=int, default=4)
    ap.add_argument("--nf_hidden_dim", type=int, default=64)
    ap.add_argument("--nf_activation", type=str, default="Tanh")

    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num_cases", type=int, default=5)
    ap.add_argument("--grid_n", type=int, default=96)
    ap.add_argument("--occ_thr", type=float, default=0.5)
    ap.add_argument("--out_dir", type=str, default="./eval_shape_outputs")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--case_mode", type=str, choices=["first", "random"], default="first")
    ap.add_argument("--opacity", type=float, default=1.0)

    return ap.parse_args()


def main():
    args = parse_args()
    set_random_seed()
    device = get_device(args.gpu)
    os.makedirs(args.out_dir, exist_ok=True)

    # load test data only to get test cases + branch inputs
    x_train, y_train, x_test, y_test, output_scalers, train_case_file, test_case_file, N_field, N_grid = \
        load_and_preprocess_data(args, args.experiment_dir)

    trunk_scaler = pickle.load(open(os.path.join(args.experiment_dir, "trunk_scaler.pkl"), "rb"))
    model = build_model(args, device, x_test, N_field)

    case_indices = choose_case_indices(
        num_test=x_test[0].shape[0],
        num_cases=args.num_cases,
        mode=args.case_mode,
        seed=args.seed,
    )

    # generate_and_visualize_occ.py처럼 공통 bbox grid 생성
    xyz_min = trunk_scaler.data_min_[:3]
    xyz_max = trunk_scaler.data_max_[:3]
    grid, pts = make_bbox_grid(args.grid_n, xyz_min, xyz_max)
    pts_scaled_xyz = apply_trunk_scaler_to_xyz(pts, trunk_scaler)  # (Ng, 3)

    gt_paths = []
    pred_paths = []
    case_keys = []

    for k, global_i in enumerate(case_indices):
        case_key = str(test_case_file[global_i])
        case_keys.append(case_key)

        gt_vtk_path = find_gt_vtk(args.volume_mesh_dir, case_key)
        gt_paths.append(gt_vtk_path)

        # branch input / point cloud는 test case 것 사용
        branch_cond = torch.from_numpy(x_test[0][global_i:global_i+1]).to(device)   # (1, C)
        point_cloud = torch.from_numpy(x_test[1][global_i:global_i+1]).to(device)   # (1, Np, d)

        # generate_and_visualize_occ.py처럼 trunk를 새 bbox grid로 직접 구성
        pts_scaled_xyz_t = torch.from_numpy(pts_scaled_xyz).unsqueeze(0).to(device)  # (1, Ng, 3)

        # xyzd 입력이면 d 채널 0으로 padding
        trunk_input_dim = x_test[2].shape[-1]
        if trunk_input_dim == 4:
            zero_d = torch.zeros((1, pts_scaled_xyz.shape[0], 1), dtype=pts_scaled_xyz_t.dtype, device=device)
            trunk_in = torch.cat([pts_scaled_xyz_t, zero_d], dim=-1)  # (1, Ng, 4)
        elif trunk_input_dim == 3:
            trunk_in = pts_scaled_xyz_t
        else:
            raise ValueError(f"Unsupported trunk_input_dim={trunk_input_dim}")

        with torch.no_grad():
            y_out = model((branch_cond, point_cloud, trunk_in)).detach().cpu().numpy()[0]  # (Ng, 5)

        pred_logit = y_out[:, 4]
        pred_prob = sigmoid(pred_logit).astype(np.float32)

        # generate_and_visualize_occ.py와 동일하게 같은 grid에 넣음
        grid_i = grid.copy(deep=True)
        grid_i.point_data["occ_prob"] = pred_prob

        grid_path = os.path.join(args.out_dir, f"{case_key}_pred_grid.vtk")
        grid_i.save(grid_path)

        inside = grid_i.threshold(value=float(args.occ_thr), scalars="occ_prob")
        surf = inside.extract_surface()

        surf_path = os.path.join(args.out_dir, f"{case_key}_pred_surface.vtk")
        surf.save(surf_path)
        pred_paths.append(surf_path)

        print(f"[OK] saved: {surf_path}")

    render_gt_pred_2xN(
        gt_paths=gt_paths,
        pred_paths=pred_paths,
        case_keys=case_keys,
        out_dir=args.out_dir,
        out_name="0_eval_shape_compare.png",
        opacity=args.opacity,
    )


if __name__ == "__main__":
    main()