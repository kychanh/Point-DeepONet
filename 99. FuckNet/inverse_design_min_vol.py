import os
import sys
import argparse
import pickle
import numpy as np

import torch
import torch.nn.functional as F

from main import DesignGenPointDeepONet


# =========================================================
# Basic utils
# =========================================================
def set_seed(seed=2024):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def inverse_minmax_transform_torch(x_scaled, scaler):
    """
    x_scaled: torch tensor (..., D), scaled in [-1,1]
    returns:  torch tensor (..., D), inverse transformed to physical scale
    """
    shp = x_scaled.shape
    device = x_scaled.device
    dtype = x_scaled.dtype

    x_np = x_scaled.detach().cpu().reshape(-1, shp[-1]).numpy()
    x_inv = scaler.inverse_transform(x_np)
    x_inv = torch.tensor(x_inv, dtype=dtype, device=device)
    return x_inv.reshape(shp)


# =========================================================
# NF inverse: prior z -> latent h
# =========================================================
def nf_inverse_from_z(nf, z):
    """
    NFModel.forward(x): latent -> prior z
    inverse design needs: prior z -> latent

    z: (B, latent_dim)
    return h: (B, latent_dim)
    """
    h = z
    for flow in nf.flow_net[::-1]:
        h, _ = flow.inverse(h)
    return h


# =========================================================
# Load artifacts
# =========================================================
def load_scalers(exp_dir):
    branch_scaler = pickle.load(open(os.path.join(exp_dir, "branch_scaler.pkl"), "rb"))
    trunk_scaler = pickle.load(open(os.path.join(exp_dir, "trunk_scaler.pkl"), "rb"))
    output_scaler = pickle.load(open(os.path.join(exp_dir, "output_scaler.pkl"), "rb"))
    return branch_scaler, trunk_scaler, output_scaler


def infer_branch_dim_from_scaler(branch_scaler):
    return int(branch_scaler.n_features_in_)


def build_net_from_artifacts(args, device, branch_scaler):
    branch_dim = infer_branch_dim_from_scaler(branch_scaler)

    net = DesignGenPointDeepONet(
        branch_condition_input_dim=branch_dim,
        pointnet_input_dim=3,
        trunk_input_dim=4,
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

    state = torch.load(args.model_path, map_location=device)
    net.load_state_dict(state, strict=True)
    net.eval()

    for p in net.parameters():
        p.requires_grad_(False)

    return net


# =========================================================
# Condition loading from xyzdmlc.npz using case_key
# =========================================================
def load_condition_from_case_key_xyzdmlc(xyzdmlc_path, case_key, branch_scaler):
    """
    Assumes xyzdmlc.npz contains:
      - each case_key maps to array-like data
      - data[0][4:9] == [m, l, cx, cy, cz]

    Returns scaled branch condition of shape (1,5)
    """
    data = np.load(xyzdmlc_path, allow_pickle=True)[case_key]
    cond_raw = data[0][4:9].reshape(1, -1)
    cond_scaled = branch_scaler.transform(cond_raw).astype(np.float32)
    return cond_scaled


# =========================================================
# Build inverse grid from trunk_scaler range
# =========================================================
def build_scaled_xyz_grid_from_trunk_scaler(trunk_scaler, nx, ny, nz, device):
    """
    trunk_scaler was fit on xyzd (4D).
    We use its original xyz data range, then set d=0 and transform with the same scaler.
    """
    data_min = trunk_scaler.data_min_
    data_max = trunk_scaler.data_max_

    xmin, ymin, zmin = data_min[:3]
    xmax, ymax, zmax = data_max[:3]

    xs = np.linspace(xmin, xmax, nx, dtype=np.float32)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float32)
    zs = np.linspace(zmin, zmax, nz, dtype=np.float32)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    xyz_phys = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    xyzd_phys = np.zeros((xyz_phys.shape[0], 4), dtype=np.float32)
    xyzd_phys[:, :3] = xyz_phys
    xyzd_scaled = trunk_scaler.transform(xyzd_phys)

    xyz_scaled = torch.tensor(xyzd_scaled[:, :3], dtype=torch.float32, device=device)
    xyz_phys_t = torch.tensor(xyz_phys, dtype=torch.float32, device=device)
    return xyz_scaled, xyz_phys_t


def encode_trunk_in_chunks(net, xyz_scaled, chunk_size=200000):
    """
    xyz_scaled: (N,3)
    returns: (1,N,H)
    """
    outs = []
    for s in range(0, xyz_scaled.shape[0], chunk_size):
        e = min(xyz_scaled.shape[0], s + chunk_size)
        outs.append(net.trunk_encoding(xyz_scaled[s:e]))
    trunk_xyz_encoded = torch.cat(outs, dim=0).unsqueeze(0)
    return trunk_xyz_encoded


# =========================================================
# Field selection
# =========================================================
FIELD_NAME_TO_INDEX = {
    "ux": 0,
    "uy": 1,
    "uz": 2,
    "vm": 3,
}


def get_selected_field(field_phys, field_component="vm", use_abs_for_disp=True):
    """
    field_phys: (1, N, 4) in physical scale
    returns selected field values: (1, N)

    For ux/uy/uz, default is to use absolute displacement magnitude per-axis.
    For vm, returns raw vm.
    """
    idx = FIELD_NAME_TO_INDEX[field_component]
    vals = field_phys[..., idx]

    if field_component in ["ux", "uy", "uz"] and use_abs_for_disp:
        vals = vals.abs()

    return vals


# =========================================================
# Inverse objective
# =========================================================
def compute_inverse_loss(
    net,
    z,
    branch_condition_input,
    trunk_xyz_encoded,
    output_scaler,
    field_component,
    field_allow,
    lambda_constraint=100.0,
    lambda_z=1e-1,
    hard_threshold=0.5,
    use_abs_for_disp=True,
):
    """
    Optimize over prior-space variable z.

    Objective:
      L = V + lambda * (1/N) * sum_i [max(f_i - field_allow, 0)]^2 + lambda_z * ||z||^2

    where f_i is one of:
      - vm
      - |ux| / |uy| / |uz| (default for displacement components)
    """
    latent = nf_inverse_from_z(net.nf, z)                          # (1,H)

    occ_logit = net.shape_decoder(latent, trunk_xyz_encoded)       # (1,N,1)
    occ_prob = torch.sigmoid(occ_logit[..., 0])                    # (1,N)

    field_scaled = net.field_decoder(
        branch_condition_input, latent, trunk_xyz_encoded
    )                                                              # (1,N,4)

    field_phys = inverse_minmax_transform_torch(field_scaled, output_scaler)  # (1,N,4)
    selected_field = get_selected_field(
        field_phys,
        field_component=field_component,
        use_abs_for_disp=use_abs_for_disp,
    )                                                              # (1,N)

    # hard mask for monitoring
    mask_hard = (occ_prob >= hard_threshold).float()               # (1,N)
    field_inside_hard = selected_field * mask_hard                 # (1,N)

    # volume term: soft volume fraction
    vol_soft = occ_prob.mean()

    # ReLU square constraint penalty
    exceed_sq = F.relu(selected_field - field_allow) ** 2          # (1,N)

    # apply soft occupancy mask so only material region contributes
    exceed_sq_inside_soft = exceed_sq * occ_prob                   # (1,N)

    # exact requested form: (1/N) sum_i [...]
    constraint_penalty = exceed_sq_inside_soft.mean()

    z_reg = (z ** 2).mean()

    total = vol_soft + lambda_constraint * constraint_penalty + lambda_z * z_reg

    # monitoring metrics
    exceed_hard = F.relu(field_inside_hard - field_allow)
    viol_hard_ratio = ((field_inside_hard > field_allow) * mask_hard).sum() / (mask_hard.sum() + 1e-8)

    stats = {
        "loss": float(total.detach().cpu()),
        "field_component": field_component,
        "field_allow": float(field_allow),
        "vol_soft": float(vol_soft.detach().cpu()),
        "constraint_penalty": float(constraint_penalty.detach().cpu()),
        "z_reg": float(z_reg.detach().cpu()),
        "field_max_soft_inside": float((selected_field * occ_prob).max().detach().cpu()),
        "field_mean_soft_inside": float((selected_field * occ_prob).mean().detach().cpu()),
        "field_max_hard_inside": float(field_inside_hard.max().detach().cpu()),
        "field_mean_hard_inside": float(
            field_inside_hard.sum().div(mask_hard.sum() + 1e-8).detach().cpu()
        ),
        "hard_fill_ratio": float(mask_hard.mean().detach().cpu()),
        "viol_hard_ratio": float(viol_hard_ratio.detach().cpu()),
        "max_exceed_hard": float(exceed_hard.max().detach().cpu()),
    }

    aux = {
        "latent": latent.detach(),
        "occ_logit": occ_logit.detach(),
        "occ_prob": occ_prob.detach(),
        "mask_hard": mask_hard.detach(),
        "field_scaled": field_scaled.detach(),
        "field_phys": field_phys.detach(),
        "selected_field": selected_field.detach(),
        "field_inside_hard": field_inside_hard.detach(),
        "exceed_sq_inside_soft": exceed_sq_inside_soft.detach(),
    }

    return total, stats, aux


def make_snapshot(it, z, stats, aux):
    return {
        "iter": int(it),
        "z": z.detach().cpu().numpy().copy(),
        "latent": aux["latent"].detach().cpu().numpy().copy(),
        "occ_prob": aux["occ_prob"].detach().cpu().numpy().copy(),
        "mask_hard": aux["mask_hard"].detach().cpu().numpy().copy(),
        "field_phys": aux["field_phys"].detach().cpu().numpy().copy(),
        "selected_field": aux["selected_field"].detach().cpu().numpy().copy(),
        "field_inside_hard": aux["field_inside_hard"].detach().cpu().numpy().copy(),
        "exceed_sq_inside_soft": aux["exceed_sq_inside_soft"].detach().cpu().numpy().copy(),
        "stats": dict(stats),
    }


# =========================================================
# Optimization loop
# =========================================================
def run_single_restart(
    net,
    branch_condition_input,
    trunk_xyz_encoded,
    output_scaler,
    field_component,
    field_allow,
    latent_dim,
    n_iters,
    lr,
    lambda_constraint,
    lambda_z,
    hard_threshold,
    use_abs_for_disp,
    seed,
    print_every=50,
    save_every=20,
):
    set_seed(seed)
    trajectory = []

    z = torch.randn(1, latent_dim, device=branch_condition_input.device, requires_grad=True)

    with torch.no_grad():
        loss0, stats0, aux0 = compute_inverse_loss(
            net=net,
            z=z,
            branch_condition_input=branch_condition_input,
            trunk_xyz_encoded=trunk_xyz_encoded,
            output_scaler=output_scaler,
            field_component=field_component,
            field_allow=field_allow,
            lambda_constraint=lambda_constraint,
            lambda_z=lambda_z,
            hard_threshold=hard_threshold,
            use_abs_for_disp=use_abs_for_disp,
        )
    trajectory.append(make_snapshot(0, z, stats0, aux0))

    optimizer = torch.optim.Adam([z], lr=lr)

    best = {
        "loss": float("inf"),
        "z": None,
        "stats": None,
        "aux": None,
    }

    for it in range(1, n_iters + 1):
        optimizer.zero_grad()

        loss, stats, aux = compute_inverse_loss(
            net=net,
            z=z,
            branch_condition_input=branch_condition_input,
            trunk_xyz_encoded=trunk_xyz_encoded,
            output_scaler=output_scaler,
            field_component=field_component,
            field_allow=field_allow,
            lambda_constraint=lambda_constraint,
            lambda_z=lambda_z,
            hard_threshold=hard_threshold,
            use_abs_for_disp=use_abs_for_disp,
        )

        loss.backward()
        optimizer.step()

        if stats["loss"] < best["loss"]:
            best["loss"] = stats["loss"]
            best["z"] = z.detach().clone()
            best["stats"] = stats
            best["aux"] = {k: v.detach().clone() for k, v in aux.items()}

        if it == 1 or it % print_every == 0 or it == n_iters:
            print(
                f"[Iter {it:04d}] "
                f"loss={stats['loss']:.6f} | "
                f"field={field_component} | "
                f"field_max_hard={stats['field_max_hard_inside']:.6f} | "
                f"constraint_pen={stats['constraint_penalty']:.6f} | "
                f"vol_soft={stats['vol_soft']:.6f} | "
                f"viol_hard_ratio={stats['viol_hard_ratio']:.6f} | "
                f"fill_hard={stats['hard_fill_ratio']:.6f}"
            )

        if (it % save_every == 0) or (it == n_iters):
            trajectory.append(make_snapshot(it, z, stats, aux))

    return best, trajectory


# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()

    # experiment artifacts
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="directory containing model_final.pth and scaler pkl files")
    parser.add_argument("--model_path", type=str, default="",
                        help="default: {exp_dir}/model_final.pth")

    # condition source
    parser.add_argument("--xyzdmlc_path", type=str, required=True,
                        help="path to xyzdmlc.npz")
    parser.add_argument("--case_key", type=str, required=True,
                        help="case key existing in xyzdmlc.npz")

    # model architecture (must match training)
    parser.add_argument("--branch_hidden_dim", type=int, default=128)
    parser.add_argument("--trunk_hidden_dim", type=int, default=128)
    parser.add_argument("--fc_hidden_dim", type=int, default=128)
    parser.add_argument("--trunk_encoding_hidden_dim", type=int, default=128)
    parser.add_argument("--nf_num_flows", type=int, default=4)
    parser.add_argument("--nf_hidden_dim", type=int, default=64)
    parser.add_argument("--nf_activation", type=str, default="Tanh")

    # inverse optimization
    parser.add_argument("--n_iters", type=int, default=800)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--n_restarts", type=int, default=5)

    parser.add_argument(
        "--field_component",
        type=str,
        default="vm",
        choices=["ux", "uy", "uz", "vm"],
        help="which field to constrain: ux, uy, uz, or vm"
    )
    parser.add_argument(
        "--field_allow",
        type=float,
        required=True,
        help="allowable threshold in physical scale for selected field"
    )
    parser.add_argument(
        "--use_abs_for_disp",
        action="store_true",
        help="for ux/uy/uz, use absolute value before thresholding"
    )

    parser.add_argument("--lambda_constraint", type=float, default=100.0,
                        help="weight for ReLU-square field constraint penalty")
    parser.add_argument("--lambda_z", type=float, default=1e-3)
    parser.add_argument("--hard_threshold", type=float, default=0.5)

    # inverse grid
    parser.add_argument("--nx", type=int, default=48)
    parser.add_argument("--ny", type=int, default=48)
    parser.add_argument("--nz", type=int, default=48)
    parser.add_argument("--trunk_chunk", type=int, default=200000)

    # runtime
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--print_every", type=int, default=50)

    # save
    parser.add_argument("--save_dir", type=str, default="./inverse_results")
    parser.add_argument("--save_name", type=str, default="inverse_best_vol.npz")

    return parser.parse_args()


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()
    set_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_path == "":
        args.model_path = os.path.join(args.exp_dir, "model_final.pth")

    os.makedirs(args.save_dir, exist_ok=True)

    # load scalers
    branch_scaler, trunk_scaler, output_scaler = load_scalers(args.exp_dir)

    # build net and load weights
    net = build_net_from_artifacts(args, device, branch_scaler)
    net.trunk_chunk = args.trunk_chunk

    # load branch condition from case_key
    cond_scaled = load_condition_from_case_key_xyzdmlc(
        xyzdmlc_path=args.xyzdmlc_path,
        case_key=args.case_key,
        branch_scaler=branch_scaler,
    )
    branch_condition_input = torch.tensor(cond_scaled, dtype=torch.float32, device=device)

    # build inverse grid in physical range seen during training
    xyz_scaled, xyz_phys = build_scaled_xyz_grid_from_trunk_scaler(
        trunk_scaler=trunk_scaler,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        device=device,
    )

    # precompute trunk encoding once
    trunk_xyz_encoded = encode_trunk_in_chunks(
        net, xyz_scaled, chunk_size=args.trunk_chunk
    )

    latent_dim = args.trunk_encoding_hidden_dim

    global_best = {
        "loss": float("inf"),
        "z": None,
        "stats": None,
        "aux": None,
    }

    # vm이면 어차피 abs 불필요, ux/uy/uz이면 기본 True 추천
    use_abs_for_disp = args.use_abs_for_disp or (args.field_component in ["ux", "uy", "uz"])

    for r in range(args.n_restarts):
        print(f"\n========== Restart {r+1}/{args.n_restarts} ==========")

        best, trajectory = run_single_restart(
            net=net,
            branch_condition_input=branch_condition_input,
            trunk_xyz_encoded=trunk_xyz_encoded,
            output_scaler=output_scaler,
            field_component=args.field_component,
            field_allow=args.field_allow,
            latent_dim=latent_dim,
            n_iters=args.n_iters,
            lr=args.lr,
            lambda_constraint=args.lambda_constraint,
            lambda_z=args.lambda_z,
            hard_threshold=args.hard_threshold,
            use_abs_for_disp=use_abs_for_disp,
            seed=args.seed + r,
            print_every=args.print_every,
        )

        print(f"[Restart {r+1}] best stats = {best['stats']}")

        if best["loss"] < global_best["loss"]:
            global_best = best

        traj_path = os.path.join(args.save_dir, f"trajectory_vol_restart_{r+1}.pkl")
        with open(traj_path, "wb") as f:
            pickle.dump(trajectory, f)

    save_path = os.path.join(args.save_dir, args.save_name)

    np.savez_compressed(
        save_path,
        case_key=np.array(args.case_key),
        field_component=np.array(args.field_component),
        field_allow=np.array(args.field_allow, dtype=np.float32),
        use_abs_for_disp=np.array(use_abs_for_disp),
        xyz_phys=xyz_phys.detach().cpu().numpy(),
        xyz_scaled=xyz_scaled.detach().cpu().numpy(),
        best_z=global_best["z"].detach().cpu().numpy(),
        latent=global_best["aux"]["latent"].detach().cpu().numpy(),
        occ_logit=global_best["aux"]["occ_logit"].detach().cpu().numpy(),
        occ_prob=global_best["aux"]["occ_prob"].detach().cpu().numpy(),
        mask_hard=global_best["aux"]["mask_hard"].detach().cpu().numpy(),
        field_scaled=global_best["aux"]["field_scaled"].detach().cpu().numpy(),
        field_phys=global_best["aux"]["field_phys"].detach().cpu().numpy(),
        selected_field=global_best["aux"]["selected_field"].detach().cpu().numpy(),
        field_inside_hard=global_best["aux"]["field_inside_hard"].detach().cpu().numpy(),
        exceed_sq_inside_soft=global_best["aux"]["exceed_sq_inside_soft"].detach().cpu().numpy(),
        best_stats=np.array(global_best["stats"], dtype=object),
    )

    print("\n========== Global Best ==========")
    print(global_best["stats"])
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    main()