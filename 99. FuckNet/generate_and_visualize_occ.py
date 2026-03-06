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


# -------------------------
# Clip ranges (same as your code)
# -------------------------
component_names = {0: 'ux', 1: 'uy', 2: 'uz', 3: 'vm', 4: 'occ_prob'}

def get_clip_values(direction):
    if direction == 'ver':
        return {0: (-0.068, 0.473), 1: (-0.093, 0.073), 2: (-0.003, 0.824), 3: (0., 232.19)}
    elif direction == 'hor':
        return {0: (-0.421, 0.008), 1: (-0.024, 0.029), 2: (-0.388, 0.109), 3: (0., 227.78)}
    elif direction == 'dia':
        return {0: (-0.079, 0.016), 1: (-0.057, 0.056), 2: (-0.006, 0.214), 3: (0., 172.19)}
    else:
        raise ValueError(f"Unknown direction: {direction}")


# -------------------------
# Model blocks (must match main.py)
# -------------------------
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
            nn.init.uniform_(self.linear.weight,
                             -np.sqrt(6 / self.in_features) / self.w0,
                             np.sqrt(6 / self.in_features) / self.w0)

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
        branch_output = self.branch_net_global(branch_condition_input)
        combined_output = branch_output + latent

        mix = combined_output.unsqueeze(1) * trunk_xyz_encoded
        mix_reduced = mix.mean(dim=1)
        combined_output2 = self.combined_branch(mix_reduced)

        B, N, D = trunk_xyz_encoded.shape
        x_trunk = self.trunk_net(trunk_xyz_encoded.reshape(B * N, D)).view(B, N, -1)
        x_trunk = x_trunk.view(B, N, D, -1)

        out = torch.einsum("bd,bndc->bnc", combined_output2, x_trunk) + self.b
        return self.tanh(out)  # scaled


class ShapeDecoder(nn.Module):
    """occupancy logits head (NO tanh/sigmoid here)"""
    def __init__(self, latent_dim, trunk_hidden_dim, fc_hidden_dim, activation):
        super().__init__()
        self.activation = activation
        self.b = nn.Parameter(torch.zeros(1))

        assert trunk_hidden_dim == latent_dim
        assert fc_hidden_dim == latent_dim

        self.trunk_net = nn.Sequential(
            nn.Linear(latent_dim, trunk_hidden_dim),
            self.activation,
            nn.Linear(trunk_hidden_dim, trunk_hidden_dim * 2),
            self.activation,
            nn.Linear(trunk_hidden_dim * 2, trunk_hidden_dim * 1),
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

    def forward(self, latent, trunk_xyz_encoded):
        mix = latent.unsqueeze(1) * trunk_xyz_encoded
        mix_reduced = mix.mean(dim=1)
        combined_output2 = self.combined_branch(mix_reduced)

        B, N, D = trunk_xyz_encoded.shape
        x_trunk = self.trunk_net(trunk_xyz_encoded.reshape(B * N, D)).view(B, N, -1)
        x_trunk = x_trunk.view(B, N, D, 1)

        occ_logit = torch.einsum("bd,bndc->bnc", combined_output2, x_trunk) + self.b
        return occ_logit  # (B,N,1)


# ---- NF (must match main.py) ----
from torch.distributions import MultivariateNormal

class FCNN_NF(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, activation):
        super().__init__()
        act_map = {"Tanh": nn.Tanh(), "SiLU": nn.SiLU(), "ReLU": nn.ReLU()}
        self.activation = act_map[activation] if isinstance(activation, str) else activation
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x):
        return self.network(x)

class RealNVP(nn.Module):
    def __init__(self, dim, hidden_dim=64, activation="Tanh", base_network=FCNN_NF):
        super().__init__()
        self.dim = dim
        self.t1 = base_network(dim // 2, dim // 2, hidden_dim, activation)
        self.s1 = base_network(dim // 2, dim // 2, hidden_dim, activation)
        self.t2 = base_network(dim // 2, dim // 2, hidden_dim, activation)
        self.s2 = base_network(dim // 2, dim // 2, hidden_dim, activation)

    def inverse(self, z):
        lower, upper = z[:, :self.dim // 2], z[:, self.dim // 2:]
        t2 = self.t2(upper)
        s2 = self.s2(upper)
        lower = (lower - t2) * torch.exp(-s2)
        t1 = self.t1(lower)
        s1 = self.s1(lower)
        upper = (upper - t1) * torch.exp(-s1)
        x = torch.cat([lower, upper], dim=1)
        log_det = torch.sum(-s1, dim=1) + torch.sum(-s2, dim=1)
        return x, log_det

class NFModel(nn.Module):
    def __init__(self, dim, hidden_dim, activation, num_flows, device):
        super().__init__()
        self.prior = MultivariateNormal(torch.zeros(dim).to(device), torch.eye(dim).to(device))
        self.flow_net = nn.Sequential(*[RealNVP(dim, hidden_dim, activation) for _ in range(num_flows)])

    def sample(self, n):
        z = self.prior.sample((n,))
        for flow in self.flow_net[::-1]:
            z, _ = flow.inverse(z)
        return z


# -------------------------
# Grid helper
# -------------------------
def make_uniform_grid(n: int, extent: float = 1.0):
    grid = pv.ImageData()
    grid.dimensions = (n, n, n)
    grid.origin = (-extent, -extent, -extent)
    spacing = (2.0 * extent) / (n - 1)
    grid.spacing = (spacing, spacing, spacing)
    pts = grid.points.astype(np.float32)
    return grid, pts

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


def inv_field(data_arr, scaler):
    shape = data_arr.shape
    flat = data_arr.reshape(-1, shape[-1])
    inv = scaler.inverse_transform(flat)
    return inv.reshape(shape)


def apply_trunk_scaler_to_xyz(pts_xyz, trunk_scaler):
    """
    trunk_scaler was fit on xyzd. We apply it to xyz by padding d=0 and taking xyz cols back.
    pts_xyz: (N,3)
    returns: (N,3) scaled
    """
    N = pts_xyz.shape[0]
    tmp = np.zeros((N, trunk_scaler.n_features_in_), dtype=np.float32)
    tmp[:, :3] = pts_xyz
    tmp_scaled = trunk_scaler.transform(tmp)
    return tmp_scaled[:, :3].astype(np.float32)


# -------------------------
# Visualization utilities
# -------------------------
def save_global_colorbar(a_min, a_max, label, out_path):
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    cmap = plt.get_cmap('jet')
    norm = plt.Normalize(vmin=a_min, vmax=a_max)
    cb1 = cbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
    cb1.set_label(label)

    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Colorbar saved: {out_path}")


def visualize_generated_surfaces(
    surface_paths,
    case_key,
    out_dir,
    direction,
    component=3,
    num_samples=10,
    opacity=1.0,
    cmap="jet",
    use_clip=True,
):
    from IPython.display import display, Image
    os.makedirs(out_dir, exist_ok=True)

    try:
        pv.start_xvfb()
        print("[DEBUG] pv.start_xvfb(): OK")
    except Exception as e:
        print("[DEBUG] pv.start_xvfb(): SKIP:", e)

    per_subplot_pixel = 300
    num_cols = min(num_samples, len(surface_paths))
    window_width = num_cols * per_subplot_pixel
    window_height = per_subplot_pixel

    plotter = pv.Plotter(shape=(1, num_cols), window_size=(window_width, window_height), off_screen=True)

    scalar_name = component_names.get(component, f"comp_{component}")
    clim = None
    if use_clip and component in [0, 1, 2, 3]:
        clip_dict = get_clip_values(direction)
        a_min, a_max = clip_dict[component]
        clim = [a_min, a_max]

    for j in range(num_cols):
        surf_path = surface_paths[j]
        surf = pv.read(surf_path)
        plotter.subplot(0, j)

        if surf.n_points == 0:
            plotter.add_text(f"EMPTY\n{os.path.basename(surf_path)}", font_size=10)
            continue

        if scalar_name not in surf.point_data:
            available = list(surf.point_data.keys())
            plotter.add_text(f"Missing '{scalar_name}'\nHave: {available}", font_size=9)
            plotter.add_mesh(surf, color="lightgray", opacity=opacity)
            continue

        plotter.add_mesh(
            surf,
            scalars=scalar_name,
            cmap=cmap,
            clim=clim,
            show_scalar_bar=(j == num_cols - 1),
            opacity=opacity
        )

        txt = f"{os.path.basename(surf_path)}\n{scalar_name}"
        if clim is not None:
            txt += f"\nclim=[{clim[0]}, {clim[1]}]"
        plotter.add_text(txt, font_size=10, position="upper_left", shadow=True)
        plotter.add_axes()

    save_path = os.path.join(out_dir, f"generated_{case_key}_{scalar_name}.png")
    plotter.render()
    plotter.screenshot(save_path)
    plotter.close()

    if os.path.exists(save_path):
        print(f"[OK] Visualization saved: {save_path}")
        display(Image(filename=save_path))

    if clim is not None:
        cb_path = os.path.join(out_dir, f"colorbar_{direction}_{scalar_name}.png")
        save_global_colorbar(clim[0], clim[1], f"{scalar_name} ({direction.upper()})", cb_path)
        if os.path.exists(cb_path):
            display(Image(filename=cb_path))

    import matplotlib.image as mpimg
    img = mpimg.imread(save_path)
    plt.figure(figsize=(14, 4))
    plt.axis("off")
    plt.imshow(img)
    plt.show()


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_dir", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True, help=".../data (contains npy/xyzdmlc.npz)")
    ap.add_argument("--case_key", type=str, required=True, help="key in xyzdmlc.npz (e.g., ver_xxx)")
    ap.add_argument("--out_dir", type=str, default="gen_outputs")

    # generation
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--grid_n", type=int, default=96)
    ap.add_argument("--grid_extent", type=float, default=1.0)
    ap.add_argument("--occ_thr", type=float, default=0.5, help="occupancy prob threshold for surface extraction")

    # NF (loaded FROM model_final.pth, but hyperparams needed to build module)
    ap.add_argument("--latent_dim", type=int, default=128)
    ap.add_argument("--nf_num_flows", type=int, default=4)
    ap.add_argument("--nf_hidden_dim", type=int, default=64)
    ap.add_argument("--nf_activation", type=str, default="Tanh")

    # viz
    ap.add_argument("--viz_num", type=int, default=8)
    ap.add_argument("--viz_component", type=int, default=3, help="0 ux,1 uy,2 uz,3 vm,4 occ_prob")
    ap.add_argument("--viz_opacity", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    direction = args.case_key.split("_")[0]

    # scalers
    branch_scaler = pickle.load(open(os.path.join(args.experiment_dir, "branch_scaler.pkl"), "rb"))
    trunk_scaler = pickle.load(open(os.path.join(args.experiment_dir, "trunk_scaler.pkl"), "rb"))
    output_scaler = pickle.load(open(os.path.join(args.experiment_dir, "output_scaler.pkl"), "rb"))

    # condition from case_key
    xyzd_full = np.load(os.path.join(args.data_root, "npy/xyzdmlc.npz"), allow_pickle=True)
    Xcase = xyzd_full[args.case_key]
    cond = Xcase[0, 4:9].astype(np.float32)              # (5,)
    cond_scaled = branch_scaler.transform(cond[None, :]).astype(np.float32)
    cond_t = torch.from_numpy(cond_scaled).to(device)

    # load state dict (contains trunk_encoding, field_decoder, shape_decoder, nf)
    state = torch.load(os.path.join(args.experiment_dir, "model_final.pth"), map_location=device)

    act = nn.SiLU()

    trunk_encoding = nn.Sequential(
        SineDenseLayer(3, args.latent_dim, w0=10.0, is_first=True),
        SineDenseLayer(args.latent_dim, args.latent_dim * 2, w0=10.0, is_first=False),
        SineDenseLayer(args.latent_dim * 2, args.latent_dim, w0=10.0, is_first=False),
    ).to(device).eval()

    field_dec = FieldDecoder(
        branch_condition_input_dim=cond_scaled.shape[1],
        latent_dim=args.latent_dim,
        trunk_hidden_dim=args.latent_dim,
        fc_hidden_dim=args.latent_dim,
        num_output_components=4,
        activation=act
    ).to(device).eval()

    shape_dec = ShapeDecoder(
        latent_dim=args.latent_dim,
        trunk_hidden_dim=args.latent_dim,
        fc_hidden_dim=args.latent_dim,
        activation=act
    ).to(device).eval()

    nf = NFModel(
        dim=args.latent_dim,
        hidden_dim=args.nf_hidden_dim,
        activation=args.nf_activation,
        num_flows=args.nf_num_flows,
        device=device
    ).to(device).eval()

    # load weights from the joint model checkpoint
    trunk_encoding.load_state_dict({k.replace("trunk_encoding.", ""): v for k, v in state.items() if k.startswith("trunk_encoding.")}, strict=True)
    field_dec.load_state_dict({k.replace("field_decoder.", ""): v for k, v in state.items() if k.startswith("field_decoder.")}, strict=True)
    shape_dec.load_state_dict({k.replace("shape_decoder.", ""): v for k, v in state.items() if k.startswith("shape_decoder.")}, strict=True)

    # NF weights
    nf_sd = {k.replace("nf.", ""): v for k, v in state.items() if k.startswith("nf.")}
    if len(nf_sd) == 0:
        raise RuntimeError("No NF weights found in model_final.pth (expected keys starting with 'nf.').")
    nf.load_state_dict(nf_sd, strict=True)

    # build uniform grid in ORIGINAL coordinate space, then scale xyz using trunk_scaler
    xyz_min = trunk_scaler.data_min_[:3]
    xyz_max = trunk_scaler.data_max_[:3]
    grid, pts = make_bbox_grid(args.grid_n, xyz_min, xyz_max)  # pts in [-extent,extent]
    pts_scaled = apply_trunk_scaler_to_xyz(pts, trunk_scaler)
    pts_t = torch.from_numpy(pts_scaled).to(device)

    with torch.no_grad():
        trunk_enc = trunk_encoding(pts_t).unsqueeze(0)  # (1,Ng,D)

    surface_paths = []

    for i in range(args.num_samples):
        with torch.no_grad():
            latent = nf.sample(1).to(device)                         # (1,D)

            field_scaled = field_dec(cond_t, latent, trunk_enc)      # (1,Ng,4) scaled
            occ_logit = shape_dec(latent, trunk_enc).squeeze(-1)     # (1,Ng)
            occ_prob = torch.sigmoid(occ_logit)

            field_scaled_np = field_scaled.squeeze(0).cpu().numpy()
            occ_prob_np = occ_prob.squeeze(0).cpu().numpy().astype(np.float32)

        field_orig = inv_field(field_scaled_np[None, :, :], output_scaler).squeeze(0)  # (Ng,4)

        grid_i = grid.copy(deep=True)
        grid_i.point_data["occ_prob"] = occ_prob_np
        grid_i.point_data["ux"] = field_orig[:, 0].astype(np.float32)
        grid_i.point_data["uy"] = field_orig[:, 1].astype(np.float32)
        grid_i.point_data["uz"] = field_orig[:, 2].astype(np.float32)
        grid_i.point_data["vm"] = field_orig[:, 3].astype(np.float32)

        grid_path = os.path.join(args.out_dir, f"{args.case_key}_sample_{i:03d}_grid.vtk")
        grid_i.save(grid_path)

        # threshold to get inside region then extract surface
        thr = float(args.occ_thr)
        inside = grid_i.threshold(value=thr, scalars="occ_prob")  # keep points with occ_prob >= thr
        surf = inside.extract_surface()

        surf_path = os.path.join(args.out_dir, f"{args.case_key}_sample_{i:03d}_surface.vtk")
        surf.save(surf_path)
        surface_paths.append(surf_path)

        print(f"[OK] sample {i}: saved {grid_path} and {surf_path} | occ_prob min={occ_prob_np.min():.4f}, max={occ_prob_np.max():.4f}")

    return visualize_generated_surfaces(
        surface_paths=surface_paths,
        out_dir=args.out_dir,
        case_key=args.case_key,
        direction=direction,
        component=args.viz_component,
        num_samples=args.viz_num,
        opacity=args.viz_opacity,
        cmap="jet",
        use_clip=True
    )


if __name__ == "__main__":
    main()