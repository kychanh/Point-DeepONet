# main.py (DesignGenPointDeepONet version)
# Drop-in replacement for the code you pasted.
# Changes vs original:
#  - Replaces DeepONetCartesianProd with DesignGenPointDeepONet (= Encoder + FieldDecoder + ShapeDecoder + NF prior)
#  - Trunk uses xyz-only for both decoders (NO sdf concatenation to trunk)
#  - Adds SDF prediction head (1 dim) and NF latent prior training
#  - Keeps preprocessing/data pipeline identical
#  - SDF GT comes from trunk_input[:, :, 3] (xyzd's d) ONLY for loss/metrics

import os
import sys
import time
import logging
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from functools import partial
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import pyvista as pv
import vtk
vtk.vtkObject.GlobalWarningDisplayOff()

os.environ["DDE_BACKEND"] = "pytorch"

import deepxde as dde
from deepxde.data.data import Data
from deepxde.data.sampler import BatchSampler

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = '12'


# ============================
# SIREN layer (unchanged)
# ============================
class SineDenseLayer(nn.Module):
    def __init__(self, in_features, out_features, w0=1.0, is_first=False):
        super(SineDenseLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w0 = w0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        if self.is_first:
            nn.init.uniform_(self.linear.weight, -1 / self.in_features, 1 / self.in_features)
        else:
            nn.init.uniform_(self.linear.weight, -np.sqrt(6 / self.in_features) / self.w0,
                             np.sqrt(6 / self.in_features) / self.w0)

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))


# ============================
# CLI args (unchanged)
# ============================
def parse_arguments():
    parser = argparse.ArgumentParser(description='Train DeepONet with PointNet branch for 3D point cloud data')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('-r', '--RUN', type=int, default=0, help='Run number')
    parser.add_argument('-bic', '--branch_condition_input_components', type=str, default='mlc', help='Branch input components')
    parser.add_argument('-tic', '--trunk_input_components', type=str, default='xyzd', help='Trunk input components')
    parser.add_argument('-oc', '--output_components', type=str, default='xyzs', help='Output components')
    parser.add_argument('-N_p', '--N_pt', type=int, default=5000, help='Number of points per sample')
    parser.add_argument('-i', '--N_iterations', type=int, default=40000, help='Number of training iterations')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-dir', '--dir_base_load_data', type=str, default='../data/sampled', help='Directory to load data from')
    parser.add_argument('-N_d', '--N_samples', type=int, default=3000, help='Number of samples')
    parser.add_argument('-sm', '--split_method', type=str, choices=['mass', 'random'], default='random', help='Data split method')
    parser.add_argument('--base_dir', type=str, default='../experiments', help='Base directory to save experiments')
    parser.add_argument('--branch_hidden_dim', type=int, default=128, help='Branch network hidden dimension')
    parser.add_argument('--trunk_hidden_dim', type=int, default=128, help='Trunk network hidden dimension')
    parser.add_argument('--trunk_encoding_hidden_dim', type=int, default=128, help='Trunk encoding hidden dimension (= latent dim)')
    parser.add_argument('--fc_hidden_dim', type=int, default=128, help='Fully connected hidden dimension')
    args = parser.parse_args()
    return args


def set_random_seed():
    seed = 2024
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(args, name):
    experiment_dir = os.path.join(args.base_dir, name)
    os.makedirs(experiment_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
        logging.FileHandler(f"{experiment_dir}/training_log.txt"),
        logging.StreamHandler(sys.stdout)
    ])
    return experiment_dir


def get_device(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log_parameters(args, device, name):
    logging.info('\n\nModel Parameters:')
    logging.info(f'Device: {device}')
    logging.info(f'Run Number: {args.RUN}')
    logging.info(f'Folder Name: {name}')
    logging.info(f'Iterations: {args.N_iterations}')
    logging.info(f'Batch Size: {args.batch_size}')
    logging.info(f'Learning Rate: {args.learning_rate}')
    logging.info(f'Number of Samples: {args.N_samples}')
    logging.info(f'Data Split Method: {args.split_method}')
    logging.info(f'Branch Condition Input Components: {args.branch_condition_input_components}')
    logging.info(f'Trunk Input Components: {args.trunk_input_components}')
    logging.info(f'Output Components: {args.output_components}')
    logging.info(f'Trunk Encoding Hidden Dimension: {args.trunk_encoding_hidden_dim}')
    logging.info(f'Branch Hidden Dimension: {args.branch_hidden_dim}')
    logging.info(f'Trunk Hidden Dimension: {args.trunk_hidden_dim}')
    logging.info(f'Fully Connected Hidden Dimension: {args.fc_hidden_dim}')


# ============================
# Preprocess helpers (unchanged)
# ============================
def process_branch_condition_input(input_components, tmp):
    input_mapping = {'m': 4, 'l': 5, 'c': [6, 7, 8]}
    selected_indices = []
    for comp in input_components:
        if comp == 'c':
            selected_indices.extend(input_mapping[comp])
        elif comp in input_mapping:
            selected_indices.append(input_mapping[comp])
    branch_condition_input = tmp['a'][:, 0, selected_indices].astype(np.float32)
    return branch_condition_input


def process_branch_pointnet_input(input_components, tmp):
    input_mapping = {'x': 0, 'y': 1, 'z': 2, 'd': 3, 'm': 4, 'l': 5, 'c': [6, 7, 8]}
    selected_indices = []
    for comp in input_components:
        if comp == 'c':
            selected_indices.extend(input_mapping[comp])
        elif comp in input_mapping:
            selected_indices.append(input_mapping[comp])
    branch_pointnet_input = tmp['a'][:, :, selected_indices].astype(np.float32)
    return branch_pointnet_input


def process_trunk_input(input_components, tmp):
    input_mapping = {'x': 0, 'y': 1, 'z': 2, 'd': 3, 'm': 4, 'l': 5, 'c': [6, 7, 8]}
    selected_indices = []
    for comp in input_components:
        if comp == 'c':
            selected_indices.extend(input_mapping[comp])
        elif comp in input_mapping:
            selected_indices.append(input_mapping[comp])
    trunk_input = tmp['a'][:, :, selected_indices].astype(np.float32)
    return trunk_input


def process_output(output_components, output_vals, identifiers):
    output_mapping = {'x': 0, 'y': 1, 'z': 2, 's': 3}
    selected_indices = [output_mapping[comp] for comp in output_components]
    combined_output = output_vals[:, :, selected_indices]

    unique_directions = set(id.split('_')[0] for id in identifiers)
    clipping_ranges_dict = {direction: get_clipping_ranges_for_direction(direction) for direction in unique_directions}

    for direction, clipping_ranges in clipping_ranges_dict.items():
        indices = [i for i, id in enumerate(identifiers) if id.split('_')[0] == direction]
        if not indices:
            continue
        indices = np.array(indices)
        for j, idx in enumerate(selected_indices):
            min_val, max_val = clipping_ranges[idx]
            combined_output[indices, :, j] = np.clip(combined_output[indices, :, j], min_val, max_val)

    if combined_output.shape[-1] == 1:
        combined_output = combined_output.squeeze(-1)

    return combined_output


def map_keys_to_indices(keys, all_keys):
    return [np.where(all_keys == key)[0][0] for key in keys]


def get_clipping_ranges_for_direction(direction):
    if direction == 'ver':
        return {
            0: (-0.068, 0.473),
            1: (-0.093, 0.073),
            2: (-0.003, 0.824),
            3: (0., 232.19),
        }
    elif direction == 'hor':
        return {
            0: (-0.421, 0.008),
            1: (-0.024, 0.029),
            2: (-0.388, 0.109),
            3: (0., 227.78),
        }
    elif direction == 'dia':
        return {
            0: (-0.079, 0.016),
            1: (-0.057, 0.056),
            2: (-0.006, 0.214),
            3: (0., 172.19),
        }
    else:
        raise ValueError(f"Unknown direction: {direction}")


def calculate_r2(true_vals, pred_vals):
    ss_res = np.sum((true_vals - pred_vals) ** 2)
    ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def load_and_preprocess_data(args, dir_base_save_model):
    tmp = np.load(f'{args.dir_base_load_data}/Rpt{str(0)}_N{str(args.N_pt)}.npz')

    branch_condition_input = process_branch_condition_input(
        args.branch_condition_input_components,
        tmp
    )
    branch_pointnet_input = process_branch_pointnet_input('xyz', tmp)
    trunk_input = process_trunk_input(
        args.trunk_input_components,
        tmp
    )
    combined_output = process_output(args.output_components, tmp['b'], tmp['c'])

    scaler_fun = partial(MinMaxScaler, feature_range=(-1, 1))

    branch_scaler = scaler_fun()
    branch_scaler.fit(branch_condition_input)
    branch_condition_input = branch_scaler.transform(branch_condition_input)

    pointnet_scaler = scaler_fun()
    ss = branch_pointnet_input.shape
    tmp_pointnet_input = branch_pointnet_input.reshape([ss[0] * ss[1], ss[2]])
    pointnet_scaler.fit(tmp_pointnet_input)
    branch_pointnet_input = pointnet_scaler.transform(tmp_pointnet_input).reshape(ss)

    trunk_scaler = scaler_fun()
    ss = trunk_input.shape
    tmp_input_trunk = trunk_input.reshape([ss[0] * ss[1], ss[2]])
    trunk_scaler.fit(tmp_input_trunk)
    trunk_input = trunk_scaler.transform(tmp_input_trunk).reshape(ss)

    output_scalers = scaler_fun()
    ss = combined_output.shape
    if combined_output.ndim == 3:
        tmp_output = combined_output.reshape([ss[0] * ss[1], ss[2]])
    elif combined_output.ndim == 2:
        tmp_output = combined_output.reshape([ss[0] * ss[1], 1])
    else:
        raise ValueError(f"Unexpected number of dimensions: {combined_output.ndim}")
    output_scalers.fit(tmp_output)
    combined_output = output_scalers.transform(tmp_output).reshape(ss)

    pickle.dump(branch_scaler, open(f'{dir_base_save_model}/branch_scaler.pkl', 'wb'))
    pickle.dump(pointnet_scaler, open(f'{dir_base_save_model}/pointnet_scaler.pkl', 'wb'))
    pickle.dump(trunk_scaler, open(f'{dir_base_save_model}/trunk_scaler.pkl', 'wb'))
    pickle.dump(output_scalers, open(f'{dir_base_save_model}/output_scaler.pkl', 'wb'))

    if args.split_method == 'random':
        split_file = f'../data/npy/combined_{args.N_samples}_split_random_train_valid.npz'
    else:
        split_file = f'../data/npy/combined_{args.N_samples}_split_mass_train_valid.npz'
    split_data = np.load(split_file)
    train_case = map_keys_to_indices(split_data['train'], np.array([key for key in tmp['c']]))
    test_case = map_keys_to_indices(split_data['valid'], np.array([key for key in tmp['c']]))

    train_case_file = tmp['c'][train_case]
    test_case_file = tmp['c'][test_case]

    branch_condition_input_train = branch_condition_input[train_case].astype(np.float32)
    branch_condition_input_test = branch_condition_input[test_case].astype(np.float32)
    branch_pointnet_input_train = branch_pointnet_input[train_case].astype(np.float32)
    branch_pointnet_input_test = branch_pointnet_input[test_case].astype(np.float32)
    trunk_input_train = trunk_input[train_case].astype(np.float32)
    trunk_input_test = trunk_input[test_case].astype(np.float32)
    s_train = combined_output[train_case].astype(np.float32)
    s_test = combined_output[test_case].astype(np.float32)

    if s_train.ndim == 3 and s_train.shape[-1] == 1:
        s_train = s_train.squeeze(-1)
    if s_test.ndim == 3 and s_test.shape[-1] == 1:
        s_test = s_test.squeeze(-1)

    logging.info(f'branch_condition_input_train.shape = {branch_condition_input_train.shape}')
    logging.info(f'branch_pointnet_input_train.shape = {branch_pointnet_input_train.shape}')
    logging.info(f'branch_condition_input_test.shape = {branch_condition_input_test.shape}')
    logging.info(f'branch_pointnet_input_test.shape = {branch_pointnet_input_test.shape}')
    logging.info(f'trunk_input_train.shape = {trunk_input_train.shape}')
    logging.info(f'trunk_input_test.shape = {trunk_input_test.shape}')
    logging.info(f's_train.shape = {s_train.shape}')
    logging.info(f's_test.shape = {s_test.shape}')

    x_train = (branch_condition_input_train, branch_pointnet_input_train, trunk_input_train)
    x_test  = (branch_condition_input_test,  branch_pointnet_input_test,  trunk_input_test)

    # ---- ADD: sdf target from trunk_input[...,3] (already scaled by trunk_scaler) ----
    d_train = trunk_input_train[:, :, 3:4].astype(np.float32)  # (Ns, Npt, 1)
    d_test  = trunk_input_test[:, :, 3:4].astype(np.float32)   # (Nt, Npt, 1)

    # y = [field(4), sdf(1)] -> (..,5)
    y_train = np.concatenate([s_train, d_train], axis=2).astype(np.float32)
    y_test  = np.concatenate([s_test,  d_test],  axis=2).astype(np.float32)
    # -------------------------------------------------------------------------------
    logging.info(f'y_train(with sdf).shape = {y_train.shape}')
    logging.info(f'y_test(with sdf).shape = {y_test.shape}')

    return x_train, y_train, x_test, y_test, output_scalers, train_case_file, test_case_file


# ============================
# Data wrapper (modified losses)
# ============================
class TripleCartesianProd(Data):
    def __init__(self, X_train, y_train, X_test, y_test, lambda_nf=1e-3):
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test
        self.num_train_samples = self.train_x[0].shape[0]
        self.sampler = BatchSampler(self.num_train_samples, shuffle=True)
        self.lambda_nf = lambda_nf

        self._step = 0
        self._log_every = 100
        self._alm_update_every = 25

        # Augmented-Lagrangian states for adaptive weighting
        self._alm_cfg = {
            "sdf": {"lambda": 0.0, "rho": 2.0},
            "zero": {"lambda": 0.0, "rho": 2.0},
            "eik": {"lambda": 0.0, "rho": 0.5},
            "conn": {"lambda": 0.0, "rho": 1.0},
            "curv": {"lambda": 0.0, "rho": 0.2},
        }

    def _alm_term(self, key, penalty):
        cfg = self._alm_cfg[key]
        return cfg["lambda"] * penalty + 0.5 * cfg["rho"] * penalty * penalty

    @torch.no_grad()
    def _alm_update(self, penalties):
        for key, val in penalties.items():
            cfg = self._alm_cfg[key]
            cfg["lambda"] = max(0.0, cfg["lambda"] + cfg["rho"] * float(val))

    def _connectivity_loss(self, xyz, sdf, sample_n=256, k=8, band=0.1, temp=0.05):
        # Surrogate connectivity regularizer:
        # near the zero-level set, neighboring points should have consistent sign transitions.
        B, N, _ = xyz.shape
        if N <= sample_n:
            xyz_s = xyz
            sdf_s = sdf
        else:
            idx = torch.randint(0, N, (B, sample_n), device=xyz.device)
            xyz_s = torch.gather(xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
            sdf_s = torch.gather(sdf, 1, idx)

        dist = torch.cdist(xyz_s, xyz_s)
        knn_idx = dist.topk(k=min(k + 1, xyz_s.shape[1]), largest=False).indices[:, :, 1:]
        neigh_sign = torch.gather(
            torch.tanh(sdf_s / temp).unsqueeze(1).expand(-1, xyz_s.shape[1], -1),
            2,
            knn_idx,
        )
        center_sign = torch.tanh(sdf_s / temp).unsqueeze(-1)
        consistency = (center_sign - neigh_sign).pow(2).mean(dim=-1)

        band_mask = (sdf_s.abs() < band).float()
        denom = band_mask.sum() + 1e-6
        return (consistency * band_mask).sum() / denom

    def _curvature_loss(self, model, latent, num_samples, device):
        x = (torch.rand(latent.shape[0], num_samples, 3, device=device) * 2.0 - 1.0).requires_grad_(True)
        x_enc = model.net.trunk_encoding(x.reshape(-1, 3)).reshape(latent.shape[0], num_samples, -1)
        sdf = model.net.shape_decoder(latent, x_enc).squeeze(-1)

        grad = torch.autograd.grad(
            outputs=sdf.sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        lap = 0.0
        for i in range(3):
            second = torch.autograd.grad(
                outputs=grad[..., i].sum(),
                inputs=x,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0][..., i]
            lap = lap + second
        return (lap ** 2).mean()

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        self._step += 1

        u_gt = targets[:, :, :4]
        d_gt = targets[:, :, 4]
        u_pred = outputs[:, :, :4]
        d_pred = outputs[:, :, 4]

        # Main objective
        loss_field = torch.mean((u_pred - u_gt) ** 2)

        # Constraint penalties
        # SDF supervised loss (requested): plain MSE
        p_sdf = torch.mean((d_pred - d_gt) ** 2)

        eps = 0.02
        mask = (torch.abs(d_gt) < eps)
        p_zero = (d_pred[mask] ** 2).mean() if mask.any() else 0.0 * d_pred.mean()

        latent = getattr(model.net, "_cached_latent", None)
        if latent is None:
            latent = model.net.encoder(inputs[1])

        # Eikonal constraint (GINN-style): random points in scaled domain, ||grad d||=1
        B = targets.shape[0]
        Ne = 2048
        x_eik = (torch.rand(B, Ne, 3, device=outputs.device) * 2.0 - 1.0).requires_grad_(True)
        x_eik_enc = model.net.trunk_encoding(x_eik.reshape(-1, 3)).reshape(B, Ne, -1)
        d_eik = model.net.shape_decoder(latent, x_eik_enc).squeeze(-1)
        grad = torch.autograd.grad(
            outputs=d_eik.sum(),
            inputs=x_eik,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        p_eik = ((grad.norm(dim=-1) - 1.0) ** 2).mean()

        xyz = inputs[2][:, :, :3]
        p_conn = self._connectivity_loss(xyz, d_pred, sample_n=256, k=8)
        p_curv = self._curvature_loss(model, latent, num_samples=128, device=outputs.device)

        penalties = {
            "sdf": p_sdf,
            "zero": p_zero,
            "eik": p_eik,
            "conn": p_conn,
            "curv": p_curv,
        }

        total = loss_field
        for key, p in penalties.items():
            total = total + self._alm_term(key, p)

        if self._step % self._alm_update_every == 0:
            self._alm_update({k: v.detach().item() for k, v in penalties.items()})

        if self._step % self._log_every == 0:
            terms = " ".join([
                f"{k}={float(v.detach().cpu()):.3e}(λ={self._alm_cfg[k]['lambda']:.2e},ρ={self._alm_cfg[k]['rho']:.2e})"
                for k, v in penalties.items()
            ])
            logging.info(
                f"[loss step {self._step}] total={float(total.detach().cpu()):.6e} "
                f"field={float(loss_field.detach().cpu()):.3e} {terms}"
            )

        return total

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        indices = self.sampler.get_next(batch_size)
        return (
            (
                self.train_x[0][indices],
                self.train_x[1][indices],
                self.train_x[2][indices]
            ),
            self.train_y[indices],
        )

    def test(self):
        return self.test_x, self.test_y


# ============================
# Encoder / FieldDecoder / ShapeDecoder / Full model
# ============================
class Encoder(nn.Module):
    """
    PointNet-like: (B, N, 3) -> (B, latent_dim)
    """
    def __init__(self, pointnet_input_dim, latent_dim, activation):
        super().__init__()
        self.activation = activation
        self.pointnet_branch = nn.Sequential(
            nn.Conv1d(pointnet_input_dim, 32, 1),
            nn.BatchNorm1d(32),
            self.activation,
            nn.Conv1d(32, 64, 1),
            nn.BatchNorm1d(64),
            self.activation,
            nn.Conv1d(64, latent_dim, 1),
            nn.BatchNorm1d(latent_dim),
            self.activation,
        )

    def forward(self, pointnet_input):
        x = pointnet_input.transpose(2, 1)      # (B, 3, N)
        x = self.pointnet_branch(x)             # (B, latent_dim, N)
        latent = x.max(dim=2)[0]                # (B, latent_dim)
        return latent


class FieldDecoder(nn.Module):
    """
    Keeps the "two connections" pattern from the original code:
      mix = (branch_output + latent) * trunk_xyz_encoded
      mix_reduced -> combined_branch
      trunk_net(trunk_xyz_encoded) -> basis -> einsum
    Trunk uses xyz-only encoding (NO sdf concatenation).
    """
    def __init__(self, branch_condition_input_dim, latent_dim, branch_hidden_dim,
                 trunk_encoding_output_dim, trunk_hidden_dim, fc_hidden_dim,
                 num_output_components, activation):
        super().__init__()
        self.activation = activation
        self.tanh = nn.Tanh()
        self.b = nn.Parameter(torch.zeros(num_output_components))

        # enforce shape consistency of original design
        assert branch_hidden_dim == latent_dim == trunk_encoding_output_dim == fc_hidden_dim, \
            "Require branch_hidden_dim == latent_dim == trunk_encoding_output_dim == fc_hidden_dim (for element-wise mix)."
        assert trunk_hidden_dim == trunk_encoding_output_dim, \
            "Require trunk_hidden_dim == trunk_encoding_output_dim so x_trunk view uses enc_dim cleanly."

        self.branch_net_global = nn.Sequential(
            nn.Linear(branch_condition_input_dim, branch_hidden_dim),
            self.activation,
            nn.Linear(branch_hidden_dim, branch_hidden_dim * 2),
            self.activation,
            nn.Linear(branch_hidden_dim * 2, branch_hidden_dim),
            self.activation,
        )

        # trunk_net now takes only trunk_xyz_encoded (enc_dim)
        self.trunk_net = nn.Sequential(
            nn.Linear(trunk_encoding_output_dim, trunk_hidden_dim),
            self.activation,
            nn.Linear(trunk_hidden_dim, trunk_hidden_dim * 2),
            self.activation,
            nn.Linear(trunk_hidden_dim * 2, trunk_hidden_dim * num_output_components),
            self.activation,
        )

        self.combined_branch = nn.Sequential(
            nn.Linear(branch_hidden_dim, fc_hidden_dim),
            self.activation,
            nn.Linear(fc_hidden_dim, fc_hidden_dim * 2),
            self.activation,
            nn.Linear(fc_hidden_dim * 2, fc_hidden_dim),
            self.activation,
        )

    def forward(self, branch_condition_input, latent, trunk_xyz_encoded):
        branch_output = self.branch_net_global(branch_condition_input)   # (B, enc_dim)
        combined_output = branch_output + latent                          # (B, enc_dim)

        mix = combined_output.unsqueeze(1) * trunk_xyz_encoded            # (B,N,enc_dim)
        mix_reduced = mix.mean(dim=1)                                     # (B,enc_dim)
        combined_output2 = self.combined_branch(mix_reduced)              # (B,enc_dim)

        B, N, enc_dim = trunk_xyz_encoded.shape
        x_trunk = self.trunk_net(trunk_xyz_encoded.reshape(B * N, -1)).view(B, N, -1)
        x_trunk = x_trunk.view(B, N, enc_dim, -1)                         # (B,N,enc_dim,num_out)

        output = torch.einsum("bh,bnhc->bnc", combined_output2, x_trunk) + self.b
        return self.tanh(output)


class ShapeDecoder(nn.Module):
    """
    Same two-connection structure, output dim=1 (SDF).
    Trunk uses xyz-only encoding (NO sdf concatenation).
    """
    def __init__(self, latent_dim, trunk_encoding_output_dim, trunk_hidden_dim, fc_hidden_dim, activation):
        super().__init__()
        self.activation = activation
        self.tanh = nn.Tanh()
        self.b = nn.Parameter(torch.zeros(1))

        assert latent_dim == trunk_encoding_output_dim == fc_hidden_dim, \
            "Require latent_dim == trunk_encoding_output_dim == fc_hidden_dim."
        assert trunk_hidden_dim == trunk_encoding_output_dim, \
            "Require trunk_hidden_dim == trunk_encoding_output_dim so x_trunk view uses enc_dim cleanly."

        self.trunk_net = nn.Sequential(
            nn.Linear(trunk_encoding_output_dim, trunk_hidden_dim),
            self.activation,
            nn.Linear(trunk_hidden_dim, trunk_hidden_dim * 2),
            self.activation,
            nn.Linear(trunk_hidden_dim * 2, trunk_hidden_dim * 1),
            self.activation,
        )

        self.combined_branch = nn.Sequential(
            nn.Linear(trunk_encoding_output_dim, fc_hidden_dim),
            self.activation,
            nn.Linear(fc_hidden_dim, fc_hidden_dim * 2),
            self.activation,
            nn.Linear(fc_hidden_dim * 2, fc_hidden_dim),
            self.activation,
        )

    def forward(self, latent, trunk_xyz_encoded):
        mix = latent.unsqueeze(1) * trunk_xyz_encoded       # (B,N,enc_dim)
        mix_reduced = mix.mean(dim=1)                       # (B,enc_dim)
        combined_output2 = self.combined_branch(mix_reduced)  # (B,enc_dim)

        B, N, enc_dim = trunk_xyz_encoded.shape
        x_trunk = self.trunk_net(trunk_xyz_encoded.reshape(B * N, -1)).view(B, N, -1)
        x_trunk = x_trunk.view(B, N, enc_dim, 1)            # (B,N,enc_dim,1)

        sdf = torch.einsum("bh,bnhc->bnc", combined_output2, x_trunk) + self.b
        return sdf  # (B,N,1)


class DesignGenPointDeepONet(dde.maps.NN, nn.Module):
    """
    Replaces DeepONetCartesianProd.
    Inputs unchanged: (branch_condition_input, pointnet_input, trunk_input)
    Outputs: (B,N,5) = field(4) + sdf(1)
    Adds NF prior on latent.
    """
    def __init__(self, branch_condition_input_dim, pointnet_input_dim, trunk_input_dim,
                 branch_hidden_dim=128, trunk_hidden_dim=128, fc_hidden_dim=128,
                 trunk_encoding_hidden_dim=128, regularization=None,
                 num_output_components=4,device_str="cuda:0"):
        super(DesignGenPointDeepONet, self).__init__()
        nn.Module.__init__(self)
        dde.maps.NN.__init__(self)

        self.activation = nn.SiLU()
        self.tanh = nn.Tanh()

        # Trunk encoding (xyz -> high-freq features) 그대로
        self.trunk_encoding = nn.Sequential(
            SineDenseLayer(in_features=3, out_features=trunk_encoding_hidden_dim, w0=10.0, is_first=True),
            SineDenseLayer(in_features=trunk_encoding_hidden_dim, out_features=trunk_encoding_hidden_dim * 2, w0=10.0, is_first=False),
            SineDenseLayer(in_features=trunk_encoding_hidden_dim * 2, out_features=trunk_encoding_hidden_dim, w0=10.0, is_first=False),
        )
        self.trunk_encoding_output_dim = trunk_encoding_hidden_dim

        # Encoder (PointNet) -> latent (B,128)
        self.encoder = Encoder(pointnet_input_dim=pointnet_input_dim,
                               latent_dim=trunk_encoding_hidden_dim,
                               activation=self.activation)

        # Decoders
        self.field_decoder = FieldDecoder(
            branch_condition_input_dim=branch_condition_input_dim,
            latent_dim=trunk_encoding_hidden_dim,
            branch_hidden_dim=branch_hidden_dim,
            trunk_encoding_output_dim=self.trunk_encoding_output_dim,
            trunk_hidden_dim=trunk_hidden_dim,
            fc_hidden_dim=fc_hidden_dim,
            num_output_components=num_output_components,
            activation=self.activation
        )

        self.shape_decoder = ShapeDecoder(
            latent_dim=trunk_encoding_hidden_dim,
            trunk_encoding_output_dim=self.trunk_encoding_output_dim,
            trunk_hidden_dim=trunk_hidden_dim,
            fc_hidden_dim=fc_hidden_dim,
            activation=self.activation
        )

        # cache for Data.losses
        self._cached_latent = None
        self.regularizer = regularization

    def forward(self, inputs):
        branch_condition_input, pointnet_input, trunk_input = inputs
        # trunk_input may be xyzd, but we only use xyz for the network
        trunk_xyz = trunk_input[:, :, :3]

        # Encoder -> latent
        latent = self.encoder(pointnet_input)  # (B,enc_dim)
        self._cached_latent = latent 

        # Trunk xyz encoding
        B, N, _ = trunk_xyz.shape
        trunk_xyz_encoded = self.trunk_encoding(trunk_xyz.reshape(-1, 3)).reshape(B, N, -1)

        # Decode field + sdf
        field_out = self.field_decoder(branch_condition_input, latent, trunk_xyz_encoded)  # (B,N,4)
        sdf_out = self.shape_decoder(latent, trunk_xyz_encoded)                            # (B,N,1)

        return torch.cat([field_out, sdf_out], dim=-1)                                     # (B,N,5)

    def parameters(self):
        # deepxde may call this; return torch params
        return list(super().parameters())


# ============================
# define_model (uses DesignGenPointDeepONet)
# ============================
def define_model(args, device, data, output_scalers):
    branch_condition_input_dim = data.train_x[0].shape[1]
    pointnet_input_dim = data.train_x[1].shape[-1]
    num_points = args.N_pt
    trunk_input_dim = data.train_x[2].shape[-1]
    num_output_components = 4

    # Important: to match the diagrams and the elementwise-mix structure
    # keep dims equal (recommended args already default to 128)
    assert args.branch_hidden_dim == args.trunk_encoding_hidden_dim == args.fc_hidden_dim, \
        "Set branch_hidden_dim == trunk_encoding_hidden_dim == fc_hidden_dim (e.g., all 128)."
    assert args.trunk_hidden_dim == args.trunk_encoding_hidden_dim, \
        "Set trunk_hidden_dim == trunk_encoding_hidden_dim (e.g., all 128)."

    model = DesignGenPointDeepONet(
        branch_condition_input_dim=branch_condition_input_dim,
        pointnet_input_dim=pointnet_input_dim,
        trunk_input_dim=trunk_input_dim,
        branch_hidden_dim=args.branch_hidden_dim,
        trunk_hidden_dim=args.trunk_hidden_dim,
        fc_hidden_dim=args.fc_hidden_dim,
        trunk_encoding_hidden_dim=args.trunk_encoding_hidden_dim,
        regularization=None,
        num_output_components=num_output_components,
        device_str=str(device)
    ).to(device)

    # Dummy loss; actual is computed in Data.losses (we keep this to satisfy deepxde)
    def loss_func(outputs, targets):
        # outputs: (B,N,5), targets: (B,N,5)
        return torch.mean((outputs - targets) ** 2)

    # -------- metrics (field only; slice outputs[:,:,:4]) --------
    def err_MAE(true_vals, pred_vals):
        return np.mean(np.abs(true_vals - pred_vals), axis=1)

    def err_RMSE(true_vals, pred_vals):
        return np.sqrt(np.mean((true_vals - pred_vals) ** 2, axis=1))

    def to_numpy(d):
        if isinstance(d, torch.Tensor):
            return d.detach().cpu().numpy()
        elif isinstance(d, np.ndarray):
            return d
        else:
            raise TypeError('Data must be a torch.Tensor or np.ndarray.')

    def inv(data_arr, scaler):
        data_shape = data_arr.shape
        data_flat = data_arr.reshape(-1, data_shape[-1])
        data_inv = scaler.inverse_transform(data_flat)
        return data_inv.reshape(data_shape)

    def ux_MAE(outputs, targets):
        outputs_np = to_numpy(outputs)[:, :, :4]
        targets_np = to_numpy(targets)[:, :, :4]
        true_vals = inv(targets_np, output_scalers)[:, :, 0]
        pred_vals = inv(outputs_np, output_scalers)[:, :, 0]
        return np.mean(err_MAE(true_vals, pred_vals))

    def uy_MAE(outputs, targets):
        outputs_np = to_numpy(outputs)[:, :, :4]
        targets_np = to_numpy(targets)[:, :, :4]
        true_vals = inv(targets_np, output_scalers)[:, :, 1]
        pred_vals = inv(outputs_np, output_scalers)[:, :, 1]
        return np.mean(err_MAE(true_vals, pred_vals))

    def uz_MAE(outputs, targets):
        outputs_np = to_numpy(outputs)[:, :, :4]
        targets_np = to_numpy(targets)[:, :, :4]
        true_vals = inv(targets_np, output_scalers)[:, :, 2]
        pred_vals = inv(outputs_np, output_scalers)[:, :, 2]
        return np.mean(err_MAE(true_vals, pred_vals))

    def vm_MAE(outputs, targets):
        outputs_np = to_numpy(outputs)[:, :, :4]
        targets_np = to_numpy(targets)[:, :, :4]
        true_vals = inv(targets_np, output_scalers)[:, :, 3]
        pred_vals = inv(outputs_np, output_scalers)[:, :, 3]
        return np.mean(err_MAE(true_vals, pred_vals))

    def ux_R2(outputs, targets):
        outputs_np = to_numpy(outputs)[:, :, :4]
        targets_np = to_numpy(targets)[:, :, :4]
        true_vals = inv(targets_np, output_scalers)[:, :, 0]
        pred_vals = inv(outputs_np, output_scalers)[:, :, 0]
        return calculate_r2(true_vals, pred_vals)

    def uy_R2(outputs, targets):
        outputs_np = to_numpy(outputs)[:, :, :4]
        targets_np = to_numpy(targets)[:, :, :4]
        true_vals = inv(targets_np, output_scalers)[:, :, 1]
        pred_vals = inv(outputs_np, output_scalers)[:, :, 1]
        return calculate_r2(true_vals, pred_vals)

    def uz_R2(outputs, targets):
        outputs_np = to_numpy(outputs)[:, :, :4]
        targets_np = to_numpy(targets)[:, :, :4]
        true_vals = inv(targets_np, output_scalers)[:, :, 2]
        pred_vals = inv(outputs_np, output_scalers)[:, :, 2]
        return calculate_r2(true_vals, pred_vals)

    def vm_R2(outputs, targets):
        outputs_np = to_numpy(outputs)[:, :, :4]
        targets_np = to_numpy(targets)[:, :, :4]
        true_vals = inv(targets_np, output_scalers)[:, :, 3]
        pred_vals = inv(outputs_np, output_scalers)[:, :, 3]
        return calculate_r2(true_vals, pred_vals)

    if args.output_components == 'xyzs':
        metrics = [ux_MAE, uy_MAE, uz_MAE, vm_MAE, ux_R2, uy_R2, uz_R2, vm_R2]
    else:
        metrics = []

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=500, verbose=True)

    model = dde.Model(data, model)
    model.compile(
        optimizer=optimizer,
        loss=loss_func,
        decay=("inverse time", 1, args.learning_rate / 10.),
        metrics=metrics,
    )

    return model, scheduler, optimizer


# ============================
# Training (unchanged)
# ============================
def train_model(model, scheduler, optimizer, args, output_scalers, experiment_dir, device):
    callbacks = []
    torch.autograd.set_detect_anomaly(True)
    logging.info(f'y_train.shape = {model.data.train_y.shape}')
    logging.info(f'y_test.shape = {model.data.test_y.shape}')

    start_time = time.time()
    losshistory, train_state = model.train(
        iterations=args.N_iterations,
        batch_size=args.batch_size,
        display_every=100,
        model_save_path=f"{experiment_dir}/model_checkpoint.pth",
        callbacks=callbacks
    )
    total_training_time = time.time() - start_time

    scheduler.step(float(np.sum(losshistory.loss_test[-1])))
    torch.save(model.net.state_dict(), f'{experiment_dir}/model_final.pth')

    total_params = sum(p.numel() for p in model.net.parameters() if p.requires_grad)
    with open(f'{experiment_dir}/total_params.txt', 'w') as f:
        f.write(f"{total_params}")

    with open(os.path.join(experiment_dir, 'training_time.txt'), 'w') as f:
        f.write(f"{total_training_time:.2f}")

    np.save(f'{experiment_dir}/train_losses.npy', np.array(losshistory.loss_train))
    np.save(f'{experiment_dir}/val_losses.npy', np.array(losshistory.loss_test))

    return losshistory, train_state


def plot_loss_curves(losshistory, experiment_dir):
    plt.figure()
    plt.plot(losshistory.loss_train, label='Train Loss')
    plt.plot(losshistory.loss_test, label='Validation Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss over Iterations')
    plt.savefig(f'{experiment_dir}/loss_curve.jpg', dpi=300)
    plt.close()


def plot_r2_scatter(true_vals, pred_vals, comp, dataset_type, experiment_dir):
    r2 = calculate_r2(true_vals, pred_vals)
    plt.figure(figsize=(6, 6))
    plt.scatter(true_vals, pred_vals, alpha=0.3, s=10, label='Data Points')
    min_val = min(true_vals.min(), pred_vals.min())
    max_val = max(true_vals.max(), pred_vals.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Fit')
    plt.xlabel(f'True {comp}')
    plt.ylabel(f'Predicted {comp}')
    plt.title(f'{dataset_type.capitalize()} Dataset: {comp} Prediction vs True\nR² = {r2:.3f}')
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, f'{dataset_type}_scatter_{comp}.jpg'), dpi=300)
    plt.close()


# ============================
# Main
# ============================
def main():
    args = parse_arguments()
    set_random_seed()
    device = get_device(args.gpu)

    name = f'PlusGINN_RUN{args.RUN}_D{args.N_samples}_N{args.N_pt}_branchinput_{args.branch_condition_input_components}_trunkinput_{args.trunk_input_components}_output_{args.output_components}_split_{args.split_method}'
    experiment_dir = setup_logging(args, name)
    model_best_path = os.path.join(experiment_dir, 'best_model.pth')
    if os.path.exists(model_best_path):
        logging.info(f'Best model already saved at {model_best_path}. Skipping training.')
        return

    log_parameters(args, device, name)

    x_train, y_train, x_test, y_test, output_scalers, train_case_file, test_case_file = load_and_preprocess_data(args, experiment_dir)

    # Data wrapper with NF weight
    data = TripleCartesianProd(x_train, y_train, x_test, y_test)

    model, scheduler, optimizer = define_model(args, device, data, output_scalers)

    logging.info(f'\nModel Structure:\n{model}')
    total_params = sum(p.numel() for p in model.net.parameters() if p.requires_grad)
    logging.info(f'Number of Trainable Parameters: {total_params}\n')

    losshistory, train_state = train_model(model, scheduler, optimizer, args, output_scalers, experiment_dir, device)
    plot_loss_curves(losshistory, experiment_dir)

    # ============================
    # Post-eval (field)
    # ============================
    test_outputs = model.predict(model.data.test_x)  # (Nt, N, 5)
    test_targets = model.data.test_y                 # (Nt, N, 5)

    def to_numpy(d):
        if isinstance(d, torch.Tensor):
            return d.detach().cpu().numpy()
        elif isinstance(d, np.ndarray):
            return d
        else:
            raise TypeError('Data must be a torch.Tensor or np.ndarray.')

    def inv(data_arr, scaler):
        data_shape = data_arr.shape
        data_flat = data_arr.reshape(-1, data_shape[-1])
        data_inv = scaler.inverse_transform(data_flat)
        return data_inv.reshape(data_shape)

    # field only inverse
    test_outputs_field = test_outputs[:, :, :4]
    test_targets_field = test_targets[:, :, :4]
    test_outputs_inv = inv(test_outputs_field, output_scalers)
    test_targets_inv = inv(test_targets_field, output_scalers)

    components = ['ux', 'uy', 'uz', 'vm']
    directions = [tcf.split('_')[0] for tcf in test_case_file]
    unique_directions = list(set(directions))

    subset_metrics_all_dir = os.path.join(experiment_dir, 'subset_metrics_all')
    os.makedirs(subset_metrics_all_dir, exist_ok=True)
    subset_plots_dir = os.path.join(experiment_dir, 'subset_plots')
    os.makedirs(subset_plots_dir, exist_ok=True)

    for direction in unique_directions:
        indices = [i for i, dir_label in enumerate(directions) if dir_label == direction]
        if not indices:
            logging.warning(f"No test samples found for direction: {direction}")
            continue
        true_vals_dir = test_targets_inv[indices, :, :]
        pred_vals_dir = test_outputs_inv[indices, :, :]
        for i, comp in enumerate(components):
            true_comp = true_vals_dir[:, :, i].flatten()
            pred_comp = pred_vals_dir[:, :, i].flatten()
            mae = np.mean(np.abs(pred_comp - true_comp))
            rmse = np.sqrt(np.mean((pred_comp - true_comp) ** 2))
            r2 = calculate_r2(true_comp, pred_comp)
            with open(os.path.join(subset_metrics_all_dir, f'{direction}_{comp}_MAE.txt'), 'w') as f_mae:
                f_mae.write(f"{mae}")
            with open(os.path.join(subset_metrics_all_dir, f'{direction}_{comp}_RMSE.txt'), 'w') as f_rmse:
                f_rmse.write(f"{rmse}")
            with open(os.path.join(subset_metrics_all_dir, f'{direction}_{comp}_R2.txt'), 'w') as f_r2:
                f_r2.write(f"{r2:.3f}")
            logging.info(f"subset Metrics for {direction} {comp}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
            plot_r2_scatter(true_comp, pred_comp, comp, f'test_{direction}_{comp}', subset_plots_dir)

    # ============================
    # Shape (SDF) metrics
    # ============================
    trunk_scaler = pickle.load(open(f'{experiment_dir}/trunk_scaler.pkl', 'rb'))

    sdf_pred_scaled = test_outputs[:, :, 4]            # (Nt, N)
    trunk_input_test_scaled = model.data.test_x[2]     # (Nt, N, 4=xyzd)
    d_gt_scaled = trunk_input_test_scaled[:, :, 3]     # (Nt, N)

    def inv_1d(vals_scaled, scaler, col_index):
        Nt, N = vals_scaled.shape
        tmp = np.zeros((Nt * N, scaler.n_features_in_), dtype=np.float32)
        tmp[:, col_index] = vals_scaled.reshape(-1)
        tmp_inv = scaler.inverse_transform(tmp)
        return tmp_inv[:, col_index].reshape(Nt, N)

    d_gt = inv_1d(d_gt_scaled, trunk_scaler, col_index=3)
    d_pred = inv_1d(sdf_pred_scaled, trunk_scaler, col_index=3)

    sdf_mae = np.mean(np.abs(d_pred - d_gt))
    sdf_rmse = np.sqrt(np.mean((d_pred - d_gt) ** 2))
    sdf_r2 = calculate_r2(d_gt.flatten(), d_pred.flatten())
    logging.info(f"SDF Overall: MAE={sdf_mae:.6f}, RMSE={sdf_rmse:.6f}, R²={sdf_r2:.4f}")

    shape_metrics_dir = os.path.join(experiment_dir, 'shape_metrics')
    os.makedirs(shape_metrics_dir, exist_ok=True)
    with open(os.path.join(shape_metrics_dir, 'sdf_MAE.txt'), 'w') as f:
        f.write(str(sdf_mae))
    with open(os.path.join(shape_metrics_dir, 'sdf_RMSE.txt'), 'w') as f:
        f.write(str(sdf_rmse))
    with open(os.path.join(shape_metrics_dir, 'sdf_R2.txt'), 'w') as f:
        f.write(f"{sdf_r2:.4f}")

    # ============================
    # The rest of your original long "overall VTK & per-case" loop
    # Only modifications: slice y_pred[:, :, :4] before inv()
    # ============================
    overall_metrics_all_dir = os.path.join(experiment_dir, 'overall_metrics_all')
    os.makedirs(overall_metrics_all_dir, exist_ok=True)
    overall_metrics_each_dir = os.path.join(experiment_dir, 'overall_metrics_each')
    os.makedirs(overall_metrics_each_dir, exist_ok=True)
    overall_plots_dir = os.path.join(experiment_dir, 'overall_plots')
    os.makedirs(overall_plots_dir, exist_ok=True)

    base = '/workspace/Point-DeepONet/data/npy/'
    targets_full = np.load(base + 'targets.npz', allow_pickle=True)
    xyzd_full = np.load(base + 'xyzdmlc.npz', allow_pickle=True)

    branch_scaler = pickle.load(open(f'{experiment_dir}/branch_scaler.pkl', 'rb'))
    pointnet_scaler = pickle.load(open(f'{experiment_dir}/pointnet_scaler.pkl', 'rb'))
    trunk_scaler = pickle.load(open(f'{experiment_dir}/trunk_scaler.pkl', 'rb'))
    output_scalers = pickle.load(open(f'{experiment_dir}/output_scaler.pkl', 'rb'))

    overall_vtk_dir = os.path.join(experiment_dir, 'overall_vtks')
    os.makedirs(overall_vtk_dir, exist_ok=True)

    directions = [tcf.split('_')[0] if '_' in tcf else 'unknown' for tcf in test_case_file]
    unique_directions = list(set(directions))
    components = ['ux', 'uy', 'uz', 'vm']
    direction_metrics = {direction: {} for direction in directions}
    all_metrics = {direction: {comp: {'true': [], 'pred': []} for comp in components} for direction in unique_directions}

    def inv_field(data_arr, scaler):
        # data_arr: (...,4)
        data_shape = data_arr.shape
        data_flat = data_arr.reshape(-1, data_shape[-1])
        data_inv = scaler.inverse_transform(data_flat)
        return data_inv.reshape(data_shape)

    for idx, tcf in tqdm(enumerate(test_case_file), total=len(test_case_file), desc="Processing Test Cases"):
        branch_condition = xyzd_full[tcf][0, 4:9]
        branch_pointnet = xyzd_full[tcf][:, :3]
        trunk_input_case = xyzd_full[tcf][:, :4]
        outputs = targets_full[tcf]

        total_points = branch_pointnet.shape[0]
        sampled_indices = np.random.choice(total_points, 5000, replace=False)
        branch_pointnet_sampled = branch_pointnet[sampled_indices]

        branch_condition_scaled = branch_scaler.transform(branch_condition[np.newaxis, :])
        branch_pointnet_scaled = pointnet_scaler.transform(branch_pointnet_sampled)
        branch_pointnet_scaled = branch_pointnet_scaled[np.newaxis, :, :]
        trunk_input_scaled = trunk_scaler.transform(trunk_input_case)
        trunk_input_scaled = trunk_input_scaled[np.newaxis, :, :]

        x_test_sample = (
            branch_condition_scaled,
            branch_pointnet_scaled,
            trunk_input_scaled
        )

        y_pred = model.predict(x_test_sample)            # (1,N,5)
        y_pred_field = y_pred[:, :, :4]                 # slice field
        u_pred = inv_field(y_pred_field, output_scalers) # (1,N,4)
        u_pred = u_pred.squeeze(0)

        direction = directions[idx]
        clipping_ranges_gt = get_clipping_ranges_for_direction(direction)
        for i in range(outputs.shape[1]):
            min_val, max_val = clipping_ranges_gt.get(i, (-np.inf, np.inf))
            outputs[:, i] = np.clip(outputs[:, i], min_val, max_val)

        if '_' in tcf:
            filename = tcf.split('_', 1)[1]
        else:
            filename = tcf

        if filename not in direction_metrics[direction]:
            direction_metrics[direction][filename] = {}

        for i, comp in enumerate(components):
            mae = np.mean(np.abs(u_pred[:, i] - outputs[:, i]))
            rmse = np.sqrt(np.mean((u_pred[:, i] - outputs[:, i]) ** 2))
            r2 = calculate_r2(outputs[:, i], u_pred[:, i])
            direction_metrics[direction][filename][comp] = {'mae': mae, 'rmse': rmse, 'r2': r2}

        for i, comp in enumerate(components):
            all_metrics[direction][comp]['true'].extend(outputs[:, i])
            all_metrics[direction][comp]['pred'].extend(u_pred[:, i])

        vtk_filepath = f'/workspace/Point-DeepONet/data/VolumeMesh/{filename}.vtk'
        if os.path.exists(vtk_filepath):
            try:
                vtk_obj = pv.read(vtk_filepath)
                new_rows = np.zeros((5, outputs.shape[1]))
                outputs_pad = np.vstack([outputs, new_rows])
                u_pred_pad = np.vstack([u_pred, new_rows])
                vtk_obj['gt_ux'] = outputs_pad[:, 0].astype(np.float32)
                vtk_obj['gt_uy'] = outputs_pad[:, 1].astype(np.float32)
                vtk_obj['gt_uz'] = outputs_pad[:, 2].astype(np.float32)
                vtk_obj['gt_vm'] = outputs_pad[:, 3].astype(np.float32)
                vtk_obj['pred_ux'] = u_pred_pad[:, 0].astype(np.float32)
                vtk_obj['pred_uy'] = u_pred_pad[:, 1].astype(np.float32)
                vtk_obj['pred_uz'] = u_pred_pad[:, 2].astype(np.float32)
                vtk_obj['pred_vm'] = u_pred_pad[:, 3].astype(np.float32)
                vtk_save_path = os.path.join(overall_vtk_dir, f'{direction}_{filename}.vtk')
                vtk_obj.save(vtk_save_path)
            except Exception as e:
                logging.error(f"Error processing VTK file for '{tcf}': {e}")
        else:
            logging.warning(f"VTK file does not exist: {vtk_filepath}")

    for direction, metrics in direction_metrics.items():
        npz_path = os.path.join(overall_metrics_each_dir, f'{direction}_metrics.npz')
        np.savez_compressed(npz_path, metrics=metrics)
        logging.info(f"Saved metrics for direction '{direction}' to '{npz_path}'")

    for direction, metrics in direction_metrics.items():
        for comp in components:
            mae_list = [test_case_metrics[comp]['mae'] for test_case_metrics in metrics.values()]
            rmse_list = [test_case_metrics[comp]['rmse'] for test_case_metrics in metrics.values()]
            r2_list = [test_case_metrics[comp]['r2'] for test_case_metrics in metrics.values()]
            avg_mae = np.mean(mae_list)
            avg_rmse = np.mean(rmse_list)
            avg_r2 = np.mean(r2_list)
            with open(os.path.join(overall_metrics_all_dir, f'{direction}_{comp}_MAE.txt'), 'w') as f_mae:
                f_mae.write(f"{avg_mae}")
            with open(os.path.join(overall_metrics_all_dir, f'{direction}_{comp}_RMSE.txt'), 'w') as f_rmse:
                f_rmse.write(f"{avg_rmse}")
            with open(os.path.join(overall_metrics_all_dir, f'{direction}_{comp}_R2.txt'), 'w') as f_r2:
                f_r2.write(f"{avg_r2:.3f}")
            logging.info(f"Overall Metrics for {direction} {comp}: MAE={avg_mae:.4f}, RMSE={avg_rmse:.4f}, R²={avg_r2:.4f}")

    for direction, comps in all_metrics.items():
        break
        for comp, values in comps.items():
            true_vals = np.array(values['true'])
            pred_vals = np.array(values['pred'])
            r2 = calculate_r2(true_vals, pred_vals)
            plt.figure(figsize=(6, 6))
            plt.scatter(true_vals, pred_vals, alpha=0.3, s=10, label='Data Points')
            min_val = min(true_vals.min(), pred_vals.min())
            max_val = max(true_vals.max(), pred_vals.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Fit')
            plt.xlabel(f'True {comp}')
            plt.ylabel(f'Predicted {comp}')
            plt.title(f'{direction.capitalize()} Direction: {comp} Prediction vs True\nR² = {r2:.3f}')
            plt.legend(loc="upper left")
            plt.tight_layout()
            plt.savefig(os.path.join(overall_plots_dir, f'{direction}_{comp}_scatter.jpg'), dpi=300)
            plt.close()


if __name__ == "__main__":
    main()