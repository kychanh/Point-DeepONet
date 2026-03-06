# main.py (DesignGenPointDeepONet - 2-phase training)
# Phase 1: field pretrain on inside-only samples (Rpt*.npz)
# Phase 2: shape finetune on extended samples with outside band (Ext_Rpt*.npz)
#
# Key ideas:
#  - Keep model output = [field(4), occ_logit(1)] => (B,N,5)
#  - For Ext_Rpt: b has NaNs on outside points. Field loss uses inside mask only.
#  - Phase 1 trains encoder+field_decoder (optionally trunk_encoding), shape_decoder frozen.
#  - Phase 2 trains shape_decoder (optionally trunk_encoding), encoder+field_decoder frozen.
#  - Occupancy supervision comes from inside/outside mask derived from b finite-ness.
#
# NOTE: This file intentionally omits the huge VTK post-processing loop to keep it stable.
#       You can add your VTK export back once training is stable.

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

os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
from deepxde.data.data import Data
from deepxde.data.sampler import BatchSampler

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = '12'


# =========================
# SIREN layer
# =========================
class SineDenseLayer(nn.Module):
    def __init__(self, in_features, out_features, w0=1.0, is_first=False):
        super().__init__()
        self.in_features = in_features
        self.w0 = w0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        if self.is_first:
            nn.init.uniform_(self.linear.weight, -1 / self.in_features, 1 / self.in_features)
        else:
            nn.init.uniform_(
                self.linear.weight,
                -np.sqrt(6 / self.in_features) / self.w0,
                np.sqrt(6 / self.in_features) / self.w0,
            )

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))


# =========================
# CLI args
# =========================
def parse_arguments():
    parser = argparse.ArgumentParser(description='2-phase training: field then shape.')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-r', '--RUN', type=int, default=0)
    parser.add_argument('-bic', '--branch_condition_input_components', type=str, default='mlc')
    parser.add_argument('-tic', '--trunk_input_components', type=str, default='xyzd')
    parser.add_argument('-oc', '--output_components', type=str, default='xyzs')

    parser.add_argument('-N_p', '--N_pt', type=int, default=1000)
    parser.add_argument('-i', '--N_iterations', type=int, default=40000)
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)

    parser.add_argument('-N_d', '--N_samples', type=int, default=3000)
    parser.add_argument('-sm', '--split_method', type=str, choices=['mass', 'random'], default='random')
    parser.add_argument('--base_dir', type=str, default='../experiments')

    parser.add_argument('--branch_hidden_dim', type=int, default=128)
    parser.add_argument('--trunk_hidden_dim', type=int, default=128)
    parser.add_argument('--trunk_encoding_hidden_dim', type=int, default=128)
    parser.add_argument('--fc_hidden_dim', type=int, default=128)

    # 2-phase switches
    parser.add_argument('--phase', type=str, choices=['field', 'shape'], default='field',
                        help='field: pretrain field on inside-only | shape: train shape on ext data')
    parser.add_argument('--pretrained_path', type=str, default='',
                        help='Required for phase=shape. Path to Phase-1 weights .pth')

    # data roots (auto set by phase if left default)
    parser.add_argument('--dir_sampled_field', type=str, default='../data/sampled',
                        help='Directory containing Rpt*.npz (inside-only)')
    parser.add_argument('--dir_sampled_ext', type=str, default='../data/sampled',
                        help='Directory containing Ext_Rpt*.npz (inside+outside with NaNs in b)')

    # loss weights
    parser.add_argument('--lambda_sdf', type=float, default=1.0)   # kept name for minimal changes; used as lambda_occ
    parser.add_argument('--lambda_zero', type=float, default=5.0)  # unused in occupancy mode
    parser.add_argument('--lambda_eik', type=float, default=0.05)  # unused in occupancy mode
    parser.add_argument('--eps0', type=float, default=0.02)        # unused in occupancy mode

    # freeze trunk encoding during shape phase (recommended initially)
    parser.add_argument('--freeze_trunk_encoding', action='store_true', default=True)

    return parser.parse_args()


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
    logging.info(f'Phase: {args.phase}')
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
    if args.phase == 'shape':
        logging.info(f'Pretrained Path: {args.pretrained_path}')
        logging.info(f'freeze_trunk_encoding: {args.freeze_trunk_encoding}')
        logging.info(f'lambda_occ(using lambda_sdf arg)={args.lambda_sdf}')


# =========================
# Preprocess helpers
# =========================
def process_branch_condition_input(input_components, tmp):
    input_mapping = {'m': 4, 'l': 5, 'c': [6, 7, 8]}
    selected_indices = []
    for comp in input_components:
        if comp == 'c':
            selected_indices.extend(input_mapping[comp])
        else:
            selected_indices.append(input_mapping[comp])
    return tmp['a'][:, 0, selected_indices].astype(np.float32)


def process_trunk_input(input_components, tmp):
    input_mapping = {'x': 0, 'y': 1, 'z': 2, 'd': 3, 'm': 4, 'l': 5, 'c': [6, 7, 8]}
    selected_indices = []
    for comp in input_components:
        if comp == 'c':
            selected_indices.extend(input_mapping[comp])
        else:
            selected_indices.append(input_mapping[comp])
    return tmp['a'][:, :, selected_indices].astype(np.float32)


def get_clipping_ranges_for_direction(direction):
    if direction == 'ver':
        return {0: (-0.068, 0.473), 1: (-0.093, 0.073), 2: (-0.003, 0.824), 3: (0., 232.19)}
    if direction == 'hor':
        return {0: (-0.421, 0.008), 1: (-0.024, 0.029), 2: (-0.388, 0.109), 3: (0., 227.78)}
    if direction == 'dia':
        return {0: (-0.079, 0.016), 1: (-0.057, 0.056), 2: (-0.006, 0.214), 3: (0., 172.19)}
    raise ValueError(direction)


def calculate_r2(true_vals, pred_vals):
    ss_res = np.sum((true_vals - pred_vals) ** 2)
    ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-12)


def map_keys_to_indices(keys, all_keys):
    return [np.where(all_keys == key)[0][0] for key in keys]


def process_output_nanaware(output_components, output_vals, identifiers):
    """
    output_vals: (N_case, N_pt, 4) may contain NaNs (Ext data).
    Clip finite entries only.
    """
    output_mapping = {'x': 0, 'y': 1, 'z': 2, 's': 3}
    selected_indices = [output_mapping[c] for c in output_components]
    out = output_vals[:, :, selected_indices].copy()

    unique_directions = set(i.split('_')[0] for i in identifiers)
    for direction in unique_directions:
        clip = get_clipping_ranges_for_direction(direction)
        idxs = [i for i, cid in enumerate(identifiers) if cid.split('_')[0] == direction]
        if not idxs:
            continue
        idxs = np.array(idxs)
        for j, comp_idx in enumerate(selected_indices):
            mn, mx = clip[comp_idx]
            block = out[idxs, :, j]
            m = np.isfinite(block)
            block[m] = np.clip(block[m], mn, mx)
            out[idxs, :, j] = block
    return out


def fit_minmax_scaler_nan_featurewise(X, feature_range=(-1, 1)):
    """
    Feature-wise min/max ignoring NaNs.
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    X2 = X.reshape(-1, X.shape[-1])
    scaler.feature_range = feature_range
    scaler.data_min_ = np.nanmin(X2, axis=0)
    scaler.data_max_ = np.nanmax(X2, axis=0)
    scaler.data_range_ = scaler.data_max_ - scaler.data_min_
    lo, hi = feature_range
    scaler.scale_ = (hi - lo) / (scaler.data_range_ + 1e-12)
    scaler.min_ = lo - scaler.data_min_ * scaler.scale_
    scaler.n_features_in_ = X2.shape[1]
    return scaler


def transform_minmax_nan_featurewise(scaler, X):
    """
    Transform finite entries feature-wise, keep NaNs.
    """
    lo, hi = scaler.feature_range
    X2 = X.reshape(-1, X.shape[-1]).copy()
    for j in range(X2.shape[1]):
        col = X2[:, j]
        m = np.isfinite(col)
        if m.any():
            col[m] = (col[m] - scaler.data_min_[j]) * scaler.scale_[j] + lo
        X2[:, j] = col
    return X2.reshape(X.shape)


def process_branch_pointnet_input_inside_xyz(tmp, N_pt):
    """
    For Ext data: encoder input uses INSIDE xyz only (where b is finite).
    For inside-only Rpt data: all points are inside, mask selects all anyway.
    """
    a = tmp['a']  # (N_case,N,9)
    b = tmp['b']  # (N_case,N,4) maybe NaN
    N_case = a.shape[0]
    out = np.zeros((N_case, N_pt, 3), dtype=np.float32)
    for i in range(N_case):
        mask_in = np.isfinite(b[i]).all(axis=1)
        xyz = a[i, mask_in, :3]
        if xyz.shape[0] == 0:
            xyz = a[i, :, :3]
        replace = xyz.shape[0] < N_pt
        idx = np.random.choice(xyz.shape[0], N_pt, replace=replace)
        out[i] = xyz[idx].astype(np.float32)
    return out


# =========================
# Data loading (phase-aware)
# =========================
def load_and_preprocess_data(args, experiment_dir):
    if args.phase == 'field':
        # inside-only file
        tmp = np.load(f"{args.dir_sampled_field}/Rpt0_N{args.N_pt}.npz")
    else:
        # extended file
        tmp = np.load(f"{args.dir_sampled_ext}/Ext_Rpt0_N{args.N_pt}.npz")

    # inputs
    branch_condition_input = process_branch_condition_input(args.branch_condition_input_components, tmp)
    branch_pointnet_input = process_branch_pointnet_input_inside_xyz(tmp, args.N_pt)  # inside xyz only
    trunk_input = process_trunk_input(args.trunk_input_components, tmp)  # (N_case,N,4=xyzd)

    # outputs
    # For field phase, tmp['b'] has no NaNs.
    # For shape phase, tmp['b'] has NaNs on outside.
    combined_output = process_output_nanaware(args.output_components, tmp['b'], tmp['c'])  # (N_case,N,4)

    # scalers
    scaler_fun = partial(MinMaxScaler, feature_range=(-1, 1))

    branch_scaler = scaler_fun()
    branch_scaler.fit(branch_condition_input)
    branch_condition_input = branch_scaler.transform(branch_condition_input)

    pointnet_scaler = scaler_fun()
    ss = branch_pointnet_input.shape
    pointnet_flat = branch_pointnet_input.reshape(ss[0] * ss[1], ss[2])
    pointnet_scaler.fit(pointnet_flat)
    branch_pointnet_input = pointnet_scaler.transform(pointnet_flat).reshape(ss)

    trunk_scaler = scaler_fun()
    ss = trunk_input.shape
    trunk_flat = trunk_input.reshape(ss[0] * ss[1], ss[2])
    trunk_scaler.fit(trunk_flat)
    trunk_input = trunk_scaler.transform(trunk_flat).reshape(ss)

    # output scaler must be NaN-safe for Ext phase
    output_scalers = fit_minmax_scaler_nan_featurewise(combined_output, feature_range=(-1, 1))
    combined_output = transform_minmax_nan_featurewise(output_scalers, combined_output)

    # save scalers (for inference)
    pickle.dump(branch_scaler, open(f'{experiment_dir}/branch_scaler.pkl', 'wb'))
    pickle.dump(pointnet_scaler, open(f'{experiment_dir}/pointnet_scaler.pkl', 'wb'))
    pickle.dump(trunk_scaler, open(f'{experiment_dir}/trunk_scaler.pkl', 'wb'))
    pickle.dump(output_scalers, open(f'{experiment_dir}/output_scaler.pkl', 'wb'))

    # split
    if args.split_method == 'random':
        split_file = f'../data/npy/combined_{args.N_samples}_split_random_train_valid.npz'
    else:
        split_file = f'../data/npy/combined_{args.N_samples}_split_mass_train_valid.npz'
    split_data = np.load(split_file)

    train_case = map_keys_to_indices(split_data['train'], np.array([key for key in tmp['c']]))
    test_case = map_keys_to_indices(split_data['valid'], np.array([key for key in tmp['c']]))

    train_case_file = tmp['c'][train_case]
    test_case_file = tmp['c'][test_case]

    # split arrays
    bc_tr = branch_condition_input[train_case].astype(np.float32)
    bc_te = branch_condition_input[test_case].astype(np.float32)

    bp_tr = branch_pointnet_input[train_case].astype(np.float32)
    bp_te = branch_pointnet_input[test_case].astype(np.float32)

    tr_tr = trunk_input[train_case].astype(np.float32)
    tr_te = trunk_input[test_case].astype(np.float32)

    s_tr = combined_output[train_case].astype(np.float32)  # (Ns,N,4) maybe NaNs
    s_te = combined_output[test_case].astype(np.float32)

    x_train = (bc_tr, bp_tr, tr_tr)
    x_test = (bc_te, bp_te, tr_te)

    # occupancy gt: inside=1, outside=0
    # inside/outside is inferred from whether field target is finite
    occ_tr = np.isfinite(s_tr).all(axis=2, keepdims=True).astype(np.float32)
    occ_te = np.isfinite(s_te).all(axis=2, keepdims=True).astype(np.float32)

    y_train = np.concatenate([s_tr, occ_tr], axis=2).astype(np.float32)  # (Ns,N,5)
    y_test = np.concatenate([s_te, occ_te], axis=2).astype(np.float32)

    logging.info(f'branch_condition_input_train.shape = {bc_tr.shape}')
    logging.info(f'branch_pointnet_input_train.shape = {bp_tr.shape}')
    logging.info(f'trunk_input_train.shape = {tr_tr.shape}')
    logging.info(f's_train.shape = {s_tr.shape} (NaNs possible on Ext)')
    logging.info(f'y_train(with occupancy).shape = {y_train.shape}')

    # quick sanity
    u = y_train[:, :, :4]
    logging.info(f"u finite ratio: {np.isfinite(u).mean():.4f}")
    logging.info(f"u max abs finite: {np.nanmax(np.abs(u)):.4f}")

    return x_train, y_train, x_test, y_test, output_scalers, train_case_file, test_case_file


# =========================
# Data wrapper (phase-aware losses)
# =========================
class TripleCartesianProd(Data):
    def __init__(self, X_train, y_train, X_test, y_test, phase: str, args):
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test
        self.num_train_samples = self.train_x[0].shape[0]
        self.sampler = BatchSampler(self.num_train_samples, shuffle=True)
        self.phase = phase
        self.args = args
        self._step = 0
        self._log_every = 100

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        """
        targets: (B,N,5) = field(4 with possible NaN) + occupancy(1)
        outputs: (B,N,5) = field(4) + occ_logit(1)
        """
        self._step += 1
        args = self.args

        u_gt = targets[:, :, :4]
        occ_gt = targets[:, :, 4]
        u_pred = outputs[:, :, :4]
        occ_logit = outputs[:, :, 4]

        # field loss (inside only + pred finite)
        mask_gt = torch.isfinite(u_gt).all(dim=-1)
        mask_pr = torch.isfinite(u_pred).all(dim=-1)
        mask_in = mask_gt & mask_pr

        if mask_in.any():
            diff = (u_pred - u_gt)[mask_in]
            loss_field = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="mean")
        else:
            loss_field = 0.0 * u_pred.mean()

        # occupancy loss (all points)
        loss_occ = F.binary_cross_entropy_with_logits(occ_logit, occ_gt)

        if self.phase == 'field':
            total = loss_field
            locc = 0.0 * loss_occ
        else:
            # keep CLI name lambda_sdf for minimal change
            total = args.lambda_sdf * loss_occ
            locc = loss_occ

        if (self._step % self._log_every) == 0:
            logging.info(
                f"[loss step {self._step}] phase={self.phase} total={float(total.detach().cpu()):.6e} | "
                f"field={float(loss_field.detach().cpu()):.3e} occ={float(locc.detach().cpu()):.3e} "
                f"(mask_in={int(mask_in.sum().detach().cpu())})"
            )
        return total

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        idx = self.sampler.get_next(batch_size)
        return ((self.train_x[0][idx], self.train_x[1][idx], self.train_x[2][idx]), self.train_y[idx])

    def test(self):
        return self.test_x, self.test_y


# =========================
# Model blocks
# =========================
class Encoder(nn.Module):
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
        x = pointnet_input.transpose(2, 1)   # (B,C,N)
        x = self.pointnet_branch(x)          # (B,latent,N)
        return x.max(dim=2)[0]               # (B,latent)


class FieldDecoder(nn.Module):
    def __init__(self, branch_condition_input_dim, latent_dim, branch_hidden_dim,
                 trunk_encoding_output_dim, trunk_hidden_dim, fc_hidden_dim,
                 num_output_components, activation):
        super().__init__()
        self.activation = activation
        self.tanh = nn.Tanh()
        self.b = nn.Parameter(torch.zeros(num_output_components))

        assert branch_hidden_dim == latent_dim == trunk_encoding_output_dim == fc_hidden_dim
        assert trunk_hidden_dim == trunk_encoding_output_dim

        self.branch_net_global = nn.Sequential(
            nn.Linear(branch_condition_input_dim, branch_hidden_dim),
            self.activation,
            nn.Linear(branch_hidden_dim, branch_hidden_dim * 2),
            self.activation,
            nn.Linear(branch_hidden_dim * 2, branch_hidden_dim),
            self.activation,
        )

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
        branch_output = self.branch_net_global(branch_condition_input)  # (B,H)
        combined = branch_output + latent                                # (B,H)

        mix = combined.unsqueeze(1) * trunk_xyz_encoded                 # (B,N,H)
        mix_reduced = mix.mean(dim=1)                                   # (B,H)
        combined2 = self.combined_branch(mix_reduced)                   # (B,H)

        B, N, H = trunk_xyz_encoded.shape
        x_trunk = self.trunk_net(trunk_xyz_encoded.reshape(B * N, -1)).view(B, N, -1)
        x_trunk = x_trunk.view(B, N, H, -1)                             # (B,N,H,4)

        out = torch.einsum("bh,bnhc->bnc", combined2, x_trunk) + self.b
        return self.tanh(out)  # scaled output


class ShapeDecoder(nn.Module):
    def __init__(self, latent_dim, trunk_encoding_output_dim, trunk_hidden_dim, fc_hidden_dim, activation):
        super().__init__()
        self.activation = activation
        self.b = nn.Parameter(torch.zeros(1))

        assert latent_dim == trunk_encoding_output_dim == fc_hidden_dim
        assert trunk_hidden_dim == trunk_encoding_output_dim

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
        mix = latent.unsqueeze(1) * trunk_xyz_encoded
        mix_reduced = mix.mean(dim=1)
        combined2 = self.combined_branch(mix_reduced)

        B, N, H = trunk_xyz_encoded.shape
        x_trunk = self.trunk_net(trunk_xyz_encoded.reshape(B * N, -1)).view(B, N, -1)
        x_trunk = x_trunk.view(B, N, H, 1)

        occ_logit = torch.einsum("bh,bnhc->bnc", combined2, x_trunk) + self.b
        return occ_logit  # (B,N,1), NO tanh / raw logits


class DesignGenPointDeepONet(dde.maps.NN, nn.Module):
    def __init__(self, branch_condition_input_dim, pointnet_input_dim, trunk_input_dim,
                 branch_hidden_dim=128, trunk_hidden_dim=128, fc_hidden_dim=128,
                 trunk_encoding_hidden_dim=128, num_output_components=4):
        super().__init__()
        nn.Module.__init__(self)
        dde.maps.NN.__init__(self)

        self.activation = nn.SiLU()

        self.trunk_encoding = nn.Sequential(
            SineDenseLayer(3, trunk_encoding_hidden_dim, w0=10.0, is_first=True),
            SineDenseLayer(trunk_encoding_hidden_dim, trunk_encoding_hidden_dim * 2, w0=10.0),
            SineDenseLayer(trunk_encoding_hidden_dim * 2, trunk_encoding_hidden_dim, w0=10.0),
        )
        self.encoder = Encoder(pointnet_input_dim, trunk_encoding_hidden_dim, self.activation)

        self.field_decoder = FieldDecoder(
            branch_condition_input_dim=branch_condition_input_dim,
            latent_dim=trunk_encoding_hidden_dim,
            branch_hidden_dim=branch_hidden_dim,
            trunk_encoding_output_dim=trunk_encoding_hidden_dim,
            trunk_hidden_dim=trunk_hidden_dim,
            fc_hidden_dim=fc_hidden_dim,
            num_output_components=num_output_components,
            activation=self.activation,
        )

        self.shape_decoder = ShapeDecoder(
            latent_dim=trunk_encoding_hidden_dim,
            trunk_encoding_output_dim=trunk_encoding_hidden_dim,
            trunk_hidden_dim=trunk_hidden_dim,
            fc_hidden_dim=fc_hidden_dim,
            activation=self.activation,
        )

        self._cached_latent = None

    def forward(self, inputs):
        branch_condition_input, pointnet_input, trunk_input = inputs
        trunk_xyz = trunk_input[:, :, :3]  # network uses xyz only

        latent = self.encoder(pointnet_input)
        self._cached_latent = latent

        B, N, _ = trunk_xyz.shape
        trunk_xyz_encoded = self.trunk_encoding(trunk_xyz.reshape(-1, 3)).reshape(B, N, -1)

        field_out = self.field_decoder(branch_condition_input, latent, trunk_xyz_encoded)  # (B,N,4)
        occ_logit = self.shape_decoder(latent, trunk_xyz_encoded)                           # (B,N,1)
        return torch.cat([field_out, occ_logit], dim=-1)                                    # (B,N,5)

    def parameters(self):
        return list(super().parameters())


# =========================
# Freeze config
# =========================
def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def configure_phase_trainable(net: DesignGenPointDeepONet, phase: str, freeze_trunk_encoding: bool):
    if phase == 'field':
        set_requires_grad(net.encoder, True)
        set_requires_grad(net.field_decoder, True)
        set_requires_grad(net.shape_decoder, False)
        set_requires_grad(net.trunk_encoding, True)
    else:
        set_requires_grad(net.encoder, False)
        set_requires_grad(net.field_decoder, False)
        set_requires_grad(net.shape_decoder, True)
        set_requires_grad(net.trunk_encoding, not freeze_trunk_encoding)


# =========================
# Define model + metrics
# =========================
def define_model(args, device, data, output_scalers):
    branch_condition_input_dim = data.train_x[0].shape[1]
    pointnet_input_dim = data.train_x[1].shape[-1]
    trunk_input_dim = data.train_x[2].shape[-1]

    assert args.branch_hidden_dim == args.trunk_encoding_hidden_dim == args.fc_hidden_dim
    assert args.trunk_hidden_dim == args.trunk_encoding_hidden_dim

    net = DesignGenPointDeepONet(
        branch_condition_input_dim=branch_condition_input_dim,
        pointnet_input_dim=pointnet_input_dim,
        trunk_input_dim=trunk_input_dim,
        branch_hidden_dim=args.branch_hidden_dim,
        trunk_hidden_dim=args.trunk_hidden_dim,
        fc_hidden_dim=args.fc_hidden_dim,
        trunk_encoding_hidden_dim=args.trunk_encoding_hidden_dim,
        num_output_components=4,
    ).to(device)

    # load pretrained for shape phase
    if args.phase == 'shape':
        if not args.pretrained_path:
            raise ValueError("phase=shape requires --pretrained_path")
        state = torch.load(args.pretrained_path, map_location=device)
        net.load_state_dict(state, strict=True)

    configure_phase_trainable(net, args.phase, args.freeze_trunk_encoding)

    # optimizer (trainable params only)
    trainable_params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=500)

    # dummy loss (deepxde requires)
    def loss_func(y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)

    # ---- robust metrics (DeepXDE metric(y_true, y_pred)) ----
    def to_numpy(d):
        if isinstance(d, torch.Tensor):
            return d.detach().cpu().numpy()
        if isinstance(d, np.ndarray):
            return d
        raise TypeError

    def masked_field_rows(y_true, y_pred):
        yt = to_numpy(y_true)[:, :, :4]
        yp = to_numpy(y_pred)[:, :, :4]
        mask_gt = np.isfinite(yt).all(axis=-1)
        mask_pr = np.isfinite(yp).all(axis=-1)
        mask = mask_gt & mask_pr
        yt2 = yt.reshape(-1, 4)
        yp2 = yp.reshape(-1, 4)
        m2 = mask.reshape(-1)
        if m2.sum() == 0:
            return None, None
        return yp2[m2], yt2[m2]

    def inv_rows(arr_2d):
        return output_scalers.inverse_transform(arr_2d)

    def mae_comp(j, default=0.0):
        def _m(y_true, y_pred):
            pred_m, true_m = masked_field_rows(y_true, y_pred)
            if pred_m is None:
                return float(default)
            pred_inv = inv_rows(pred_m)[:, j]
            true_inv = inv_rows(true_m)[:, j]
            m = np.isfinite(pred_inv) & np.isfinite(true_inv)
            if m.sum() == 0:
                return float(default)
            return float(np.mean(np.abs(pred_inv[m] - true_inv[m])))
        return _m

    def r2_comp(j, default=0.0):
        def _m(y_true, y_pred):
            pred_m, true_m = masked_field_rows(y_true, y_pred)
            if pred_m is None:
                return float(default)
            pred_inv = inv_rows(pred_m)[:, j]
            true_inv = inv_rows(true_m)[:, j]
            m = np.isfinite(pred_inv) & np.isfinite(true_inv)
            if m.sum() < 2:
                return float(default)
            return float(calculate_r2(true_inv[m], pred_inv[m]))
        return _m

    ux_MAE, uy_MAE, uz_MAE, vm_MAE = mae_comp(0), mae_comp(1), mae_comp(2), mae_comp(3)
    ux_R2, uy_R2, uz_R2, vm_R2 = r2_comp(0), r2_comp(1), r2_comp(2), r2_comp(3)

    metrics = [ux_MAE, uy_MAE, uz_MAE, vm_MAE, ux_R2, uy_R2, uz_R2, vm_R2]

    model = dde.Model(data, net)
    model.compile(optimizer=optimizer, loss=loss_func, metrics=metrics)
    return model, scheduler, optimizer


# =========================
# Training
# =========================
def train_model(model, scheduler, args, experiment_dir):
    callbacks = []
    torch.autograd.set_detect_anomaly(False)

    logging.info(f'y_train.shape = {model.data.train_y.shape}')
    logging.info(f'y_test.shape = {model.data.test_y.shape}')
    logging.info("Training model...")

    start_time = time.time()
    losshistory, train_state = model.train(
        iterations=args.N_iterations,
        batch_size=args.batch_size,
        display_every=100,
        model_save_path=f"{experiment_dir}/model_checkpoint.pth",
        callbacks=callbacks,
    )
    total_training_time = time.time() - start_time

    # scheduler step on last test loss
    try:
        scheduler.step(float(np.sum(losshistory.loss_test[-1])))
    except Exception:
        pass

    torch.save(model.net.state_dict(), f'{experiment_dir}/model_final.pth')

    with open(os.path.join(experiment_dir, 'training_time.txt'), 'w') as f:
        f.write(f"{total_training_time:.2f}")

    np.save(f'{experiment_dir}/train_losses.npy', np.array(losshistory.loss_train))
    np.save(f'{experiment_dir}/val_losses.npy', np.array(losshistory.loss_test))

    return losshistory, train_state


# =========================
# Minimal post-eval (overall field + occupancy)
# =========================
def post_eval(model, output_scalers, experiment_dir, args):
    test_outputs = model.predict(model.data.test_x)  # (Nt,N,5)
    test_targets = model.data.test_y                 # (Nt,N,5)

    # -------- field metrics overall (NaN-safe) --------
    y_true = test_targets[:, :, :4]
    y_pred = test_outputs[:, :, :4]
    mask = np.isfinite(y_true).all(axis=-1) & np.isfinite(y_pred).all(axis=-1)
    yt = y_true.reshape(-1, 4)
    yp = y_pred.reshape(-1, 4)
    m2 = mask.reshape(-1)

    if m2.sum() > 0:
        yt_inv = output_scalers.inverse_transform(yt[m2])
        yp_inv = output_scalers.inverse_transform(yp[m2])

        comps = ['ux', 'uy', 'uz', 'vm']
        for j, c in enumerate(comps):
            mae = np.mean(np.abs(yp_inv[:, j] - yt_inv[:, j]))
            rmse = np.sqrt(np.mean((yp_inv[:, j] - yt_inv[:, j]) ** 2))
            r2 = calculate_r2(yt_inv[:, j], yp_inv[:, j])
            logging.info(f"[Overall Field] {c}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    else:
        logging.info("[Overall Field] No valid inside points found in test set for metrics.")

    # -------- occupancy metrics --------
    occ_logit = test_outputs[:, :, 4]
    occ_prob = 1.0 / (1.0 + np.exp(-occ_logit))
    occ_pred = (occ_prob >= 0.5).astype(np.float32)
    occ_gt = test_targets[:, :, 4].astype(np.float32)

    occ_acc = np.mean(occ_pred == occ_gt)

    inter = np.logical_and(occ_pred > 0.5, occ_gt > 0.5).sum()
    union = np.logical_or(occ_pred > 0.5, occ_gt > 0.5).sum()
    occ_iou = inter / (union + 1e-12)

    occ_bce = -np.mean(
        occ_gt * np.log(occ_prob + 1e-12) +
        (1.0 - occ_gt) * np.log(1.0 - occ_prob + 1e-12)
    )

    logging.info(f"[Overall Occupancy] BCE={occ_bce:.6f}, ACC={occ_acc:.6f}, IoU={occ_iou:.6f}")


# =========================
# Main
# =========================
def main():
    args = parse_arguments()
    set_random_seed()
    device = get_device(args.gpu)

    name = (
        f'Occ2Phase_{args.phase}_RUN{args.RUN}_D{args.N_samples}_N{args.N_pt}_'
        f'branchinput_{args.branch_condition_input_components}_trunkinput_{args.trunk_input_components}_'
        f'output_{args.output_components}_split_{args.split_method}'
    )
    experiment_dir = setup_logging(args, name)
    log_parameters(args, device, name)

    x_train, y_train, x_test, y_test, output_scalers, train_case_file, test_case_file = load_and_preprocess_data(args, experiment_dir)

    data = TripleCartesianProd(x_train, y_train, x_test, y_test, phase=args.phase, args=args)
    model, scheduler, optimizer = define_model(args, device, data, output_scalers)

    total_params = sum(p.numel() for p in model.net.parameters() if p.requires_grad)
    logging.info(f'Number of Trainable Parameters (this phase): {total_params}\n')

    losshistory, train_state = train_model(model, scheduler, args, experiment_dir)

    # Always run a minimal eval for sanity
    post_eval(model, output_scalers, experiment_dir, args)

    logging.info(f"Saved final weights: {experiment_dir}/model_final.pth")
    if args.phase == 'field':
        logging.info("Next: run phase=shape with --pretrained_path pointing to this model_final.pth")

if __name__ == "__main__":
    main()