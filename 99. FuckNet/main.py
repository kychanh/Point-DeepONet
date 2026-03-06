# main.py (DesignGenPointDeepONet - Design-GenNO style end-to-end, occupancy + NF)
#
# Requirements implemented:
#  - End-to-end training: encoder + field_decoder + shape_decoder + NF
#  - Field supervised on inside-only Rpt points
#  - Shape supervised on dense Grid occupancy points
#  - No masking tricks: we slice by known counts (N_field, N_grid)
#  - Grid*.npz format:
#      a: grid xyz (Ncase,Ng,3) OR (Ng,3)
#      b: occupancy (Ncase,Ng,1) OR (Ncase,Ng)
#      c: filenames (Ncase,)
#
# Output tensor format (for DeepXDE compatibility):
#  - model outputs a single tensor (B, N_field + N_grid, 5)
#  - first N_field rows: [field(4), 0]
#  - last  N_grid  rows: [0,0,0,0, occ_logit(1)]
#  - targets are built in the same layout:
#    - first N_field rows: [field_gt(4), 0]
#    - last  N_grid  rows: [0,0,0,0, occ_gt(1)]
#
# Loss (Design-GenNO Loss_data spirit):
#   L = w_occ * BCEWithLogits(occ_logit, occ_gt) + w_field * MSE(field_pred, field_gt) + lambda_nf * NF_NLL
#
# Notes:
#  - trunk_scaler is fit on Rpt trunk (xyzd). We apply it to Grid coords after padding d=0.
#  - occupancy is NOT scaled.
#  - field output is scaled with output_scaler (same as your previous pipeline).
#
# Keep your long VTK post-processing loop separate if you want; this file focuses on stable training.

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
import random
from functools import partial
from sklearn.preprocessing import MinMaxScaler

os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
from deepxde.data.data import Data
from deepxde.data.sampler import BatchSampler

import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.amp import autocast

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
                np.sqrt(6 / self.in_features) / self.w0
            )

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))


# =========================
# Normalizing Flow (RealNVP)
# =========================
class FCNN(nn.Module):
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
    def __init__(self, dim, hidden_dim=64, activation='Tanh', base_network=FCNN):
        super().__init__()
        self.dim = dim
        self.t1 = base_network(dim // 2, dim // 2, hidden_dim, activation)
        self.s1 = base_network(dim // 2, dim // 2, hidden_dim, activation)
        self.t2 = base_network(dim // 2, dim // 2, hidden_dim, activation)
        self.s2 = base_network(dim // 2, dim // 2, hidden_dim, activation)

    def forward(self, x):
        lower, upper = x[:, :self.dim // 2], x[:, self.dim // 2:]
        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        upper = t1_transformed + upper * torch.exp(s1_transformed)

        t2_transformed = self.t2(upper)
        s2_transformed = self.s2(upper)
        lower = t2_transformed + lower * torch.exp(s2_transformed)

        z = torch.cat([lower, upper], dim=1)
        log_det = torch.sum(s1_transformed, dim=1) + torch.sum(s2_transformed, dim=1)
        return z, log_det

    def inverse(self, z):
        lower, upper = z[:, :self.dim // 2], z[:, self.dim // 2:]

        t2_transformed = self.t2(upper)
        s2_transformed = self.s2(upper)
        lower = (lower - t2_transformed) * torch.exp(-s2_transformed)

        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        upper = (upper - t1_transformed) * torch.exp(-s1_transformed)

        x = torch.cat([lower, upper], dim=1)
        log_det = torch.sum(-s1_transformed, dim=1) + torch.sum(-s2_transformed, dim=1)
        return x, log_det


class NFModel(nn.Module):
    def __init__(self, dim, hidden_dim, activation, num_flows: int, device):
        super().__init__()
        self.prior = MultivariateNormal(torch.zeros(dim).to(device), torch.eye(dim).to(device))
        self.flow_net = nn.Sequential(*[RealNVP(dim, hidden_dim, activation) for _ in range(num_flows)])

    def forward(self, x):
        batch_size = x.shape[0]
        log_det = torch.zeros(batch_size, device=x.device)
        h = x
        for flow in self.flow_net:
            h, ld = flow.forward(h)
            log_det += ld
        z = h
        prior_logprob = self.prior.log_prob(z)
        return z, prior_logprob, log_det

    def sample(self, n_samples):
        z = self.prior.sample((n_samples,))
        h = z
        log_det = torch.zeros(n_samples, device=z.device)
        for flow in self.flow_net[::-1]:
            h, ld = flow.inverse(h)
            log_det += ld
        return h


# =========================
# CLI args
# =========================
def parse_arguments():
    parser = argparse.ArgumentParser(description='Design-GenNO style end-to-end: field + occupancy + NF')

    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-r', '--RUN', type=int, default=0)

    parser.add_argument('-bic', '--branch_condition_input_components', type=str, default='mlc')
    parser.add_argument('-tic', '--trunk_input_components', type=str, default='xyzd')
    parser.add_argument('-oc', '--output_components', type=str, default='xyzs')

    parser.add_argument('-N_p', '--N_pt', type=int, default=5000)
    parser.add_argument('-i', '--N_iterations', type=int, default=40000)
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)

    parser.add_argument('-dir', '--dir_base_load_data', type=str, default='../data/sampled',
                        help='Contains Rpt{RUN}_N{N_pt}.npz and Grid{RUN}.npz')
    parser.add_argument('--grid_path', type=str, default='',
                        help='If empty, uses {dir_base_load_data}/Grid{RUN}.npz')

    parser.add_argument('-N_d', '--N_samples', type=int, default=3000)
    parser.add_argument('-sm', '--split_method', type=str, choices=['mass', 'random'], default='random')
    parser.add_argument('--base_dir', type=str, default='../experiments')

    parser.add_argument('--branch_hidden_dim', type=int, default=128)
    parser.add_argument('--trunk_hidden_dim', type=int, default=128)
    parser.add_argument('--trunk_encoding_hidden_dim', type=int, default=128)
    parser.add_argument('--fc_hidden_dim', type=int, default=128)

    # NF hyperparams
    parser.add_argument('--nf_num_flows', type=int, default=4)
    parser.add_argument('--nf_hidden_dim', type=int, default=64)
    parser.add_argument('--nf_activation', type=str, default='Tanh')
    parser.add_argument('--lambda_nf', type=float, default=1e-3)

    # Loss weights (Design-GenNO Loss_data spirit)
    parser.add_argument('--w_occ', type=float, default=2.0)
    parser.add_argument('--w_field', type=float, default=1.0)

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
    logging.info(f'NF: num_flows={args.nf_num_flows}, hidden={args.nf_hidden_dim}, act={args.nf_activation}, lambda_nf={args.lambda_nf}')
    logging.info(f'Loss weights: w_occ={args.w_occ}, w_field={args.w_field}\n\n')


# =========================
# Preprocess helpers
# =========================
def process_branch_condition_input(input_components, tmp):
    input_mapping = {'m': 4, 'l': 5, 'c': [6, 7, 8]}
    selected_indices = []
    for comp in input_components:
        if comp == 'c':
            selected_indices.extend(input_mapping[comp])
        elif comp in input_mapping:
            selected_indices.append(input_mapping[comp])
    return tmp['a'][:, 0, selected_indices].astype(np.float32)


def process_branch_pointnet_input_xyz(tmp):
    # encoder input: xyz only
    return tmp['a'][:, :, :3].astype(np.float32)


def process_trunk_input(input_components, tmp):
    input_mapping = {'x': 0, 'y': 1, 'z': 2, 'd': 3, 'm': 4, 'l': 5, 'c': [6, 7, 8]}
    selected_indices = []
    for comp in input_components:
        if comp == 'c':
            selected_indices.extend(input_mapping[comp])
        elif comp in input_mapping:
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


def process_output(output_components, output_vals, identifiers):
    # output_vals: (Ncase,Npt,4)
    output_mapping = {'x': 0, 'y': 1, 'z': 2, 's': 3}
    selected_indices = [output_mapping[comp] for comp in output_components]
    combined_output = output_vals[:, :, selected_indices].copy()

    unique_directions = set(id.split('_')[0] for id in identifiers)
    for direction in unique_directions:
        clip = get_clipping_ranges_for_direction(direction)
        idxs = [i for i, cid in enumerate(identifiers) if cid.split('_')[0] == direction]
        if not idxs:
            continue
        idxs = np.array(idxs)
        for j, comp_idx in enumerate(selected_indices):
            mn, mx = clip[comp_idx]
            combined_output[idxs, :, j] = np.clip(combined_output[idxs, :, j], mn, mx)
    return combined_output.astype(np.float32)


def map_keys_to_indices(keys, all_keys):
    return [np.where(all_keys == key)[0][0] for key in keys]


def calculate_r2(true_vals, pred_vals):
    ss_res = np.sum((true_vals - pred_vals) ** 2)
    ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2) + 1e-12
    return 1 - ss_res / ss_tot


# =========================
# Load Grid*.npz (a=xyz, b=occ, c=filename)
# =========================
def load_grid_npz(path):
    g = np.load(path, allow_pickle=True)
    if 'a' not in g.files or 'b' not in g.files or 'c' not in g.files:
        raise KeyError(f"[Grid] need keys a,b,c. Found: {g.files}")
    a = g['a'].astype(np.float32)  # (Ncase,Ng,3) OR (Ng,3)
    b = g['b'].astype(np.float32)  # (Ncase,Ng,1) OR (Ncase,Ng)
    c = g['c']                     # (Ncase,)
    if b.ndim == 3 and b.shape[-1] == 1:
        b = b[..., 0]
    if b.ndim != 2:
        raise ValueError(f"[Grid] b must be (Ncase,Ng) or (Ncase,Ng,1). Got {b.shape}")
    if a.ndim == 2:
        if a.shape[1] != 3:
            raise ValueError(f"[Grid] a must be (Ng,3) if 2D. Got {a.shape}")
        # broadcast to (Ncase,Ng,3)
        a = np.broadcast_to(a[None, :, :], (b.shape[0], a.shape[0], 3)).copy()
    elif a.ndim == 3:
        if a.shape[-1] != 3:
            raise ValueError(f"[Grid] a must be (...,3). Got {a.shape}")
        if a.shape[0] != b.shape[0] or a.shape[1] != b.shape[1]:
            raise ValueError(f"[Grid] shape mismatch: a={a.shape}, b={b.shape}")
    else:
        raise ValueError(f"[Grid] a must be 2D or 3D. Got {a.shape}")
    if len(c) != b.shape[0]:
        raise ValueError(f"[Grid] c length mismatch: len(c)={len(c)} vs b.shape[0]={b.shape[0]}")
    return a.astype(np.float32), b.astype(np.float32), c


def align_grid_to_rpt(rpt_c, grid_a, grid_b, grid_c):
    # match by filename key
    grid_index = {str(k): i for i, k in enumerate(grid_c)}
    idxs = []
    missing = []
    for k in rpt_c:
        ks = str(k)
        if ks not in grid_index:
            missing.append(ks)
        else:
            idxs.append(grid_index[ks])
    if missing:
        raise KeyError(f"[Align] Missing {len(missing)} keys in Grid file. Examples: {missing[:5]}")
    idxs = np.array(idxs, dtype=np.int64)
    return grid_a[idxs], grid_b[idxs]


# =========================
# Data loading & preprocessing
# =========================
def load_and_preprocess_data(args, experiment_dir):
    # --- load Rpt (field/inside points) ---
    rpt_path = os.path.join(args.dir_base_load_data, f"Rpt0_N{args.N_pt}.npz")
    tmp = np.load(rpt_path)

    rpt_c = tmp['c']  # filenames/ids

    branch_condition_input = process_branch_condition_input(args.branch_condition_input_components, tmp)  # (Ncase,Cc)
    branch_pointnet_input = process_branch_pointnet_input_xyz(tmp)                                        # (Ncase,Npt,3)
    trunk_input_field = process_trunk_input(args.trunk_input_components, tmp)                             # (Ncase,Npt,4=xyzd)
    field_gt = process_output(args.output_components, tmp['b'], rpt_c)                                    # (Ncase,Npt,4)

    # --- load Grid (occupancy grid points) ---
    grid_path = args.grid_path.strip()
    if grid_path == "":
        grid_path = os.path.join(args.dir_base_load_data, f"Grid{args.RUN}.npz")
    grid_a, grid_b, grid_c = load_grid_npz(grid_path)  # a:(Ncase,Ng,3), b:(Ncase,Ng), c:(Ncase,)

    # align grid ordering to Rpt ordering via c
    grid_a, grid_b = align_grid_to_rpt(rpt_c, grid_a, grid_b, grid_c)  # still (Ncase,Ng,3), (Ncase,Ng)

    N_case = trunk_input_field.shape[0]
    N_field = trunk_input_field.shape[1]
    Ng = grid_a.shape[1]

    # build trunk_input_grid as xyzd-like with d=0
    trunk_input_grid = np.zeros((N_case, Ng, trunk_input_field.shape[-1]), dtype=np.float32)
    trunk_input_grid[:, :, :3] = grid_a
    # d channel = 0

    # --- scalers ---
    scaler_fun = partial(MinMaxScaler, feature_range=(-1, 1))

    branch_scaler = scaler_fun()
    branch_scaler.fit(branch_condition_input)
    branch_condition_input = branch_scaler.transform(branch_condition_input)

    pointnet_scaler = scaler_fun()
    ss = branch_pointnet_input.shape
    tmp_pn = branch_pointnet_input.reshape([ss[0] * ss[1], ss[2]])
    pointnet_scaler.fit(tmp_pn)
    branch_pointnet_input = pointnet_scaler.transform(tmp_pn).reshape(ss)

    # trunk scaler fitted on field trunk
    trunk_scaler = scaler_fun()
    ssf = trunk_input_field.shape
    tmp_tf = trunk_input_field.reshape([ssf[0] * ssf[1], ssf[2]])
    trunk_scaler.fit(tmp_tf)
    trunk_input_field = trunk_scaler.transform(tmp_tf).reshape(ssf)

    # apply same trunk scaler to grid trunk
    ssg = trunk_input_grid.shape
    tmp_tg = trunk_input_grid.reshape([ssg[0] * ssg[1], ssg[2]])
    trunk_input_grid = trunk_scaler.transform(tmp_tg).reshape(ssg)

    # field output scaler
    output_scalers = scaler_fun()
    sso = field_gt.shape
    tmp_out = field_gt.reshape([sso[0] * sso[1], sso[2]])
    output_scalers.fit(tmp_out)
    field_gt = output_scalers.transform(tmp_out).reshape(sso)

    # save scalers
    pickle.dump(branch_scaler, open(f'{experiment_dir}/branch_scaler.pkl', 'wb'))
    pickle.dump(pointnet_scaler, open(f'{experiment_dir}/pointnet_scaler.pkl', 'wb'))
    pickle.dump(trunk_scaler, open(f'{experiment_dir}/trunk_scaler.pkl', 'wb'))
    pickle.dump(output_scalers, open(f'{experiment_dir}/output_scaler.pkl', 'wb'))

    # --- split ---
    if args.split_method == 'random':
        split_file = f'../data/npy/combined_{args.N_samples}_split_random_train_valid.npz'
    else:
        split_file = f'../data/npy/combined_{args.N_samples}_split_mass_train_valid.npz'
    split_data = np.load(split_file)

    train_case = map_keys_to_indices(split_data['train'], np.array([key for key in rpt_c]))
    test_case = map_keys_to_indices(split_data['valid'], np.array([key for key in rpt_c]))

    train_case_file = rpt_c[train_case]
    test_case_file = rpt_c[test_case]

    # split arrays
    bc_tr = branch_condition_input[train_case].astype(np.float32)
    bc_te = branch_condition_input[test_case].astype(np.float32)

    bp_tr = branch_pointnet_input[train_case].astype(np.float32)
    bp_te = branch_pointnet_input[test_case].astype(np.float32)

    tf_tr = trunk_input_field[train_case].astype(np.float32)
    tf_te = trunk_input_field[test_case].astype(np.float32)

    tg_tr = trunk_input_grid[train_case].astype(np.float32)
    tg_te = trunk_input_grid[test_case].astype(np.float32)

    uf_tr = field_gt[train_case].astype(np.float32)  # (Ns,N_field,4)
    uf_te = field_gt[test_case].astype(np.float32)

    occ_tr = grid_b[train_case].astype(np.float32)   # (Ns,Ng)
    occ_te = grid_b[test_case].astype(np.float32)

    # --- build trunk concat for single-output tensor ---
    trunk_tr = np.concatenate([tf_tr, tg_tr], axis=1)  # (Ns, N_field+Ng, 4)
    trunk_te = np.concatenate([tf_te, tg_te], axis=1)

    # --- build y tensor without masking: just layout ---
    # y: (Ns, N_field+Ng, 5) where
    #  - field part: [field_gt, 0]
    #  - occ part:   [0,0,0,0, occ_gt]
    Ns = trunk_tr.shape[0]
    Nt = trunk_te.shape[0]
    Ntot_tr = trunk_tr.shape[1]
    Ntot_te = trunk_te.shape[1]

    y_tr = np.zeros((Ns, Ntot_tr, 5), dtype=np.float32)
    y_te = np.zeros((Nt, Ntot_te, 5), dtype=np.float32)

    # field block
    y_tr[:, :N_field, :4] = uf_tr
    y_te[:, :N_field, :4] = uf_te
    # occ block
    y_tr[:, N_field:, 4] = occ_tr
    y_te[:, N_field:, 4] = occ_te

    x_train = (bc_tr, bp_tr, trunk_tr)
    x_test = (bc_te, bp_te, trunk_te)

    logging.info(f"[DATA] Rpt: N_field={N_field}, Grid: Ng={Ng}, total N={N_field+Ng}")
    logging.info(f"branch_condition_input_train.shape = {bc_tr.shape}")
    logging.info(f"branch_pointnet_input_train.shape = {bp_tr.shape}")
    logging.info(f"trunk_input_train.shape = {trunk_tr.shape}")
    logging.info(f"y_train.shape = {y_tr.shape}")

    return x_train, y_tr, x_test, y_te, output_scalers, train_case_file, test_case_file, N_field, Ng


# =========================
# Data wrapper (end-to-end loss)
# =========================
class TripleCartesianProd(Data):
    def __init__(self, X_train, y_train, X_test, y_test, args, N_field, N_grid):
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test
        self.num_train_samples = self.train_x[0].shape[0]
        self.sampler = BatchSampler(self.num_train_samples, shuffle=True)

        self.lambda_nf = float(args.lambda_nf)
        self.w_occ = float(args.w_occ)
        self.w_field = float(args.w_field)

        self.N_field = int(N_field)
        self.N_grid = int(N_grid)

        self._step = 0
        self._log_every = 200

        self.test_sampler = BatchSampler(self.test_x[0].shape[0], shuffle=False)
        self.test_batch_size = min(args.batch_size, self.test_x[0].shape[0])  # val도 train batch와 동일하게

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        """
        targets, outputs: (B, N_field + N_grid, 5)
        field region: [:, :N_field, :4]
        occ region:   [:, N_field:, 4]
        """
        self._step += 1
        Nf = self.N_field

        # field loss
        u_gt = targets[:, :Nf, :4]
        u_pr = outputs[:, :Nf, :4]
        loss_field = torch.mean((u_pr - u_gt) ** 2)

        # occupancy loss (logits)
        occ_gt = targets[:, Nf:, 4]
        occ_logit = outputs[:, Nf:, 4]
        loss_occ = F.binary_cross_entropy_with_logits(occ_logit, occ_gt)

        # NF prior loss (NLL)
        prior_logprob = model.net._cached_nf_prior_logprob
        log_det = model.net._cached_nf_log_det
        loss_nf = -(prior_logprob + log_det).mean()

        total = self.w_occ * loss_occ + self.w_field * loss_field + self.lambda_nf * loss_nf

        if (self._step % self._log_every) == 0:
            logging.info(
                f"[loss] step={self._step} "
                f"occ={float(loss_occ.detach().cpu()):.4e}(w={self.w_occ:g}) | "
                f"field={float(loss_field.detach().cpu()):.4e}(w={self.w_field:g}) | "
                f"nf={float(loss_nf.detach().cpu()):.4e}(w={self.lambda_nf:g}) | "
                f"total={float(total.detach().cpu()):.4e}"
            )

        return total

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            # CPU torch로 변환
            x0 = torch.from_numpy(self.train_x[0])
            x1 = torch.from_numpy(self.train_x[1])
            x2 = torch.from_numpy(self.train_x[2])
            y  = torch.from_numpy(self.train_y)
            return (x0, x1, x2), y

        idx = self.sampler.get_next(batch_size)

        x0 = torch.from_numpy(self.train_x[0][idx])
        x1 = torch.from_numpy(self.train_x[1][idx])
        x2 = torch.from_numpy(self.train_x[2][idx])
        y  = torch.from_numpy(self.train_y[idx])

        return (x0, x1, x2), y

    def test(self):
        # ✅ 테스트를 전체가 아니라 "한 배치"만 반환
        idx = self.test_sampler.get_next(self.test_batch_size)

        # numpy -> CPU torch
        x0 = torch.from_numpy(self.test_x[0][idx])
        x1 = torch.from_numpy(self.test_x[1][idx])
        x2 = torch.from_numpy(self.test_x[2][idx])
        y  = torch.from_numpy(self.test_y[idx])

        return (x0, x1, x2), y

# =========================
# Model blocks (PointDeepONet 변형 유지)
# =========================
class Encoder(nn.Module):
    """PointNet-like: (B, N, 3) -> (B, latent_dim)"""
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
        x = pointnet_input.transpose(2, 1)   # (B,3,N)
        x = self.pointnet_branch(x)          # (B,latent,N)
        latent = x.max(dim=2)[0]             # (B,latent)
        return latent


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
        return self.tanh(out)  # scaled field in [-1,1]


class ShapeDecoder(nn.Module):
    """Occupancy logits head (Bernoulli). No tanh."""
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
        mix = latent.unsqueeze(1) * trunk_xyz_encoded     # (B,N,H)
        mix_reduced = mix.mean(dim=1)                     # (B,H)
        combined2 = self.combined_branch(mix_reduced)     # (B,H)

        B, N, H = trunk_xyz_encoded.shape
        x_trunk = self.trunk_net(trunk_xyz_encoded.reshape(B * N, -1)).view(B, N, -1)
        x_trunk = x_trunk.view(B, N, H, 1)

        occ_logit = torch.einsum("bh,bnhc->bnc", combined2, x_trunk) + self.b
        return occ_logit  # (B,N,1)


class DesignGenPointDeepONet(dde.maps.NN, nn.Module):
    """
    Inputs: (branch_condition_input, pointnet_input, trunk_input_concat)
    trunk_input_concat: (B, N_field+N_grid, 4) but network uses xyz only
    Outputs: (B, N_field+N_grid, 5) with the layout described at file top.
    """
    def __init__(self, branch_condition_input_dim, pointnet_input_dim, trunk_input_dim,
                 branch_hidden_dim=128, trunk_hidden_dim=128, fc_hidden_dim=128,
                 trunk_encoding_hidden_dim=128, num_output_components=4,
                 nf_num_flows=4, nf_hidden_dim=64, nf_activation="Tanh",
                 device_str="cuda:0"):
        super().__init__()
        nn.Module.__init__(self)
        dde.maps.NN.__init__(self)

        self.trunk_chunk = 200000  # 기본값, 상황에 따라 50k~500k 조절
        self.N_field = 5000  # 생성자에 인자로 받기

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

        self.nf = NFModel(
            dim=trunk_encoding_hidden_dim,
            hidden_dim=nf_hidden_dim,
            activation=nf_activation,
            num_flows=nf_num_flows,
            device=device_str
        )

        self._cached_nf_prior_logprob = None
        self._cached_nf_log_det = None

    def forward(self, inputs):
        branch_condition_input, pointnet_input, trunk_input = inputs

        dev = next(iter(nn.Module.parameters(self))).device
        branch_condition_input = branch_condition_input.to(dev, non_blocking=True)
        pointnet_input = pointnet_input.to(dev, non_blocking=True)
        trunk_input = trunk_input.to(dev, non_blocking=True)

        # encoder는 fp32 유지해도 되고 autocast에 포함해도 됨
        latent = self.encoder(pointnet_input)

        # NF는 logdet/exp 때문에 fp32가 더 안전 (autocast 바깥 권장)
        _, prior_logprob, log_det = self.nf.forward(latent)
        self._cached_nf_prior_logprob = prior_logprob
        self._cached_nf_log_det = log_det
        
        
        trunk_xyz = trunk_input[:, :, :3]
        B, N, _ = trunk_xyz.shape
        flat = trunk_xyz.reshape(-1, 3)

        # --- AMP: trunk encoding + decoders ---
        # bfloat16이 있으면 bfloat16이 더 안정적 (A100/H100류에서 특히)
        use_amp = torch.cuda.is_available()
        amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16

        with autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            # chunked trunk encoding (OOM 방지)
            chunk = getattr(self, "trunk_chunk", 200000)  # 50k~500k 튜닝
            outs = []
            for s in range(0, flat.shape[0], chunk):
                e = min(flat.shape[0], s + chunk)
                outs.append(self.trunk_encoding(flat[s:e]))
            trunk_xyz_encoded = torch.cat(outs, dim=0).reshape(B, N, -1)

            Nf = self.N_field
            enc_field = trunk_xyz_encoded[:, :Nf, :]
            enc_grid  = trunk_xyz_encoded[:, Nf:, :]

            field_out = self.field_decoder(branch_condition_input, latent, enc_field)  # (B,Nf,4)
            occ_logit = self.shape_decoder(latent, enc_grid)                           # (B,Ng,1)

            # pack
            out = torch.zeros((B, N, 5), device=field_out.device, dtype=field_out.dtype)
            out[:, :Nf, :4] = field_out
            out[:, Nf:, 4:5] = occ_logit

            #field_out = self.field_decoder(branch_condition_input, latent, trunk_xyz_encoded)  # (B,N,4)
            #occ_logit = self.shape_decoder(latent, trunk_xyz_encoded)                          # (B,N,1)

            #out = torch.cat([field_out, occ_logit], dim=-1)

        return out

    def parameters(self):
        return list(super().parameters())


# =========================
# define_model
# =========================
def define_model(args, device, data, output_scalers, N_field):
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
        nf_num_flows=args.nf_num_flows,
        nf_hidden_dim=args.nf_hidden_dim,
        nf_activation=args.nf_activation,
        device_str=str(device),
    ).to(device)

    def loss_func(y_true, y_pred):
        # dummy; real loss is computed in Data.losses
        return torch.mean((y_true - y_pred) ** 2)

    # metrics: field only over first N_field rows
    def to_numpy(d):
        if isinstance(d, torch.Tensor):
            return d.detach().cpu().numpy()
        if isinstance(d, np.ndarray):
            return d
        raise TypeError

    def inv_field(arr, scaler):
        shp = arr.shape
        flat = arr.reshape(-1, shp[-1])
        inv = scaler.inverse_transform(flat)
        return inv.reshape(shp)

    Nf = int(N_field)

    def mae_comp(j, default=0.0):
        def _m(y_true, y_pred):
            yt = to_numpy(y_true)[:, :Nf, :4]
            yp = to_numpy(y_pred)[:, :Nf, :4]
            yt_inv = inv_field(yt, output_scalers)[:, :, j].reshape(-1)
            yp_inv = inv_field(yp, output_scalers)[:, :, j].reshape(-1)
            return float(np.mean(np.abs(yp_inv - yt_inv))) if yt_inv.size > 0 else float(default)
        return _m

    def r2_comp(j, default=0.0):
        def _m(y_true, y_pred):
            yt = to_numpy(y_true)[:, :Nf, :4]
            yp = to_numpy(y_pred)[:, :Nf, :4]
            yt_inv = inv_field(yt, output_scalers)[:, :, j].reshape(-1)
            yp_inv = inv_field(yp, output_scalers)[:, :, j].reshape(-1)
            if yt_inv.size < 2:
                return float(default)
            return float(calculate_r2(yt_inv, yp_inv))
        return _m

    metrics = [mae_comp(0), mae_comp(1), mae_comp(2), mae_comp(3),
               r2_comp(0), r2_comp(1), r2_comp(2), r2_comp(3)]

    model = dde.Model(data, net)
    optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=500)

    model.compile(optimizer=optimizer, loss=loss_func, metrics=metrics)
    return model, scheduler, optimizer


# =========================
# Training
# =========================
def train_model(model, scheduler, args, experiment_dir):
    callbacks = []
    #torch.autograd.set_detect_anomaly(False)

    logging.info(f'y_train.shape = {model.data.train_y.shape}')
    logging.info(f'y_test.shape = {model.data.test_y.shape}')
    logging.info("Training model...")

    start_time = time.time()
    import torch, os
    print("default tensor type:", torch.tensor(0.).device)
    print("cuda available:", torch.cuda.is_available())
    print("memory allocated GB:", torch.cuda.memory_allocated()/1024**3)
    print("memory reserved  GB:", torch.cuda.memory_reserved()/1024**3)
    losshistory, train_state = model.train(
        iterations=args.N_iterations,
        batch_size=args.batch_size,
        display_every=100,
        model_save_path=f"{experiment_dir}/model_checkpoint.pth",
        callbacks=callbacks,
    )
    total_training_time = time.time() - start_time

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
# Minimal post-eval
# =========================
def post_eval(model, output_scalers, experiment_dir, N_field, batch_eval=4):
    """
    Batch evaluation to avoid OOM.
    batch_eval: number of cases per forward in eval (2~8 권장)
    """
    net = model.net
    net.eval()

    x_test = model.data.test_x
    y_test = model.data.test_y

    Nt = x_test[0].shape[0]
    Nf = int(N_field)

    preds = []

    with torch.no_grad():
        for s in range(0, Nt, batch_eval):
            e = min(Nt, s + batch_eval)

            # inputs are numpy or torch (depending on your Data.test()).
            # Make them torch CPU then move inside forward
            x0 = x_test[0][s:e]
            x1 = x_test[1][s:e]
            x2 = x_test[2][s:e]

            if isinstance(x0, np.ndarray):
                x0 = torch.from_numpy(x0)
                x1 = torch.from_numpy(x1)
                x2 = torch.from_numpy(x2)

            y_pred_t = net((x0, x1, x2))          # torch tensor
            preds.append(y_pred_t.detach().cpu().numpy())

    y_pred = np.concatenate(preds, axis=0)       # (Nt,N,5)
    y_true = y_test                               # (Nt,N,5) numpy

    # ---------------- Field metrics (first N_field) ----------------
    yt = y_true[:, :Nf, :4].reshape(-1, 4)
    yp = y_pred[:, :Nf, :4].reshape(-1, 4)
    yt_inv = output_scalers.inverse_transform(yt)
    yp_inv = output_scalers.inverse_transform(yp)

    def calculate_r2_1d(a, b):
        ss_res = np.sum((a - b) ** 2)
        ss_tot = np.sum((a - np.mean(a)) ** 2) + 1e-12
        return 1 - ss_res / ss_tot

    comps = ['ux', 'uy', 'uz', 'vm']
    for j, c in enumerate(comps):
        mae = np.mean(np.abs(yp_inv[:, j] - yt_inv[:, j]))
        rmse = np.sqrt(np.mean((yp_inv[:, j] - yt_inv[:, j]) ** 2))
        r2 = calculate_r2_1d(yt_inv[:, j], yp_inv[:, j])
        logging.info(f"[Field] {c}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

    # ---------------- Occupancy metrics (grid block) ----------------
    occ_gt = y_true[:, Nf:, 4].reshape(-1)
    occ_logit = y_pred[:, Nf:, 4].reshape(-1)
    prob = 1.0 / (1.0 + np.exp(-occ_logit))
    pred = (prob >= 0.5).astype(np.float32)
    gt = occ_gt.astype(np.float32)

    acc = np.mean(pred == gt)
    inter = np.logical_and(pred > 0.5, gt > 0.5).sum()
    union = np.logical_or(pred > 0.5, gt > 0.5).sum()
    iou = inter / (union + 1e-12)
    bce = -np.mean(gt * np.log(prob + 1e-12) + (1 - gt) * np.log(1 - prob + 1e-12))
    logging.info(f"[Occupancy] BCE={bce:.6f}, ACC={acc:.6f}, IoU={iou:.6f}")

# =========================
# Main
# =========================
def main():
    args = parse_arguments()
    set_random_seed()
    device = get_device(args.gpu)

    name = (
        f'DesignGenNOStyle_OCC_RUN{args.RUN}_D{args.N_samples}_Npt{args.N_pt}_'
        f'branch_{args.branch_condition_input_components}_trunk_{args.trunk_input_components}_'
        f'out_{args.output_components}_split_{args.split_method}'
    )
    experiment_dir = setup_logging(args, name)
    log_parameters(args, device, name)

    x_train, y_train, x_test, y_test, output_scalers, train_case_file, test_case_file, N_field, N_grid = \
        load_and_preprocess_data(args, experiment_dir)
    print("x_train trunk bytes GB:", x_train[2].nbytes/1024**3)
    print("y_train bytes GB:", y_train.nbytes/1024**3)

    data = TripleCartesianProd(x_train, y_train, x_test, y_test, args=args, N_field=N_field, N_grid=N_grid)
    model, scheduler, optimizer = define_model(args, device, data, output_scalers, N_field=N_field)

    total_params = sum(p.numel() for p in model.net.parameters() if p.requires_grad)
    logging.info(f'Number of Trainable Parameters: {total_params}\n')

    train_model(model, scheduler, args, experiment_dir)
    post_eval(model, output_scalers, experiment_dir, N_field=N_field)

    logging.info(f"Saved final weights: {experiment_dir}/model_final.pth")


if __name__ == "__main__":
    main()