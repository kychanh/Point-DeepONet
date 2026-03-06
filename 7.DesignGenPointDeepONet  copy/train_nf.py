import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# =========================
# Design-NO NF code (AS-IS) + training stabilization (outside the NF code)
# =========================
try:
    from FunActivation import FunActivation
    from NormalizingFlow_utils import unconstrained_RQS
except:
    try:
        from .FunActivation import FunActivation
        from .NormalizingFlow_utils import unconstrained_RQS
    except:
        FunActivation = None
        unconstrained_RQS = None

from torch.distributions import MultivariateNormal

import math
import torch.nn.functional as F


class FCNN(nn.Module):
    """
    Simple fully connected neural network.
    """
    def __init__(self, in_dim, out_dim, hidden_dim, activation):
        super().__init__()
        if isinstance(activation, str):
            if FunActivation is None:
                raise RuntimeError("FunActivation not found. Put FunActivation.py in the same folder or adjust imports.")
            self.activation = FunActivation()(activation)
        else:
            self.activation = activation

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
    """
    Non-volume preserving flow. [Dinh et. al. 2017]
    """
    def __init__(self, dim, hidden_dim=20, activation='Tanh', base_network=FCNN):
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
    def __init__(self, dim, hidden_dim, activation, flow_type: str,
                 num_flows: int, device='cuda:0', **kwrds):
        super().__init__()
        self.prior = MultivariateNormal(
            torch.zeros(dim).to(device),
            torch.eye(dim).to(device)
        )
        self.flow_net = []
        for _ in range(num_flows):
            flow_layer = eval(flow_type)  # e.g., 'RealNVP'
            self.flow_net.append(flow_layer(dim, hidden_dim, activation, **kwrds))
        self.flow_net = nn.Sequential(*self.flow_net)

    def forward(self, x):
        batch_size = x.shape[0]
        log_det = torch.zeros(batch_size).to(x)
        for flow in self.flow_net:
            x, ld = flow.forward(x)
            log_det += ld
        z, prior_logprob = x, self.prior.log_prob(x)
        return z, prior_logprob, log_det

    def inverse(self, z):
        batch_size = z.shape[0]
        log_det = torch.zeros(batch_size).to(z)
        for flow in self.flow_net[::-1]:
            z, ld = flow.inverse(z)
            log_det += ld
        x = z
        return x, log_det

    def sample(self, n_samples):
        z = self.prior.sample((n_samples,))
        x, _ = self.inverse(z)
        return x


# =========================
# Encoder: 그대로 (main.py와 동일)
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
        x = pointnet_input.transpose(2, 1)
        x = self.pointnet_branch(x)
        latent = x.max(dim=2)[0]
        return latent


def map_keys_to_indices(keys, all_keys):
    return [np.where(all_keys == key)[0][0] for key in keys]


# =========================
# Stability helpers (NO changes to NF core code above)
# =========================
def init_small(module: nn.Module, w_std: float = 1e-2):
    """
    Make initial s/t outputs small by initializing Linear layers with small weights.
    This does NOT modify RealNVP/FCNN source code; it only sets params after construction.
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=w_std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


@torch.no_grad()
def compute_latent_norm_stats(X: torch.Tensor):
    mu = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True).clamp_min(1e-6)
    return mu, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_dir", type=str, required=True)
    ap.add_argument("--sampled_npz", type=str, required=True)   # data/sampled/Rpt*_N*.npz
    ap.add_argument("--split_npz", type=str, required=True)     # combined_*_split_*_train_valid.npz
    ap.add_argument("--device", type=str, default="cuda:0")

    ap.add_argument("--latent_dim", type=int, default=128)

    # NF hyperparams (Design-NO NFModel args 그대로)
    ap.add_argument("--flow_type", type=str, default="RealNVP")
    ap.add_argument("--num_flows", type=int, default=4)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--activation", type=str, default="SiLU")

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)  # 🔧 safer default
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--w_std_init", type=float, default=1e-2)
    ap.add_argument("--latent_clip", type=float, default=5.0, help="clip normalized latents to [-c,c] before NF")
    ap.add_argument("--save_name", type=str, default="nf_final.pth")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # -------------------------
    # 1) Load trained deterministic model and extract encoder weights
    # -------------------------
    ckpt_path = os.path.join(args.experiment_dir, "model_final.pth")
    # weights_only=True (new torch security best practice)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)

    encoder = Encoder(pointnet_input_dim=3, latent_dim=args.latent_dim, activation=nn.SiLU()).to(device).eval()
    encoder_state = {k.replace("encoder.", ""): v for k, v in state.items() if k.startswith("encoder.")}
    encoder.load_state_dict(encoder_state, strict=True)

    # -------------------------
    # 2) Load sampled npz + split indices
    # -------------------------
    tmp = np.load(args.sampled_npz)
    all_keys = np.array([key for key in tmp["c"]])

    split = np.load(args.split_npz)
    train_idx = map_keys_to_indices(split["train"], all_keys)

    # -------------------------
    # 3) Reuse pointnet_scaler.pkl (same as main training)
    # -------------------------
    pointnet_scaler = pickle.load(open(os.path.join(args.experiment_dir, "pointnet_scaler.pkl"), "rb"))

    xyz = tmp["a"][train_idx, :, :3].astype(np.float32)  # (Ntrain, Npt, 3) UNscaled
    ss = xyz.shape
    xyz_scaled = pointnet_scaler.transform(xyz.reshape(-1, 3)).reshape(ss).astype(np.float32)

    # -------------------------
    # 4) Extract latents X
    # -------------------------
    latents = []
    bs = args.batch_size
    with torch.no_grad():
        for i in tqdm(range(0, xyz_scaled.shape[0], bs), desc="Encoding latents"):
            batch = torch.from_numpy(xyz_scaled[i:i+bs]).to(device)
            z = encoder(batch)                  # (B, latent_dim)
            latents.append(z.detach().cpu())
    X = torch.cat(latents, dim=0).to(device)   # (Ntrain, latent_dim)

    # sanity check: encoder output should be finite
    if not torch.isfinite(X).all():
        bad = (~torch.isfinite(X)).any(dim=1).nonzero(as_tuple=False).squeeze(-1)[:10].tolist()
        raise RuntimeError(f"Encoder latents contain NaN/Inf. Example bad rows idx={bad}. Fix deterministic training first.")

    print(f"[latent] absmax={X.abs().max().item():.6f}  mean={X.mean().item():.6f}  std={X.std().item():.6f}")

    # -------------------------
    # 4.5) Normalize latents (critical to prevent exp(s) overflow)
    # -------------------------
    mu, std = compute_latent_norm_stats(X)
    Xn = (X - mu) / std
    if args.latent_clip > 0:
        Xn = Xn.clamp(-args.latent_clip, args.latent_clip)

    # save normalization stats for generation-time de-normalization
    norm_path = os.path.join(args.experiment_dir, "nf_latent_norm.pth")
    torch.save({"mu": mu.detach().cpu(), "std": std.detach().cpu(), "clip": float(args.latent_clip)}, norm_path)
    print("Saved latent normalization stats to:", norm_path)

    # -------------------------
    # 5) Train NFModel (Design-NO 그대로) with safe init + grad clip
    # -------------------------
    nf = NFModel(
        dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        activation=args.activation,
        flow_type=args.flow_type,
        num_flows=args.num_flows,
        device=str(device),
    ).to(device).train()

    # 🔧 safe small init (NO source change)
    nf.apply(lambda m: init_small(m, w_std=args.w_std_init))

    opt = optim.AdamW(nf.parameters(), lr=args.lr, weight_decay=1e-5)

    N = Xn.shape[0]
    for epoch in range(args.epochs):
        perm = torch.randperm(N, device=device)
        total = 0.0
        n_skipped = 0

        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            x_batch = Xn[idx]

            # last safety: ensure finite
            if not torch.isfinite(x_batch).all():
                n_skipped += 1
                continue

            z, prior_logprob, log_det = nf(x_batch)

            # if flow exploded, skip the batch (rare once normalized)
            if (not torch.isfinite(z).all()) or (not torch.isfinite(prior_logprob).all()) or (not torch.isfinite(log_det).all()):
                n_skipped += 1
                opt.zero_grad(set_to_none=True)
                continue

            loss = -(prior_logprob + log_det).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(nf.parameters(), max_norm=args.grad_clip)
            opt.step()

            total += loss.item() * x_batch.shape[0]

        denom = max(1, (N - n_skipped * bs))
        print(f"[NF] epoch {epoch+1}/{args.epochs}  nll={total / denom:.6f}  skipped_batches={n_skipped}")

    save_path = os.path.join(args.experiment_dir, args.save_name)
    torch.save(nf.state_dict(), save_path)
    print("Saved NF to:", save_path)


if __name__ == "__main__":
    main()