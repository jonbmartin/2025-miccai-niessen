"""
Image-domain INR with Fourier-space loss via Kaiser–Bessel (KB) interpolation
-----------------------------------------------------------------------------
- Single coil
- Time **binned** (bins can include multiple shots)
- Shared spatial trunk Phi(x,y) and hypernetwork last layer conditioned on bin id
- Forward model: evaluate I_b(x,y) on Cartesian grid (N x N * oversamp), FFT2,
  then interpolate to non-Cartesian k-samples using differentiable KB gather
- All points used each step (per user request)
- Device support: MPS -> CUDA -> CPU

Defaults (per user):
  grid_size N = 100, oversamp = 2, KB width J = 5, uniform weights

Note: We implement a **device-side** differentiable KB gather (indices+weights)
so gradients flow through FFT to the INR. A CPU CSR path could be added later
if desired, but is not needed for training.
"""
from __future__ import annotations
import os as _os
# macOS OpenMP runtime conflicts protection
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("OMP_NUM_THREADS", "1")

import math
import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.io as sio
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- matplotlib for visualization ---
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for file-saving
import matplotlib.pyplot as plt

# ------------------------------
# Device
# ------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()
print("Device:", DEVICE)

# ------------------------------
# .mat loader (single coil, chosen slice); returns flattened across shots
# ------------------------------

def load_rosette_mat(mat_path: str) -> Dict[str, torch.Tensor]:
    md = loadmat(mat_path)
    def _grab(k):
        if k not in md:
            raise KeyError(f"Key '{k}' not found in {mat_path}")
        return md[k]
    kx1 = np.asarray(_grab('kx1'))  # (nro, nshot)
    ky1 = np.asarray(_grab('ky1'))  # (nro, nshot)
    t   = np.asarray(_grab('t')).squeeze()  # (nro,)
    dr1 = np.asarray(_grab('dr1'))  # (nro, nshot, ncoil, nslice) complex
    if kx1.ndim != 2 or ky1.ndim != 2:
        raise ValueError("kx1/ky1 expected (nro,nshot)")
    if t.ndim != 1:
        raise ValueError("t expected (nro,)")
    if dr1.ndim != 4:
        raise ValueError("dr1 expected (nro,nshot,ncoil,nslice)")
    return {
        'kx1': torch.from_numpy(kx1.astype(np.float32)),
        'ky1': torch.from_numpy(ky1.astype(np.float32)),
        't':   torch.from_numpy(t.astype(np.float32)),
        'dr1': torch.from_numpy(dr1.astype(np.complex64)),
    }


def select_single_coil_timebin(
    data: Dict[str, torch.Tensor], coil: int, slice_: int, n_bins: int, shot_skip: int = 1
) -> Dict[str, torch.Tensor]:
    """Flatten across (optionally undersampled) shots for a single coil/slice; create time bin ids.
    Returns dict with:
      kx: (Ntot,), ky: (Ntot,), s: (Ntot,2), t: (Ntot,), bin_id: (Ntot,) int64 in [0,n_bins-1]
    Assumes kx/ky already in [-0.5,0.5]. RMS-normalizes s.
    """
    kx1 = data['kx1']  # (nro, nshot)
    ky1 = data['ky1']
    t   = data['t']    # (nro,)
    dr1 = data['dr1']  # (nro, nshot, ncoil, nslice)

    nro, nshot = kx1.shape
    _, nshot2, ncoil, nslice = dr1.shape
    if nshot != nshot2:
        raise ValueError("shot mismatch between kx/ky and dr1")
    if not (0 <= coil < ncoil):
        raise IndexError("coil out of range")
    if not (0 <= slice_ < nslice):
        raise IndexError("slice out of range")

    # Optional even-stride shot undersampling: select shots 0:nskip:end
    if shot_skip is None or shot_skip < 1:
        shot_skip = 1
    sel_shots_np = np.arange(0, nshot, shot_skip, dtype=np.int64)
    if sel_shots_np.size == 0:
        sel_shots_np = np.array([0], dtype=np.int64)
    # Subselect shots on second dimension for kx1, ky1, dr1
    kx1 = kx1[:, sel_shots_np]
    ky1 = ky1[:, sel_shots_np]
    dr1 = dr1[:, sel_shots_np, :, :]
    # Update nshot after selection
    nshot = kx1.shape[1]

    # Flatten all shots (C-order): for each readout index r, shots 0..nshot-1
    kx = kx1.reshape(-1)               # (nro*nshot,)
    ky = ky1.reshape(-1)
    
    # tile t across shots (assume same readout times per shot) for convenience
    t_long = t.repeat(nshot)           # (nro*nshot,)

    # complex signal for this coil/slice, flattened like kx/ky
    s_cplx = dr1[:, :, coil, slice_].reshape(-1)  # (nro*nshot,)
    s_ri = torch.view_as_real(s_cplx)             # (Ntot,2)

    # RMS normalize target
    amp = torch.sqrt((s_ri[...,0]**2 + s_ri[...,1]**2).mean())
    s_ri = s_ri / (amp + 1e-12)

    # --- New binning: between maxima of k-space magnitude over readout index ---
    # kmag per readout sample (sum over shots)
    kmag = (kx1.numpy()**2 + ky1.numpy()**2).sum(axis=1)  # shape (nro,)
    prom_thresh = 0.5 * kmag.max()  # approximate MATLAB MinProminence
    # simple local-max detector
    local = np.zeros_like(kmag, dtype=bool)
    if nro >= 3:
        local[1:-1] = (kmag[1:-1] > kmag[:-2]) & (kmag[1:-1] >= kmag[2:])
    peak_idx = np.where(local & (kmag >= prom_thresh))[0]
    # bin boundaries (0-based): include first and last index
    ts = np.unique(np.concatenate(([0], peak_idx, [nro-1]))).astype(np.int64)
    # ensure strictly increasing and at least two edges
    if ts.size < 2:
        ts = np.array([0, nro-1], dtype=np.int64)

    # assign bin id per readout index: left-inclusive, right-exclusive for inner edges
    inner_edges = ts[1:-1]
    bin_id_ro = np.digitize(np.arange(nro, dtype=np.int64), inner_edges, right=False).astype(np.int64)
    # expand to flattened (nro*nshot,), repeating each readout's bin across all shots
    bin_id = np.repeat(bin_id_ro, nshot)

    bin_id = torch.from_numpy(bin_id).long()

    # shot index per sample (0..nshot-1) matching the flatten order
    # (for each readout r, shots 0..nshot-1)
    shot_idx = torch.from_numpy(np.tile(np.arange(nshot, dtype=np.int64), nro)).long()

    return {
        'kx': kx.float(),
        'ky': ky.float(),
        's': s_ri.float(),
        't': t_long.float(),
        'bin_id': bin_id,
        'shot_idx': shot_idx,
        'nro': torch.tensor(nro, dtype=torch.long),
        'nshot': torch.tensor(nshot, dtype=torch.long),
    }


# ------------------------------
# INR: spatial trunk + hypernetwork last layer (per-bin)
# ------------------------------

class FourierFeatures(nn.Module):
    """Gaussian Fourier features for coordinates.
    x: (..., in_dims). Returns [..., out_dims] with sin/cos features.
    out_dims = 2 * in_dims * n_freqs (+ in_dims if include_input).
    """
    def __init__(self, in_dims: int, n_freqs: int = 10, std: float = 8.0, include_input: bool = True, seed: int = 0):
        super().__init__()
        self.in_dims = in_dims
        self.n_freqs = n_freqs
        self.std = std
        self.include_input = include_input
        # deterministic random matrix using a fixed CPU generator
        g = torch.Generator(device='cpu')
        g.manual_seed(int(seed))
        B = torch.randn(in_dims, n_freqs, generator=g) * std  # (in_dims, n_freqs)
        self.register_buffer('B', B)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_dims)
        # project to ( ... , n_freqs ) per input dim
        # shape after matmul: (..., n_freqs) per dim, then stack dims
        # we can fold by multiplying by B for each dim jointly:
        # compute x @ B for each dim independently by elementwise multiply + sum
        # Easier: expand to (..., in_dims, 1) and multiply with B (in_dims, n_freqs)
        x_exp = x.unsqueeze(-1)                               # (..., in_dims, 1)
        proj = (x_exp * self.B)                               # (..., in_dims, n_freqs)
        proj = proj.reshape(*x.shape[:-1], -1)                # (..., in_dims*n_freqs)
        # 2π for better coverage
        proj = 2 * math.pi * proj
        sinus = torch.sin(proj)
        cosin = torch.cos(proj)
        out = torch.cat([sinus, cosin], dim=-1)               # (..., 2*in_dims*n_freqs)
        if self.include_input:
            out = torch.cat([x, out], dim=-1)
        return out

def pe_out_size(in_dims: int, n_freqs: int, include_input: bool) -> int:
    base = 2 * in_dims * n_freqs
    return base + (in_dims if include_input else 0)

class SpatialTrunk(nn.Module):
    def __init__(self, in_dims=2, hidden=256, depth=5, feat_dim=128):
        super().__init__()
        layers = [nn.Linear(in_dims, hidden), nn.ReLU(inplace=True)]
        for _ in range(depth-2):
            layers += [nn.Linear(hidden, hidden), nn.ReLU(inplace=True)]
        layers += [nn.Linear(hidden, feat_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):  # x: (...,2)
        return self.net(x)

class LastLayerHyper(nn.Module):
    """Given bin id in [0..B-1] mapped to normalized scalar u in [0,1],
    output last-layer weights W_b (feat_dim x 2) and bias b_b (2,).
    """
    def __init__(self, hidden=64, feat_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, feat_dim*2 + 2)
        )
        self.feat_dim = feat_dim
    def forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # u: (K,1) in [0,1]
        p = self.mlp(u)  # (K, feat_dim*2+2)
        W = p[..., : self.feat_dim*2].reshape(-1, self.feat_dim, 2)
        b = p[..., self.feat_dim*2 : ]  # (K,2)
        return W, b

# ------------------------------
# KB interpolation (separable Kaiser–Bessel), differentiable gather on device
# ------------------------------

def i0_approx(x: torch.Tensor) -> torch.Tensor:
    """
    MPS-safe differentiable approximation to modified Bessel I0(x).
    Coefficients from Abramowitz–Stegun / cephes.
    """
    ax = x.abs()
    # Branch 1: |x| < 3.75  → polynomial
    t  = ax / 3.75
    t2 = t * t
    p1 = 1.0 + t2*(3.5156229 + t2*(3.0899424 + t2*(1.2067492 +
         t2*(0.2659732 + t2*(0.0360768 + t2*0.0045813)))))

    # Branch 2: |x| >= 3.75 → asymptotic
    y  = 3.75 / (ax + 1e-12)
    p2 = (0.39894228 + y*(0.01328592 + y*(0.00225319 + y*(-0.00157565 +
         y*(0.00916281 + y*(-0.02057706 + y*(0.02635537 +
         y*(-0.01647633 + y*0.00392377))))))))
    p2 = torch.exp(ax) / torch.sqrt(ax + 1e-12) * p2

    return torch.where(ax < 3.75, p1, p2)

def kaiser_bessel(u: torch.Tensor, J: int, alpha: float, beta: float) -> torch.Tensor:
    """
    KB kernel value for distance u (|u| <= J/2). Outside -> 0.
    Uses i0_approx for MPS safety.
    """
    out = torch.zeros_like(u)
    mask = (u.abs() <= (J / 2))
    if mask.any():
        x = u[mask]
        # sqrt(1 - (2x/J)^2) guarded for numerical safety
        t = torch.sqrt(torch.clamp(1 - (2 * x / J) ** 2, min=0.0) + 1e-12)
        beta_t = torch.as_tensor(beta, device=u.device, dtype=u.dtype) * t
        beta_s = torch.as_tensor(beta, device=u.device, dtype=u.dtype)
        num = i0_approx(beta_t)
        den = i0_approx(beta_s) + 1e-12
        out[mask] = num / den
    return out

def kb_beta(J: int, alpha: float) -> float:
    # Fessler NUFFT heuristic for Kaiser–Bessel beta
    return math.pi * math.sqrt((J/alpha * (alpha - 0.5))**2 - 0.8)


def precompute_kb_indices_weights(
    kx: torch.Tensor, ky: torch.Tensor, Nk: int, J: int, alpha: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute neighbor indices (Nsamp, J*J) into a flattened Nk x Nk grid
    and separable weights (Nsamp, J*J) for KB kernel. All on DEVICE.
    kx, ky in [-0.5,0.5]. Grid index 0..Nk-1 with 0 at -0.5 and Nk-1 at +0.5.
    """
    device = kx.device
    dtype = torch.float32
    beta = kb_beta(J, alpha)

    # map to continuous grid coords
    gx = (kx + 0.5) * (Nk - 1)
    gy = (ky + 0.5) * (Nk - 1)

    # left-most integer index for window
    half = J // 2
    # ensure we cover J points centered around
    ix0 = torch.floor(gx).to(torch.int64) - half
    iy0 = torch.floor(gy).to(torch.int64) - half

    # neighbor offsets 0..J-1
    offs = torch.arange(0, J, device=device, dtype=torch.int64)
    ix = (ix0[:, None] + offs[None, :]).clamp(0, Nk-1)  # (Nsamp,J)
    iy = (iy0[:, None] + offs[None, :]).clamp(0, Nk-1)  # (Nsamp,J)

    # distances for kernel
    dx = (gx[:, None] - ix.to(dtype))
    dy = (gy[:, None] - iy.to(dtype))
    wx = kaiser_bessel(dx, J, alpha, beta)  # (Nsamp,J)
    wy = kaiser_bessel(dy, J, alpha, beta)  # (Nsamp,J)
    # separable 2D weights: outer product per sample -> (Nsamp,J,J)
    w2d = wx[:, :, None] * wy[:, None, :]
    w2d = w2d.reshape(-1, J*J)

    # build flat indices into Nk*Nk grid
    # flatten in row-major (y major): idx = iy*Nk + ix for each pair
    ix_b = ix[:, :, None].expand(-1, -1, J)
    iy_b = iy[:, None, :].expand(-1, J, -1)
    idx2d = iy_b * Nk + ix_b  # (Nsamp,J,J)
    idx = idx2d.reshape(-1, J*J)

    # normalize weights per sample to sum to 1 (optional; keeps scale stable)
    wsum = w2d.sum(dim=1, keepdim=True) + 1e-12
    w2d = w2d / wsum

    return idx.to(device), w2d.to(device)

def compute_kb_rolloff_map(Nk: int, J: int, alpha: float, device: torch.device) -> torch.Tensor:
    beta = kb_beta(J, alpha)

    # place window exactly at the integer center pixel
    cx = Nk // 2
    cy = Nk // 2
    half = J // 2
    offs = torch.arange(0, J, dtype=torch.int64)

    ix0 = cx - half
    iy0 = cy - half
    ix = (ix0 + offs).clamp(0, Nk-1)   # x indices (columns)
    iy = (iy0 + offs).clamp(0, Nk-1)   # y indices (rows)

    # distances from the integer center (no half-pixel offset)
    dx = (ix.to(torch.float32) - float(cx))
    dy = (iy.to(torch.float32) - float(cy))

    wx = kaiser_bessel(dx, J, alpha, beta)  # (J,)
    wy = kaiser_bessel(dy, J, alpha, beta)  # (J,)
    w2d = wx[:, None] * wy[None, :]         # (J, J)

    grid_k = torch.zeros((Nk, Nk), dtype=torch.complex64)
    for ii in range(J):
        for jj in range(J):
            grid_k[iy[jj].item(), ix[ii].item()] = complex(w2d[ii, jj].item(), 0.0)

    grid_k_cpu = grid_k.to('cpu')
    psf_cplx = torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(grid_k_cpu, dim=(-2, -1)), norm='ortho'),
        dim=(-2, -1)
    )

    # use magnitude, not real part; then normalize center to 1
    psf = torch.abs(psf_cplx)
    cval = psf[cy, cx].item() if psf[cy, cx].abs() > 0 else 1.0
    psf = (psf / cval).to(device=device, dtype=torch.float32)
    return psf

# ------------------------------
# FFT helpers (centered, orthonormal) working on 2-channel complex
# ------------------------------

def fft2c(img_ri: torch.Tensor) -> torch.Tensor:
    """
    Centered, orthonormal 2D FFT on a 2‑channel complex image (.., H, W, 2).
    On Apple MPS, complex tensors are unsupported, so we hop to CPU for FFT
    and return to the original device with gradients preserved.
    """
    if img_ri.device.type == 'mps':
        img_cpu = img_ri.to('cpu')  # differentiable device transfer
        z_cpu = torch.view_as_complex(img_cpu)
        k_cpu = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(z_cpu, dim=(-2, -1)), norm='ortho'),
            dim=(-2, -1)
        )
        return torch.view_as_real(k_cpu).to(img_ri.device)
    else:
        z = torch.view_as_complex(img_ri)
        k = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(z, dim=(-2, -1)), norm='ortho'),
            dim=(-2, -1)
        )
        return torch.view_as_real(k)
    
# ------------------------------
# Haar wavelet helpers (orthonormal, decimated)
# ------------------------------
def _haar_filters(device, dtype=torch.float32):
    s2 = math.sqrt(0.5)
    h0 = torch.tensor([s2, s2], device=device, dtype=dtype)   # low-pass
    h1 = torch.tensor([s2, -s2], device=device, dtype=dtype)  # high-pass
    return h0, h1

# 2D Haar, one level, for batch (B,1,H,W) → (LL,LH,HL,HH) each (B,1,H/2,W/2)
def _dwt2_haar_level(x: torch.Tensor):
    B, C, H, W = x.shape
    assert C == 1
    h0, h1 = _haar_filters(x.device, x.dtype)
    kLL = torch.ger(h0, h0)  # (2,2)
    kLH = torch.ger(h0, h1)
    kHL = torch.ger(h1, h0)
    kHH = torch.ger(h1, h1)
    weight = torch.stack([kLL, kLH, kHL, kHH], dim=0).unsqueeze(1)  # (4,1,2,2)
    y = F.conv2d(x, weight, stride=2, padding=0)
    LL, LH, HL, HH = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]
    return LL, LH, HL, HH

# 1D Haar, one level, along last dim for batch (Npix,1,T) → (LL,LH)
def _dwt1_haar_level_t(x: torch.Tensor):
    h0, h1 = _haar_filters(x.device, x.dtype)
    w1d = torch.stack([h0, h1], dim=0).unsqueeze(1)  # (2,1,2)
    y = F.conv1d(x, w1d, stride=2, padding=0)
    LL, LH = y[:, 0:1, :], y[:, 1:2, :]
    return LL, LH

# Multi-level L1 on 2D detail subbands for (B,H,W) images
def wavelet_l1_xy(vol: torch.Tensor, levels: int = 1) -> torch.Tensor:
    B, H, W = vol.shape
    x = vol.view(B, 1, H, W)
    total = 0.0
    LL = x
    for _ in range(max(1, int(levels))):
        LL, LH, HL, HH = _dwt2_haar_level(LL)
        total = total + LH.abs().sum() + HL.abs().sum() + HH.abs().sum()
    return total / (B * H * W)

# Multi-level L1 along t for each pixel; (H,W,B) -> scalar
def wavelet_l1_t(vol: torch.Tensor, levels: int = 1) -> torch.Tensor:
    H, W, B = vol.shape
    x = vol.reshape(H * W, 1, B)
    total = 0.0
    LL = x
    L = max(1, int(levels))
    for _ in range(L):
        if LL.shape[-1] < 2:
            break
        LL, LH = _dwt1_haar_level_t(LL)
        total = total + LH.abs().sum()
    return total / (H * W * B)

# ------------------------------
# Config
# ------------------------------

@dataclass
class TrainCfg:
    steps: int = 2000
    lr: float = 1e-3
    verbose: int = 100

    # grid & interpolation
    grid_size: int = 100     # image grid (pre-oversampling)
    oversamp: float = 2.0    # oversampling factor for FFT grid
    kb_width: int = 5        # Kaiser–Bessel J

    # model
    trunk_hidden: int = 256
    trunk_depth: int = 5
    feat_dim: int = 128
    hyper_hidden: int = 64

    # bins
    n_bins: int = 8

    # visualization
    viz_dir: str = 'viz_img_inr'
    viz_window: int = 2000  # number of samples to plot in the |s| panel
    viz_shot: int = 0  # which shot index to visualize along Nro

    # density compensation in loss
    dcw: str = 'radial'  # options: 'none', 'radial'
    dcw_eps: float = 1e-3

    # spatial Fourier features (x,y)
    use_xy_pe: bool = True
    xy_pe_nfreqs: int = 10
    xy_pe_std: float = 8.0
    xy_pe_include_input: bool = True

    # Fourier features seed
    xy_pe_seed: int = 0

    # wavelet sparsity (Haar) on image magnitude
    lambda_wav_xy: float = 0.0    # L1 of 2D detail subbands per bin
    lambda_wav_t: float = 0.0     # L1 of 1D temporal detail per pixel across bins
    wav_levels_xy: int = 1        # levels for 2D Haar
    wav_levels_t: int = 1         # levels for 1D Haar along bins

# ------------------------------
# Build Cartesian coords
# ------------------------------

def make_xy_grid(N: int, device: torch.device) -> torch.Tensor:
    xs = torch.linspace(-0.5, 0.5, N, device=device)
    ys = torch.linspace(-0.5, 0.5, N, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # (N*N,2)

# ------------------------------
# Visualization helper
# ------------------------------
from typing import Optional
def save_training_viz(step: int,
                    s_true_ri: torch.Tensor,
                    s_pred_ri: torch.Tensor,
                    shot_idx: torch.Tensor,
                    img_mid_ri: Optional[torch.Tensor],
                    cfg: TrainCfg) -> None:
    """Save a figure with (1) |s_meas| vs |s_pred| **along Nro for a given shot**
    and (2) magnitude image from the middle bin.
    - s_true_ri, s_pred_ri: (Nsamp, 2)
    - shot_idx: (Nsamp,) long, values in [0..nshot-1]
    - img_mid_ri: (Nk, Nk, 2) or None
    """
    import os
    os.makedirs(cfg.viz_dir, exist_ok=True)
    # Select the requested shot
    si = shot_idx.detach().cpu().numpy().astype(np.int64)
    s_true = s_true_ri.detach().cpu().numpy()
    s_pred = s_pred_ri.detach().cpu().numpy()
    mask = (si == int(cfg.viz_shot))
    if not np.any(mask):
        # nothing to plot for this shot
        return
    s_true_shot = s_true[mask]
    s_pred_shot = s_pred[mask]
    mag_true = np.sqrt(s_true_shot[...,0]**2 + s_true_shot[...,1]**2)
    mag_pred = np.sqrt(s_pred_shot[...,0]**2 + s_pred_shot[...,1]**2)
    nro_len = mag_true.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    # Panel 1: full Nro window for this shot
    ax = axes[0]
    ax.plot(np.arange(nro_len), mag_true, label='|meas|', linewidth=1.0)
    ax.plot(np.arange(nro_len), mag_pred, label='|pred|', linewidth=1.0)
    ax.set_title(f'|s| along Nro — shot {cfg.viz_shot} — step {step}')
    ax.set_xlabel('readout index (within shot)')
    ax.set_ylabel('|s|')
    ax.legend()
    # Panel 2: middle-bin image magnitude (if provided)
    ax = axes[1]
    if img_mid_ri is not None:
        img_mid = img_mid_ri.detach().cpu().numpy()
        img_mag = np.sqrt(img_mid[...,0]**2 + img_mid[...,1]**2)
        im = ax.imshow(img_mag, origin='lower', cmap='gray')
        ax.set_title('Middle-bin image |I|')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.text(0.5, 0.5, 'no middle-bin image', ha='center', va='center')
        ax.set_axis_off()
    out_path = os.path.join(cfg.viz_dir, f'viz_step_{step:05d}.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# ------------------------------
# Simple image saver for 2D arrays
# ------------------------------
def save_png(arr2d: torch.Tensor, out_path: str, title: Optional[str] = None):
    arr = arr2d.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(arr, origin='lower')
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# ------------------------------
# Training
# ------------------------------

def train_img_inr_kb(
    kx: torch.Tensor, ky: torch.Tensor, s_ri: torch.Tensor, bin_id: torch.Tensor,
    shot_idx: torch.Tensor, cfg: TrainCfg
) -> Dict:
    """Train image-domain INR with Fourier-space KB loss.
    kx,ky: (Nsamp,), s_ri: (Nsamp,2) (R,I), bin_id: (Nsamp,) in [0..B-1]
    """
    # Move inputs to device (keep bin_id on CPU for grouping convenience too)
    kx = kx.to(DEVICE)
    ky = ky.to(DEVICE)
    s_ri = s_ri.to(DEVICE)
    bin_id = bin_id.to(torch.long).to(DEVICE)
    shot_idx = shot_idx.to(torch.long).to(DEVICE)

    # Density compensation weights per sample
    if cfg.dcw == 'radial':
        r = torch.sqrt(kx**2 + ky**2)
        dcw = 1.0 * (r + cfg.dcw_eps)
        dcw = dcw / (dcw.mean() + 1e-12)
    else:
        dcw = torch.ones_like(kx)

    # Prep grid and trunk
    N = cfg.grid_size
    Nk = int(round(cfg.grid_size * cfg.oversamp))

    # Build high-res coordinate grid for INR eval (Nk x Nk) and optional PE
    xy_lo = make_xy_grid(N, DEVICE)  # (Nk*Nk, 2)  # FOV handled via k-space scaling
    if cfg.use_xy_pe:
        pe = FourierFeatures(
            in_dims=2,
            n_freqs=cfg.xy_pe_nfreqs,
            std=cfg.xy_pe_std,
            include_input=cfg.xy_pe_include_input,
            seed=cfg.xy_pe_seed,
        ).to(DEVICE)
        xy_feat_lo = pe(xy_lo)
        trunk_in_dims = xy_feat_lo.shape[-1]
    else:
        xy_feat_lo = xy_lo
        trunk_in_dims = 2

    trunk = SpatialTrunk(in_dims=trunk_in_dims, hidden=cfg.trunk_hidden, depth=cfg.trunk_depth, feat_dim=cfg.feat_dim).to(DEVICE)
    hyper = LastLayerHyper(hidden=cfg.hyper_hidden, feat_dim=cfg.feat_dim).to(DEVICE)

    opt = torch.optim.Adam(trunk.parameters(), lr=cfg.lr, weight_decay=1e-4)
    opt_h = torch.optim.Adam(hyper.parameters(), lr=cfg.lr, weight_decay=1e-4)

    # Precompute per-bin sample lists and KB indices/weights
    B = int(bin_id.max().item()) + 1
    # Map sample times to bins already done; ensure only non-empty bins used
    bins_present = torch.unique(bin_id.cpu()).tolist()

    print('number of bins =', B)

    # Ensure viz dir exists
    _os.makedirs(cfg.viz_dir, exist_ok=True)

    # Precompute indices/weights per bin on device
    kb_cache = {}
    for b in bins_present:
        mask = (bin_id == b)
        idx, w = precompute_kb_indices_weights(kx[mask], ky[mask], Nk=Nk, J=cfg.kb_width, alpha=2.34*cfg.kb_width)
        kb_cache[b] = (mask.to(DEVICE), idx, w)

    # Normalized bin scalars u in [0,1] for hypernetwork
    if B > 1:
        u_bins = torch.linspace(0.0, 1.0, B, device=DEVICE).view(B, 1)
    else:
        u_bins = torch.zeros((1,1), device=DEVICE)

    # Precompute KB roll-off map on Nk and crop to N for image-domain de-apodization
    rolloff_Nk = compute_kb_rolloff_map(Nk=Nk, J=cfg.kb_width, alpha=cfg.oversamp, device=DEVICE)  # (Nk,Nk)
    if Nk == N:
        rolloff_lo = rolloff_Nk
    else:
        pad = Nk - N
        h0 = pad // 2
        w0 = pad - h0
        # central crop to N×N
        rolloff_lo = rolloff_Nk[h0:h0+N, h0:h0+N]
    # add channel dim for broadcasting with (N,N,2)
    rolloff_lo = torch.clamp(rolloff_lo, min=0.1).unsqueeze(-1)
    
    # Save diagnostic PNGs for PSF and rolloff
    try:
        _os.makedirs(cfg.viz_dir, exist_ok=True)
        rl = rolloff_lo.squeeze(-1) if rolloff_lo.ndim == 3 else rolloff_lo  # (N,N)
        save_png(rl, _os.path.join(cfg.viz_dir, 'rolloff_lo.png'), title='KB rolloff (N×N)')
    except Exception as _e:
        # keep training even if viz save fails
        pass

    # training loop
    for it in range(1, cfg.steps + 1):
        opt.zero_grad(set_to_none=True)
        opt_h.zero_grad(set_to_none=True)

        # 1) Evaluate shared spatial trunk features on (N x N)
        feat = trunk(xy_feat_lo).reshape(N, N, cfg.feat_dim)  # (N,N,D)

        total_loss = 0.0

        # storage for full predicted s over all samples (for plotting)
        full_pred = torch.zeros_like(s_ri)
        # choose the middle bin among bins_present for image display
        mid_b = 2#bins_present[len(bins_present)//2] if len(bins_present) > 0 else None
        mid_img_for_viz = None

        imgs_mag_list = []  # collect |I_b_lo| per bin this iteration

        # 2) For each present bin, form image via hyper last layer, FFT, and KB gather
        for b in bins_present:
            mask, neigh_idx, neigh_w = kb_cache[b]  # mask over samples, (Ns,J*J), (Ns,J*J)
            if mask.sum() == 0:
                continue
            # Hyper last layer params for bin b
            u = u_bins[b:b+1]  # (1,1)
            W, bias = hyper(u)  # W: (1,D,2), bias: (1,2)
            W = W[0]            # (D,2)
            bias = bias[0]      # (2,)

            # Image for bin b on base grid: I_b_lo(x,y) = feat @ W + bias
            # feat: (N,N,D) -> (N,N,2)
            I_b_lo = torch.einsum('xyd,dc->xyc', feat, W) + bias  # (N,N,2)
            #I_b_lo = I_b_lo / rolloff_lo # kernel rolloff correction

            # collect magnitude image for wavelet sparsity
            I_b_mag = torch.sqrt(I_b_lo[...,0]**2 + I_b_lo[...,1]**2 + 1e-12)
            imgs_mag_list.append(I_b_mag)

            # Zero‑pad to Nk×Nk **before** FFT to achieve oversampling without changing FOV
            if Nk == N:
                I_b_pad = I_b_lo
            else:
                pad = Nk - N
                pad_h_left = pad // 2
                pad_h_right = pad - pad_h_left
                pad_w_left = pad // 2
                pad_w_right = pad - pad_w_left
                I_b_pad = torch.zeros((Nk, Nk, 2), dtype=I_b_lo.dtype, device=I_b_lo.device)
                I_b_pad[pad_h_left:pad_h_left+N, pad_w_left:pad_w_left+N, :] = I_b_lo

            # FFT to k-space (Nk,Nk,2)
            K_b = fft2c(I_b_pad)

            # Gather predicted samples using KB weights (differentiable)
            K_flat = K_b.reshape(-1, 2)  # (Nk*Nk,2)
            # indices: (Ns,J*J)
            vals = K_flat.index_select(0, neigh_idx.reshape(-1))  # (Ns*JJ,2)
            vals = vals.reshape(neigh_idx.shape[0], neigh_idx.shape[1], 2)  # (Ns,JJ,2)
            w = neigh_w.unsqueeze(-1)  # (Ns,JJ,1)
            pred = (vals * w).sum(dim=1)  # (Ns,2)

            # fill predicted values for this bin into the full array
            full_pred[mask] = pred
            # keep the middle bin image for plotting at verbose steps
            if mid_b is not None and b == mid_b:
                mid_img_for_viz = I_b_lo

            # Compute weighted MSE with density compensation weight
            target = s_ri[mask]
            w_b = dcw[mask]
            res = pred - target                 # (Nb,2)
            mse = (res**2).sum(dim=-1)         # (Nb,) complex magnitude squared
            loss_b = (w_b * mse).mean()
            total_loss = total_loss + loss_b
        
        # Wavelet sparsity on image magnitude
        if cfg.lambda_wav_xy > 0.0 and len(imgs_mag_list) > 0:
            vol_xy = torch.stack(imgs_mag_list, dim=0)  # (B_present,N,N)
            total_loss = total_loss + cfg.lambda_wav_xy * wavelet_l1_xy(vol_xy, levels=cfg.wav_levels_xy)
        if cfg.lambda_wav_t > 0.0 and len(imgs_mag_list) > 1:
            vol_t = torch.stack(imgs_mag_list, dim=0).permute(1, 2, 0)  # (N,N,B_present)
            total_loss = total_loss + cfg.lambda_wav_t * wavelet_l1_t(vol_t, levels=cfg.wav_levels_t)

        # Visualization at verbose cadence
        if it % cfg.verbose == 0:
            try:
                save_training_viz(it, s_ri, full_pred, shot_idx, mid_img_for_viz, cfg)
            except Exception as _e:
                # keep training even if viz fails
                pass

        # 3) Backprop
        total_loss.backward()
        opt.step(); opt_h.step()

        if it % cfg.verbose == 0 or it in (1, cfg.steps):
            print(f"[img-INR] step {it}/{cfg.steps} loss={float(total_loss):.4e}")

    # -------------------------------------------------------------
    # Save reconstructed images for all bins to a .mat file
    #   - I_re, I_im, I_mag have shape (N, N, B_present)
    #   - u_bins saved as (B,1)
    # -------------------------------------------------------------
    try:
        _os.makedirs(cfg.viz_dir, exist_ok=True)
        # Recompute features on base grid with final trunk
        feat_final = trunk(xy_feat_lo).reshape(N, N, cfg.feat_dim)  # (N,N,D)
        B_present = len(bins_present)
        I_re = np.zeros((N, N, B_present), dtype=np.float32)
        I_im = np.zeros((N, N, B_present), dtype=np.float32)
        I_mag = np.zeros((N, N, B_present), dtype=np.float32)
        u_save = []
        for bi, b in enumerate(bins_present):
            u = u_bins[b:b+1]  # (1,1)
            W, bias = hyper(u)  # W: (1,D,2), bias: (1,2)
            W = W[0]
            bias = bias[0]
            I_b_lo = torch.einsum('xyd,dc->xyc', feat_final, W) + bias  # (N,N,2)
            I_b_np = I_b_lo.detach().cpu().numpy()
            I_re[..., bi] = I_b_np[..., 0]
            I_im[..., bi] = I_b_np[..., 1]
            I_mag[..., bi] = np.sqrt(I_b_np[..., 0]**2 + I_b_np[..., 1]**2)
            u_save.append(float(u.item()))
        mat_out_path = _os.path.join(cfg.viz_dir, 'inr_images.mat')
        sio.savemat(mat_out_path, {
            'I_re': I_re,
            'I_im': I_im,
            'I_mag': I_mag,
            'u_bins': np.array(u_save, dtype=np.float32).reshape(-1, 1),
            'grid_size': np.array([N], dtype=np.int32),
            'bins_present': np.array(bins_present, dtype=np.int32),
        })
        print(f"Saved reconstructed INR images to {mat_out_path}")
    except Exception as _e:
        print(f"[warn] Could not save .mat images: {_e}")
        
    # Return latest features and heads for potential rendering
    return {
        'trunk': trunk,
        'hyper': hyper,
        'Nk': Nk,
        'bins_present': bins_present,
        'kb_cache': kb_cache,
    }

# ------------------------------
# Main
# ------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image-domain INR with KB Fourier loss (single coil)')
    parser.add_argument('--mat', type=str, required=True, help='Path to rosette_raw_data.mat')
    parser.add_argument('--coil', type=int, default=0)
    parser.add_argument('--slice', dest='slice_', type=int, default=2)
    parser.add_argument('--n-bins', type=int, default=9, help='Number of time bins (across t in ms)')
    parser.add_argument('--grid-size', type=int, default=100)
    parser.add_argument('--oversamp', type=float, default=2.0)
    parser.add_argument('--kb-width', type=int, default=5)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--verbose', type=int, default=100)
    parser.add_argument('--dcw', type=str, default='radial', choices=['none', 'radial'],
                        help='Density compensation weighting in loss')
    parser.add_argument('--dcw-eps', type=float, default=1e-3, help='Epsilon for radial DCW (1/(r+eps))')
    parser.add_argument('--no-xy-pe', action='store_true', help='Disable (x,y) Fourier features')
    parser.add_argument('--xy-pe-nfreqs', type=int, default=64, help='Number of Fourier feature bands for (x,y)')
    parser.add_argument('--xy-pe-std', type=float, default=8.0, help='Std of Gaussian projection for PE')
    parser.add_argument('--xy-pe-include-input', action='store_true', help='Include raw (x,y) alongside PE')
    parser.add_argument('--xy-pe-seed', type=int, default=0, help='Seed for Gaussian Fourier features')
    parser.add_argument('--shot-skip', type=int, default=6,
                        help='Even-stride shot undersampling: keep shots 0:shot-skip:end (1 = keep all)')
    parser.add_argument('--lambda-wav-xy', type=float, default=0e-1, help='L1 weight for 2D Haar sparsity on image magnitude (x,y)')
    parser.add_argument('--lambda-wav-t', type=float, default=0e-1, help='L1 weight for 1D Haar sparsity along bins (t) of image magnitude')
    parser.add_argument('--wav-levels-xy', type=int, default=2, help='Levels for 2D Haar on (x,y)')
    parser.add_argument('--wav-levels-t', type=int, default=2, help='Levels for 1D Haar along t')
    args = parser.parse_args()

    # Load & prepare data
    data = load_rosette_mat(args.mat)
    pick = select_single_coil_timebin(data, coil=args.coil, slice_=args.slice_, n_bins=args.n_bins, shot_skip=args.shot_skip)

    cfg = TrainCfg(
        steps=args.steps, lr=args.lr, verbose=max(100, args.verbose),
        grid_size=args.grid_size, oversamp=args.oversamp, kb_width=args.kb_width,
        trunk_hidden=256, trunk_depth=5, feat_dim=128, hyper_hidden=64,
        #trunk_hidden=64, trunk_depth=2, feat_dim=128, hyper_hidden=64,
        n_bins=args.n_bins,
        dcw=args.dcw,
        dcw_eps=args.dcw_eps,
        use_xy_pe=not args.no_xy_pe,
        xy_pe_nfreqs=args.xy_pe_nfreqs,
        xy_pe_std=args.xy_pe_std,
        xy_pe_include_input=args.xy_pe_include_input or True,
        xy_pe_seed=args.xy_pe_seed,
        lambda_wav_xy=args.lambda_wav_xy,
        lambda_wav_t=args.lambda_wav_t,
        wav_levels_xy=args.wav_levels_xy,
        wav_levels_t=args.wav_levels_t,
    )

    out = train_img_inr_kb(pick['kx'], pick['ky'], pick['s'], pick['bin_id'], pick['shot_idx'], cfg)
