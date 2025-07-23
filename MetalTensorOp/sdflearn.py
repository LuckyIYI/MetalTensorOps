#!/usr/bin/env python3
# siren_svg_sdf.py  –  (c) 2025  MIT-licensed
# -------------------------------------------------------------
import argparse, math, numpy as np, matplotlib.pyplot as plt, torch, torch.nn as nn
from svgpathtools import svg2paths2, Path, Line, CubicBezier, QuadraticBezier, Arc

# ---------------------------------------------------------------------------
# 1)  Sample every SVG path densely  → list[2-D points] (poly-line)
# ---------------------------------------------------------------------------
def sample_svg(svg_file, pts_per_unit=2.0):
    """Return a Nx2 float32 array of densely-sampled points along *all* paths."""
    paths, attrs, _ = svg2paths2(svg_file)
    all_pts = []
    for path, attr in zip(paths, attrs):
        # svgpathtools works in the SVG's native coordinate system.
        length = path.length()
        n      = max(8, int(length * pts_per_unit))
        ts     = np.linspace(0.0, 1.0, n, endpoint=False)
        pts    = np.stack([path.point(t) for t in ts]).view(np.float64).reshape(-1, 2)
        all_pts.append(pts)
    poly = np.concatenate(all_pts, axis=0)
    # Normalise to roughly fit [-1,1]^2 – easy for networks:
    cx, cy = poly.mean(axis=0)
    scale  = (poly.max() - poly.min()) * 0.6          # shrink a bit
    poly   = (poly - (cx, cy)) / scale
    return poly.astype(np.float32)

# ---------------------------------------------------------------------------
# 2)  SDF from a poly-line – brute-force but robust for ≤ few 10k points
# ---------------------------------------------------------------------------
def sdf_from_poly(poly, res=256):
    """Return grid coords (N²×2) and SDF; uses even spacing in [-1,1]²."""
    xs = np.linspace(-1, 1, res); ys = np.linspace(-1, 1, res)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)

    # Precompute edges for distance queries
    seg_a = poly
    seg_b = np.roll(poly, -1, axis=0)

    def dist_point_to_segment(p, a, b):
        pa, ba = p - a, b - a
        h = np.clip(np.einsum('...d,...d', pa, ba) / (ba@ba + 1e-12), 0, 1)
        proj = a + h[...,None]*ba
        return np.linalg.norm(p - proj, axis=-1)

    # Compute unsigned distance: min over segments
    dists = np.min([dist_point_to_segment(grid, a, b) for a, b in zip(seg_a, seg_b)], axis=0)

    # Sign: winding number (ray-casting)
    x, y = grid.T
    inside = np.zeros_like(dists, dtype=bool)
    for a, b in zip(seg_a, seg_b):
        ax, ay, bx, by = *a, *b
        cond = ((ay > y) != (by > y)) & \
               (x < (bx-ax)*(y-ay)/(by-ay+1e-12) + ax)
        inside ^= cond
    sdf = np.where(inside, -dists, dists)
    return grid.astype(np.float32), sdf.astype(np.float32)

# ---------------------------------------------------------------------------
# 3)  Tiny SIREN -------------------------------------------------------------
class SineLayer(nn.Module):
    def __init__(self, m, n, first=False, ω0=10.):
        super().__init__()
        self.ω0, self.first = ω0, first
        self.lin = nn.Linear(m, n)
        with torch.no_grad():
            bound = 1/m if first else math.sqrt(6/m)/ω0
            self.lin.weight.uniform_(-bound, bound)

    def forward(self, x): return torch.sin(self.ω0 * self.lin(x))

class SIREN(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=3):
        super().__init__()
        layers = []
        # First layer with ω0=30
        layers.append(SineLayer(2, hidden_dim, first=True, ω0=30.))
        # Hidden layers with ω0=1
        for _ in range(num_layers - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim, first=False, ω0=1.))
        # Final linear layer
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def export_weights(net, path):
    import json
    layers = []
    for layer in net.net:
        # 1. Plain Linear (the output layer)
        if isinstance(layer, nn.Linear):
            lin = layer
        # 2. SineLayer – grab the internal .lin
        elif hasattr(layer, "lin") and isinstance(layer.lin, nn.Linear):
            lin = layer.lin
        else:
            continue

        W = lin.weight.detach().cpu().numpy()
        layers.append({
            "weight": W.tolist(),    # [out, in] – no transpose
            "bias":   lin.bias.detach().cpu().numpy().tolist()
        })
    with open(path, "w") as f:
        json.dump(layers, f)

# ---------------------------------------------------------------------------
# 4)  Training loop + plotting
# ---------------------------------------------------------------------------
def train_and_show(coords, sdf, res, steps=2000, lr=1e-3, weights_path="weights.json", hidden_dim=128, num_layers=3):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    X = torch.tensor(coords, dtype=torch.float32, device=device)
    Y = torch.tensor(sdf,    dtype=torch.float32, device=device).unsqueeze(1)

    net = SIREN(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=steps)
    mse   = nn.MSELoss()

    for it in range(1, steps+1):
        loss = mse(net(X), Y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()
        if it % (steps // 10) == 0 or it == 1:
            print(f"step {it:4d}   loss = {loss.item():.6f}")

    with torch.no_grad():
        pred = net(X).squeeze().cpu().numpy()
        export_weights(net, weights_path)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.title("Ground truth SDF")
    plt.imshow(sdf.reshape(res,res), extent=[-1,1,-1,1], origin='lower'); plt.colorbar()
    plt.subplot(1,2,2); plt.title("SIREN prediction")
    plt.imshow(pred.reshape(res,res), extent=[-1,1,-1,1], origin='lower'); plt.colorbar()
    plt.tight_layout(); plt.show()

# ---------------------------------------------------------------------------
# 5)  CLI glue
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--svg", required=True, help="SVG file to learn (any logo!)")
    ap.add_argument("--res", type=int, default=64, help="grid resolution (default 64)")
    ap.add_argument("--weights", default="weights.json",
                    help="where to write trained weights (default: weights.json)")
    ap.add_argument("--steps", type=int, default=2000,
                    help="number of training steps (default: 2000)")
    ap.add_argument("--hidden_dim", type=int, default=128,
                    help="width of hidden layers (default: 128)")
    ap.add_argument("--num_layers", type=int, default=3,
                    help="number of SIREN hidden layers (default: 3)")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="learning rate (default: 1e-3)")
    args = ap.parse_args()

    poly         = sample_svg(args.svg)
    coords, sdf  = sdf_from_poly(poly, res=args.res)
    train_and_show(coords, sdf, res=args.res,
                   steps=args.steps, lr=args.lr,
                   weights_path=args.weights,
                   hidden_dim=args.hidden_dim,
                   num_layers=args.num_layers)
