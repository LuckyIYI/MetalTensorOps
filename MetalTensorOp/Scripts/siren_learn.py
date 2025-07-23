#!/usr/bin/env python3
# siren_learn.py — Unified SVG SDF and Image SIREN trainer (c) 2025 MIT-licensed
# -----------------------------------------------------------------------------
import argparse, math, numpy as np, matplotlib.pyplot as plt, torch, torch.nn as nn
from pathlib import Path

# Optional imports for features depending on input type
try:
    from svgpathtools import svg2paths2
except ImportError:
    svg2paths2 = None
try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None
    ImageOps = None

# -----------------------------------------------------------------------------
# 1. Sampling functions
# -----------------------------------------------------------------------------
def sample_svg(svg_file, pts_per_unit=2.0):
    if svg2paths2 is None:
        raise ImportError("svgpathtools is required for SVG input.")
    paths, attrs, _ = svg2paths2(svg_file)
    all_pts = []
    for path, attr in zip(paths, attrs):
        length = path.length()
        n      = max(8, int(length * pts_per_unit))
        ts     = np.linspace(0.0, 1.0, n, endpoint=False)
        pts    = np.stack([path.point(t) for t in ts]).view(np.float64).reshape(-1, 2)
        all_pts.append(pts)
    poly = np.concatenate(all_pts, axis=0)
    cx, cy = poly.mean(axis=0)
    scale  = (poly.max() - poly.min()) * 0.6
    poly   = (poly - (cx, cy)) / scale
    return poly.astype(np.float32)

def sample_image(image_file, max_dim=64):
    if Image is None:
        raise ImportError("Pillow (PIL) is required for image input.")
    img = Image.open(image_file)
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    scale = max_dim / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = img.convert('RGB').resize((new_w, new_h), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0  # (new_h, new_w, 3)
    xs = np.linspace(-1, 1, new_w)
    ys = np.linspace(-1, 1, new_h)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)  # (new_w*new_h, 2)
    rgb = arr.reshape(-1, 3)  # (new_w*new_h, 3)
    return grid, rgb, new_w, new_h

# -----------------------------------------------------------------------------
# 2. SDF from poly-line (for SVG)
# -----------------------------------------------------------------------------
def sdf_from_poly(poly, res=256):
    xs = np.linspace(-1, 1, res); ys = np.linspace(-1, 1, res)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    seg_a = poly
    seg_b = np.roll(poly, -1, axis=0)
    def dist_point_to_segment(p, a, b):
        pa, ba = p - a, b - a
        h = np.clip(np.einsum('...d,...d', pa, ba) / (ba@ba + 1e-12), 0, 1)
        proj = a + h[...,None] * ba
        return np.linalg.norm(p - proj, axis=-1)
    dists = np.min([dist_point_to_segment(grid, a, b) for a, b in zip(seg_a, seg_b)], axis=0)
    x, y = grid.T
    inside = np.zeros_like(dists, dtype=bool)
    for a, b in zip(seg_a, seg_b):
        ax, ay, bx, by = *a, *b
        cond = ((ay > y) != (by > y)) & \
               (x < (bx-ax)*(y-ay)/(by-ay+1e-12) + ax)
        inside ^= cond
    sdf = np.where(inside, -dists, dists)
    return grid.astype(np.float32), sdf.astype(np.float32)

# -----------------------------------------------------------------------------
# 3. SIREN and utilities
# -----------------------------------------------------------------------------
class SineLayer(nn.Module):
    def __init__(self, m, n, first=False, ω0=30.):
        super().__init__()
        self.ω0, self.first = ω0, first
        self.lin = nn.Linear(m, n)
        with torch.no_grad():
            bound = 1/m if first else math.sqrt(6/m)/ω0
            self.lin.weight.uniform_(-bound, bound)
    def forward(self, x):
        return torch.sin(self.ω0 * self.lin(x))

class SIREN(nn.Module):
    def __init__(self, hidden_dim, num_layers, out_dim):
        super().__init__()
        layers = []
        layers.append(SineLayer(2, hidden_dim, first=True, ω0=30.))
        for _ in range(num_layers - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim, first=False, ω0=1.))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class FourierFeatureLayer(nn.Module):
    def __init__(self, in_dim, num_frequencies, sigma=10.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_dim, num_frequencies) * sigma, requires_grad=False)
        self.sigma = sigma
    def forward(self, x):
        x_proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class FourierMLP(nn.Module):
    def __init__(self, hidden_dim, num_layers, out_dim, num_frequencies, sigma=10.0):
        super().__init__()
        self.ff = FourierFeatureLayer(2, num_frequencies, sigma)
        layers = [nn.Linear(num_frequencies*2, hidden_dim), nn.ReLU()]
        for _ in range(num_layers-1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        x = self.ff(x)
        return self.net(x)

def export_weights(net, path, metadata=None):
    import json
    layers = []
    for layer in getattr(net, "net", []):
        if isinstance(layer, nn.Linear):
            lin = layer
        elif hasattr(layer, "lin") and isinstance(layer.lin, nn.Linear):
            lin = layer.lin
        else:
            continue
        W = lin.weight.detach().cpu().numpy()
        layers.append({
            "weights": W.tolist(),
            "biases": lin.bias.detach().cpu().numpy().tolist()
        })
    # Compose output dictionary with new top-level format
    out = {
        "mlp": {
            "layers": layers,
        },
        "metadata": {}
    }
    # Add Fourier params if present
    if hasattr(net, "ff") and hasattr(net.ff, "B"):
        out["fourier"] = {
            "B": net.ff.B.detach().cpu().numpy().tolist(),
            "sigma": net.ff.sigma,
        }
    # Restructure metadata to match Swift Metadata struct if provided
    if metadata is not None:
        mode = metadata.get("mode", None)
        if mode == "image":
            image_meta = {
                "width": metadata.get("width"),
                "height": metadata.get("height"),
                "aspect_ratio": metadata.get("aspect_ratio"),
            }
            out["metadata"].update({
                "mode": "image",
                "image": image_meta
            })
        elif mode == "sdf":
            sdf_meta = {
                "resolution": metadata.get("resolution")
            }
            out["metadata"].update({
                "mode": "sdf",
                "sdf": sdf_meta
            })
        else:
            out["metadata"].update(metadata)
    with open(path, "w") as f:
        json.dump(out, f)

# -----------------------------------------------------------------------------
# 4. Training loop and plotting
# -----------------------------------------------------------------------------
def train_and_show(coords, targets, out_dim, shape_or_res, steps, lr, weights_path, hidden_dim, num_layers, metadata, model_type='siren'):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    X = torch.tensor(coords, dtype=torch.float32, device=device)
    Y = torch.tensor(targets, dtype=torch.float32, device=device)
    if out_dim == 1:
        Y = Y.unsqueeze(1)
    if model_type == 'fourier':
        net = FourierMLP(hidden_dim=hidden_dim, num_layers=num_layers, out_dim=out_dim, num_frequencies=128, sigma=10.0).to(device)
    else:
        net = SIREN(hidden_dim=hidden_dim, num_layers=num_layers, out_dim=out_dim).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=steps)
    mse = nn.MSELoss()
    for it in range(1, steps+1):
        loss = mse(net(X), Y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()
        if it % (steps // 10) == 0 or it == 1:
            print(f"step {it:4d}   loss = {loss.item():.6f}")
    with torch.no_grad():
        pred = net(X).cpu().numpy()
        export_weights(net, weights_path, metadata=metadata)
    plt.figure(figsize=(10,4))
    if out_dim == 1:
        sdf = Y.cpu().numpy().reshape(shape_or_res, shape_or_res)
        pred_img = pred.reshape(shape_or_res, shape_or_res)
        plt.subplot(1,2,1); plt.title("Ground truth SDF")
        plt.imshow(sdf, extent=[-1,1,-1,1], origin='lower'); plt.colorbar()
        plt.subplot(1,2,2); plt.title("SIREN prediction")
        plt.imshow(pred_img, extent=[-1,1,-1,1], origin='lower'); plt.colorbar()
    else:
        w, h = shape_or_res
        plt.subplot(1,2,1); plt.title("Ground truth image")
        plt.imshow(Y.cpu().numpy().reshape(h, w, 3), extent=[-1,1,-1,1], origin='lower')
        plt.subplot(1,2,2); plt.title("SIREN prediction")
        plt.imshow(pred.reshape(h, w, out_dim), extent=[-1,1,-1,1], origin='lower')
    plt.tight_layout(); plt.show()

# -----------------------------------------------------------------------------
# 5. CLI glue
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--svg", help="SVG file to learn (SDF mode)")
    group.add_argument("--image", help="Image file to learn (image mode)")
    ap.add_argument("--fourier", action='store_true', help="Use Fourier feature MLP instead of SIREN")
    ap.add_argument("--res", type=int, default=64, help="grid resolution for SDF mode (default: 64)")
    ap.add_argument("--max_dim", type=int, default=64, help="image max dim for image mode (default: 64)")
    ap.add_argument("--weights", default="model.json", help="where to write trained model (default: model.json)")
    ap.add_argument("--steps", type=int, default=2000, help="number of training steps (default: 2000)")
    ap.add_argument("--hidden_dim", type=int, default=64, help="width of hidden layers (default: 128)")
    ap.add_argument("--num_layers", type=int, default=8, help="number of SIREN hidden layers (default: 3)")
    ap.add_argument("--lr", type=float, default=1e-3, help="learning rate (default: 1e-3)")
    args = ap.parse_args()

    if not ("--weights" in [a for a in vars(args) if getattr(args, a) == args.weights] and args.weights != "model.json"):
        if args.weights == "model.json":
            args.weights = "fourier.json" if args.fourier else "siren.json"
    model_type = 'fourier' if args.fourier else 'siren'
    if args.svg is not None:
        poly = sample_svg(args.svg)
        coords, sdf = sdf_from_poly(poly, res=args.res)
        metadata = {
            "resolution": args.res,
            "mode": "sdf"
        }
        train_and_show(coords, sdf, 1, args.res,
                       steps=args.steps, lr=args.lr,
                       weights_path=args.weights,
                       hidden_dim=args.hidden_dim,
                       num_layers=args.num_layers,
                       metadata=metadata,
                       model_type=model_type)
    elif args.image is not None:
        coords, rgb, w, h = sample_image(args.image, max_dim=args.max_dim)
        metadata = {
            "width": w,
            "height": h,
            "aspect_ratio": float(w)/float(h),
            "mode": "image"
        }
        train_and_show(coords, rgb, 3, (w, h),
                       steps=args.steps, lr=args.lr,
                       weights_path=args.weights,
                       hidden_dim=args.hidden_dim,
                       num_layers=args.num_layers,
                       metadata=metadata,
                       model_type=model_type)

if __name__ == "__main__":
    main()

