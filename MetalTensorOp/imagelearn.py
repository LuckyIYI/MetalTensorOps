import argparse, math, numpy as np, matplotlib.pyplot as plt, torch, torch.nn as nn
from PIL import Image
from PIL import ImageOps

def sample_image(image_file, max_dim=64):
    """Return Nx2 coordinates and Nx3 RGB targets for an image resized so its longest side is max_dim, preserving orientation and aspect ratio."""
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

class SineLayer(nn.Module):
    def __init__(self, m, n, first=False, ω0=30.):
        super().__init__()
        self.ω0, self.first = ω0, first
        self.lin = nn.Linear(m, n)
        with torch.no_grad():
            bound = 1/m if first else math.sqrt(6/m)/ω0
            self.lin.weight.uniform_(-bound, bound)
    def forward(self, x): return torch.sin(self.ω0 * self.lin(x))

class SIREN(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        layers = []
        # First layer with ω0=30
        layers.append(SineLayer(2, hidden_dim, first=True, ω0=30.))
        # Hidden layers with ω0=1
        for _ in range(num_layers - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim, first=False, ω0=1.))
        # Final linear layer
        layers.append(nn.Linear(hidden_dim, 3))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def export_weights(net, path):
    import json
    layers = []
    for layer in net.net:
        if isinstance(layer, nn.Linear):
            lin = layer
        elif hasattr(layer, "lin") and isinstance(layer.lin, nn.Linear):
            lin = layer.lin
        else:
            continue
        W = lin.weight.detach().cpu().numpy()
        layers.append({
            "weight": W.tolist(),
            "bias":   lin.bias.detach().cpu().numpy().tolist()
        })
    with open(path, "w") as f:
        json.dump(layers, f)

def train_and_show(coords, rgb, shape, steps, lr, weights_path, hidden_dim, num_layers):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    X = torch.tensor(coords, dtype=torch.float32, device=device)
    Y = torch.tensor(rgb,    dtype=torch.float32, device=device)
    net = SIREN(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
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
        export_weights(net, weights_path)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Ground truth image")
    plt.imshow(Y.cpu().numpy().reshape(shape[1], shape[0], 3), extent=[-1, 1, -1, 1], origin='lower')
    plt.subplot(1, 2, 2)
    plt.title("SIREN prediction")
    plt.imshow(pred.reshape(shape[1], shape[0], 3), extent=[-1, 1, -1, 1], origin='lower')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Image file to learn (png, jpg, etc.)")
    ap.add_argument("--max_dim", type=int, default=512,
                    help="maximum dimension for the longer side of the image (default: 64)")
    ap.add_argument("--steps", type=int, default=20000,
                    help="number of training steps (default: 4000)")
    ap.add_argument("--hidden_dim", type=int, default=64,
                    help="width of hidden layers (default: 256)")
    ap.add_argument("--num_layers", type=int, default=8,
                    help="number of SIREN hidden layers (default: 3)")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="learning rate (default: 1e-3)")
    ap.add_argument("--weights", default="weights.json",
                    help="where to write trained weights (default: weights.json)")
    args = ap.parse_args()
    coords, rgb, w, h = sample_image(args.image, max_dim=args.max_dim)
    train_and_show(coords, rgb, (w, h),
                   steps=args.steps, lr=args.lr,
                   weights_path=args.weights,
                   hidden_dim=args.hidden_dim,
                   num_layers=args.num_layers)
