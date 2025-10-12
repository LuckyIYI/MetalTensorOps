#!/usr/bin/env python3
"""
Instant NGP training using MLX for image fitting
Exports weights compatible with MetalTensorOp testing framework plus sparse GT samples.
"""

import argparse
import json
import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from pathlib import Path
from mlx.utils import tree_flatten

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None
    ImageOps = None

# Config matching Metal implementation
NUM_LEVELS = 16
FEATURES_PER_LEVEL = 2
LOG2_HASHMAP_SIZE = 12
BASE_RESOLUTION = 16.0
MAX_RESOLUTION = 2048.0
TOTAL_FEATURES = NUM_LEVELS * FEATURES_PER_LEVEL
MLP_HIDDEN_WIDTH = 64
MLP_OUTPUT_DIM = 3


def serialize_linear_layers(linear_layers):
    layers = []
    for layer in linear_layers:
        weight_np = np.array(layer.weight, dtype=np.float32)
        weight_shape = tuple(int(dim) for dim in layer.weight.shape)
        if len(weight_shape) != 2:
            weight_np = weight_np.reshape(weight_shape)
        else:
            weight_np = weight_np.reshape(weight_shape[0], weight_shape[1])

        bias_np = np.array(layer.bias, dtype=np.float32).reshape(-1)

        layers.append({
            "weights": weight_np.tolist(),
            "biases": bias_np.tolist()
        })
    return layers


def compute_psnr(loss_value: float) -> float:
    if loss_value <= 0.0:
        return 100.0
    return -10.0 * math.log10(loss_value)


def reconstruct_and_save(model, positions_np, image_shape, batch_size, output_path):
    width, height = image_shape
    total = positions_np.shape[0]
    if total == 0 or Image is None:
        return

    positions_mx = mx.array(positions_np)
    recon_chunks = []
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        preds = model(positions_mx[start:end])
        mx.eval(preds)
        recon_chunks.append(np.array(preds))

    reconstruction = np.concatenate(recon_chunks, axis=0)
    reconstruction = reconstruction.reshape(height, width, 3)
    reconstruction = np.clip(reconstruction, 0.0, 1.0)
    output_image = (reconstruction * 255.0).astype(np.uint8)
    Image.fromarray(output_image).save(output_path)
    print(f"Reconstructed image saved to {output_path}")

class HashEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_levels = NUM_LEVELS
        self.features_per_level = FEATURES_PER_LEVEL
        self.hashmap_size = 2 ** LOG2_HASHMAP_SIZE
        self.dim = 2  # 2D for images

        scale_values = np.exp(
            np.linspace(
                np.floor(np.log(BASE_RESOLUTION)),
                np.floor(np.log(MAX_RESOLUTION)),
                NUM_LEVELS,
                dtype=np.float32,
            )
        )
        self.level_scales = mx.array(scale_values, dtype=mx.float32)

        # Create embedding tables for each resolution level
        self.embeddings = [
            mx.random.uniform(
                low=-1e-4,
                high=1e-4,
                shape=(self.hashmap_size, FEATURES_PER_LEVEL),
            )
            for _ in range(NUM_LEVELS)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [B, 2] UV coordinates in [0, 1]
        Returns:
            [B, num_levels * features_per_level] concatenated features
        """
        feats = []

        for l in range(self.num_levels):
            # Scale coordinates to resolution
            scaled = x * self.level_scales[l]
            floored = mx.floor(scaled).astype(mx.int32)
            dx = scaled - floored.astype(mx.float32)

            # 4 corners of grid cell
            corners = mx.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=mx.int32)

            # Spatial hash function
            def hash_corners(pos):
                primes = mx.array([1, 2654435761], dtype=mx.uint32)
                hashed_parts = (pos * primes).astype(mx.uint32)
                return mx.bitwise_xor(hashed_parts[:, 0], hashed_parts[:, 1]) % self.hashmap_size

            # Hash all 4 corners
            corner_indices = [hash_corners(floored + c) for c in corners]

            # Fetch features
            f = [self.embeddings[l][h] for h in corner_indices]

            # Bilinear interpolation
            f_x0 = f[0] * (1 - dx[:, 0:1]) + f[2] * dx[:, 0:1]
            f_x1 = f[1] * (1 - dx[:, 0:1]) + f[3] * dx[:, 0:1]
            f_interp = f_x0 * (1 - dx[:, 1:2]) + f_x1 * dx[:, 1:2]

            feats.append(f_interp)

        return mx.concatenate(feats, axis=-1)


class InstantNGPMLP(nn.Module):
    def __init__(self, hidden_width=MLP_HIDDEN_WIDTH, num_hidden_layers=1):
        super().__init__()
        in_features = NUM_LEVELS * FEATURES_PER_LEVEL

        # First hidden layer
        self.hidden_layers = [nn.Linear(in_features, hidden_width)]

        # Additional hidden layers
        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_width, hidden_width))

        # Output layer
        self.output_layer = nn.Linear(hidden_width, MLP_OUTPUT_DIM)

    def __call__(self, x):
        # Pass through all hidden layers with ReLU activation
        for layer in self.hidden_layers:
            x = nn.relu(layer(x))
        # Final layer with sigmoid activation
        x = nn.sigmoid(self.output_layer(x))
        return x


class InstantNGP(nn.Module):
    def __init__(self, hidden_width=MLP_HIDDEN_WIDTH, num_hidden_layers=1):
        super().__init__()
        self.encoder = HashEncoder()
        self.mlp = InstantNGPMLP(hidden_width=hidden_width, num_hidden_layers=num_hidden_layers)

    def __call__(self, x):
        features = self.encoder(x)
        return self.mlp(features)


class SineLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, omega0: float, first: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        bound = 1.0 / in_dim if first else math.sqrt(6.0 / in_dim) / omega0
        weight_init = mx.random.uniform(low=-bound, high=bound, shape=self.linear.weight.shape)
        bias_init = mx.random.uniform(low=-bound, high=bound, shape=self.linear.bias.shape)
        self.linear.weight = weight_init
        self.linear.bias = bias_init
        self.omega0 = omega0

    def __call__(self, x: mx.array) -> mx.array:
        return mx.sin(self.omega0 * self.linear(x))


class SirenMLP(nn.Module):
    def __init__(self, hidden_width: int, num_hidden_layers: int, out_dim: int):
        super().__init__()
        self.first = SineLayer(2, hidden_width, omega0=30.0, first=True)
        self.hidden = [SineLayer(hidden_width, hidden_width, omega0=1.0) for _ in range(num_hidden_layers)]
        self.output = nn.Linear(hidden_width, out_dim)
        bound = math.sqrt(6.0 / hidden_width) / 1.0
        self.output.weight = mx.random.uniform(low=-bound, high=bound, shape=self.output.weight.shape)
        self.output.bias = mx.zeros_like(self.output.bias)

    @property
    def linear_layers(self):
        return [self.first.linear, *[layer.linear for layer in self.hidden], self.output]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.first(x)
        for layer in self.hidden:
            x = layer(x)
        x = self.output(x)
        return mx.sigmoid(x)


def sample_image(image_file, max_dim=512, space="zero_one"):
    """Load and prepare image data"""
    if Image is None:
        raise ImportError("PIL required")
    img = Image.open(image_file)
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    scale = max_dim / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = img.convert('RGB').resize((new_w, new_h), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0

    if space == "zero_one":
        xs = np.linspace(0, 1, new_w)
        ys = np.linspace(0, 1, new_h)
    elif space == "minus_one_one":
        xs = np.linspace(-1, 1, new_w)
        ys = np.linspace(-1, 1, new_h)
    else:
        raise ValueError(f"Unknown coordinate space '{space}'")
    xx, yy = np.meshgrid(xs, ys)
    positions = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    colors = arr.reshape(-1, 3)

    return positions.astype(np.float32), colors.astype(np.float32), new_w, new_h


def export_instant_ngp_weights(net, path, metadata=None, sample_positions=None, sample_colors=None, sample_count=64, sample_seed=1337):
    """Export weights in Metal-compatible format with optional sparse GT samples."""
    hash_table_data = []
    for level in range(NUM_LEVELS):
        level_table = np.array(net.encoder.embeddings[level])  # (hashmap_size, features_per_level)
        hash_table_data.append(level_table)
    hash_table = np.stack(hash_table_data, axis=0)  # (16, 524288, 2)

    def to_fp16_list(arr):
        return arr.astype(np.float16).flatten().tolist()

    out = {
        "metadata": metadata or {},
        "model": {
            "type": "instant-ngp"
        },
        "encoding": {
            "num_levels": NUM_LEVELS,
            "features_per_level": FEATURES_PER_LEVEL,
            "log2_hashmap_size": LOG2_HASHMAP_SIZE,
            "base_resolution": int(BASE_RESOLUTION),
            "max_resolution": int(MAX_RESOLUTION),
            "hash_table": {
                "shape": list(hash_table.shape),
                "data": to_fp16_list(hash_table)
            }
        },
        "mlp": {
            "layers": serialize_linear_layers(net.mlp.hidden_layers + [net.mlp.output_layer])
        }
    }

    if sample_positions is not None and sample_colors is not None:
        total_positions = sample_positions.shape[0]
        if total_positions == 0:
            raise ValueError("No positions available to sample")
        sample_count = int(min(sample_count, total_positions))
        rng = np.random.default_rng(sample_seed)
        indices = rng.choice(total_positions, size=sample_count, replace=False)
        samples = []
        for idx in indices:
            samples.append({
                "position": sample_positions[idx].astype(np.float32).tolist(),
                "value": sample_colors[idx].astype(np.float32).tolist()
            })
        out["sample_count"] = sample_count
        out["sample_seed"] = int(sample_seed)
        out["samples"] = samples

    with open(path, 'w') as f:
        json.dump(out, f, indent=2)


def export_basic_mlp_weights(
    linear_layers,
    path,
    metadata,
    model_type,
    sample_positions,
    sample_colors,
    sample_count=64,
    sample_seed=1337,
    extra_fields=None,
):
    out = {
        "metadata": metadata or {},
        "model": {
            "type": model_type
        },
        "mlp": {
            "layers": serialize_linear_layers(linear_layers)
        }
    }

    if extra_fields:
        out.update(extra_fields)

    if sample_positions is not None and sample_colors is not None:
        total_positions = sample_positions.shape[0]
        sample_count = int(min(sample_count, total_positions))
        rng = np.random.default_rng(sample_seed)
        indices = rng.choice(total_positions, size=sample_count, replace=False)
        out["sample_count"] = sample_count
        out["sample_seed"] = int(sample_seed)
        out["samples"] = [
            {
                "position": sample_positions[i].astype(np.float32).tolist(),
                "value": sample_colors[i].astype(np.float32).tolist()
            }
            for i in indices
        ]

    with open(path, 'w') as f:
        json.dump(out, f, indent=2)




def train_siren(
    image_path: Path,
    weights_path: Path,
    steps: int,
    lr: float,
    hidden_width: int,
    num_layers: int,
    max_dim: int,
    sample_count: int,
    sample_seed: int,
    batch_size: int | None,
):
    print("Training SIREN (MLX)")
    positions, colors, width, height = sample_image(image_path, max_dim=max_dim, space="minus_one_one")

    X = mx.array(positions)
    Y = mx.array(colors)

    model = SirenMLP(hidden_width=hidden_width, num_hidden_layers=num_layers, out_dim=3)
    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(model, x, y):
        prediction = model(x)
        return mx.mean(mx.square(prediction - y))

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    batch_size = min(batch_size or positions.shape[0], positions.shape[0])
    num_samples = positions.shape[0]

    for it in range(1, steps + 1):
        indices = mx.random.randint(0, num_samples, (batch_size,))
        batch_X = X[indices]
        batch_Y = Y[indices]
        loss, grads = loss_and_grad(model, batch_X, batch_Y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        if it % max(steps // 10, 1) == 0 or it == 1:
            loss_val = float(loss)
            psnr = compute_psnr(loss_val)
            print(f"step {it:4d}   loss = {loss_val:.6f}   PSNR = {psnr:.2f} dB")

    metadata = {
        "mode": "image",
        "image": {
            "width": int(width),
            "height": int(height),
            "aspect_ratio": float(width) / float(height)
        }
    }

    export_basic_mlp_weights(
        model.linear_layers,
        weights_path,
        metadata,
        "siren",
        sample_positions=positions,
        sample_colors=colors,
        sample_count=sample_count,
        sample_seed=sample_seed,
    )

    if Image is not None:
        recon_path = weights_path.with_name(f"{weights_path.stem}_reconstructed.png")
        reconstruct_and_save(
            model,
            positions,
            (int(width), int(height)),
            batch_size,
            str(recon_path)
        )

def train_instant_ngp(positions, colors, image_shape, steps, lr, weights_path, metadata, sample_count, sample_seed, hidden_width=MLP_HIDDEN_WIDTH, num_layers=1, batch_size=None):
    """Train Instant NGP on image using MLX"""
    print("Training on MLX (Apple Silicon)")

    X = mx.array(positions)
    Y = mx.array(colors)

    model = InstantNGP(hidden_width=hidden_width, num_hidden_layers=num_layers)
    optimizer = optim.Adam(learning_rate=lr)

    # Print parameter count
    params = tree_flatten(model.parameters())
    total_params = sum(p.size for _, p in params)
    hash_encoder_params = tree_flatten(model.encoder.parameters())
    hash_encoder_total = sum(p.size for _, p in hash_encoder_params)
    mlp_params = tree_flatten(model.mlp.parameters())
    mlp_total = sum(p.size for _, p in mlp_params)
    print(f"Total parameters: {total_params:,}")
    print(f"  - Hash Encoder: {hash_encoder_total:,}")
    print(f"  - MLP:          {mlp_total:,}")

    # Loss and gradient
    def loss_fn(model, x, y):
        prediction = model(x)
        return mx.mean(mx.square(prediction - y))

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    batch_size = min(batch_size or 99999, positions.shape[0])
    num_samples = len(positions)

    for it in range(1, steps + 1):
        # Random batch
        indices = mx.random.randint(0, num_samples, (batch_size,))
        batch_X = X[indices]
        batch_Y = Y[indices]

        # Forward + backward
        loss, grads = loss_and_grad(model, batch_X, batch_Y)
        optimizer.update(model, grads)

        # Evaluate
        mx.eval(model.parameters(), optimizer.state, loss)

        # Log progress
        if it % max(steps // 10, 1) == 0 or it == 1:
            loss_val = loss.item()
            psnr = compute_psnr(loss_val)
            print(f"step {it:4d}   loss = {loss_val:.6f}   PSNR = {psnr:.2f} dB")

    # Export weights and sparse ground-truth samples for testing
    export_instant_ngp_weights(
        model,
        weights_path,
        metadata=metadata,
        sample_positions=positions,
        sample_colors=colors,
        sample_count=sample_count,
        sample_seed=sample_seed
    )
    print(f"Weights and samples exported to {weights_path}")

    if Image is not None:
        recon_path = weights_path.replace('.json', '_reconstructed.png') if isinstance(weights_path, str) else str(Path(weights_path).with_name(f"{Path(weights_path).stem}_reconstructed.png"))
        reconstruct_and_save(
            model,
            positions,
            image_shape,
            batch_size,
            recon_path
        )


def main():
    ap = argparse.ArgumentParser(description='Train neural field models with MLX')
    ap.add_argument('--model', choices=['instant-ngp', 'siren'], default='instant-ngp', help='Model to train')
    ap.add_argument('--image', required=True, help='Input image file')
    ap.add_argument('--weights', help='Output weights file (default depends on model)')
    ap.add_argument('--max_dim', type=int, default=1080, help='Max image dimension (default: 384)')
    ap.add_argument('--steps', type=int, default=2000, help='Training steps (default: 2000)')
    ap.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    ap.add_argument('--hidden-width', type=int, default=64, help='Hidden width for MLPs (default: 64)')
    ap.add_argument('--num-layers', type=int, default=1, help='Hidden layer count for MLPs (default: 1)')
    ap.add_argument('--sample-count', type=int, default=256, help='Number of embedded samples (default: 256)')
    ap.add_argument('--sample-seed', type=int, default=1337, help='Sample selection seed (default: 1337)')
    ap.add_argument('--batch-size', type=int, help='Override training batch size (default: full batch)')
    args = ap.parse_args()

    image_path = Path(args.image)
    default_weights = {
        'instant-ngp': 'instant_ngp.json',
        'siren': 'siren.json'
    }
    weights_path = Path(args.weights if args.weights else default_weights[args.model])

    if args.model == 'instant-ngp':
        print(f"Loading {image_path}...")
        positions, colors, w, h = sample_image(image_path, max_dim=args.max_dim, space='zero_one')
        print(f"Image: {w}×{h} = {positions.shape[0]} pixels")
        metadata = {
            'mode': 'image',
            'image': {
                'width': int(w),
                'height': int(h),
                'aspect_ratio': float(w) / float(h)
            }
        }
        train_instant_ngp(
            positions,
            colors,
            (w, h),
            args.steps,
            args.lr,
            str(weights_path),
            metadata,
            args.sample_count,
            args.sample_seed,
            hidden_width=args.hidden_width,
            num_layers=args.num_layers,
            batch_size=args.batch_size
        )
    elif args.model == 'siren':
        train_siren(
            image_path=image_path,
            weights_path=weights_path,
            steps=args.steps,
            lr=args.lr,
            hidden_width=args.hidden_width,
            num_layers=args.num_layers,
            max_dim=args.max_dim,
            sample_count=args.sample_count,
            sample_seed=args.sample_seed,
            batch_size=args.batch_size,
        )


if __name__ == '__main__':
    main()
