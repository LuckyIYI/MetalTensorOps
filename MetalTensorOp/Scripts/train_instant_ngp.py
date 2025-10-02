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

class HashEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_levels = NUM_LEVELS
        self.features_per_level = FEATURES_PER_LEVEL
        self.hashmap_size = 2 ** LOG2_HASHMAP_SIZE
        self.dim = 2  # 2D for images

        self.level_scales = mx.exp(
            mx.linspace(mx.log(BASE_RESOLUTION), mx.log(MAX_RESOLUTION), NUM_LEVELS)
        )

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
    def __init__(self):
        super().__init__()
        in_features = NUM_LEVELS * FEATURES_PER_LEVEL
        self.layer1 = nn.Linear(in_features, MLP_HIDDEN_WIDTH)
        self.layer2 = nn.Linear(MLP_HIDDEN_WIDTH, MLP_OUTPUT_DIM)

    def __call__(self, x):
        x = nn.relu(self.layer1(x))
        x = nn.sigmoid(self.layer2(x))
        return x


class InstantNGP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = HashEncoder()
        self.mlp = InstantNGPMLP()

    def __call__(self, x):
        features = self.encoder(x)
        return self.mlp(features)


def sample_image(image_file, max_dim=512):
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

    # Create UV grid [0, 1] × [0, 1]
    xs = np.linspace(0, 1, new_w)
    ys = np.linspace(0, 1, new_h)
    xx, yy = np.meshgrid(xs, ys)
    positions = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    colors = arr.reshape(-1, 3)

    return positions.astype(np.float32), colors.astype(np.float32), new_w, new_h


def export_weights(net, path, metadata=None, sample_positions=None, sample_colors=None, sample_count=64, sample_seed=1337):
    """Export weights in Metal-compatible format with optional sparse GT samples."""
    layer1_weight = np.array(net.mlp.layer1.weight)  # (64, 32)
    layer1_bias = np.array(net.mlp.layer1.bias)
    layer2_weight = np.array(net.mlp.layer2.weight)  # (3, 64)
    layer2_bias = np.array(net.mlp.layer2.bias)

    hash_table_data = []
    for level in range(NUM_LEVELS):
        level_table = np.array(net.encoder.embeddings[level])  # (hashmap_size, features_per_level)
        hash_table_data.append(level_table)
    hash_table = np.stack(hash_table_data, axis=0)  # (16, 524288, 2)

    def to_fp16_list(arr):
        return arr.astype(np.float16).flatten().tolist()

    out = {
        "metadata": metadata or {},
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
            "hidden_width": MLP_HIDDEN_WIDTH,
            "output_dim": MLP_OUTPUT_DIM,
            "layers": [
                {
                    "weights": to_fp16_list(layer1_weight.T),  # Transpose for Metal
                    "biases": to_fp16_list(layer1_bias)
                },
                {
                    "weights": to_fp16_list(layer2_weight.T),
                    "biases": to_fp16_list(layer2_bias)
                }
            ]
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



def train_instant_ngp(positions, colors, image_shape, steps, lr, weights_path, metadata, sample_count, sample_seed):
    """Train Instant NGP on image using MLX"""
    print("Training on MLX (Apple Silicon)")

    X = mx.array(positions)
    Y = mx.array(colors)

    model = InstantNGP()
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

    batch_size = 4096
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
        if it % (steps // 10) == 0 or it == 1:
            loss_val = loss.item()
            psnr = -10 * math.log10(loss_val) if loss_val > 0 else 100
            print(f"step {it:4d}   loss = {loss_val:.6f}   PSNR = {psnr:.2f} dB")

    # Export weights and sparse ground-truth samples for testing
    export_weights(
        model,
        weights_path,
        metadata=metadata,
        sample_positions=positions,
        sample_colors=colors,
        sample_count=sample_count,
        sample_seed=sample_seed
    )
    print(f"Weights and samples exported to {weights_path}")

    # Generate and save reconstructed image
    w, h = image_shape
    reconstructed_rgb = []
    for i in range(0, len(X), batch_size):
        batch_uv = X[i:i+batch_size]
        batch_rgb = model(batch_uv)
        mx.eval(batch_rgb)
        reconstructed_rgb.append(np.array(batch_rgb))

    reconstructed_img_np = np.concatenate(reconstructed_rgb, axis=0)
    reconstructed_img_np = reconstructed_img_np.reshape(h, w, 3)
    reconstructed_img_np = (reconstructed_img_np * 255).astype(np.uint8)

    if Image is not None:
        output_path = weights_path.replace('.json', '_reconstructed.png')
        Image.fromarray(reconstructed_img_np).save(output_path)
        print(f"Reconstructed image saved to {output_path}")


def main():
    ap = argparse.ArgumentParser(description='Train Instant NGP on image (MLX)')
    ap.add_argument('--image', required=True, help='Input image file')
    ap.add_argument('--weights', default='instant_ngp.json', help='Output weights (default: instant_ngp.json)')
    ap.add_argument('--max_dim', type=int, default=512, help='Max image dimension (default: 512)')
    ap.add_argument('--steps', type=int, default=2000, help='Training steps (default: 2000)')
    ap.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    ap.add_argument('--sample-count', type=int, default=64, help='Number of samples to export (default: 64)')
    ap.add_argument('--sample-seed', type=int, default=1337, help='Random seed for sample selection (default: 1337)')
    args = ap.parse_args()

    print(f"Loading {args.image}...")
    positions, colors, w, h = sample_image(args.image, max_dim=args.max_dim)
    print(f"Image: {w}×{h} = {positions.shape[0]} pixels")

    metadata = {
        "mode": "image",
        "image": {
            "width": w,
            "height": h,
            "aspect_ratio": float(w) / float(h)
        }
    }

    train_instant_ngp(positions, colors, (w, h), args.steps, args.lr, args.weights, metadata, args.sample_count, args.sample_seed)


if __name__ == '__main__':
    main()
