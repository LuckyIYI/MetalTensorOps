# Training Scripts

## train_siren.swift

Standalone CLI script for training SIREN (Sinusoidal Representation Networks) on images using Metal 4.

### Quick Start

```bash
# 1. Build the executable
./Scripts/build_train_siren.sh

# 2. Train on an image
./build/train_siren --input your_image.png --steps 500
```

### Requirements

- **macOS 26.0+** - Metal 4 tensor operations are only available on macOS 26.0 and later
- **Swift compiler** - Included with Xcode
- **Metal 4-compatible GPU** - Apple Silicon or supported GPU

### Command-Line Options

```
-i, --input <path>           Input image path (required)
-w, --weights <path>         Output JSON weights (default: trained_siren.json)
-o, --output-image <path>    Reconstructed PNG (default: trained_output.png)
-s, --steps <int>            Training steps (default: 500)
-l, --log-interval <int>     Log loss every N steps (default: 50)
-e, --eval-interval <int>    Evaluate PSNR every N steps (default: 50)
--lr <float>                 Adam learning rate (default: 1e-3)
--limit <int>                Limit dataset samples
--train-batch <int>          Batch size (default: 2048)
-h, --help                   Show help
```

### Examples

Train on a small image for 100 steps:
```bash
./build/train_siren -i test.png -s 100
```

Train with custom learning rate and batch size:
```bash
./build/train_siren -i image.png --lr 5e-4 --train-batch 4096 -s 1000
```

Train on a subset of the data:
```bash
./build/train_siren -i large_image.png --limit 2048 -s 500
```

### Output

The script produces:
1. **JSON weights file** - Network weights in JSON format for inference
2. **Reconstructed PNG** - The network's learned representation of the input
3. **Training metrics** - Loss, PSNR, and timing information logged to stdout
