# MetalTensorOp

 **Exploring the Metal 4 compute APIs with a focus on Metal Performance Primitives (MPP) & tensors.**


## Why this exists

Metal 4 introduces a modern, **low‑overhead compute stack**: reusable command buffers, argument tables, a unified ML/compute encoder, texture‑view pools, and more.
This repo is a playground where I am planning to implement a few examples. The first examples are Neural Implicits trained in Python and executed on‑device with Swift + Metal.

## MTL4 Tips

### **Resource residency & lifetime**  
* **Add backing MTLBuffers to residency** (not just tensor<> handles). 
* **Keep strong references** to anything you bind in argument tables for the entire encode/execute window.  
* Prefer **queue-level residency sets** when possible so all command buffers inherit the same residency.  
  
### **Encoding**  
* **Triple-buffer command allocators**; only reuse/reset an allocator once the prior command buffer is finished.  
  
### **Dispatch & kernel configuration**  
* **Respect maxTotalThreadsPerThreadgroup** from the compiled pipeline. Don’t assume; **query it** and size TGs accordingly.  
* **M3/M4** can allow larger TGs thanks to Dynamic Caching, anyway pick TG sizes **per architecture at runtime**.  
* For dynamic tensor sizes/descriptors, use **dynamic_length<int>** rather than placeholders like 0.  
  
  
### **Algorithm selection & performance**  
- **Compute kernels (e.g., per-pixel MLP)**: prefer **batched cooperative (threadgroup) matmul**; this is fastest on current and upcoming chips.  
- **Fragment shaders**: use **single-threaded tensor ops** for tiny MLPs or when avoiding compute dispatch overhead.  
- On newer hardware, **Tensor Ops** leverage “neural accelerators” and generally match or beat hand-tuned SIMD; keep custom kernels only for **older OS/devices** that can’t use Metal 4 features.  


## Running the SIREN Training Script

The project includes a standalone CLI script for training SIREN networks on images using Metal 4 tensor operations.

### Prerequisites

- macOS 26.0 or later (required for Metal 4 APIs)
- Xcode with Swift compiler
- Apple Silicon or Metal 4-compatible GPU

### Building the Executable

From the project root, run:

```bash
./Scripts/build_train_siren.sh
```

This will compile the training script and all dependencies into a standalone executable at `build/train_siren`.

### Running Training

Basic usage:

```bash
./build/train_siren --input path/to/image.png
```

Full options:

```bash
./build/train_siren \
  --input image.png \
  --weights trained_siren.json \
  --output-image trained_output.png \
  --steps 500 \
  --log-interval 50 \
  --eval-interval 50 \
  --lr 1e-3 \
  --train-batch 2048
```

**Options:**
- `--input <path>` - Input image path (required)
- `--weights <path>` - Output JSON weights file (default: `trained_siren.json`)
- `--output-image <path>` - Reconstructed PNG output (default: `trained_output.png`)
- `--steps <int>` - Number of training steps (default: 500)
- `--log-interval <int>` - Print loss every N steps (default: 50)
- `--eval-interval <int>` - Evaluate PSNR every N steps (default: 50)
- `--lr <float>` - Adam learning rate (default: 1e-3)
- `--limit <int>` - Limit dataset to N samples
- `--train-batch <int>` - Batch size per step (default: 2048)

### Example

```bash
# Create a test image
mkdir -p test_data
# (Use any PNG image)

# Train for 500 steps
./build/train_siren --input test_data/image.png --steps 500

# Results:
# - trained_siren.json (model weights)
# - trained_output.png (reconstructed image)
```

The script uses Metal 4's cooperative threadgroup matmul for efficient GPU training and reports loss, PSNR, and timing metrics.

## TODO List

* Provide demos for texture compression, denoising, etc.
* Replace hard‑coded `#define`s with compile‑time values.
* Report the issues to Apple.

