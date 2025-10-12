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


## TODO List

* Provide demos for texture compression, denoising, etc.
* Replace hard‑coded `#define`s with compile‑time values.
* Report the issues to Apple.

