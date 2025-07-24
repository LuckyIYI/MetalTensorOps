# MetalTensorOp

 **Exploring the Metal 4 compute APIs with a focus on Metal Performance Primitives (MPP) & tensors.**


## Why this exists

Metal 4 introduces a modern, **low‑overhead compute stack**: reusable command buffers, argument tables, a unified ML/compute encoder, texture‑view pools, and more.
This repo is a playground where I am planning to implement a few examples. The first examples are Neural Implicits trained in Python and executed on‑device with Swift + Metal.

> *WARNING: the tensor API is still unstable. It fails on M1 Macs, works on iPhone (with plenty of error logs), and crashes on visionOS.*


## Findings

* Thread‑level matmul doesn’t run on M1 (black screen); likely GPU‑family limitation or beta bug.
* The same shader fails to compile on visionOS.
* Matmul behaviour is sensitive to tensor shape *and* operation order.
* Headers show matmul lacks `int4` support.
* Operation: **M × K** tensor `A` × **K × N** tensor `B` → accumulated into **M × N** output.
* The Metal‑Shading‑Language specification is the main source of truth.
* `MPPTensorOpsMatMul2d.h` explains cooperative tensors well, though its sample code chunks are already outdated.
* Argument tables let you bind resources by ID or attach argument buffers containing multiple GPU resources.



## TODO List

* Implement a cooperative‑tensor matmul example.
* Add an example using both `simdgroup` execution scope.
* Provide demos for texture compression, denoising, etc.
* Add performance tests.
* Double‑check correct use of the CPU‑side API.
* Experiment with the new resource‑allocator workflow.
* Replace hard‑coded `#define`s with compile‑time flags.
* Report the issues to Apple

