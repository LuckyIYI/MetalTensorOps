#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal/metal_simdgroup_matrix>

using namespace metal;
using namespace mpp::tensor_ops;

kernel void simdgroupMatrixMatrix(
    tensor<device half, dextents<int, 2>> A [[buffer(0)]],
    tensor<device half, dextents<int, 2>> B [[buffer(1)]],
    tensor<device half, dextents<int, 2>> C [[buffer(2)]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    constexpr int TILE_M = 64, TILE_N = 32;

    constexpr auto matmulDescriptor = matmul2d_descriptor(
        TILE_M, TILE_N, dynamic_length_v<int>,
        false, false, false,
        matmul2d_descriptor::mode::multiply
    );

    matmul2d<matmulDescriptor, execution_simdgroups<4>> matmulOp;

    auto mA = A.slice(0, tgid.y * TILE_M);
    auto mB = B.slice(tgid.x * TILE_N, 0);
    auto mC = C.slice(tgid.x * TILE_N, tgid.y * TILE_M);

    matmulOp.run(mA, mB, mC);
}

kernel void simdgroupMatrixVector(
    tensor<device half, dextents<int, 2>> A [[buffer(0)]],
    tensor<device half, dextents<int, 2>> B [[buffer(1)]],
    tensor<device half, dextents<int, 2>> C [[buffer(2)]],
    uint tgy [[threadgroup_position_in_grid]])
{
    constexpr auto matmulDescriptor = matmul2d_descriptor(64, 1, dynamic_length_v<int>, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<matmulDescriptor, execution_simdgroups<4>> matmulOp;

    auto mA = A.slice(0, tgy * 64);
    auto mB = B;
    auto mC = C.slice(0, tgy * 64);

    matmulOp.run(mA, mB, mC);
}

kernel void simdgroupMatrixMatrixMetal3(
    device const half *A [[buffer(0)]],
    device const half *B [[buffer(1)]],
    device float      *C [[buffer(2)]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    constexpr int N = 64, K = 256;
    constexpr uint TILE_M = 8, TILE_N = 8, TILE_K = 8;

    const uint rowBase = tgid.y * TILE_M;
    const uint colBase = tgid.x * TILE_N;

    auto acc = make_filled_simdgroup_matrix<float, TILE_M, TILE_N>(float(0));

    simdgroup_half8x8 aFrag;
    simdgroup_half8x8 bFrag;

    for (uint k = 0; k < K; k += TILE_K) {
        simdgroup_load(aFrag, A + rowBase * K + k, K);
        simdgroup_load(bFrag, B + k * N + colBase, N);
        simdgroup_multiply_accumulate(acc, aFrag, bFrag, acc);
    }

    simdgroup_store(acc, C + rowBase * N + colBase, N);
}

kernel void threadMatrixMatrix(
    tensor<device half, dextents<int, 2>> A [[buffer(0)]],
    tensor<device half, dextents<int, 2>> B [[buffer(1)]],
    tensor<device half, dextents<int, 2>> C [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    constexpr matmul2d_descriptor descriptor(64, 32, dynamic_length_v<int>, false, false, false);
    matmul2d<descriptor, execution_thread> matmulOp;
    matmulOp.run(A, B, C);
}

kernel void threadVectorMatrix(
    tensor<device half, dextents<int, 2>> A [[buffer(0)]], // (1 x K)
    tensor<device half, dextents<int, 2>> B [[buffer(1)]], // (K x N)
    tensor<device half, dextents<int, 2>> C [[buffer(2)]], // (1 x N)
    uint gid [[thread_position_in_grid]]
) {
    constexpr matmul2d_descriptor descriptor(1, 17, dynamic_length_v<int>, false, false, false);
    matmul2d<descriptor, execution_thread> matmulOp;
    matmulOp.run(A, B, C);
}

kernel void threadMatrixVector(
    tensor<device half, dextents<int, 2>> A [[buffer(0)]], // (M x K)
    tensor<device half, dextents<int, 2>> B [[buffer(1)]], // (K x 1)
    tensor<device half, dextents<int, 2>> C [[buffer(2)]], // (M x 1)
    uint gid [[thread_position_in_grid]]
) {
    constexpr matmul2d_descriptor descriptor(29, 1, dynamic_length_v<int>, false, false, false);
    matmul2d<descriptor, execution_thread> matmulOp;
    matmulOp.run(A, B, C);
}
