#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal/metal_simdgroup_matrix>

using namespace metal;
using namespace mpp::tensor_ops;

kernel void simdgroupMatmul(
    tensor<device half, dextents<int, 2>> A [[buffer(0)]],
    tensor<device half, dextents<int, 2>> B [[buffer(1)]],
    tensor<device half, dextents<int, 2>> C [[buffer(2)]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    constexpr auto matmulDescriptor = matmul2d_descriptor(64, 32, dynamic_length_v<int>, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<matmulDescriptor, execution_simdgroups<4>> matmulOp;

    auto mA = A.slice(0, tgid.y * 64);
    auto mB = B.slice(tgid.x * 32, 0);
    auto mC = C.slice(tgid.x * 32, tgid.y * 64);

    matmulOp.run(mA, mB, mC);
}

kernel void simdgroupMatmulMetal3(
    device const half *A [[buffer(0)]],
    device const half *B [[buffer(1)]],
    device float      *C [[buffer(2)]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    constexpr int N = 64, K = 256;
    constexpr uint TILE_M = 8, TILE_N = 8, TILE_K = 8;

    const uint rowBase = tgid.y * TILE_M;
    const uint colBase = tgid.x * TILE_N;

    using sg_mat = simdgroup_matrix<float, TILE_M, TILE_N>;

    sg_mat acc = make_filled_simdgroup_matrix<float, TILE_M, TILE_N>(float(0));

    for (uint k = 0; k < K; k += TILE_K) {
        simdgroup_half8x8 aFrag;
        simdgroup_half8x8 bFrag;

        // Load   A[rowBase .. rowBase+7][k .. k+7]
        simdgroup_load(aFrag, A + rowBase * K + k, K);
        // Load   B[k .. k+7][colBase .. colBase+7]
        simdgroup_load(bFrag, B + k * N + colBase, N);
        // acc += aFrag × bFrag
        simdgroup_multiply_accumulate(acc, aFrag, bFrag, acc);
    }

    // Store the 8×8 tile back to C
    simdgroup_store(acc, C + rowBase * N + colBase, N);
}
