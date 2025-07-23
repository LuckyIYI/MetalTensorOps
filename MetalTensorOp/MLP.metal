#include <metal_stdlib>
using namespace metal;
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_numeric>
#include "Shared.h"

using namespace mpp;
#define HIDDEN_DIM 128


struct Tensors {
    tensor<device half, dextents<int,2>> W[8];
    tensor<device half, dextents<int,1>> B[8];
};


kernel void mlp(
    texture2d<float, access::write> outTexture [[texture(0)]],
    constant Tensors &layers [[buffer(0)]],
    constant uint &layerCount  [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    const half2 xy = (half2(gid)+0.5h)/half2(outTexture.get_width(),outTexture.get_height())*2.0h-1.0h;
    
    thread half vInMem[HIDDEN_DIM] = {xy.x, xy.y};
    auto  vIn = tensor(vInMem, dextents<int,2>(HIDDEN_DIM,1));

    thread half tmpHiddenMem[HIDDEN_DIM];
    auto  tmpHidden = tensor(tmpHiddenMem, dextents<int,2>(HIDDEN_DIM,1));

    thread half tmpOutMem[1];
    auto  tmpOut = tensor(tmpOutMem, dextents<int,2>(1,1));

    constexpr half OMEGA0 = 10.0h;

    constexpr tensor_ops::matmul2d_descriptor kHiddenDesc(
        /* M N K */ 1, HIDDEN_DIM, HIDDEN_DIM,
        /* transpose */ false, false,
        /* reduced-precision */ true);

    constexpr tensor_ops::matmul2d_descriptor kOutDesc(
        /* M N K */ 1, 1, HIDDEN_DIM,
        /* transpose */ false, false,
        /* reduced-precision */ true);
    
    for (uint l = 0; l < layerCount; ++l) {
        auto W = layers.W[l];
        auto B = layers.B[l];

        bool isLast = (l == layerCount - 1);

        if (isLast) {
            tensor_ops::matmul2d<kOutDesc, execution_thread>{}.run(vIn, W, tmpOut);
        } else {
            tensor_ops::matmul2d<kHiddenDesc, execution_thread>{}.run(vIn, W, tmpHidden);
        }

        uint outDim = isLast ? 1 : HIDDEN_DIM;
        for (uint i = 0; i < outDim; ++i) {
            half val = (isLast ? tmpOut[i,0] : tmpHidden[i,0]) + B[i];
            vIn[i,0] = isLast ? val : sin(OMEGA0 * val);
        }
    }

    const half sdf = vIn[0,0];
    float3 col = sin(sdf * 100.0h) * 0.5 + 0.5;
    outTexture.write(float4(col,1.0), gid);
}

