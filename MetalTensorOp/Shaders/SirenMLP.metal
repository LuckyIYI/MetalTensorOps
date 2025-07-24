#include <metal_stdlib>
using namespace metal;
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_numeric>
using namespace mpp;

#define INPUT_DIM 2
#define HIDDEN_DIM 32
#define OUTPUT_DIM 1

struct MLPLayers {
    tensor<device half, dextents<int, 2>> W[16];
    tensor<device half, dextents<int, 1>> B[16];
};

kernel void sirenMLP(
    texture2d<float, access::write> outTexture [[texture(0)]],
    constant MLPLayers &mlpLayers [[buffer(0)]],
    constant uint &mlpLayerCount [[buffer(1)]],
    constant float &time [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const half2 xy = (half2(gid) + 0.5h) / half2(outTexture.get_width(), outTexture.get_height()) * 2.0h - 1.0h;

    thread half inputMem[HIDDEN_DIM] = { xy.x, xy.y }; // doesn't work when set to INPUT_DIM
    auto input = tensor(inputMem, dextents<int, 2>(HIDDEN_DIM, 1)); //  doesn't work when set to INPUT_DIM

    thread half hiddenMem[HIDDEN_DIM];
    auto hidden = tensor(hiddenMem, dextents<int, 2>(HIDDEN_DIM, 1));

    thread half outputMem[OUTPUT_DIM];
    auto output = tensor(outputMem, dextents<int, 2>(OUTPUT_DIM, 1));

    constexpr half FIRST_OMEGA0 = 30.0h;
    constexpr half HIDDEN_OMEGA0 = 1.0h;

    constexpr tensor_ops::matmul2d_descriptor inputDesc(1, HIDDEN_DIM,  INPUT_DIM, false, false, true);
    constexpr tensor_ops::matmul2d_descriptor hiddenDesc(1, HIDDEN_DIM, HIDDEN_DIM, false, false, true);
    constexpr tensor_ops::matmul2d_descriptor outputDesc(1, OUTPUT_DIM, HIDDEN_DIM, false, false, true);

    for (uint layerIndex = 0; layerIndex < mlpLayerCount; ++layerIndex) {
        auto W = mlpLayers.W[layerIndex];
        auto B = mlpLayers.B[layerIndex];

        bool isLastLayer = (layerIndex == mlpLayerCount - 1);

        if (layerIndex == 0) {
            // First layer: input -> hidden
            tensor_ops::matmul2d<inputDesc, execution_thread>{}.run(input, W, hidden);
        } else
            if (isLastLayer) {
            // Last layer: hidden -> output
            tensor_ops::matmul2d<outputDesc, execution_thread>{}.run(input, W, output);
        } else {
            // Hidden layers: hidden -> hidden
            tensor_ops::matmul2d<hiddenDesc, execution_thread>{}.run(input, W, hidden);
        }

        const uint outputDim = isLastLayer ? OUTPUT_DIM : HIDDEN_DIM;
        for (uint i = 0; i < outputDim; ++i) {
            half val = (isLastLayer ? output[i, 0] : hidden[i, 0]) + B[i];
            if (isLastLayer) {
                input[i, 0] = val;
            } else if (layerIndex == 0) {
                input[i, 0] = sin(FIRST_OMEGA0 * val);
            } else {
                input[i, 0] = sin(HIDDEN_OMEGA0 * val);
            }
        }
    }

    float3 col;
    if  (OUTPUT_DIM == 1) {
        const half sdf = input[0, 0];
        float t = time;
        float2 uv = float2(gid) / float2(outTexture.get_width(), outTexture.get_height());
        float phase = sin(dot(float2(1.0), float2(-uv.y, uv.x)) * 300.0 * mix(sdf, 0.6h, 0.95h) + t * 3.0);
        float r = 0.5 + 0.5 * sin(2.0 * M_PI_F * (length(uv) + phase * 0.31));
        float g = 0.5 + 0.5 * sin(2.0 * M_PI_F * (length(uv) + phase * 0.3));
        float b = 0.5 + 0.5 * sin(2.0 * M_PI_F * (length(uv) + phase * 0.32));
        col = mix(float3(1.0, 0.445, 0.186), float3(r, g, b), smoothstep(0.0h, 0.01h, sdf));
    } else if  (OUTPUT_DIM == 3) {
        float r = float(input[0, 0]);
        float g = float(input[1, 0]);
        float b = float(input[2, 0]);
        col = float3(r, g, b);
    }

    outTexture.write(float4(col, 1.0), gid);
}
