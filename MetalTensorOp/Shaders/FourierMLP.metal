#include <metal_stdlib>
using namespace metal;
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_numeric>
using namespace mpp;


#define IN_DIM 2
#define NUM_FREQ 256
#define HIDDEN_DIM 64
#define OUTPUT_DIM 3

struct FourierLayers {
    tensor<device half, dextents<int, 2>> W[16];
    tensor<device half, dextents<int, 1>> B[16];
    tensor<device float, dextents<int, 2>> BMatrix;
};

kernel void fourierMLP(
    texture2d<float, access::write> outTexture [[texture(0)]],
    constant FourierLayers &mlpLayers [[buffer(0)]],
    constant uint &mlpLayerCount [[buffer(1)]],
    constant float &sigma [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // Parameters

    const int fourierDim = NUM_FREQ * 2;
    // 1. Compute normalized input coords
    float2 xy = (float2(gid) + 0.5f) / float2(outTexture.get_width(), outTexture.get_height()) * 2.0f - 1.0f;

    // 2. Apply Fourier feature mapping:
    thread float fourierFeature[fourierDim];
    for (int i = 0; i < NUM_FREQ; ++i) {
        float proj = 0.0f;
        for (int j = 0; j < IN_DIM; ++j) {
            proj += xy[j] * mlpLayers.BMatrix[j, i];
        }
        proj *= 2.0f * float(M_PI_F);
        fourierFeature[i] = sin(proj);
        fourierFeature[i + NUM_FREQ] = cos(proj);
    }

    // 3. Feed into MLP
    thread half current_activation[NUM_FREQ * 2];
    for (int i = 0; i < NUM_FREQ * 2; ++i) {
        current_activation[i] = half(fourierFeature[i]);
    }

    thread half hiddenMem[HIDDEN_DIM];
    auto hidden = tensor(hiddenMem, dextents<int, 2>(HIDDEN_DIM, 1));
    thread half outputMem[OUTPUT_DIM];
    auto output = tensor(outputMem, dextents<int, 2>(OUTPUT_DIM, 1));

    constexpr tensor_ops::matmul2d_descriptor inputDesc(1, HIDDEN_DIM, NUM_FREQ * 2, false, false, true);
    constexpr tensor_ops::matmul2d_descriptor hiddenDesc(1, HIDDEN_DIM, HIDDEN_DIM, false, false, true);
    constexpr tensor_ops::matmul2d_descriptor outputDesc(1, OUTPUT_DIM, HIDDEN_DIM, false, false, true);

    for (uint layerIndex = 0; layerIndex < mlpLayerCount; ++layerIndex) {
        auto W = mlpLayers.W[layerIndex];
        auto B = mlpLayers.B[layerIndex];
        bool isLastLayer = (layerIndex == mlpLayerCount - 1);

        if (layerIndex == 0) {
            auto input_tensor = tensor(current_activation, dextents<int, 2>(NUM_FREQ * 2, 1));
            tensor_ops::matmul2d<inputDesc, execution_thread>{}.run(input_tensor, W, hidden);
            for (uint i = 0; i < HIDDEN_DIM; ++i) {
                float val = float(hidden[i, 0] + B[i]);
                current_activation[i] = half(max(val, 0.0f));
            }
        } else {
            auto input_tensor = tensor(current_activation, dextents<int, 2>(HIDDEN_DIM, 1));
            if (isLastLayer) {
                tensor_ops::matmul2d<outputDesc, execution_thread>{}.run(input_tensor, W, output);
                for (uint i = 0; i < OUTPUT_DIM; ++i) {
                    current_activation[i] = output[i, 0] + B[i];
                }
            } else {
                tensor_ops::matmul2d<hiddenDesc, execution_thread>{}.run(input_tensor, W, hidden);
                for (uint i = 0; i < HIDDEN_DIM; ++i) {
                    float val = float(hidden[i, 0] + B[i]);
                    current_activation[i] = half(max(val, 0.0f));
                }
            }
        }
    }

    float3 col;
    if constexpr (OUTPUT_DIM == 1) {
        const half sdf = current_activation[0];
        col = sin(float(sdf) * 100.0f) * 0.5f + 0.5f;
    } else if constexpr (OUTPUT_DIM == 3) {
        float r = float(current_activation[0]);
        float g = float(current_activation[1]);
        float b = float(current_activation[2]);
        col = float3(r, g, b);
    }
    outTexture.write(float4(col, 1.0), gid);
}
