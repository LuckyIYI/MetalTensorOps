#include <metal_stdlib>
using namespace metal;
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_numeric>
using namespace mpp;

#define INPUT_DIM 2
#define NUM_FREQ 64
#define HIDDEN_DIM 32
#define OUTPUT_DIM 1

struct FourierLayers {
    tensor<device half, dextents<int, 2>> weigts[16];
    tensor<constant half, dextents<int, 1>> biases[16];
    tensor<constant float, dextents<int, 2>> BMatrix;
};

kernel void fourierMLP(
    texture2d<float, access::write> outTexture [[texture(0)]],
    constant FourierLayers &mlpLayers [[buffer(0)]],
    constant uint &mlpLayerCount [[buffer(1)]],
    constant float &sigma [[buffer(2)]],
    constant float &time [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const int fourierDim = NUM_FREQ * 2;
    // 1. Compute normalized input coords
    float2 xy = (float2(gid) + 0.5f) / float2(outTexture.get_width(), outTexture.get_height()) * 2.0f - 1.0f;

    // 2. Apply Fourier feature mapping:
    thread float fourierFeature[fourierDim];
    for (int i = 0; i < NUM_FREQ; ++i) {
        float proj = 0.0f;
        for (int j = 0; j < INPUT_DIM; ++j) {
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
        auto weigts = mlpLayers.weigts[layerIndex];
        auto biases = mlpLayers.biases[layerIndex];
        bool isLastLayer = (layerIndex == mlpLayerCount - 1);

        if (layerIndex == 0) {
            auto input_tensor = tensor(current_activation, dextents<int, 2>(NUM_FREQ * 2, 1));
            tensor_ops::matmul2d<inputDesc, execution_thread>{}.run(input_tensor, weigts, hidden);
            for (uint i = 0; i < HIDDEN_DIM; ++i) {
                float val = float(hidden[i, 0] + biases[i]);
                current_activation[i] = half(max(val, 0.0f));
            }
        } else {
            auto input_tensor = tensor(current_activation, dextents<int, 2>(HIDDEN_DIM, 1));
            if (isLastLayer) {
                tensor_ops::matmul2d<outputDesc, execution_thread>{}.run(input_tensor, weigts, output);
                for (uint i = 0; i < OUTPUT_DIM; ++i) {
                    current_activation[i] = output[i, 0] + biases[i];
                }
            } else {
                tensor_ops::matmul2d<hiddenDesc, execution_thread>{}.run(input_tensor, weigts, hidden);
                for (uint i = 0; i < HIDDEN_DIM; ++i) {
                    float val = float(hidden[i, 0] + biases[i]);
                    current_activation[i] = half(max(val, 0.0f));
                }
            }
        }
    }

    float3 col;
    if  (OUTPUT_DIM == 1) {
        const half sdf = current_activation[0];
        float t = time;
        float2 uv = float2(gid) / float2(outTexture.get_width(), outTexture.get_height());
        float phase = sin(dot(float2(1.0), float2(-uv.y, uv.x)) * 300.0 * mix(sdf, 0.65h, 0.95h) + t * 3.0);
        float r = 0.5 + 0.5 * sin(2.0 * M_PI_F * (length(uv) + phase * 0.3));
        float g = 0.5 + 0.5 * sin(2.0 * M_PI_F * (length(uv) + phase * 0.31));
        float b = 0.5 + 0.5 * sin(2.0 * M_PI_F * (length(uv) + phase * 0.32));
        col = mix(float3(1.0, 0.445, 0.186), float3(r, g, b), smoothstep(0.0h, 0.01h, sdf));
    } else if  (OUTPUT_DIM == 3) {
        float r = float(current_activation[0]);
        float g = float(current_activation[1]);
        float b = float(current_activation[2]);
        col = float3(r, g, b);
    }

    outTexture.write(float4(col, 1.0), gid);
}
