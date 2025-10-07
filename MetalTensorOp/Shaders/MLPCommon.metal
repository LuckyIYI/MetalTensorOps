#ifndef MLP_COMMON_METAL
#define MLP_COMMON_METAL

#include <metal_tensor>
#include <metal_numeric>
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#define MAX_MLP_LAYERS 16

struct DeviceMLPLayers {
    tensor<device half, dextents<int, 2>> weights[MAX_MLP_LAYERS];
    tensor<device half, dextents<int, 1>> biases[MAX_MLP_LAYERS];
};

struct DeviceFloatMLPLayers {
    tensor<device float, dextents<int, 2>> weights[MAX_MLP_LAYERS];
    tensor<device float, dextents<int, 1>> biases[MAX_MLP_LAYERS];
};

inline float activation_relu(float value) {
    return max(value, 0.0f);
}

inline float activation_sigmoid(float value) {
    return 1.f / (1.f + exp(-value));
}

#endif // MLP_COMMON_METAL
