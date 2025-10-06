#include <metal_stdlib>
using namespace metal;
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_numeric>
#include "MLPCommon.metal"
using namespace mpp::tensor_ops;

#define INPUT_DIM 2
#define NUM_FREQ 64
#define HIDDEN_DIM 64
#define OUTPUT_DIM 3
#define FOURIER_BATCH_SIZE 32
#define FOURIER_INPUT_DIM (NUM_FREQ * 2)
#define FOURIER_MAX_DIM FOURIER_INPUT_DIM

struct FourierLayers {
    DeviceMLPLayers mlp;
    tensor<constant float, dextents<int, 2>> BMatrix;
};

template<typename LoadPosition, typename StoreOutput>
inline void fourierRunCooperativeBatch(
    constant FourierLayers &mlpLayers,
    constant uint &mlpLayerCount,
    LoadPosition loadPosition,
    StoreOutput storeOutput,
    uint batchStart,
    uint actualBatch,
    uint linearThreadId,
    uint threadgroupSize,
    threadgroup half *activationA,
    threadgroup half *activationB,
    threadgroup float *accumStorage,
    threadgroup half *outputStorage,
    threadgroup uchar *sampleMask
)
{
    for (uint idx = linearThreadId; idx < FOURIER_BATCH_SIZE * FOURIER_MAX_DIM; idx += threadgroupSize) {
        activationA[idx] = half(0.0f);
    }
    for (uint idx = linearThreadId; idx < FOURIER_BATCH_SIZE * HIDDEN_DIM; idx += threadgroupSize) {
        activationB[idx] = half(0.0f);
    }
    for (uint idx = linearThreadId; idx < FOURIER_BATCH_SIZE * FOURIER_MAX_DIM; idx += threadgroupSize) {
        accumStorage[idx] = 0.0f;
    }
    for (uint idx = linearThreadId; idx < FOURIER_BATCH_SIZE * OUTPUT_DIM; idx += threadgroupSize) {
        outputStorage[idx] = half(0.0f);
    }
    for (uint idx = linearThreadId; idx < FOURIER_BATCH_SIZE; idx += threadgroupSize) {
        sampleMask[idx] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto bMatrix = mlpLayers.BMatrix;
    const float twoPi = 2.0f * float(M_PI_F);

    for (uint localIdx = linearThreadId; localIdx < actualBatch; localIdx += threadgroupSize) {
        const uint sampleIndex = batchStart + localIdx;
        thread float2 xy;
        const bool active = loadPosition(sampleIndex, xy);
        sampleMask[localIdx] = active ? 1 : 0;
        if (!active) {
            continue;
        }

        for (uint freq = 0; freq < NUM_FREQ; ++freq) {
            float proj = 0.0f;
            proj += xy.x * float(bMatrix[0, int(freq)]);
            proj += xy.y * float(bMatrix[1, int(freq)]);
            proj *= twoPi;
            activationA[localIdx * FOURIER_MAX_DIM + freq] = half(sin(proj));
            activationA[localIdx * FOURIER_MAX_DIM + freq + NUM_FREQ] = half(cos(proj));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (mlpLayerCount == 0) {
        return;
    }

    constexpr auto firstLayerDesc = matmul2d_descriptor(
        FOURIER_BATCH_SIZE,
        HIDDEN_DIM,
        dynamic_length_v<int>,
        false,
        false,
        false,
        matmul2d_descriptor::mode::multiply
    );

    constexpr auto hiddenLayerDesc = matmul2d_descriptor(
        FOURIER_BATCH_SIZE,
        HIDDEN_DIM,
        dynamic_length_v<int>,
        false,
        false,
        false,
        matmul2d_descriptor::mode::multiply
    );

    constexpr auto outputLayerDesc = matmul2d_descriptor(
        FOURIER_BATCH_SIZE,
        OUTPUT_DIM,
        dynamic_length_v<int>,
        false,
        false,
        false,
        matmul2d_descriptor::mode::multiply
    );

    matmul2d<firstLayerDesc, execution_simdgroups<4>> firstLayerMatmul;
    matmul2d<hiddenLayerDesc, execution_simdgroups<4>> hiddenLayerMatmul;
    matmul2d<outputLayerDesc, execution_simdgroups<4>> outputLayerMatmul;

    threadgroup half *currentActivation = activationA;
    threadgroup half *nextActivation = activationB;

    for (uint layerIndex = 0; layerIndex < mlpLayerCount; ++layerIndex) {
        auto weights = mlpLayers.mlp.weights[layerIndex];
        auto biases = mlpLayers.mlp.biases[layerIndex];
        const bool isFirstLayer = (layerIndex == 0);
        const bool isLastLayer = (layerIndex == mlpLayerCount - 1);
        const int inputDim = isFirstLayer ? FOURIER_INPUT_DIM : HIDDEN_DIM;
        const int outputDim = isLastLayer ? OUTPUT_DIM : HIDDEN_DIM;

        auto inputTensor = tensor(currentActivation, dextents<int, 2>(inputDim, FOURIER_BATCH_SIZE));
        auto accumTensor = tensor(accumStorage, dextents<int, 2>(outputDim, FOURIER_BATCH_SIZE));

        if (isFirstLayer) {
            firstLayerMatmul.run(inputTensor, weights, accumTensor);
        } else if (isLastLayer) {
            outputLayerMatmul.run(inputTensor, weights, accumTensor);
        } else {
            hiddenLayerMatmul.run(inputTensor, weights, accumTensor);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint idx = linearThreadId; idx < actualBatch * outputDim; idx += threadgroupSize) {
            const uint batchIdx = idx / outputDim;
            if (!sampleMask[batchIdx]) {
                continue;
            }
            const uint outputIdx = idx % outputDim;
            float value = accumStorage[outputIdx + batchIdx * outputDim] + float(biases[outputIdx]);

            if (isLastLayer) {
                outputStorage[batchIdx * OUTPUT_DIM + outputIdx] = half(value);
            } else {
                float activated = max(value, 0.0f);
                nextActivation[batchIdx * HIDDEN_DIM + outputIdx] = half(activated);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (!isLastLayer) {
            threadgroup half *temp = currentActivation;
            currentActivation = nextActivation;
            nextActivation = temp;

            for (uint idx = linearThreadId; idx < FOURIER_BATCH_SIZE * HIDDEN_DIM; idx += threadgroupSize) {
                nextActivation[idx] = half(0.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint localIdx = linearThreadId; localIdx < actualBatch; localIdx += threadgroupSize) {
        if (!sampleMask[localIdx]) {
            continue;
        }
        const uint sampleIndex = batchStart + localIdx;
        storeOutput(sampleIndex, localIdx, outputStorage);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

struct FourierTexturePositionLoader {
    uint textureWidth;
    uint textureHeight;

    inline bool operator()(uint sampleIndex, thread float2 &xy) const {
        const uint x = sampleIndex % textureWidth;
        const uint y = sampleIndex / textureWidth;
        if (x >= textureWidth || y >= textureHeight) {
            xy = float2(0.0f);
            return false;
        }
        xy = (float2(float(x), float(y)) + float2(0.5f)) / float2(float(textureWidth), float(textureHeight)) * 2.0f - 1.0f;
        return true;
    }
};

struct FourierBufferPositionLoader {
    device const float2 *positions;

    inline bool operator()(uint sampleIndex, thread float2 &xy) const {
        xy = positions[sampleIndex];
        return true;
    }
};

struct FourierTextureOutputWriter {
    texture2d<float, access::write> outTexture;
    uint textureWidth;
    uint textureHeight;

    inline void operator()(uint sampleIndex, uint localIdx, threadgroup half *outputStorage) const {
        const uint x = sampleIndex % textureWidth;
        const uint y = sampleIndex / textureWidth;
        if (x >= textureWidth || y >= textureHeight) {
            return;
        }
        float3 rgb;
        rgb.x = float(outputStorage[localIdx * OUTPUT_DIM + 0]);
        rgb.y = float(outputStorage[localIdx * OUTPUT_DIM + 1]);
        rgb.z = float(outputStorage[localIdx * OUTPUT_DIM + 2]);
        outTexture.write(float4(rgb, 1.0f), uint2(x, y));
    }
};

struct FourierBufferOutputWriter {
    device float3 *outputs;

    inline void operator()(uint sampleIndex, uint localIdx, threadgroup half *outputStorage) const {
        float3 rgb;
        rgb.x = float(outputStorage[localIdx * OUTPUT_DIM + 0]);
        rgb.y = float(outputStorage[localIdx * OUTPUT_DIM + 1]);
        rgb.z = float(outputStorage[localIdx * OUTPUT_DIM + 2]);
        outputs[sampleIndex] = rgb;
    }
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

    constexpr matmul2d_descriptor inputDesc(1, HIDDEN_DIM, NUM_FREQ * 2, false, false, true);
    constexpr matmul2d_descriptor hiddenDesc(1, HIDDEN_DIM, HIDDEN_DIM, false, false, true);
    constexpr matmul2d_descriptor outputDesc(1, OUTPUT_DIM, HIDDEN_DIM, false, false, true);

    for (uint layerIndex = 0; layerIndex < mlpLayerCount; ++layerIndex) {
        auto weights = mlpLayers.mlp.weights[layerIndex];
        auto biases = mlpLayers.mlp.biases[layerIndex];
        bool isLastLayer = (layerIndex == mlpLayerCount - 1);

        if (layerIndex == 0) {
            auto input_tensor = tensor(current_activation, dextents<int, 2>(NUM_FREQ * 2, 1));
            matmul2d<inputDesc, execution_thread>{}.run(input_tensor, weights, hidden);
            for (uint i = 0; i < HIDDEN_DIM; ++i) {
                float val = float(hidden[i, 0] + biases[i]);
                current_activation[i] = half(max(val, 0.0f));
            }
        } else {
            auto input_tensor = tensor(current_activation, dextents<int, 2>(HIDDEN_DIM, 1));
            if (isLastLayer) {
                matmul2d<outputDesc, execution_thread>{}.run(input_tensor, weights, output);
                for (uint i = 0; i < OUTPUT_DIM; ++i) {
                    current_activation[i] = output[i, 0] + biases[i];
                }
            } else {
                matmul2d<hiddenDesc, execution_thread>{}.run(input_tensor, weights, hidden);
                for (uint i = 0; i < HIDDEN_DIM; ++i) {
                    float val = float(hidden[i, 0] + biases[i]);
                    current_activation[i] = half(max(val, 0.0f));
                }
            }
        }
    }

    float3 col;
    float r = float(current_activation[0]);
    float g = float(current_activation[1]);
    float b = float(current_activation[2]);
    col = float3(r, g, b);

    outTexture.write(float4(col, 1.0), gid);
}

kernel void fourierMLPCoop(
    texture2d<float, access::write> outTexture [[texture(0)]],
    constant FourierLayers &mlpLayers [[buffer(0)]],
    constant uint &mlpLayerCount [[buffer(1)]],
    constant float &sigma [[buffer(2)]],
    constant float &time [[buffer(3)]],
    uint3 threadgroupPositionInGrid [[threadgroup_position_in_grid]],
    ushort threadsPerSimdgroup [[threads_per_simdgroup]],
    ushort3 threadsPerThreadgroup [[threads_per_threadgroup]],
    ushort simdLaneId [[thread_index_in_simdgroup]],
    ushort simdGroupId [[simdgroup_index_in_threadgroup]]
) {
    const uint threadgroupSize =
        uint(threadsPerThreadgroup.x) * uint(threadsPerThreadgroup.y) * uint(threadsPerThreadgroup.z);
    const uint linearThreadId = uint(simdGroupId) * uint(threadsPerSimdgroup) + uint(simdLaneId);

    const uint textureWidth = outTexture.get_width();
    const uint textureHeight = outTexture.get_height();
    const uint numPixels = textureWidth * textureHeight;
    (void)sigma;
    (void)time;

    const uint batchStart = threadgroupPositionInGrid.x * FOURIER_BATCH_SIZE;
    if (batchStart >= numPixels) {
        return;
    }

    const uint remaining = numPixels - batchStart;
    const uint actualBatch = remaining < FOURIER_BATCH_SIZE ? remaining : (uint)FOURIER_BATCH_SIZE;

    threadgroup half activationA[FOURIER_BATCH_SIZE * FOURIER_MAX_DIM];
    threadgroup half activationB[FOURIER_BATCH_SIZE * HIDDEN_DIM];
    threadgroup float accumStorage[FOURIER_BATCH_SIZE * FOURIER_MAX_DIM];
    threadgroup half outputStorage[FOURIER_BATCH_SIZE * OUTPUT_DIM];
    threadgroup uchar sampleMask[FOURIER_BATCH_SIZE];

    FourierTexturePositionLoader loader{textureWidth, textureHeight};
    FourierTextureOutputWriter writer{outTexture, textureWidth, textureHeight};

    fourierRunCooperativeBatch(
        mlpLayers,
        mlpLayerCount,
        loader,
        writer,
        batchStart,
        actualBatch,
        linearThreadId,
        threadgroupSize,
        activationA,
        activationB,
        accumStorage,
        outputStorage,
        sampleMask
    );
}

kernel void fourierMLPCoopBuffer(
    constant FourierLayers &mlpLayers [[buffer(0)]],
    constant uint &mlpLayerCount [[buffer(1)]],
    constant float &sigma [[buffer(2)]],
    device const float2 *positions [[buffer(3)]],
    device float3 *outputs [[buffer(4)]],
    constant uint &numSamples [[buffer(5)]],
    uint3 threadgroupPositionInGrid [[threadgroup_position_in_grid]],
    ushort threadsPerSimdgroup [[threads_per_simdgroup]],
    ushort3 threadsPerThreadgroup [[threads_per_threadgroup]],
    ushort simdLaneId [[thread_index_in_simdgroup]],
    ushort simdGroupId [[simdgroup_index_in_threadgroup]]
) {
    (void)sigma;
    const uint threadgroupSize =
        uint(threadsPerThreadgroup.x) * uint(threadsPerThreadgroup.y) * uint(threadsPerThreadgroup.z);
    const uint linearThreadId = uint(simdGroupId) * uint(threadsPerSimdgroup) + uint(simdLaneId);

    const uint batchStart = threadgroupPositionInGrid.x * FOURIER_BATCH_SIZE;
    if (batchStart >= numSamples) {
        return;
    }

    const uint remaining = numSamples - batchStart;
    const uint actualBatch = remaining < FOURIER_BATCH_SIZE ? remaining : (uint)FOURIER_BATCH_SIZE;

    threadgroup half activationA[FOURIER_BATCH_SIZE * FOURIER_MAX_DIM];
    threadgroup half activationB[FOURIER_BATCH_SIZE * HIDDEN_DIM];
    threadgroup float accumStorage[FOURIER_BATCH_SIZE * FOURIER_MAX_DIM];
    threadgroup half outputStorage[FOURIER_BATCH_SIZE * OUTPUT_DIM];
    threadgroup uchar sampleMask[FOURIER_BATCH_SIZE];

    FourierBufferPositionLoader loader{positions};
    FourierBufferOutputWriter writer{outputs};

    fourierRunCooperativeBatch(
        mlpLayers,
        mlpLayerCount,
        loader,
        writer,
        batchStart,
        actualBatch,
        linearThreadId,
        threadgroupSize,
        activationA,
        activationB,
        accumStorage,
        outputStorage,
        sampleMask
    );
}
