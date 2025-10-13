#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_numeric>
#include "MLPCommon.metal"

using namespace metal;
using namespace mpp::tensor_ops;

struct InstantNGPRenderUniforms {
    float time;
    uint trainingWidth;
    uint trainingHeight;
    uint _padding;
};

// Instant NGP Configuration
// Multi-resolution hash encoding + compact MLP
// Based on "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"
// https://nvlabs.github.io/instant-ngp/

// Hash encoding parameters
#define NGP_NUM_LEVELS 16
#define NGP_FEATURES_PER_LEVEL 2
#define NGP_LOG2_HASHMAP_SIZE 14
#define NGP_BASE_RESOLUTION 16
#define NGP_MAX_RESOLUTION 2048
#define NGP_TOTAL_FEATURES (NGP_NUM_LEVELS * NGP_FEATURES_PER_LEVEL)


// MLP configuration for cooperative execution
#define NGP_MLP_HIDDEN_WIDTH 64
#define NGP_MLP_OUTPUT_DIM 3  // RGB output
#define NGP_BATCH_SIZE 64  // Process 64 positions per threadgroup

// Hash function: 2D spatial hash matching MLX implementation
inline uint hashPosition2d(uint2 pos) {
    constexpr uint primes[2] = {1u, 2654435761u};
    uint result = 0;
    result ^= pos.x * primes[0];
    result ^= pos.y * primes[1];
    return result;
}

// Compute multi-resolution hash encoding for 2D position (MLX-compatible)
// Bilinear interpolation with 4 corners
inline void computeHashEncoding2d(
    float2 position,
    device half *hashTable,
    thread float *encodedFeatures,
    uint level
) {
    const float lnMin = floor(log((float)NGP_BASE_RESOLUTION));
    const float lnMax = floor(log((float)NGP_MAX_RESOLUTION));
    const float t = (NGP_NUM_LEVELS > 1) ? (float(level) / float(NGP_NUM_LEVELS - 1)) : 0.0f;
    const float scaleFactor = exp(mix(lnMin, lnMax, t));

    const float2 scaledPos = position * scaleFactor;
    const float2 posFloor = floor(scaledPos);
    const float2 posFract = scaledPos - posFloor;

    const uint2 posGrid = uint2(posFloor);

    // Hash encoding with bilinear interpolation (4 corners)
    float accumulatedFeatures[NGP_FEATURES_PER_LEVEL] = {};

    const uint tableSize = 1u << NGP_LOG2_HASHMAP_SIZE;

    // Process 4 corners: [0,0], [0,1], [1,0], [1,1]
    for (uint corner = 0; corner < 4; ++corner) {
        const uint2 offset = uint2(
            (corner >> 0) & 1,
            (corner >> 1) & 1
        );

        const uint2 cornerPos = posGrid + offset;

        // Compute bilinear weight for this corner
        const float wx = (offset.x == 0) ? (1.0f - posFract.x) : posFract.x;
        const float wy = (offset.y == 0) ? (1.0f - posFract.y) : posFract.y;
        const float weight = wx * wy;

        // Hash to table index (matching MLX)
        const uint hash = hashPosition2d(cornerPos);
        const uint hashIndex = hash % tableSize;

        // Accumulate features from hash table
        const uint featureOffset = (level * tableSize + hashIndex) * NGP_FEATURES_PER_LEVEL;

        for (uint f = 0; f < NGP_FEATURES_PER_LEVEL; ++f) {
            accumulatedFeatures[f] += weight * float(hashTable[featureOffset + f]);
        }
    }

    // Write to output
    const uint outputOffset = level * NGP_FEATURES_PER_LEVEL;
    for (uint f = 0; f < NGP_FEATURES_PER_LEVEL; ++f) {
        encodedFeatures[outputOffset + f] = accumulatedFeatures[f];
    }
}

// Full hash encoding across all levels (2D version)
inline void computeFullHashEncoding2d(
    float2 position,
    device half *hashTable,
    thread float *encodedFeatures
) {
    for (uint level = 0; level < NGP_NUM_LEVELS; ++level) {
        computeHashEncoding2d(position, hashTable, encodedFeatures, level);
    }
}

template<typename LoadPosition, typename StoreOutput>
inline void instantNGPRunCooperativeBatch(
    constant DeviceMLPLayers &mlpLayers,
    constant uint &mlpLayerCount,
    device half *hashTable,
    LoadPosition loadPosition,
    StoreOutput storeOutput,
    uint batchStart,
    uint actualBatch,
    uint linearThreadId,
    uint threadgroupSize,
    threadgroup half *encodedStorage,
    threadgroup float *hiddenAccumStorage,
    threadgroup half *hiddenStorage,
    threadgroup float *outputAccumStorage,
    threadgroup half *outputStorage,
    threadgroup uchar *sampleMask
)
{
    if (mlpLayerCount == 0) {
        return;
    }

    for (uint idx = linearThreadId; idx < NGP_BATCH_SIZE * NGP_TOTAL_FEATURES; idx += threadgroupSize) {
        encodedStorage[idx] = half(0.0f);
    }
    for (uint idx = linearThreadId; idx < NGP_BATCH_SIZE * NGP_MLP_HIDDEN_WIDTH; idx += threadgroupSize) {
        hiddenAccumStorage[idx] = 0.0f;
        hiddenStorage[idx] = half(0.0f);
    }
    for (uint idx = linearThreadId; idx < NGP_BATCH_SIZE * NGP_MLP_OUTPUT_DIM; idx += threadgroupSize) {
        outputAccumStorage[idx] = 0.0f;
        outputStorage[idx] = half(0.0f);
    }
    for (uint idx = linearThreadId; idx < NGP_BATCH_SIZE; idx += threadgroupSize) {
        sampleMask[idx] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint localIdx = linearThreadId; localIdx < actualBatch; localIdx += threadgroupSize) {
        const uint sampleIndex = batchStart + localIdx;
        thread float2 position;
        const bool active = loadPosition(sampleIndex, position);
        sampleMask[localIdx] = active ? 1 : 0;
        if (!active) {
            continue;
        }

        thread float encodedFeatures[NGP_TOTAL_FEATURES];
        computeFullHashEncoding2d(position, hashTable, encodedFeatures);

        #pragma unroll
        for (uint f = 0; f < NGP_TOTAL_FEATURES; ++f) {
            encodedStorage[localIdx * NGP_TOTAL_FEATURES + f] = half(encodedFeatures[f]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto encodedTensor = tensor(encodedStorage, dextents<int, 2>(NGP_TOTAL_FEATURES, NGP_BATCH_SIZE));
    auto hiddenAccumTensor = tensor(hiddenAccumStorage, dextents<int, 2>(NGP_MLP_HIDDEN_WIDTH, NGP_BATCH_SIZE));
    auto hiddenTensor = tensor(hiddenStorage, dextents<int, 2>(NGP_MLP_HIDDEN_WIDTH, NGP_BATCH_SIZE));
    auto outputAccumTensor = tensor(outputAccumStorage, dextents<int, 2>(NGP_MLP_OUTPUT_DIM, NGP_BATCH_SIZE));

    constexpr auto firstLayerDescriptor = matmul2d_descriptor(
        NGP_BATCH_SIZE,
        NGP_MLP_HIDDEN_WIDTH,
        dynamic_length_v<int>,
        false,
        false,
        false,
        matmul2d_descriptor::mode::multiply
    );

    constexpr auto hiddenLayerDescriptor = matmul2d_descriptor(
        NGP_BATCH_SIZE,
        NGP_MLP_HIDDEN_WIDTH,
        dynamic_length_v<int>,
        false,
        false,
        false,
        matmul2d_descriptor::mode::multiply
    );

    constexpr auto outputLayerDescriptor = matmul2d_descriptor(
        NGP_BATCH_SIZE,
        NGP_MLP_OUTPUT_DIM,
        dynamic_length_v<int>,
        false,
        false,
        false,
        matmul2d_descriptor::mode::multiply
    );

    matmul2d<firstLayerDescriptor, execution_simdgroups<4>> firstLayerMatmul;
    matmul2d<hiddenLayerDescriptor, execution_simdgroups<4>> hiddenLayerMatmul;
    matmul2d<outputLayerDescriptor, execution_simdgroups<4>> outputLayerMatmul;

    for (uint layerIndex = 0; layerIndex < mlpLayerCount; ++layerIndex) {
        auto weights = mlpLayers.weights[layerIndex];
        auto biases = mlpLayers.biases[layerIndex];
        const bool isFirstLayer = (layerIndex == 0);
        const bool isLastLayer = (layerIndex == mlpLayerCount - 1);

        if (isFirstLayer) {
            firstLayerMatmul.run(encodedTensor, weights, hiddenAccumTensor);
        } else if (isLastLayer) {
            outputLayerMatmul.run(hiddenTensor, weights, outputAccumTensor);
        } else {
            hiddenLayerMatmul.run(hiddenTensor, weights, hiddenAccumTensor);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (isLastLayer) {
            for (uint idx = linearThreadId; idx < actualBatch * NGP_MLP_OUTPUT_DIM; idx += threadgroupSize) {
                const uint batchIdx = idx / NGP_MLP_OUTPUT_DIM;
                if (!sampleMask[batchIdx]) {
                    continue;
                }
                const uint outputIdx = idx % NGP_MLP_OUTPUT_DIM;
                float value = outputAccumStorage[outputIdx + batchIdx * NGP_MLP_OUTPUT_DIM] + float(biases[outputIdx]);
                outputStorage[outputIdx + batchIdx * NGP_MLP_OUTPUT_DIM] = half(activation_sigmoid(value));
            }
        } else {
            for (uint idx = linearThreadId; idx < actualBatch * NGP_MLP_HIDDEN_WIDTH; idx += threadgroupSize) {
                const uint batchIdx = idx / NGP_MLP_HIDDEN_WIDTH;
                if (!sampleMask[batchIdx]) {
                    continue;
                }
                const uint hiddenIdx = idx % NGP_MLP_HIDDEN_WIDTH;
                float value = hiddenAccumStorage[hiddenIdx + batchIdx * NGP_MLP_HIDDEN_WIDTH] + float(biases[hiddenIdx]);
                hiddenStorage[hiddenIdx + batchIdx * NGP_MLP_HIDDEN_WIDTH] = half(activation_relu(value));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
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

struct InstantNGPBufferPositionLoader {
    tensor<device float, dextents<int, 2>> positions;

    inline bool operator()(uint sampleIndex, thread float2 &position) const {
        position = float2(positions[sampleIndex, 0], positions[sampleIndex, 1]);
        return true;
    }
};

struct InstantNGPBufferOutputWriter {
    tensor<device half, dextents<int, 2>> outputs;

    inline void operator()(uint sampleIndex, uint localIdx, threadgroup half *outputStorage) const {
        for (uint channel = 0; channel < NGP_MLP_OUTPUT_DIM; ++channel) {
            outputs[sampleIndex, channel] = outputStorage[localIdx * NGP_MLP_OUTPUT_DIM + channel];
        }
    }
};

struct InstantNGPTexturePositionLoader {
    uint textureWidth;
    uint textureHeight;
    uint trainingWidth;
    uint trainingHeight;

    inline bool operator()(uint sampleIndex, thread float2 &uv) const {
        const uint x = sampleIndex % textureWidth;
        const uint y = sampleIndex / textureWidth;
        if (x >= textureWidth || y >= textureHeight) {
            uv = float2(0.0f);
            return false;
        }

        const float denomX = (textureWidth > 1u) ? float(textureWidth - 1u) : 1.0f;
        const float denomY = (textureHeight > 1u) ? float(textureHeight - 1u) : 1.0f;
        uv = float2(float(x), float(y)) / float2(denomX, denomY);

        if (trainingWidth > 0u && trainingHeight > 0u) {
            const float targetAspect = float(trainingWidth) / float(trainingHeight);
            const float textureAspect = float(textureWidth) / float(textureHeight);

            if (textureAspect > targetAspect) {
                const float visible = targetAspect / textureAspect;
                const float left = 0.5f - 0.5f * visible;
                uv.x = (uv.x - left) / visible;
            } else if (textureAspect < targetAspect) {
                const float visible = textureAspect / targetAspect;
                const float top = 0.5f - 0.5f * visible;
                uv.y = (uv.y - top) / visible;
            }
        }

        if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) {
            return false;
        }

        return true;
    }
};

struct InstantNGPTextureOutputWriter {
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
        rgb.x = float(outputStorage[localIdx * NGP_MLP_OUTPUT_DIM + 0]);
        rgb.y = float(outputStorage[localIdx * NGP_MLP_OUTPUT_DIM + 1]);
        rgb.z = float(outputStorage[localIdx * NGP_MLP_OUTPUT_DIM + 2]);
        outTexture.write(float4(rgb, 1.0f), uint2(x, y));
    }
};

// Cooperative Instant NGP inference that writes into a buffer
// Processes a batch of positions using cooperative matmul
kernel void instantNGPCoopBuffer(
    constant DeviceMLPLayers &mlpLayers [[buffer(0)]],
    constant uint &mlpLayerCount [[buffer(1)]],
    device half *hashTable [[buffer(2)]],
    tensor<device float, dextents<int, 2>> positions [[buffer(3)]],
    tensor<device half, dextents<int, 2>> outputs [[buffer(4)]],
    constant uint &numPositions [[buffer(5)]],
    uint3 threadgroupPositionInGrid [[threadgroup_position_in_grid]],
    ushort threadsPerSimdgroup [[threads_per_simdgroup]],
    ushort3 threadsPerThreadgroup [[threads_per_threadgroup]],
    ushort simdLaneId [[thread_index_in_simdgroup]],
    ushort simdGroupId [[simdgroup_index_in_threadgroup]]
) {
    const uint threadgroupSize =
        uint(threadsPerThreadgroup.x) * uint(threadsPerThreadgroup.y) * uint(threadsPerThreadgroup.z);
    const uint linearThreadId = uint(simdGroupId) * uint(threadsPerSimdgroup) + uint(simdLaneId);

    const uint batchStart = threadgroupPositionInGrid.y * NGP_BATCH_SIZE;
    if (batchStart >= numPositions) {
        return;
    }

    const uint remaining = numPositions - batchStart;
    const uint actualBatch = remaining < NGP_BATCH_SIZE ? remaining : (uint)NGP_BATCH_SIZE;

    threadgroup half encodedStorage[NGP_BATCH_SIZE * NGP_TOTAL_FEATURES];
    threadgroup float hiddenAccumStorage[NGP_BATCH_SIZE * NGP_MLP_HIDDEN_WIDTH];
    threadgroup half hiddenStorage[NGP_BATCH_SIZE * NGP_MLP_HIDDEN_WIDTH];
    threadgroup float outputAccumStorage[NGP_BATCH_SIZE * NGP_MLP_OUTPUT_DIM];
    threadgroup half outputStorage[NGP_BATCH_SIZE * NGP_MLP_OUTPUT_DIM];
    threadgroup uchar sampleMask[NGP_BATCH_SIZE];

    InstantNGPBufferPositionLoader loader{positions};
    InstantNGPBufferOutputWriter writer{outputs};

    instantNGPRunCooperativeBatch(
        mlpLayers,
        mlpLayerCount,
        hashTable,
        loader,
        writer,
        batchStart,
        actualBatch,
        linearThreadId,
        threadgroupSize,
        encodedStorage,
        hiddenAccumStorage,
        hiddenStorage,
        outputAccumStorage,
        outputStorage,
        sampleMask
    );
}


kernel void instantNGPRender(
    texture2d<float, access::write> outTexture [[texture(0)]],
    constant DeviceMLPLayers &mlpLayers [[buffer(0)]],
    constant uint &mlpLayerCount [[buffer(1)]],
    device half *hashTable [[buffer(2)]],
    constant InstantNGPRenderUniforms &uniforms [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (mlpLayerCount == 0) {
        return;
    }

    const uint textureWidth = outTexture.get_width();
    const uint textureHeight = outTexture.get_height();


    if (gid.x >= textureWidth || gid.y >= textureHeight) {
        return;
    }

    const uint2 pixel = gid;
    const float textureWidthF = float(textureWidth);
    const float textureHeightF = float(textureHeight);

    float2 uv = float2(float(pixel.x), float(pixel.y)) / float2(textureWidthF - 1.0, textureHeightF - 1.0);

    if (uniforms.trainingWidth > 0 && uniforms.trainingHeight > 0) {
        const float targetAspect = float(uniforms.trainingWidth) / float(uniforms.trainingHeight);
        const float textureAspect = textureWidthF / textureHeightF;

        if (textureAspect > targetAspect) {
            const float visible = targetAspect / textureAspect;
            const float left = 0.5f - 0.5f * visible;
            uv.x = (uv.x - left) / visible;
        } else if (textureAspect < targetAspect) {
            const float visible = textureAspect / targetAspect;
            const float top = 0.5f - 0.5f * visible;
            uv.y = (uv.y - top) / visible;
        }
    }

    if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) {
        outTexture.write(float4(0.0f, 0.0f, 0.0f, 1.0f), pixel);
        return;
    }

    thread float encodedFeaturesF32[NGP_TOTAL_FEATURES];
    computeFullHashEncoding2d(uv, hashTable, encodedFeaturesF32);

    thread half encodedFeatures[NGP_TOTAL_FEATURES];
    for (uint f = 0; f < NGP_TOTAL_FEATURES; ++f) {
        encodedFeatures[f] = half(encodedFeaturesF32[f]);
    }
    auto encodedTensor = tensor(encodedFeatures, dextents<int, 2>(NGP_TOTAL_FEATURES, 1));

    thread half hiddenActivation[NGP_MLP_HIDDEN_WIDTH];
    thread half nextActivation0[NGP_MLP_HIDDEN_WIDTH];
    auto hiddenTensor = tensor(hiddenActivation, dextents<int, 2>(NGP_MLP_HIDDEN_WIDTH, 1));
    auto nextTensor = tensor(nextActivation0, dextents<int, 2>(NGP_MLP_HIDDEN_WIDTH, 1));

    thread half outputActivation[NGP_MLP_OUTPUT_DIM];
    auto outputTensor = tensor(outputActivation, dextents<int, 2>(NGP_MLP_OUTPUT_DIM, 1));

    constexpr auto firstLlayerDesc = matmul2d_descriptor(
        1,
        NGP_MLP_HIDDEN_WIDTH,
        NGP_TOTAL_FEATURES,
        false,
        false,
        false,
        matmul2d_descriptor::mode::multiply
    );

    constexpr auto hiddenLayerDesc = matmul2d_descriptor(
        1,
        NGP_MLP_HIDDEN_WIDTH,
        NGP_MLP_HIDDEN_WIDTH,
        false,
        false,
        false,
        matmul2d_descriptor::mode::multiply
    );

    constexpr auto outputLayerDesc = matmul2d_descriptor(
        1,
        NGP_MLP_OUTPUT_DIM,
        NGP_MLP_HIDDEN_WIDTH,
        false,
        false,
        false,
        matmul2d_descriptor::mode::multiply
    );

    matmul2d<firstLlayerDesc, execution_thread> firstLayerMatmul;
    matmul2d<hiddenLayerDesc, execution_thread> hiddenLayerMatmul;
    matmul2d<outputLayerDesc, execution_thread> outputLayerMatmul;

    auto firstLayerWeights = mlpLayers.weights[0];
    auto firstLayerBias = mlpLayers.biases[0];
    firstLayerMatmul.run(encodedTensor, firstLayerWeights, hiddenTensor);
    for (uint i = 0; i < NGP_MLP_HIDDEN_WIDTH; ++i) {
        float value = float(hiddenTensor[i, 0]) + float(firstLayerBias[i]);
        hiddenActivation[i] = half(activation_relu(value));
    }

    thread half *currentActivation = hiddenActivation;
    thread half *nextActivation = nextActivation0;

    for (uint layerIndex = 1; layerIndex + 1 < mlpLayerCount; ++layerIndex) {
        auto inputTensor = tensor(currentActivation, dextents<int, 2>(NGP_MLP_HIDDEN_WIDTH, 1));
        auto outputTempTensor = tensor(nextActivation, dextents<int, 2>(NGP_MLP_HIDDEN_WIDTH, 1));
        auto weights = mlpLayers.weights[layerIndex];
        auto biases = mlpLayers.biases[layerIndex];
        hiddenLayerMatmul.run(inputTensor, weights, outputTempTensor);
        for (uint i = 0; i < NGP_MLP_HIDDEN_WIDTH; ++i) {
            float value = float(outputTempTensor[i, 0]) + float(biases[i]);
            nextActivation[i] = half(activation_relu(value));
        }
        thread half *temp = currentActivation;
        currentActivation = nextActivation;
        nextActivation = temp;
    }

    auto finalInputTensor = tensor(currentActivation, dextents<int, 2>(NGP_MLP_HIDDEN_WIDTH, 1));
    const uint lastLayerIndex = mlpLayerCount - 1;
    auto outputWeights = mlpLayers.weights[lastLayerIndex];
    auto outputBias = mlpLayers.biases[lastLayerIndex];
    outputLayerMatmul.run(finalInputTensor, outputWeights, outputTensor);

    float3 rgb;
    for (uint channel = 0; channel < NGP_MLP_OUTPUT_DIM; ++channel) {
        float value = float(outputTensor[channel, 0]) + float(outputBias[channel]);
        rgb[channel] = activation_sigmoid(value);
    }

    outTexture.write(float4(rgb, 1.0f), pixel);
}

kernel void instantNGPRenderCoop(
    texture2d<float, access::write> outTexture [[texture(0)]],
    constant DeviceMLPLayers &mlpLayers [[buffer(0)]],
    constant uint &mlpLayerCount [[buffer(1)]],
    device half *hashTable [[buffer(2)]],
    constant InstantNGPRenderUniforms &uniforms [[buffer(3)]],
    constant uint &numPixels [[buffer(4)]],
    uint3 threadgroupPositionInGrid [[threadgroup_position_in_grid]],
    ushort threadsPerSimdgroup [[threads_per_simdgroup]],
    ushort3 threadsPerThreadgroup [[threads_per_threadgroup]],
    ushort simdLaneId [[thread_index_in_simdgroup]],
    ushort simdGroupId [[simdgroup_index_in_threadgroup]]
)
{
    const uint threadgroupSize =
        uint(threadsPerThreadgroup.x) * uint(threadsPerThreadgroup.y) * uint(threadsPerThreadgroup.z);
    const uint linearThreadId = uint(simdGroupId) * uint(threadsPerSimdgroup) + uint(simdLaneId);

    const uint textureWidth = outTexture.get_width();
    const uint textureHeight = outTexture.get_height();

    const uint batchStart = threadgroupPositionInGrid.x * NGP_BATCH_SIZE;
    if (batchStart >= numPixels) {
        return;
    }

    const uint remaining = numPixels - batchStart;
    const uint actualBatch = remaining < NGP_BATCH_SIZE ? remaining : (uint)NGP_BATCH_SIZE;

    threadgroup half encodedStorage[NGP_BATCH_SIZE * NGP_TOTAL_FEATURES];
    threadgroup float hiddenAccumStorage[NGP_BATCH_SIZE * NGP_MLP_HIDDEN_WIDTH];
    threadgroup half hiddenStorage[NGP_BATCH_SIZE * NGP_MLP_HIDDEN_WIDTH];
    threadgroup float outputAccumStorage[NGP_BATCH_SIZE * NGP_MLP_OUTPUT_DIM];
    threadgroup half outputStorage[NGP_BATCH_SIZE * NGP_MLP_OUTPUT_DIM];
    threadgroup uchar sampleMask[NGP_BATCH_SIZE];

    InstantNGPTexturePositionLoader loader{
        textureWidth,
        textureHeight,
        uniforms.trainingWidth,
        uniforms.trainingHeight
    };
    InstantNGPTextureOutputWriter writer{outTexture, textureWidth, textureHeight};

    instantNGPRunCooperativeBatch(
        mlpLayers,
        mlpLayerCount,
        hashTable,
        loader,
        writer,
        batchStart,
        actualBatch,
        linearThreadId,
        threadgroupSize,
        encodedStorage,
        hiddenAccumStorage,
        hiddenStorage,
        outputAccumStorage,
        outputStorage,
        sampleMask
    );
}
