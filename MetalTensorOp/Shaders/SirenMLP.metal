#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_numeric>
#include "MLPCommon.metal"

using namespace metal;
using namespace mpp::tensor_ops;

// Compile-time dimensions
#define INPUT_DIM 2
#define HIDDEN_DIM 64
#define OUTPUT_DIM 3
#define SIREN_BATCH_SIZE 32
#define SIREN_MAX_DIM HIDDEN_DIM

struct SirenRenderUniforms {
    float time;
    uint trainingWidth;
    uint trainingHeight;
    uint _padding;
};

template<typename LoadInput, typename StoreOutput>
inline void sirenRunCooperativeBatch(
    constant DeviceMLPLayers &mlpLayers,
    constant uint &mlpLayerCount,
    LoadInput loadInput,
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
    for (uint idx = linearThreadId; idx < SIREN_BATCH_SIZE * SIREN_MAX_DIM; idx += threadgroupSize) {
        activationA[idx] = half(0.0f);
        activationB[idx] = half(0.0f);
    }
    for (uint idx = linearThreadId; idx < SIREN_BATCH_SIZE * SIREN_MAX_DIM; idx += threadgroupSize) {
        accumStorage[idx] = 0.0f;
    }
    for (uint idx = linearThreadId; idx < SIREN_BATCH_SIZE * OUTPUT_DIM; idx += threadgroupSize) {
        outputStorage[idx] = half(0.0f);
    }
    for (uint idx = linearThreadId; idx < SIREN_BATCH_SIZE; idx += threadgroupSize) {
        sampleMask[idx] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint localIdx = linearThreadId; localIdx < actualBatch; localIdx += threadgroupSize) {
        const uint sampleIndex = batchStart + localIdx;
        sampleMask[localIdx] = loadInput(sampleIndex, localIdx, activationA);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (mlpLayerCount == 0) {
        return;
    }

    constexpr auto firstLayerDesc = matmul2d_descriptor(
        SIREN_BATCH_SIZE,
        HIDDEN_DIM,
        dynamic_length_v<int>,
        false,
        false,
        false,
        matmul2d_descriptor::mode::multiply
    );

    constexpr auto hiddenLayerDesc = matmul2d_descriptor(
        SIREN_BATCH_SIZE,
        HIDDEN_DIM,
        dynamic_length_v<int>,
        false,
        false,
        false,
        matmul2d_descriptor::mode::multiply
    );

    constexpr auto outputLayerDesc = matmul2d_descriptor(
        SIREN_BATCH_SIZE,
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
        auto weights = mlpLayers.weights[layerIndex];
        auto biases = mlpLayers.biases[layerIndex];
        const bool isFirstLayer = (layerIndex == 0);
        const bool isLastLayer = (layerIndex == mlpLayerCount - 1);
        const int inputDim = (isFirstLayer) ? INPUT_DIM : HIDDEN_DIM;
        const int outputDim = isLastLayer ? OUTPUT_DIM : HIDDEN_DIM;

        auto inputTensor = tensor(currentActivation, dextents<int, 2>(inputDim, SIREN_BATCH_SIZE));
        auto accumTensor = tensor(accumStorage, dextents<int, 2>(outputDim, SIREN_BATCH_SIZE));

        if (isFirstLayer) {
            firstLayerMatmul.run(inputTensor, weights, accumTensor);
        } else if (isLastLayer) {
            outputLayerMatmul.run(inputTensor, weights, accumTensor);
        } else {
            hiddenLayerMatmul.run(inputTensor, weights, accumTensor);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float omega = (layerIndex == 0) ? 30.0f : 1.0f;

        for (uint idx = linearThreadId; idx < actualBatch * outputDim; idx += threadgroupSize) {
            const uint batchIdx = idx / outputDim;
            if (!sampleMask[batchIdx]) {
                continue;
            }
            const uint outputIdx = idx % outputDim;
            float value = accumStorage[outputIdx + batchIdx * outputDim] + float(biases[outputIdx]);

            if (isLastLayer) {
                outputStorage[batchIdx * OUTPUT_DIM + outputIdx] = half(activation_sigmoid(value));
            } else {
                float activated = sin(omega * value);
                nextActivation[batchIdx * SIREN_MAX_DIM + outputIdx] = half(activated);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (!isLastLayer) {
            threadgroup half *temp = currentActivation;
            currentActivation = nextActivation;
            nextActivation = temp;

            for (uint idx = linearThreadId; idx < SIREN_BATCH_SIZE * SIREN_MAX_DIM; idx += threadgroupSize) {
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

struct SirenTextureInputLoader {
    uint textureWidth;
    uint textureHeight;
    uint trainingWidth;
    uint trainingHeight;

    inline uchar operator()(uint sampleIndex, uint localIdx, threadgroup half *activationBase) const {
        const uint x = sampleIndex % textureWidth;
        const uint y = sampleIndex / textureWidth;
        if (x >= textureWidth || y >= textureHeight || textureWidth == 0u || textureHeight == 0u) {
            activationBase[localIdx * INPUT_DIM + 0] = half(0.0f);
            activationBase[localIdx * INPUT_DIM + 1] = half(0.0f);
            return 0;
        }

        float2 uv = (float2(float(x), float(y)) + float2(0.5f)) /
            float2(max(float(textureWidth), 1.0f), max(float(textureHeight), 1.0f));

        if (trainingWidth > 0u && trainingHeight > 0u) {
            const float targetAspect = float(trainingWidth) / float(trainingHeight);
            const float textureAspect = float(textureWidth) / float(textureHeight);

            if (textureAspect > targetAspect) {
                const float visible = targetAspect / textureAspect;
                const float left = 0.5f - 0.5f * visible;
                uv.x = (uv.x - left) / max(visible, 1e-6f);
            } else if (textureAspect < targetAspect) {
                const float visible = textureAspect / targetAspect;
                const float top = 0.5f - 0.5f * visible;
                uv.y = (uv.y - top) / max(visible, 1e-6f);
            }
        }

        if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) {
            activationBase[localIdx * INPUT_DIM + 0] = half(0.0f);
            activationBase[localIdx * INPUT_DIM + 1] = half(0.0f);
            return 0;
        }

        const float2 mapped = uv * 2.0f - 1.0f;
        activationBase[localIdx * INPUT_DIM + 0] = half(mapped.x);
        activationBase[localIdx * INPUT_DIM + 1] = half(mapped.y);
        return 1;
    }
};

struct SirenTextureOutputWriter {
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

struct SirenBufferInputLoader {
    device const float2 *positions;

    inline uchar operator()(uint sampleIndex, uint localIdx, threadgroup half *activationBase) const {
        float2 xy = positions[sampleIndex];
        activationBase[localIdx * INPUT_DIM + 0] = half(xy.x);
        activationBase[localIdx * INPUT_DIM + 1] = half(xy.y);
        return 1;
    }
};

struct SirenBufferOutputWriter {
    device float3 *outputs;

    inline void operator()(uint sampleIndex, uint localIdx, threadgroup half *outputStorage) const {
        float3 rgb;
        rgb.x = float(outputStorage[localIdx * OUTPUT_DIM + 0]);
        rgb.y = float(outputStorage[localIdx * OUTPUT_DIM + 1]);
        rgb.z = float(outputStorage[localIdx * OUTPUT_DIM + 2]);
        outputs[sampleIndex] = rgb;
    }
};

kernel void sirenMLP(
    texture2d<float, access::write> outTexture [[texture(0)]],
    constant DeviceMLPLayers &mlpLayers [[buffer(0)]],
    constant uint &mlpLayerCount [[buffer(1)]],
    constant SirenRenderUniforms &uniforms [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint textureWidth = outTexture.get_width();
    const uint textureHeight = outTexture.get_height();

    if (gid.x >= textureWidth || gid.y >= textureHeight || textureWidth == 0u || textureHeight == 0u) {
        return;
    }

    float2 uv = (float2(float(gid.x), float(gid.y)) + float2(0.5f)) /
        float2(float(textureWidth), float(textureHeight));

    if (uniforms.trainingWidth > 0u && uniforms.trainingHeight > 0u) {
        const float targetAspect = float(uniforms.trainingWidth) / float(uniforms.trainingHeight);
        const float textureAspect = float(textureWidth) / float(textureHeight);

        if (textureAspect > targetAspect) {
            const float visible = targetAspect / textureAspect;
            const float left = 0.5f - 0.5f * visible;
            uv.x = (uv.x - left) / max(visible, 1e-6f);
        } else if (textureAspect < targetAspect) {
            const float visible = textureAspect / targetAspect;
            const float top = 0.5f - 0.5f * visible;
            uv.y = (uv.y - top) / max(visible, 1e-6f);
        }
    }

    if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) {
        outTexture.write(float4(0.0f, 0.0f, 0.0f, 1.0f), gid);
        return;
    }

    const half2 xy = half2(uv * 2.0f - 1.0f);

    thread half inputMem[HIDDEN_DIM] = { xy.x, xy.y };
    auto input = tensor(inputMem, dextents<int, 2>(HIDDEN_DIM, 1));

    thread half hiddenMem[HIDDEN_DIM];
    auto hidden = tensor(hiddenMem, dextents<int, 2>(HIDDEN_DIM, 1));

    thread half outputMem[OUTPUT_DIM];
    auto output = tensor(outputMem, dextents<int, 2>(OUTPUT_DIM, 1));

    constexpr half FIRST_OMEGA0 = 30.0h;
    constexpr half HIDDEN_OMEGA0 = 1.0h;

    constexpr matmul2d_descriptor inputDesc(1, HIDDEN_DIM,  INPUT_DIM, false, false, true);
    constexpr matmul2d_descriptor hiddenDesc(1, HIDDEN_DIM, HIDDEN_DIM, false, false, true);
    constexpr matmul2d_descriptor outputDesc(1, OUTPUT_DIM, HIDDEN_DIM, false, false, true);

    for (uint layerIndex = 0; layerIndex < mlpLayerCount; ++layerIndex) {
        auto weights = mlpLayers.weights[layerIndex];
        auto biases = mlpLayers.biases[layerIndex];

        bool isLastLayer = (layerIndex == mlpLayerCount - 1);

        if (layerIndex == 0) {
            // First layer: input -> hidden
            matmul2d<inputDesc, execution_thread>{}.run(input, weights, hidden);
        } else if (isLastLayer) {
            // Last layer: hidden -> output
            matmul2d<outputDesc, execution_thread>{}.run(input, weights, output);
        } else {
            // Hidden layers: hidden -> hidden
            matmul2d<hiddenDesc, execution_thread>{}.run(input, weights, hidden);
        }

        const uint outputDim = isLastLayer ? OUTPUT_DIM : HIDDEN_DIM;
        for (uint i = 0; i < outputDim; ++i) {
            half val = (isLastLayer ? output[i, 0] : hidden[i, 0]) + biases[i];
            if (isLastLayer) {
                input[i, 0] = half(activation_sigmoid(float(val)));
            } else if (layerIndex == 0) {
                input[i, 0] = sin(FIRST_OMEGA0 * val);
            } else {
                input[i, 0] = sin(HIDDEN_OMEGA0 * val);
            }
        }
    }

    float3 col;
    float r = float(input[0, 0]);
    float g = float(input[1, 0]);
    float b = float(input[2, 0]);
    col = float3(r, g, b);

    outTexture.write(float4(col, 1.0), gid);
}

kernel void sirenMLPCoop(
    texture2d<float, access::write> outTexture [[texture(0)]],
    constant DeviceMLPLayers &mlpLayers [[buffer(0)]],
    constant uint &mlpLayerCount [[buffer(1)]],
    constant SirenRenderUniforms &uniforms [[buffer(2)]],
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

    const uint batchStart = threadgroupPositionInGrid.x * SIREN_BATCH_SIZE;
    if (batchStart >= numPixels) {
        return;
    }

    const uint remaining = numPixels - batchStart;
    const uint actualBatch = remaining < SIREN_BATCH_SIZE ? remaining : (uint)SIREN_BATCH_SIZE;

    threadgroup half activationA[SIREN_BATCH_SIZE * SIREN_MAX_DIM];
    threadgroup half activationB[SIREN_BATCH_SIZE * SIREN_MAX_DIM];
    threadgroup float accumStorage[SIREN_BATCH_SIZE * SIREN_MAX_DIM];
    threadgroup half outputStorage[SIREN_BATCH_SIZE * OUTPUT_DIM];
    threadgroup uchar sampleMask[SIREN_BATCH_SIZE];

    SirenTextureInputLoader loader{
        textureWidth,
        textureHeight,
        uniforms.trainingWidth,
        uniforms.trainingHeight
    };
    SirenTextureOutputWriter writer{outTexture, textureWidth, textureHeight};

    sirenRunCooperativeBatch(
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

kernel void sirenMLPCoopBuffer(
    constant DeviceMLPLayers &mlpLayers [[buffer(0)]],
    constant uint &mlpLayerCount [[buffer(1)]],
    device const float2 *positions [[buffer(2)]],
    device float3 *outputs [[buffer(3)]],
    constant uint &numSamples [[buffer(4)]],
    uint3 threadgroupPositionInGrid [[threadgroup_position_in_grid]],
    ushort threadsPerSimdgroup [[threads_per_simdgroup]],
    ushort3 threadsPerThreadgroup [[threads_per_threadgroup]],
    ushort simdLaneId [[thread_index_in_simdgroup]],
    ushort simdGroupId [[simdgroup_index_in_threadgroup]]
) {
    const uint threadgroupSize =
        uint(threadsPerThreadgroup.x) * uint(threadsPerThreadgroup.y) * uint(threadsPerThreadgroup.z);
    const uint linearThreadId = uint(simdGroupId) * uint(threadsPerSimdgroup) + uint(simdLaneId);

    const uint batchStart = threadgroupPositionInGrid.x * SIREN_BATCH_SIZE;
    if (batchStart >= numSamples) {
        return;
    }

    const uint remaining = numSamples - batchStart;
    const uint actualBatch = remaining < SIREN_BATCH_SIZE ? remaining : (uint)SIREN_BATCH_SIZE;

    threadgroup half activationA[SIREN_BATCH_SIZE * SIREN_MAX_DIM];
    threadgroup half activationB[SIREN_BATCH_SIZE * SIREN_MAX_DIM];
    threadgroup float accumStorage[SIREN_BATCH_SIZE * SIREN_MAX_DIM];
    threadgroup half outputStorage[SIREN_BATCH_SIZE * OUTPUT_DIM];
    threadgroup uchar sampleMask[SIREN_BATCH_SIZE];

    SirenBufferInputLoader loader{positions};
    SirenBufferOutputWriter writer{outputs};

    sirenRunCooperativeBatch(
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
