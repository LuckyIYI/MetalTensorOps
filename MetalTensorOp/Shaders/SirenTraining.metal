#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_numeric>
#include <metal_atomic>
#include "MLPCommon.metal"

using namespace metal;
using namespace mpp::tensor_ops;

constant uint SIREN_TRAIN_BATCH_SIZE = 32;
constant uint SIREN_TRAIN_MAX_DIM = 64;
constant uint SIREN_TRAIN_OUTPUT_DIM = 3;
constant uint SIREN_TRAIN_INPUT_DIM = 2;
constant uint SIREN_TRAIN_THREADS = 128; // matches dispatch configuration (threadExecutionWidth * 4)

struct SirenTrainingParams {
    uint batchStart;
    uint totalSamples;
    uint sliceBatchSize;
    uint layerCount;
    uint globalBatchSize;
};

struct SirenAdamParams {
    uint layerCount;
    uint totalWeightCount;
    uint totalBiasCount;
    float learningRate;
    float beta1;
    float beta2;
    float epsilon;
    float beta1Denom;
    float beta2Denom;
    float weightDecay;
    uint preserveGradients;
};

static_assert(SIREN_TRAIN_BATCH_SIZE <= 64, "Training batch size exceeds supported maximum for threadgroup storage");

inline uint activationIndex(
    uint layerIndex,
    uint sampleIndex,
    uint neuronIndex,
    uint chunkIndex,
    uint layerStride,
    uint chunkStride
) {
    return chunkIndex * chunkStride
        + layerIndex * layerStride
        + sampleIndex * SIREN_TRAIN_MAX_DIM
        + neuronIndex;
}

inline uint layerOffset(uint layerIndex, uint chunkIndex, uint layerStride, uint chunkStride) {
    return chunkIndex * chunkStride + layerIndex * layerStride;
}

kernel void sirenTrainStep(
    device DeviceMLPLayers &mlpLayers              [[buffer(0)]],
    device DeviceFloatMLPLayers &gradients         [[buffer(1)]],
    device const float2 *positions                 [[buffer(2)]],
    device const float3 *targets                   [[buffer(3)]],
    constant SirenTrainingParams &params           [[buffer(4)]],
    constant uint *layerInputDims                  [[buffer(5)]],
    constant uint *layerOutputDims                 [[buffer(6)]],
    device half *activationHistory                 [[buffer(7)]],
    device half *preActivationHistory              [[buffer(8)]],
    device float *lossBuffer                       [[buffer(9)]],
    uint3 threadgroupPosInGrid                     [[threadgroup_position_in_grid]],
    ushort threadsPerSimdgroup                     [[threads_per_simdgroup]],
    ushort3 threadsPerThreadgroup                  [[threads_per_threadgroup]],
    ushort simdLaneId                              [[thread_index_in_simdgroup]],
    ushort simdGroupId                             [[simdgroup_index_in_threadgroup]])
{
    const uint threadgroupSize = uint(threadsPerThreadgroup.x) * uint(threadsPerThreadgroup.y) * uint(threadsPerThreadgroup.z);
    const uint linearThreadId = uint(simdGroupId) * uint(threadsPerSimdgroup) + uint(simdLaneId);

    if (params.layerCount == 0) {
        return;
    }

    const uint datasetRemaining = (params.batchStart < params.totalSamples) ? (params.totalSamples - params.batchStart) : 0;
    const uint sliceBatch = min(params.sliceBatchSize, datasetRemaining);
    if (sliceBatch == 0) {
        return;
    }

    const uint chunkIndex = threadgroupPosInGrid.x;
    const uint chunkBase = chunkIndex * SIREN_TRAIN_BATCH_SIZE;
    if (chunkBase >= sliceBatch) {
        return;
    }
    const uint chunkSize = min((uint)SIREN_TRAIN_BATCH_SIZE, sliceBatch - chunkBase);
    const uint targetBase = params.batchStart + chunkBase;

    const uint layerStride = SIREN_TRAIN_BATCH_SIZE * SIREN_TRAIN_MAX_DIM;
    const uint activationChunkStride = (params.layerCount + 1) * layerStride;
    const uint preActivationChunkStride = params.layerCount * layerStride;

    threadgroup half activationA[SIREN_TRAIN_BATCH_SIZE * SIREN_TRAIN_MAX_DIM];
    threadgroup half activationB[SIREN_TRAIN_BATCH_SIZE * SIREN_TRAIN_MAX_DIM];
    threadgroup float accumStorage[SIREN_TRAIN_BATCH_SIZE * SIREN_TRAIN_MAX_DIM];
    threadgroup half deltaCurrentHalf[SIREN_TRAIN_BATCH_SIZE * SIREN_TRAIN_MAX_DIM];
    threadgroup half deltaNextHalf[SIREN_TRAIN_BATCH_SIZE * SIREN_TRAIN_MAX_DIM];
    threadgroup float lossShared[SIREN_TRAIN_THREADS];

    const uint activationLayerStride = SIREN_TRAIN_BATCH_SIZE * SIREN_TRAIN_MAX_DIM;

    if (linearThreadId < SIREN_TRAIN_THREADS) {
        lossShared[linearThreadId] = 0.0f;
    }

    float localLoss = 0.0f;
    const float invGlobalBatch = (params.globalBatchSize > 0)
        ? (1.0f / float(params.globalBatchSize))
        : 0.0f;

    for (uint idx = linearThreadId; idx < activationLayerStride; idx += threadgroupSize) {
        activationA[idx] = half(0.0f);
        activationB[idx] = half(0.0f);
        accumStorage[idx] = 0.0f;
        deltaCurrentHalf[idx] = half(0.0f);
        deltaNextHalf[idx] = half(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint currentWidth = SIREN_TRAIN_INPUT_DIM;
    threadgroup half *currentActivation = activationA;
    threadgroup half *nextActivation = activationB;

    for (uint localSample = linearThreadId; localSample < chunkSize; localSample += threadgroupSize) {
        const uint sampleIndex = targetBase + localSample;
        float2 xy = positions[sampleIndex];
        const uint base = localSample * currentWidth;
        currentActivation[base + 0] = half(xy.x);
        currentActivation[base + 1] = half(xy.y);
    }
    for (uint localSample = linearThreadId + chunkSize; localSample < SIREN_TRAIN_BATCH_SIZE; localSample += threadgroupSize) {
        const uint base = localSample * currentWidth;
        currentActivation[base + 0] = half(0.0f);
        currentActivation[base + 1] = half(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = linearThreadId; idx < chunkSize * currentWidth; idx += threadgroupSize) {
        const uint sample = idx / currentWidth;
        const uint feature = idx % currentWidth;
        activationHistory[activationIndex(0, sample, feature, chunkIndex, layerStride, activationChunkStride)] = currentActivation[sample * currentWidth + feature];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

        constexpr auto firstLayerDescriptor = matmul2d_descriptor(
            SIREN_TRAIN_BATCH_SIZE,
            SIREN_TRAIN_MAX_DIM,
            dynamic_length_v<int>,
            false,
            false,
            false,
            matmul2d_descriptor::mode::multiply
        );

        constexpr auto hiddenLayerDescriptor = matmul2d_descriptor(
            SIREN_TRAIN_BATCH_SIZE,
            SIREN_TRAIN_MAX_DIM,
            dynamic_length_v<int>,
            false,
            false,
            false,
            matmul2d_descriptor::mode::multiply
        );

        constexpr auto outputLayerDescriptor = matmul2d_descriptor(
            SIREN_TRAIN_BATCH_SIZE,
            SIREN_TRAIN_OUTPUT_DIM,
            dynamic_length_v<int>,
            false,
            false,
            false,
            matmul2d_descriptor::mode::multiply
        );

        matmul2d<firstLayerDescriptor, execution_simdgroups<4>> firstLayerMatmul;
        matmul2d<hiddenLayerDescriptor, execution_simdgroups<4>> hiddenLayerMatmul;
        matmul2d<outputLayerDescriptor, execution_simdgroups<4>> outputLayerMatmul;

        for (uint layerIndex = 0; layerIndex < params.layerCount; ++layerIndex) {
            auto weights = mlpLayers.weights[layerIndex];
            auto layerBiases = mlpLayers.biases[layerIndex];

            const uint outputDim = layerOutputDims[layerIndex];
            const uint inputDim = layerInputDims[layerIndex];
            const bool isLastLayer = (layerIndex == params.layerCount - 1);
            const float omega = (layerIndex == 0) ? 30.0f : 1.0f;

            auto inputTensor = tensor(currentActivation, dextents<int, 2>(int(currentWidth), int(SIREN_TRAIN_BATCH_SIZE)));
            auto accumTensor = tensor(accumStorage, dextents<int, 2>(int(outputDim), int(SIREN_TRAIN_BATCH_SIZE)));

            if (chunkSize < SIREN_TRAIN_BATCH_SIZE) {
                for (uint idx = linearThreadId + chunkSize * currentWidth; idx < SIREN_TRAIN_BATCH_SIZE * currentWidth; idx += threadgroupSize) {
                    accumStorage[idx] = 0.0f;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (layerIndex == 0) {
                firstLayerMatmul.run(inputTensor, weights, accumTensor);
            } else if (isLastLayer) {
                outputLayerMatmul.run(inputTensor, weights, accumTensor);
            } else {
                hiddenLayerMatmul.run(inputTensor, weights, accumTensor);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint sample = linearThreadId; sample < chunkSize; sample += threadgroupSize) {
                const uint computeBase = sample * outputDim;
                const uint historyBase = activationIndex(layerIndex + 1, sample, 0, chunkIndex, layerStride, activationChunkStride);
                float activationSum = 0.0f;

                for (uint outIdx = 0; outIdx < outputDim; ++outIdx) {
                    const uint accumIndex = outIdx + sample * outputDim;
                    float dotValue = accumStorage[accumIndex];
                    float z = dotValue + float(layerBiases[outIdx]);
                    accumStorage[accumIndex] = z;

                    float activated = isLastLayer ? activation_sigmoid(z) : sin(omega * z);
                    nextActivation[computeBase + outIdx] = half(activated);
                    activationHistory[historyBase + outIdx] = half(activated);
                    activationSum += activated;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint idx = linearThreadId; idx < chunkSize * outputDim; idx += threadgroupSize) {
                const uint sample = idx / outputDim;
                const uint feature = idx % outputDim;
                const uint preBase = layerOffset(layerIndex, chunkIndex, layerStride, preActivationChunkStride) + sample * SIREN_TRAIN_MAX_DIM;
                const uint accumIndex = feature + sample * outputDim;
                preActivationHistory[preBase + feature] = half(accumStorage[accumIndex]);
            }

            for (uint sample = linearThreadId; sample < chunkSize; sample += threadgroupSize) {
                const uint historyBase = activationIndex(layerIndex + 1, sample, 0, chunkIndex, layerStride, activationChunkStride);
                const uint preBase = layerOffset(layerIndex, chunkIndex, layerStride, preActivationChunkStride) + sample * SIREN_TRAIN_MAX_DIM;
                for (uint padding = outputDim; padding < SIREN_TRAIN_MAX_DIM; ++padding) {
                    activationHistory[historyBase + padding] = half(0.0f);
                    preActivationHistory[preBase + padding] = half(0.0f);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint sample = linearThreadId + chunkSize; sample < SIREN_TRAIN_BATCH_SIZE; sample += threadgroupSize) {
                const uint computeBase = sample * outputDim;
                const uint historyBase = activationIndex(layerIndex + 1, sample, 0, chunkIndex, layerStride, activationChunkStride);
                const uint preBase = layerOffset(layerIndex, chunkIndex, layerStride, preActivationChunkStride) + sample * SIREN_TRAIN_MAX_DIM;
                for (uint feature = 0; feature < outputDim; ++feature) {
                    nextActivation[computeBase + feature] = half(0.0f);
                }
                for (uint feature = 0; feature < SIREN_TRAIN_MAX_DIM; ++feature) {
                    activationHistory[historyBase + feature] = half(0.0f);
                    preActivationHistory[preBase + feature] = half(0.0f);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (!isLastLayer) {
                threadgroup half *temp = currentActivation;
                currentActivation = nextActivation;
                nextActivation = temp;
            } else {
                currentActivation = nextActivation;
            }
            currentWidth = outputDim;
        }

        threadgroup_barrier(mem_flags::mem_device);

        const uint lastLayerIndex = params.layerCount - 1;
        threadgroup half *outputActivation = currentActivation;

        for (uint sample = linearThreadId; sample < chunkSize; sample += threadgroupSize) {
            const uint computeBase = sample * currentWidth;
            const uint sampleOffset = sample * SIREN_TRAIN_MAX_DIM;
            const float3 target = targets[targetBase + sample];

            float3 prediction = float3(0.0f);
            for (uint c = 0; c < SIREN_TRAIN_OUTPUT_DIM; ++c) {
                prediction[c] = float(outputActivation[computeBase + c]);
            }

            const uint lastPreBase = layerOffset(lastLayerIndex, chunkIndex, layerStride, preActivationChunkStride) + sampleOffset;
            for (uint c = 0; c < SIREN_TRAIN_OUTPUT_DIM; ++c) {
                float activated = clamp(prediction[c], 1e-6f, 1.0f - 1e-6f);
                float preValue = log(activated / (1.0f - activated));
                preActivationHistory[lastPreBase + c] = half(preValue);
            }
            for (uint padding = SIREN_TRAIN_OUTPUT_DIM; padding < SIREN_TRAIN_MAX_DIM; ++padding) {
                preActivationHistory[lastPreBase + padding] = half(0.0f);
            }

            float3 diff = prediction - target;
            localLoss += dot(diff, diff);

            for (uint c = 0; c < SIREN_TRAIN_OUTPUT_DIM; ++c) {
                float activated = prediction[c];
                float derivative = activated * (1.0f - activated);
                float deltaValue = diff[c] * derivative;
                deltaCurrentHalf[sampleOffset + c] = half(deltaValue);
            }
            for (uint c = SIREN_TRAIN_OUTPUT_DIM; c < SIREN_TRAIN_MAX_DIM; ++c) {
                deltaCurrentHalf[sampleOffset + c] = half(0.0f);
            }
        }

        for (uint sample = linearThreadId + chunkSize; sample < SIREN_TRAIN_BATCH_SIZE; sample += threadgroupSize) {
            for (uint feature = 0; feature < SIREN_TRAIN_MAX_DIM; ++feature) {
                deltaCurrentHalf[sample * SIREN_TRAIN_MAX_DIM + feature] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup half *currentDelta = deltaCurrentHalf;
        threadgroup half *nextDelta = deltaNextHalf;

        for (int layerIndex = int(params.layerCount) - 1; layerIndex >= 0; --layerIndex) {
            auto weights = mlpLayers.weights[layerIndex];
            auto gWeights = gradients.weights[layerIndex];
            auto gBiases = gradients.biases[layerIndex];

            const uint outputDim = layerOutputDims[layerIndex];
            const uint inputDim = layerInputDims[layerIndex];
            const uint prevActivationBase = activationIndex(uint(layerIndex), 0, 0, chunkIndex, layerStride, activationChunkStride);

            for (uint idx = linearThreadId; idx < outputDim * inputDim; idx += threadgroupSize) {
                const uint row = idx / inputDim;
                const uint col = idx % inputDim;

                float gradSum = 0.0f;
                for (uint sample = 0; sample < chunkSize; ++sample) {
                    const uint sampleOffset = sample * SIREN_TRAIN_MAX_DIM;
                    float deltaVal = float(currentDelta[sampleOffset + row]);
                    float actVal = float(activationHistory[prevActivationBase + sampleOffset + col]);
                    gradSum += deltaVal * actVal;
                }
                float contribution = gradSum * invGlobalBatch;
                device atomic_float *gradPtr = reinterpret_cast<device atomic_float *>(&gWeights[row, col]);
                atomic_fetch_add_explicit(gradPtr, contribution, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint idx = linearThreadId; idx < outputDim; idx += threadgroupSize) {
                float gradBiasSum = 0.0f;
                for (uint sample = 0; sample < chunkSize; ++sample) {
                    gradBiasSum += float(currentDelta[sample * SIREN_TRAIN_MAX_DIM + idx]);
                }
                float contribution = gradBiasSum * invGlobalBatch;
                device atomic_float *biasPtr = reinterpret_cast<device atomic_float *>(&gBiases[idx]);
                atomic_fetch_add_explicit(biasPtr, contribution, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (layerIndex == 0) {
                break;
            }

            for (uint idx = linearThreadId; idx < chunkSize * inputDim; idx += threadgroupSize) {
                const uint sample = idx / inputDim;
                const uint neuron = idx % inputDim;
                float sum = 0.0f;
                for (uint row = 0; row < outputDim; ++row) {
                    float weightVal = float(weights[row, neuron]);
                    float deltaVal = float(currentDelta[sample * SIREN_TRAIN_MAX_DIM + row]);
                    sum += weightVal * deltaVal;
                }
                const uint offset = sample * SIREN_TRAIN_MAX_DIM + neuron;
                float preAct = float(preActivationHistory[layerOffset(uint(layerIndex - 1), chunkIndex, layerStride, preActivationChunkStride) + offset]);
                const float omega = (layerIndex - 1 == 0) ? 30.0f : 1.0f;
                float actDerivative = omega * cos(omega * preAct);
                float value = sum * actDerivative;
                nextDelta[offset] = half(value);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint sample = linearThreadId; sample < chunkSize; sample += threadgroupSize) {
                for (uint neuron = inputDim; neuron < SIREN_TRAIN_MAX_DIM; ++neuron) {
                    const uint offset = sample * SIREN_TRAIN_MAX_DIM + neuron;
                    nextDelta[offset] = half(0.0f);
                }
            }
            for (uint sample = linearThreadId + chunkSize; sample < SIREN_TRAIN_BATCH_SIZE; sample += threadgroupSize) {
                for (uint neuron = 0; neuron < SIREN_TRAIN_MAX_DIM; ++neuron) {
                    nextDelta[sample * SIREN_TRAIN_MAX_DIM + neuron] = half(0.0f);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup half *temp = currentDelta;
            currentDelta = nextDelta;
            nextDelta = temp;
        }

    uint lossSlot = min(linearThreadId, SIREN_TRAIN_THREADS - 1);
    if (lossSlot < SIREN_TRAIN_THREADS) {
        lossShared[lossSlot] = localLoss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (linearThreadId == 0) {
        float totalLoss = 0.0f;
        const uint lanes = min(threadgroupSize, SIREN_TRAIN_THREADS);
        for (uint i = 0; i < lanes; ++i) {
            totalLoss += lossShared[i];
        }
        device atomic_float *lossAtomic = reinterpret_cast<device atomic_float *>(lossBuffer);
        atomic_fetch_add_explicit(lossAtomic, totalLoss, memory_order_relaxed);
    }
}

kernel void sirenAdamUpdateWeights(
    device DeviceMLPLayers &mlpLayers              [[buffer(0)]],
    device DeviceFloatMLPLayers &gradients         [[buffer(1)]],
    device DeviceFloatMLPLayers &moments1          [[buffer(2)]],
    device DeviceFloatMLPLayers &moments2          [[buffer(3)]],
    constant SirenAdamParams &params               [[buffer(4)]],
    constant uint *layerInputDims                  [[buffer(5)]],
    constant uint *layerOutputDims                 [[buffer(6)]],
    uint gid                                       [[thread_position_in_grid]])
{
    if (gid >= params.totalWeightCount) {
        return;
    }

    uint remaining = gid;
    uint layerIndex = 0;
    for (; layerIndex < params.layerCount; ++layerIndex) {
        uint inputDim = layerInputDims[layerIndex];
        uint outputDim = layerOutputDims[layerIndex];
        uint layerWeights = inputDim * outputDim;
        if (remaining < layerWeights) {
            auto weights = mlpLayers.weights[layerIndex];
            auto gWeights = gradients.weights[layerIndex];
            auto mWeights = moments1.weights[layerIndex];
            auto vWeights = moments2.weights[layerIndex];

            uint row = remaining / inputDim;
            uint col = remaining % inputDim;

            float grad = gWeights[row, col];
            float weightValue = float(weights[row, col]);
            grad += params.weightDecay * weightValue;

            float m = params.beta1 * mWeights[row, col] + (1.0f - params.beta1) * grad;
            float v = params.beta2 * vWeights[row, col] + (1.0f - params.beta2) * grad * grad;

            mWeights[row, col] = m;
            vWeights[row, col] = v;

            float mHat = m / params.beta1Denom;
            float vHat = v / params.beta2Denom;
            float update = params.learningRate * (mHat / (sqrt(vHat) + params.epsilon));
            weights[row, col] = half(weightValue - update);
            if (params.preserveGradients == 0) {
                gWeights[row, col] = 0.0f;
            }
            return;
        }
        remaining -= layerWeights;
    }
}

kernel void sirenAdamUpdateBiases(
    device DeviceMLPLayers &mlpLayers              [[buffer(0)]],
    device DeviceFloatMLPLayers &gradients         [[buffer(1)]],
    device DeviceFloatMLPLayers &moments1          [[buffer(2)]],
    device DeviceFloatMLPLayers &moments2          [[buffer(3)]],
    constant SirenAdamParams &params               [[buffer(4)]],
    constant uint *layerInputDims                  [[buffer(5)]],
    constant uint *layerOutputDims                 [[buffer(6)]],
    uint gid                                       [[thread_position_in_grid]])
{
    if (gid >= params.totalBiasCount) {
        return;
    }

    uint remaining = gid;
    uint layerIndex = 0;
    for (; layerIndex < params.layerCount; ++layerIndex) {
        uint outputDim = layerOutputDims[layerIndex];
        if (remaining < outputDim) {
            auto layerBiases = mlpLayers.biases[layerIndex];
            auto gBiases = gradients.biases[layerIndex];
            auto mBiases = moments1.biases[layerIndex];
            auto vBiases = moments2.biases[layerIndex];

            uint idx = remaining;
            float grad = gBiases[idx];
            float biasValue = float(layerBiases[idx]);
            grad += params.weightDecay * biasValue;

            float m = params.beta1 * mBiases[idx] + (1.0f - params.beta1) * grad;
            float v = params.beta2 * vBiases[idx] + (1.0f - params.beta2) * grad * grad;

            mBiases[idx] = m;
            vBiases[idx] = v;

            float mHat = m / params.beta1Denom;
            float vHat = v / params.beta2Denom;
            float update = params.learningRate * (mHat / (sqrt(vHat) + params.epsilon));
            layerBiases[idx] = half(biasValue - update);
            if (params.preserveGradients == 0) {
                gBiases[idx] = 0.0f;
            }
            return;
        }
        remaining -= outputDim;
    }
}
