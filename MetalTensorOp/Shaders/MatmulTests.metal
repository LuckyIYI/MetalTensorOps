#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal/metal_simdgroup_matrix>
#include <metal/metal_simdgroup>

using namespace metal;
using namespace mpp::tensor_ops;

kernel void simdgroupMatrixMatrix(
    tensor<device half, dextents<int, 2>> A [[buffer(0)]],
    tensor<device half, dextents<int, 2>> B [[buffer(1)]],
    tensor<device half, dextents<int, 2>> C [[buffer(2)]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    constexpr int TILE_M = 64, TILE_N = 32;

    constexpr auto matmulDescriptor = matmul2d_descriptor(
        TILE_M, TILE_N, dynamic_length_v<int>,
        false, false, false,
        matmul2d_descriptor::mode::multiply
    );

    matmul2d<matmulDescriptor, execution_simdgroups<4>> matmulOp;

    auto mA = A.slice(0, tgid.y * TILE_M);
    auto mB = B.slice(tgid.x * TILE_N, 0);
    auto mC = C.slice(tgid.x * TILE_N, tgid.y * TILE_M);

    matmulOp.run(mA, mB, mC);
}

kernel void simdgroupMatrixVector(
    tensor<device half, dextents<int, 2>> A [[buffer(0)]],
    tensor<device half, dextents<int, 2>> B [[buffer(1)]],
    tensor<device half, dextents<int, 2>> C [[buffer(2)]],
    uint tgy [[threadgroup_position_in_grid]])
{
    constexpr auto matmulDescriptor = matmul2d_descriptor(64, 1, dynamic_length_v<int>);
    matmul2d<matmulDescriptor, execution_simdgroups<4>> matmulOp;

    auto mA = A.slice(0, tgy * 64);
    auto mB = B;
    auto mC = C.slice(0, tgy * 64);

    matmulOp.run(mA, mB, mC);
}

struct GemvConfig {
    uint rows;
    uint cols;
    uint rowsPerSimdgroup;
    uint simdgroupsPerThreadgroup;
};

kernel void simdHierarchicalMatrixVector(
    device const half *A [[buffer(0)]],
    device const half *B [[buffer(1)]],
    device half *C [[buffer(2)]],
    constant GemvConfig &config [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdWidth [[threads_per_simdgroup]],
    uint simdGroupIndex [[simdgroup_index_in_threadgroup]],
    uint tgIndex [[threadgroup_position_in_grid]]
) {
    constexpr uint VALUES_PER_THREAD = 4;
    constexpr uint MAX_ROWS_PER_SIMDGROUP = 8;
    constexpr uint MAX_SIMDGROUPS_PER_THREADGROUP = 8;

    const uint tileWidth = VALUES_PER_THREAD * simdWidth;

    uint rowsPerSimdgroup = max(1u, min(config.rowsPerSimdgroup, MAX_ROWS_PER_SIMDGROUP));
    uint simdgroupsPerThreadgroup = max(1u, min(config.simdgroupsPerThreadgroup, MAX_SIMDGROUPS_PER_THREADGROUP));

    if (simdGroupIndex >= simdgroupsPerThreadgroup) {
        return;
    }

    const uint globalGroup = tgIndex * simdgroupsPerThreadgroup + simdGroupIndex;
    const uint baseRow = globalGroup * rowsPerSimdgroup;

    const uint totalRows = config.rows;
    if (baseRow >= totalRows) {
        return;
    }

    uint remaining = totalRows - baseRow;
    uint activeRows = remaining < rowsPerSimdgroup ? remaining : rowsPerSimdgroup;

    float accumulators[MAX_ROWS_PER_SIMDGROUP];
    const device half *rowPointers[MAX_ROWS_PER_SIMDGROUP];
    for (uint r = 0; r < activeRows; ++r) {
        accumulators[r] = 0.0f;
        rowPointers[r] = A + (baseRow + r) * config.cols;
    }
    for (uint r = activeRows; r < MAX_ROWS_PER_SIMDGROUP; ++r) {
        accumulators[r] = 0.0f;
        rowPointers[r] = nullptr;
    }

    for (uint base = 0; base < config.cols; base += tileWidth) {
        float bVals[VALUES_PER_THREAD];
        for (uint i = 0; i < VALUES_PER_THREAD; ++i) {
            const uint column = base + lane * VALUES_PER_THREAD + i;
            bVals[i] = (column < config.cols) ? float(B[column]) : 0.0f;
        }

        for (uint r = 0; r < activeRows; ++r) {
            const device half *rowPtr = rowPointers[r];
            float partial = accumulators[r];

            for (uint i = 0; i < VALUES_PER_THREAD; ++i) {
                const uint column = base + lane * VALUES_PER_THREAD + i;
                if (column < config.cols) {
                    const float a = float(rowPtr[column]);
                    partial = fma(a, bVals[i], partial);
                }
            }

            accumulators[r] = partial;
        }
    }

    for (uint r = 0; r < activeRows; ++r) {
        const float sum = simd_sum(accumulators[r]);
        if (lane == 0) {
            C[baseRow + r] = half(sum);
        }
    }
}

kernel void simdgroupMatrixMatrixMetal3(
    device const half *A [[buffer(0)]],
    device const half *B [[buffer(1)]],
    device float      *C [[buffer(2)]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    constexpr int N = 64, K = 256;
    constexpr uint TILE_M = 8, TILE_N = 8, TILE_K = 8;

    const uint rowBase = tgid.y * TILE_M;
    const uint colBase = tgid.x * TILE_N;

    auto acc = make_filled_simdgroup_matrix<float, TILE_M, TILE_N>(float(0));

    simdgroup_half8x8 aFrag;
    simdgroup_half8x8 bFrag;

    for (uint k = 0; k < K; k += TILE_K) {
        simdgroup_load(aFrag, A + rowBase * K + k, K);
        simdgroup_load(bFrag, B + k * N + colBase, N);
        simdgroup_multiply_accumulate(acc, aFrag, bFrag, acc);
    }

    simdgroup_store(acc, C + rowBase * N + colBase, N);
}

kernel void threadMatrixMatrix(
    tensor<device half, dextents<int, 2>> A [[buffer(0)]],
    tensor<device half, dextents<int, 2>> B [[buffer(1)]],
    tensor<device half, dextents<int, 2>> C [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    constexpr matmul2d_descriptor descriptor(64, 32, dynamic_length_v<int>, false, false, false);
    matmul2d<descriptor, execution_thread> matmulOp;
    matmulOp.run(A, B, C);
}

kernel void threadVectorMatrix(
    tensor<device half, dextents<int, 2>> A [[buffer(0)]], // (1 x K)
    tensor<device half, dextents<int, 2>> B [[buffer(1)]], // (K x N)
    tensor<device half, dextents<int, 2>> C [[buffer(2)]], // (1 x N)
    uint gid [[thread_position_in_grid]]
) {
    constexpr matmul2d_descriptor descriptor(1, 17, dynamic_length_v<int>, false, false, false);
    matmul2d<descriptor, execution_thread> matmulOp;
    matmulOp.run(A, B, C);
}

kernel void threadMatrixVector(
    tensor<device half, dextents<int, 2>> A [[buffer(0)]], // (M x K)
    tensor<device half, dextents<int, 2>> B [[buffer(1)]], // (K x 1)
    tensor<device half, dextents<int, 2>> C [[buffer(2)]], // (M x 1)
    uint gid [[thread_position_in_grid]]
) {
    constexpr matmul2d_descriptor descriptor(29, 1, dynamic_length_v<int>, false, false, false);
    matmul2d<descriptor, execution_thread> matmulOp;
    matmulOp.run(A, B, C);
}

#define THREAD_MLP_TEST_MAX_INPUT  4
#define THREAD_MLP_TEST_MAX_HIDDEN 256
#define THREAD_MLP_TEST_MAX_OUTPUT 16

struct ThreadMLPTestMetadata {
    uint inputDim;
    uint hiddenDim;
    uint outputDim;
};

kernel void threadMLP(
    tensor<device half, dextents<int, 2>> weightsInputHidden [[buffer(0)]],
    tensor<device half, dextents<int, 2>> weightsHiddenOutput [[buffer(1)]],
    tensor<device half, dextents<int, 2>> inputTensor [[buffer(2)]],
    tensor<device half, dextents<int, 2>> outputTensor [[buffer(3)]],
    tensor<device half, dextents<int, 1>> biasHidden [[buffer(4)]],
    tensor<device half, dextents<int, 1>> biasOutput [[buffer(5)]],
    constant ThreadMLPTestMetadata &metadata [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint inputDim = metadata.inputDim;
    const uint hiddenDim = metadata.hiddenDim;
    const uint outputDim = metadata.outputDim;

    if (inputDim > THREAD_MLP_TEST_MAX_INPUT ||
        hiddenDim > THREAD_MLP_TEST_MAX_HIDDEN ||
        outputDim > THREAD_MLP_TEST_MAX_OUTPUT) {
        return;
    }

    thread half inputActivations[THREAD_MLP_TEST_MAX_HIDDEN] = {};
    thread half hiddenActivations[THREAD_MLP_TEST_MAX_HIDDEN] = {};
    thread half outputActivations[THREAD_MLP_TEST_MAX_OUTPUT] = {};

    auto input = tensor(inputActivations, dextents<int, 2>(THREAD_MLP_TEST_MAX_HIDDEN, 1));
    auto hidden = tensor(hiddenActivations, dextents<int, 2>(THREAD_MLP_TEST_MAX_HIDDEN, 1));
    auto output = tensor(outputActivations, dextents<int, 2>(THREAD_MLP_TEST_MAX_OUTPUT, 1));

    for (uint i = 0; i < inputDim; ++i) {
        input[i, 0] = inputTensor[i, 0];
    }

    constexpr matmul2d_descriptor inputHiddenDescriptor(
        1,
        THREAD_MLP_TEST_MAX_HIDDEN,
        THREAD_MLP_TEST_MAX_INPUT,
        false,
        false,
        true,
        matmul2d_descriptor::mode::multiply
    );

    constexpr matmul2d_descriptor hiddenOutputDescriptor(
        1,
        THREAD_MLP_TEST_MAX_OUTPUT,
        THREAD_MLP_TEST_MAX_HIDDEN,
        false,
        false,
        true,
        matmul2d_descriptor::mode::multiply
    );

    matmul2d<inputHiddenDescriptor, execution_thread>{}.run(input, weightsInputHidden, hidden);

    for (uint h = 0; h < hiddenDim; ++h) {
        float sum = float(hidden[h, 0]) + float(biasHidden[h]);
        input[h, 0] = half(max(sum, 0.0f));
    }

    matmul2d<hiddenOutputDescriptor, execution_thread>{}.run(input, weightsHiddenOutput, output);

    for (uint o = 0; o < outputDim; ++o) {
        float sum = float(output[o, 0]) + float(biasOutput[o]);
        outputTensor[o, 0] = half(sum);
    }
}


#define THREAD_DYNAMIC_MLP_MAX_LAYERS 8
#define THREAD_DYNAMIC_MLP_INPUT_DIM 4
#define THREAD_DYNAMIC_MLP_HIDDEN_DIM 32
#define THREAD_DYNAMIC_MLP_OUTPUT_DIM 8

struct ThreadDynamicMLPMetadata {
    uint inputDim;
    uint hiddenDim;
    uint outputDim;
    uint hiddenLayerCount; // Number of hidden layers (not including input/output)
};

// Dynamic MLP with loop and runtime tensor dimensions (EXACT pattern from SirenMLP)
struct DynamicMLPLayers {
    tensor<device half, dextents<int, 2>> weights[THREAD_DYNAMIC_MLP_MAX_LAYERS];
    tensor<constant half, dextents<int, 1>> biases[THREAD_DYNAMIC_MLP_MAX_LAYERS];
};

kernel void threadDynamicMLP(
    constant DynamicMLPLayers &layers [[buffer(0)]],
    tensor<device half, dextents<int, 2>> inputTensor [[buffer(1)]],
    tensor<device half, dextents<int, 2>> outputTensor [[buffer(2)]],
    constant ThreadDynamicMLPMetadata &metadata [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint inputDim = metadata.inputDim;
    const uint hiddenDim = metadata.hiddenDim;
    const uint outputDim = metadata.outputDim;
    const uint hiddenLayerCount = metadata.hiddenLayerCount;

    // Total layers = 1 input->hidden + (hiddenLayerCount-1) hidden->hidden + 1 hidden->output
    const uint totalLayers = hiddenLayerCount + 1;

    if (totalLayers == 0 || totalLayers > THREAD_DYNAMIC_MLP_MAX_LAYERS) {
        return;
    }

    // Allocate buffers (input goes in one, hidden in the other, ping-pong)
    thread half inputBuffer[THREAD_DYNAMIC_MLP_INPUT_DIM] = {};
    thread half hiddenA[THREAD_DYNAMIC_MLP_HIDDEN_DIM] = {};
    thread half hiddenB[THREAD_DYNAMIC_MLP_HIDDEN_DIM] = {};
    thread half outputBuffer[THREAD_DYNAMIC_MLP_OUTPUT_DIM] = {};

    // Load input
    for (uint i = 0; i < inputDim; ++i) {
        inputBuffer[i] = inputTensor[i, 0];
    }

    thread half *currentActivations = inputBuffer;
    thread half *nextHiddenBuffer = hiddenA;
    uint currentDim = inputDim;

    // MatMul descriptors matching SirenMLP
    constexpr matmul2d_descriptor matDesc_in2hid(1, THREAD_DYNAMIC_MLP_HIDDEN_DIM, THREAD_DYNAMIC_MLP_INPUT_DIM, false, false, true);
    constexpr matmul2d_descriptor matDesc_hid2hid(1, THREAD_DYNAMIC_MLP_HIDDEN_DIM, THREAD_DYNAMIC_MLP_HIDDEN_DIM, false, false, true);
    constexpr matmul2d_descriptor matDesc_hid2out(1, THREAD_DYNAMIC_MLP_OUTPUT_DIM, THREAD_DYNAMIC_MLP_HIDDEN_DIM, false, false, true);

    // Process all layers
    for (uint layerIdx = 0; layerIdx < totalLayers; ++layerIdx) {
        const bool isFirst = (layerIdx == 0);
        const bool isLast = (layerIdx == totalLayers - 1);

        // Determine output dimension and buffer
        const uint targetDim = isLast ? outputDim : hiddenDim;
        thread half *layerOut = isLast ? outputBuffer : nextHiddenBuffer;

        // Create tensors with RUNTIME dimensions
        auto inT = tensor(currentActivations, dextents<int, 2>(currentDim, 1));
        auto outT = tensor(layerOut, dextents<int, 2>(targetDim, 1));

        // Get weight and bias for this layer
        auto W = layers.weights[layerIdx];
        auto b = layers.biases[layerIdx];

        // Run matmul with appropriate descriptor
        if (isLast) {
            matmul2d<matDesc_hid2out, execution_thread>{}.run(inT, W, outT);
        } else if (isFirst) {
            matmul2d<matDesc_in2hid, execution_thread>{}.run(inT, W, outT);
        } else {
            matmul2d<matDesc_hid2hid, execution_thread>{}.run(inT, W, outT);
        }

        // Apply bias and activation (ReLU for hidden, linear for output)
        for (uint i = 0; i < targetDim; ++i) {
            float sum = float(outT[i, 0]) + float(b[i]);
            if (isLast) {
                layerOut[i] = half(sum); // Linear output
            } else {
                layerOut[i] = half(max(sum, 0.0f)); // ReLU
            }
        }

        // Advance pointers
        currentActivations = layerOut;
        currentDim = targetDim;

        // Ping-pong hidden buffers for next hidden layer
        if (!isLast) {
            nextHiddenBuffer = (layerOut == hiddenA) ? hiddenB : hiddenA;
        }
    }

    // Write output
    for (uint o = 0; o < outputDim; ++o) {
        outputTensor[o, 0] = outputBuffer[o];
    }
}
