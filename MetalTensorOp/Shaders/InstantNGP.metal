#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_numeric>

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
#define NGP_LOG2_HASHMAP_SIZE 12
#define NGP_BASE_RESOLUTION 16
#define NGP_MAX_RESOLUTION 2048
#define NGP_TOTAL_FEATURES (NGP_NUM_LEVELS * NGP_FEATURES_PER_LEVEL)


// MLP configuration for cooperative execution
#define NGP_MLP_HIDDEN_WIDTH 64
#define NGP_MLP_OUTPUT_DIM 3  // RGB output
#define NGP_MLP_NUM_LAYERS 2

#define NGP_BATCH_SIZE 64  // Process 64 positions per threadgroup

// Hash function: 2D spatial hash matching MLX implementation
inline uint hash_position_2d(uint2 pos) {
    constexpr uint primes[2] = {1u, 2654435761u};
    uint result = 0;
    result ^= pos.x * primes[0];
    result ^= pos.y * primes[1];
    return result;
}

// Compute multi-resolution hash encoding for 2D position (MLX-compatible)
// Bilinear interpolation with 4 corners
inline void compute_hash_encoding_2d(
    float2 position,
    device half *hash_table,
    thread float *encoded_features,
    uint level
) {
    const float ln_min = floor(log((float)NGP_BASE_RESOLUTION));
    const float ln_max = floor(log((float)NGP_MAX_RESOLUTION));
    const float t = (NGP_NUM_LEVELS > 1) ? (float(level) / float(NGP_NUM_LEVELS - 1)) : 0.0f;
    const float scale_factor = exp(mix(ln_min, ln_max, t));

    const float2 scaled_pos = position * scale_factor;
    const float2 pos_floor = floor(scaled_pos);
    const float2 pos_fract = scaled_pos - pos_floor;

    const uint2 pos_grid = uint2(pos_floor);

    // Hash encoding with bilinear interpolation (4 corners)
    float accumulated_features[NGP_FEATURES_PER_LEVEL] = {};

    const uint table_size = 1u << NGP_LOG2_HASHMAP_SIZE;

    // Process 4 corners: [0,0], [0,1], [1,0], [1,1]
    for (uint corner = 0; corner < 4; ++corner) {
        const uint2 offset = uint2(
            (corner >> 0) & 1,
            (corner >> 1) & 1
        );

        const uint2 corner_pos = pos_grid + offset;

        // Compute bilinear weight for this corner
        const float wx = (offset.x == 0) ? (1.0f - pos_fract.x) : pos_fract.x;
        const float wy = (offset.y == 0) ? (1.0f - pos_fract.y) : pos_fract.y;
        const float weight = wx * wy;

        // Hash to table index (matching MLX)
        const uint hash = hash_position_2d(corner_pos);
        const uint hash_index = hash % table_size;

        // Accumulate features from hash table
        const uint feature_offset = (level * table_size + hash_index) * NGP_FEATURES_PER_LEVEL;

        for (uint f = 0; f < NGP_FEATURES_PER_LEVEL; ++f) {
            accumulated_features[f] += weight * float(hash_table[feature_offset + f]);
        }
    }

    // Write to output
    const uint output_offset = level * NGP_FEATURES_PER_LEVEL;
    for (uint f = 0; f < NGP_FEATURES_PER_LEVEL; ++f) {
        encoded_features[output_offset + f] = accumulated_features[f];
    }
}

// Full hash encoding across all levels (2D version)
inline void compute_full_hash_encoding_2d(
    float2 position,
    device half *hash_table,
    thread float *encoded_features
) {
    for (uint level = 0; level < NGP_NUM_LEVELS; ++level) {
        compute_hash_encoding_2d(position, hash_table, encoded_features, level);
    }
}

// Cooperative Instant NGP Inference Kernel
// Processes a batch of positions using cooperative matmul
kernel void instantNGPInference(
    tensor<device half, dextents<int, 2>> layer1_weights [[buffer(0)]],  // (features, hidden)
    tensor<device half, dextents<int, 1>> layer1_bias [[buffer(1)]],     // (hidden)
    tensor<device half, dextents<int, 2>> layer2_weights [[buffer(2)]],  // (hidden, output)
    tensor<device half, dextents<int, 1>> layer2_bias [[buffer(3)]],     // (output)
    device half *hash_table [[buffer(4)]],         // Multi-resolution hash table
    tensor<device float, dextents<int, 2>> positions [[buffer(5)]],  // Input positions (N, 2)
    tensor<device half, dextents<int, 2>> outputs [[buffer(6)]],     // Output RGB (N, 3)
    constant uint &num_positions [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint THREADS_PER_SIMDGROUP = 32u;
    constexpr uint SIMDGROUPS_PER_THREADGROUP = 4u;
    constexpr uint THREADGROUP_SIZE = THREADS_PER_SIMDGROUP * SIMDGROUPS_PER_THREADGROUP;

    const uint batch_start = tgid.y * NGP_BATCH_SIZE;
    if (batch_start >= num_positions) {
        return;
    }

    const uint remaining = num_positions - batch_start;
    const uint actual_batch = remaining < NGP_BATCH_SIZE ? remaining : uint(NGP_BATCH_SIZE);

    threadgroup half encoded_storage[NGP_BATCH_SIZE * NGP_TOTAL_FEATURES];
    threadgroup float hidden_accum_storage[NGP_BATCH_SIZE * NGP_MLP_HIDDEN_WIDTH];
    threadgroup half hidden_storage[NGP_BATCH_SIZE * NGP_MLP_HIDDEN_WIDTH];
    threadgroup float output_accum_storage[NGP_BATCH_SIZE * NGP_MLP_OUTPUT_DIM];
    threadgroup half output_storage[NGP_BATCH_SIZE * NGP_MLP_OUTPUT_DIM];

    const uint linear_thread_id = simd_group_id * THREADS_PER_SIMDGROUP + simd_lane_id;

    for (uint idx = linear_thread_id; idx < NGP_BATCH_SIZE * NGP_TOTAL_FEATURES; idx += THREADGROUP_SIZE) {
        encoded_storage[idx] = half(0.0f);
    }
    for (uint idx = linear_thread_id; idx < NGP_BATCH_SIZE * NGP_MLP_HIDDEN_WIDTH; idx += THREADGROUP_SIZE) {
        hidden_accum_storage[idx] = 0.0f;
        hidden_storage[idx] = half(0.0f);
    }
    for (uint idx = linear_thread_id; idx < NGP_BATCH_SIZE * NGP_MLP_OUTPUT_DIM; idx += THREADGROUP_SIZE) {
        output_accum_storage[idx] = 0.0f;
        output_storage[idx] = half(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint local_idx = linear_thread_id; local_idx < actual_batch; local_idx += THREADGROUP_SIZE) {
        const uint position_index = batch_start + local_idx;
        const float2 position = float2(
            positions[position_index, 0],
            positions[position_index, 1]
        );

        thread float encoded_features[NGP_TOTAL_FEATURES];
        compute_full_hash_encoding_2d(position, hash_table, encoded_features);

        #pragma unroll
        for (uint f = 0; f < NGP_TOTAL_FEATURES; ++f) {
            encoded_storage[local_idx * NGP_TOTAL_FEATURES + f] = half(encoded_features[f]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto encoded_tensor = tensor(encoded_storage, dextents<int, 2>(NGP_TOTAL_FEATURES, NGP_BATCH_SIZE));
    auto hidden_accum_tensor = tensor(hidden_accum_storage, dextents<int, 2>(NGP_MLP_HIDDEN_WIDTH, NGP_BATCH_SIZE));
    auto hidden_tensor = tensor(hidden_storage, dextents<int, 2>(NGP_MLP_HIDDEN_WIDTH, NGP_BATCH_SIZE));
    auto output_accum_tensor = tensor(output_accum_storage, dextents<int, 2>(NGP_MLP_OUTPUT_DIM, NGP_BATCH_SIZE));

    // Use consistent layout: C(N, M) = B(N, K) * A(K, M)
    // A = encoded(K, batch), B1 = W1(N, K), B2 = W2(N, K)
    constexpr auto layer1_descriptor = matmul2d_descriptor(
        NGP_BATCH_SIZE,            // M (batch)
        NGP_MLP_HIDDEN_WIDTH,      // N (hidden)
        dynamic_length_v<int>,     // K (features)
        false,                     // transpose A
        false,                     // transpose B
        false,
        matmul2d_descriptor::mode::multiply
    );

    constexpr auto layer2_descriptor = matmul2d_descriptor(
        NGP_BATCH_SIZE,            // M (batch)
        NGP_MLP_OUTPUT_DIM,        // N (output)
        dynamic_length_v<int>,     // K (hidden)
        false,
        false,
        false,
        matmul2d_descriptor::mode::multiply
    );

    matmul2d<layer1_descriptor, execution_simdgroups<4>> layer1_matmul;
    matmul2d<layer2_descriptor, execution_simdgroups<4>> layer2_matmul;

    layer1_matmul.run(encoded_tensor, layer1_weights, hidden_accum_tensor);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = linear_thread_id; idx < actual_batch * NGP_MLP_HIDDEN_WIDTH; idx += THREADGROUP_SIZE) {
        const uint batch_idx = idx / NGP_MLP_HIDDEN_WIDTH;
        const uint hidden_idx = idx % NGP_MLP_HIDDEN_WIDTH;
        float value = hidden_accum_storage[hidden_idx + batch_idx * NGP_MLP_HIDDEN_WIDTH] + float(layer1_bias[hidden_idx]);
        value = max(value, 0.0f);
        hidden_storage[hidden_idx + batch_idx * NGP_MLP_HIDDEN_WIDTH] = half(value);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    layer2_matmul.run(hidden_tensor, layer2_weights, output_accum_tensor);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = linear_thread_id; idx < actual_batch * NGP_MLP_OUTPUT_DIM; idx += THREADGROUP_SIZE) {
        const uint batch_idx = idx / NGP_MLP_OUTPUT_DIM;
        const uint output_idx = idx % NGP_MLP_OUTPUT_DIM;
        float value = output_accum_storage[output_idx + batch_idx * NGP_MLP_OUTPUT_DIM] + float(layer2_bias[output_idx]);
        float sigmoid = 1.f / (1.f + exp(-value));
        output_storage[output_idx + batch_idx * NGP_MLP_OUTPUT_DIM] = half(sigmoid);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = linear_thread_id; idx < actual_batch * NGP_MLP_OUTPUT_DIM; idx += THREADGROUP_SIZE) {
        const uint batch_idx = idx / NGP_MLP_OUTPUT_DIM;
        const uint output_idx = idx % NGP_MLP_OUTPUT_DIM;
        const uint position_index = batch_start + batch_idx;
        outputs[position_index, output_idx] = output_storage[output_idx + batch_idx * NGP_MLP_OUTPUT_DIM];
    }
}


// Debug variant: dumps per-stage activations for comparison against MLX
kernel void instantNGPInferenceDebug(
    tensor<device half, dextents<int, 2>> layer1_weights [[buffer(0)]],  // (hidden, features)
    tensor<device half, dextents<int, 1>> layer1_bias [[buffer(1)]],     // (hidden)
    tensor<device half, dextents<int, 2>> layer2_weights [[buffer(2)]],  // (output, hidden)
    tensor<device half, dextents<int, 1>> layer2_bias [[buffer(3)]],     // (output)
    device half *hash_table [[buffer(4)]],
    tensor<device float, dextents<int, 2>> positions [[buffer(5)]],      // (N, 2)
    tensor<device half, dextents<int, 2>> outputs [[buffer(6)]],         // (N, 3)
    constant uint &num_positions [[buffer(7)]],
    tensor<device float, dextents<int, 2>> debug_encoded [[buffer(8)]],  // (N, NGP_TOTAL_FEATURES)
    tensor<device float, dextents<int, 2>> debug_hidden [[buffer(9)]],   // (N, NGP_MLP_HIDDEN_WIDTH)
    tensor<device float, dextents<int, 2>> debug_output [[buffer(10)]],  // (N, NGP_MLP_OUTPUT_DIM)
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint THREADS_PER_SIMDGROUP = 32u;
    constexpr uint SIMDGROUPS_PER_THREADGROUP = 4u;
    constexpr uint THREADGROUP_SIZE = THREADS_PER_SIMDGROUP * SIMDGROUPS_PER_THREADGROUP;

    const uint batch_start = tgid.y * NGP_BATCH_SIZE;
    if (batch_start >= num_positions) {
        return;
    }

    const uint remaining = num_positions - batch_start;
    const uint actual_batch = remaining < NGP_BATCH_SIZE ? remaining : uint(NGP_BATCH_SIZE);

    threadgroup half encoded_storage[NGP_BATCH_SIZE * NGP_TOTAL_FEATURES];
    threadgroup float hidden_accum_storage[NGP_BATCH_SIZE * NGP_MLP_HIDDEN_WIDTH];
    threadgroup half hidden_storage[NGP_BATCH_SIZE * NGP_MLP_HIDDEN_WIDTH];
    threadgroup float output_accum_storage[NGP_BATCH_SIZE * NGP_MLP_OUTPUT_DIM];
    threadgroup half output_storage[NGP_BATCH_SIZE * NGP_MLP_OUTPUT_DIM];

    const uint linear_thread_id = simd_group_id * THREADS_PER_SIMDGROUP + simd_lane_id;

    for (uint idx = linear_thread_id; idx < NGP_BATCH_SIZE * NGP_TOTAL_FEATURES; idx += THREADGROUP_SIZE) {
        encoded_storage[idx] = half(0.0f);
    }
    for (uint idx = linear_thread_id; idx < NGP_BATCH_SIZE * NGP_MLP_HIDDEN_WIDTH; idx += THREADGROUP_SIZE) {
        hidden_accum_storage[idx] = 0.0f;
        hidden_storage[idx] = half(0.0f);
    }
    for (uint idx = linear_thread_id; idx < NGP_BATCH_SIZE * NGP_MLP_OUTPUT_DIM; idx += THREADGROUP_SIZE) {
        output_accum_storage[idx] = 0.0f;
        output_storage[idx] = half(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 1) Hash encoding per sample
    for (uint local_idx = linear_thread_id; local_idx < actual_batch; local_idx += THREADGROUP_SIZE) {
        const uint position_index = batch_start + local_idx;
        const float2 position = float2(
            positions[position_index, 0],
            positions[position_index, 1]
        );

        thread float encoded_features[NGP_TOTAL_FEATURES];
        compute_full_hash_encoding_2d(position, hash_table, encoded_features);

        #pragma unroll
        for (uint f = 0; f < NGP_TOTAL_FEATURES; ++f) {
            encoded_storage[local_idx * NGP_TOTAL_FEATURES + f] = half(encoded_features[f]);
            // Dump encoded features in column-major: (N, F)
            debug_encoded[position_index, f] = encoded_features[f];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto encoded_tensor = tensor(encoded_storage, dextents<int, 2>(NGP_TOTAL_FEATURES, NGP_BATCH_SIZE));
    auto hidden_accum_tensor = tensor(hidden_accum_storage, dextents<int, 2>(NGP_MLP_HIDDEN_WIDTH, NGP_BATCH_SIZE));
    auto hidden_tensor = tensor(hidden_storage, dextents<int, 2>(NGP_MLP_HIDDEN_WIDTH, NGP_BATCH_SIZE));
    auto output_accum_tensor = tensor(output_accum_storage, dextents<int, 2>(NGP_MLP_OUTPUT_DIM, NGP_BATCH_SIZE));

    // Use consistent layout: C(N, M) = B(N, K) * A(K, M)
    constexpr auto layer1_descriptor = matmul2d_descriptor(
        NGP_BATCH_SIZE,
        NGP_MLP_HIDDEN_WIDTH,
        dynamic_length_v<int>,
        false,
        false,
        false,
        matmul2d_descriptor::mode::multiply
    );

    constexpr auto layer2_descriptor = matmul2d_descriptor(
        NGP_BATCH_SIZE,
        NGP_MLP_OUTPUT_DIM,
        dynamic_length_v<int>,
        false,
        false,
        false,
        matmul2d_descriptor::mode::multiply
    );

    // 2) Layer 1 matmul
    matmul2d<layer1_descriptor, execution_simdgroups<4>> layer1_matmul;
    matmul2d<layer2_descriptor, execution_simdgroups<4>> layer2_matmul;
    layer1_matmul.run(encoded_tensor, layer1_weights, hidden_accum_tensor);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bias + ReLU and dump hidden activations
    for (uint idx = linear_thread_id; idx < actual_batch * NGP_MLP_HIDDEN_WIDTH; idx += THREADGROUP_SIZE) {
        const uint batch_idx = idx / NGP_MLP_HIDDEN_WIDTH;
        const uint hidden_idx = idx % NGP_MLP_HIDDEN_WIDTH;
        float value = hidden_accum_storage[hidden_idx + batch_idx * NGP_MLP_HIDDEN_WIDTH] + float(layer1_bias[hidden_idx]);
        value = max(value, 0.0f);
        hidden_storage[hidden_idx + batch_idx * NGP_MLP_HIDDEN_WIDTH] = half(value);
        const uint position_index = batch_start + batch_idx;
        debug_hidden[position_index, hidden_idx] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3) Layer 2 matmul
    layer2_matmul.run(hidden_tensor, layer2_weights, output_accum_tensor);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bias + sigmoid and dump outputs
    for (uint idx = linear_thread_id; idx < actual_batch * NGP_MLP_OUTPUT_DIM; idx += THREADGROUP_SIZE) {
        const uint batch_idx = idx / NGP_MLP_OUTPUT_DIM;
        const uint output_idx = idx % NGP_MLP_OUTPUT_DIM;
        float value = output_accum_storage[output_idx + batch_idx * NGP_MLP_OUTPUT_DIM] + float(layer2_bias[output_idx]);
        float sigmoid = 1.f / (1.f + exp(-value));
        output_storage[output_idx + batch_idx * NGP_MLP_OUTPUT_DIM] = half(sigmoid);
        const uint position_index = batch_start + batch_idx;
        debug_output[position_index, output_idx] = sigmoid;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write final outputs (half) for completeness
    for (uint idx = linear_thread_id; idx < actual_batch * NGP_MLP_OUTPUT_DIM; idx += THREADGROUP_SIZE) {
        const uint batch_idx = idx / NGP_MLP_OUTPUT_DIM;
        const uint output_idx = idx % NGP_MLP_OUTPUT_DIM;
        const uint position_index = batch_start + batch_idx;
        outputs[position_index, output_idx] = output_storage[output_idx + batch_idx * NGP_MLP_OUTPUT_DIM];
    }
}

kernel void instantNGPEncodeDebug(
    device half *hash_table [[buffer(0)]],
    tensor<device float, dextents<int, 2>> positions [[buffer(1)]],
    tensor<device float, dextents<int, 2>> encodings [[buffer(2)]],
    constant uint &num_positions [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= num_positions) {
        return;
    }

    const float2 position = float2(positions[gid, 0], positions[gid, 1]);
    thread float encoded_features[NGP_TOTAL_FEATURES];
    compute_full_hash_encoding_2d(position, hash_table, encoded_features);

    for (uint f = 0; f < NGP_TOTAL_FEATURES; ++f) {
        encodings[gid, f] = encoded_features[f];
    }
}

kernel void instantNGPRender(
    texture2d<float, access::write> outTexture [[texture(0)]],
    tensor<device half, dextents<int, 2>> layer1_weights [[buffer(0)]],
    tensor<device half, dextents<int, 1>> layer1_bias [[buffer(1)]],
    tensor<device half, dextents<int, 2>> layer2_weights [[buffer(2)]],
    tensor<device half, dextents<int, 1>> layer2_bias [[buffer(3)]],
    device half *hash_table [[buffer(4)]],
    constant InstantNGPRenderUniforms &uniforms [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint texture_width = outTexture.get_width();
    const uint texture_height = outTexture.get_height();

    if (gid.x >= texture_width || gid.y >= texture_height) {
        return;
    }

    const uint2 pixel = gid;
    const float texture_width_f = float(texture_width);
    const float texture_height_f = float(texture_height);

    float2 uv = float2(float(pixel.x), float(pixel.y)) / float2(texture_width_f - 1.0, texture_height_f - 1.0);

    if (uniforms.trainingWidth > 0 && uniforms.trainingHeight > 0) {
        const float target_aspect = float(uniforms.trainingWidth) / float(uniforms.trainingHeight);
        const float texture_aspect = texture_width_f / texture_height_f;

        if (texture_aspect > target_aspect) {
            const float visible = target_aspect / texture_aspect;
            const float left = 0.5f - 0.5f * visible;
            uv.x = (uv.x - left) / visible;
        } else if (texture_aspect < target_aspect) {
            const float visible = texture_aspect / target_aspect;
            const float top = 0.5f - 0.5f * visible;
            uv.y = (uv.y - top) / visible;
        }
    }

    if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) {
        outTexture.write(float4(0.0f, 0.0f, 0.0f, 1.0f), pixel);
        return;
    }

    thread float encoded_features_f32[NGP_TOTAL_FEATURES];
    compute_full_hash_encoding_2d(uv, hash_table, encoded_features_f32);

    thread half encoded_features[NGP_TOTAL_FEATURES];
    for (uint f = 0; f < NGP_TOTAL_FEATURES; ++f) {
        encoded_features[f] = half(encoded_features_f32[f]);
    }
    auto encoded_tensor = tensor(encoded_features, dextents<int, 2>(NGP_TOTAL_FEATURES, 1));

    thread half hidden_matmul[NGP_MLP_HIDDEN_WIDTH];
    auto hidden_tensor = tensor(hidden_matmul, dextents<int, 2>(NGP_MLP_HIDDEN_WIDTH, 1));

    constexpr auto layer1_desc = matmul2d_descriptor(
        1,
        NGP_MLP_HIDDEN_WIDTH,
        NGP_TOTAL_FEATURES,
        false,
        false,
        false,
        matmul2d_descriptor::mode::multiply
    );

    matmul2d<layer1_desc, execution_thread> layer1_matmul;
    layer1_matmul.run(encoded_tensor, layer1_weights, hidden_tensor);

    for (uint hidden_idx = 0; hidden_idx < NGP_MLP_HIDDEN_WIDTH; ++hidden_idx) {
        float value = float(hidden_tensor[hidden_idx, 0]) + float(layer1_bias[hidden_idx]);
        value = max(value, 0.0f);
        hidden_tensor[hidden_idx, 0] = half(value);
    }

    thread half output_matmul[NGP_MLP_OUTPUT_DIM];
    auto output_tensor = tensor(output_matmul, dextents<int, 2>(NGP_MLP_OUTPUT_DIM, 1));

    constexpr auto layer2_desc = matmul2d_descriptor(
        1,
        NGP_MLP_OUTPUT_DIM,
        NGP_MLP_HIDDEN_WIDTH,
        false,
        false,
        false,
        matmul2d_descriptor::mode::multiply
    );

    matmul2d<layer2_desc, execution_thread> layer2_matmul;
    layer2_matmul.run(hidden_tensor, layer2_weights, output_tensor);

    float3 rgb;
    for (uint channel = 0; channel < NGP_MLP_OUTPUT_DIM; ++channel) {
        float value = float(output_tensor[channel, 0]) + float(layer2_bias[channel]);
        rgb[channel] = 1.f / (1.f + exp(-value));
    }

    outTexture.write(float4(rgb, 1.0f), pixel);
}

kernel void instantNGPTrainStep(
    tensor<device half, dextents<int, 2>> layer1_weights [[buffer(0)]],
    tensor<device half, dextents<int, 1>> layer1_bias [[buffer(1)]],
    tensor<device half, dextents<int, 2>> layer2_weights [[buffer(2)]],
    tensor<device half, dextents<int, 1>> layer2_bias [[buffer(3)]],
    device half *hash_table [[buffer(4)]],
    device float3 *positions [[buffer(5)]],
    device half3 *target_colors [[buffer(6)]],
    device half *hash_gradients [[buffer(7)]],
    device half *weight_gradients [[buffer(8)]],
    constant uint &batch_size [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    // Forward pass + backward pass implementation
    // Left as exercise - would implement:
    // 1. Forward pass to compute outputs
    // 2. Compute loss (MSE or similar)
    // 3. Backward pass through MLP
    // 4. Backward pass through hash encoding
    // 5. Accumulate gradients
}
