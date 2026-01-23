/**
 * ONNX Zig - C API Header
 *
 * This header provides C bindings for the ONNX Zig inference library.
 * Use this for FFI integration with Swift, Java/JNI, Node.js N-API, etc.
 */

#ifndef ONNX_ZIG_H
#define ONNX_ZIG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Error Codes
// =============================================================================

typedef enum {
    ONNX_ZIG_OK = 0,
    ONNX_ZIG_ERROR_ALLOCATION_FAILED = 1,
    ONNX_ZIG_ERROR_INVALID_INPUT = 2,
    ONNX_ZIG_ERROR_INVALID_OUTPUT = 3,
    ONNX_ZIG_ERROR_MODEL_NOT_FOUND = 4,
    ONNX_ZIG_ERROR_INVALID_MODEL = 5,
    ONNX_ZIG_ERROR_INFERENCE_FAILED = 6,
    ONNX_ZIG_ERROR_SHAPE_MISMATCH = 7,
    ONNX_ZIG_ERROR_NULL_POINTER = 8,
    ONNX_ZIG_ERROR_UNKNOWN = 99,
} OnnxZigError;

// =============================================================================
// Opaque Handle Types
// =============================================================================

typedef struct OnnxZigSession* OnnxZigSessionHandle;
typedef struct OnnxZigTensor* OnnxZigTensorHandle;

// =============================================================================
// Session API
// =============================================================================

/**
 * Create a new inference session from an ONNX model file.
 *
 * @param model_path Path to the ONNX model file (null-terminated string)
 * @param out_session Pointer to receive the session handle
 * @return ONNX_ZIG_OK on success, error code otherwise
 */
OnnxZigError onnx_zig_session_create(const char* model_path, OnnxZigSessionHandle* out_session);

/**
 * Create a new inference session with options.
 *
 * @param model_path Path to the ONNX model file
 * @param intra_op_threads Number of threads for intra-op parallelism (0 = auto)
 * @param inter_op_threads Number of threads for inter-op parallelism (0 = auto)
 * @param out_session Pointer to receive the session handle
 * @return ONNX_ZIG_OK on success, error code otherwise
 */
OnnxZigError onnx_zig_session_create_with_options(
    const char* model_path,
    uint32_t intra_op_threads,
    uint32_t inter_op_threads,
    OnnxZigSessionHandle* out_session
);

/**
 * Destroy an inference session and free its resources.
 *
 * @param session Session handle to destroy
 */
void onnx_zig_session_destroy(OnnxZigSessionHandle session);

/**
 * Get the number of model inputs.
 *
 * @param session Session handle
 * @return Number of inputs
 */
size_t onnx_zig_session_get_input_count(OnnxZigSessionHandle session);

/**
 * Get the number of model outputs.
 *
 * @param session Session handle
 * @return Number of outputs
 */
size_t onnx_zig_session_get_output_count(OnnxZigSessionHandle session);

/**
 * Get the name of an input tensor.
 *
 * @param session Session handle
 * @param index Input index
 * @return Input name (null-terminated string, valid until session is destroyed)
 */
const char* onnx_zig_session_get_input_name(OnnxZigSessionHandle session, size_t index);

/**
 * Get the name of an output tensor.
 *
 * @param session Session handle
 * @param index Output index
 * @return Output name (null-terminated string, valid until session is destroyed)
 */
const char* onnx_zig_session_get_output_name(OnnxZigSessionHandle session, size_t index);

// =============================================================================
// Inference API
// =============================================================================

/**
 * Run inference with float32 input data.
 *
 * @param session Session handle
 * @param input_data Array of pointers to input data arrays
 * @param input_shapes Array of input shapes (each shape is an array of int64)
 * @param input_shape_dims Array of dimension counts for each input
 * @param input_count Number of inputs
 * @param out_tensors Pointer to receive array of output tensor handles
 * @param out_count Pointer to receive number of output tensors
 * @return ONNX_ZIG_OK on success, error code otherwise
 */
OnnxZigError onnx_zig_session_run_f32(
    OnnxZigSessionHandle session,
    const float** input_data,
    const int64_t** input_shapes,
    const size_t* input_shape_dims,
    size_t input_count,
    OnnxZigTensorHandle** out_tensors,
    size_t* out_count
);

/**
 * Run inference with a single float32 input (convenience function).
 *
 * @param session Session handle
 * @param input_data Pointer to input data
 * @param input_shape Array of dimension sizes
 * @param shape_dims Number of dimensions
 * @param out_tensors Pointer to receive array of output tensor handles
 * @param out_count Pointer to receive number of output tensors
 * @return ONNX_ZIG_OK on success, error code otherwise
 */
OnnxZigError onnx_zig_session_run_f32_simple(
    OnnxZigSessionHandle session,
    const float* input_data,
    const int64_t* input_shape,
    size_t shape_dims,
    OnnxZigTensorHandle** out_tensors,
    size_t* out_count
);

// =============================================================================
// Tensor API
// =============================================================================

/**
 * Create a new float32 tensor.
 *
 * @param shape Array of dimension sizes
 * @param ndim Number of dimensions
 * @param out_tensor Pointer to receive tensor handle
 * @return ONNX_ZIG_OK on success, error code otherwise
 */
OnnxZigError onnx_zig_tensor_create_f32(
    const size_t* shape,
    size_t ndim,
    OnnxZigTensorHandle* out_tensor
);

/**
 * Create a float32 tensor from existing data (copies data).
 *
 * @param data Pointer to data
 * @param shape Array of dimension sizes
 * @param ndim Number of dimensions
 * @param out_tensor Pointer to receive tensor handle
 * @return ONNX_ZIG_OK on success, error code otherwise
 */
OnnxZigError onnx_zig_tensor_create_f32_from_data(
    const float* data,
    const size_t* shape,
    size_t ndim,
    OnnxZigTensorHandle* out_tensor
);

/**
 * Destroy a tensor and free its resources.
 *
 * @param tensor Tensor handle to destroy
 */
void onnx_zig_tensor_destroy(OnnxZigTensorHandle tensor);

/**
 * Free an array of output tensors from inference.
 *
 * @param tensors Array of tensor handles
 * @param count Number of tensors
 */
void onnx_zig_tensors_destroy(OnnxZigTensorHandle* tensors, size_t count);

/**
 * Get tensor data pointer (read-only).
 *
 * @param tensor Tensor handle
 * @return Pointer to tensor data (float32)
 */
const float* onnx_zig_tensor_get_data_f32(OnnxZigTensorHandle tensor);

/**
 * Get tensor data pointer (mutable).
 *
 * @param tensor Tensor handle
 * @return Pointer to tensor data (float32)
 */
float* onnx_zig_tensor_get_data_f32_mut(OnnxZigTensorHandle tensor);

/**
 * Get the total number of elements in a tensor.
 *
 * @param tensor Tensor handle
 * @return Number of elements
 */
size_t onnx_zig_tensor_get_numel(OnnxZigTensorHandle tensor);

/**
 * Get the number of dimensions.
 *
 * @param tensor Tensor handle
 * @return Number of dimensions
 */
size_t onnx_zig_tensor_get_ndim(OnnxZigTensorHandle tensor);

/**
 * Get a specific dimension size.
 *
 * @param tensor Tensor handle
 * @param dim Dimension index
 * @return Size of the dimension
 */
size_t onnx_zig_tensor_get_dim(OnnxZigTensorHandle tensor, size_t dim);

/**
 * Get the shape array.
 *
 * @param tensor Tensor handle
 * @param out_shape Array to receive shape (must be at least ndim elements)
 * @param max_dims Maximum number of dimensions to copy
 * @return Number of dimensions copied
 */
size_t onnx_zig_tensor_get_shape(OnnxZigTensorHandle tensor, size_t* out_shape, size_t max_dims);

// =============================================================================
// Tensor Operations
// =============================================================================

/**
 * Matrix multiplication: C = A @ B
 *
 * @param a First matrix tensor [M, K]
 * @param b Second matrix tensor [K, N]
 * @param out_result Pointer to receive result tensor [M, N]
 * @return ONNX_ZIG_OK on success, error code otherwise
 */
OnnxZigError onnx_zig_matmul(
    OnnxZigTensorHandle a,
    OnnxZigTensorHandle b,
    OnnxZigTensorHandle* out_result
);

/**
 * Transpose a 2D tensor.
 *
 * @param tensor Input tensor [M, N]
 * @param out_result Pointer to receive transposed tensor [N, M]
 * @return ONNX_ZIG_OK on success, error code otherwise
 */
OnnxZigError onnx_zig_transpose(
    OnnxZigTensorHandle tensor,
    OnnxZigTensorHandle* out_result
);

/**
 * Apply softmax activation in-place.
 *
 * @param tensor Tensor to apply softmax to
 */
void onnx_zig_softmax_inplace(OnnxZigTensorHandle tensor);

/**
 * Get argmax index.
 *
 * @param tensor Tensor to find argmax of
 * @return Index of maximum element
 */
size_t onnx_zig_argmax(OnnxZigTensorHandle tensor);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Get the library version string.
 *
 * @return Version string (e.g., "0.1.0")
 */
const char* onnx_zig_version(void);

/**
 * Get the last error message.
 *
 * @return Error message string (valid until next API call)
 */
const char* onnx_zig_get_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // ONNX_ZIG_H
