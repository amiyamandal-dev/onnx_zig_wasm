//! ONNX Runtime C API bindings for Zig
//! This module provides FFI bindings to the ONNX Runtime C API using @cImport
//! for automatic struct layout matching.
//!
//! Reference: https://onnxruntime.ai/docs/api/c/

const std = @import("std");
const builtin = @import("builtin");

// =============================================================================
// C API Import
// =============================================================================

pub const c = @cImport({
    @cInclude("onnxruntime/onnxruntime_c_api.h");
});

// =============================================================================
// Re-export commonly used types
// =============================================================================

pub const OrtEnv = c.OrtEnv;
pub const OrtStatus = c.OrtStatus;
pub const OrtMemoryInfo = c.OrtMemoryInfo;
pub const OrtSession = c.OrtSession;
pub const OrtValue = c.OrtValue;
pub const OrtRunOptions = c.OrtRunOptions;
pub const OrtTypeInfo = c.OrtTypeInfo;
pub const OrtTensorTypeAndShapeInfo = c.OrtTensorTypeAndShapeInfo;
pub const OrtSessionOptions = c.OrtSessionOptions;
pub const OrtAllocator = c.OrtAllocator;
pub const OrtApi = c.OrtApi;
pub const OrtApiBase = c.OrtApiBase;

// =============================================================================
// Enums (re-exported for convenience)
// =============================================================================

pub const OrtLoggingLevel = c.OrtLoggingLevel;
pub const OrtErrorCode = c.OrtErrorCode;
pub const ONNXTensorElementDataType = c.ONNXTensorElementDataType;
pub const ONNXType = c.ONNXType;
pub const OrtAllocatorType = c.OrtAllocatorType;
pub const OrtMemType = c.OrtMemType;
pub const GraphOptimizationLevel = c.GraphOptimizationLevel;
pub const ExecutionMode = c.ExecutionMode;

// =============================================================================
// API Version
// =============================================================================

pub const ORT_API_VERSION: u32 = c.ORT_API_VERSION;

// =============================================================================
// External Function Declaration
// =============================================================================

pub const OrtGetApiBase = c.OrtGetApiBase;

// =============================================================================
// Zig Error Types
// =============================================================================

pub const OrtError = error{
    Fail,
    InvalidArgument,
    NoSuchFile,
    NoModel,
    EngineError,
    RuntimeException,
    InvalidProtobuf,
    ModelLoaded,
    NotImplemented,
    InvalidGraph,
    ExecutionProviderFail,
    NullApi,
    NullResult,
};

// =============================================================================
// Helper Functions
// =============================================================================

var global_api: ?*const OrtApi = null;

/// Get the ONNX Runtime API, initializing if necessary
pub fn getApi() OrtError!*const OrtApi {
    if (global_api) |api| {
        return api;
    }

    const api_base = OrtGetApiBase();
    const api = api_base.*.GetApi.?(ORT_API_VERSION);

    if (api) |a| {
        global_api = a;
        return a;
    }

    return OrtError.NullApi;
}

/// Convert OrtStatus to Zig error, releasing the status
pub fn checkStatus(api: *const OrtApi, status: ?*OrtStatus) OrtError!void {
    if (status) |s| {
        const error_code = api.GetErrorCode.?(s);
        api.ReleaseStatus.?(s);

        return switch (error_code) {
            c.ORT_OK => {},
            c.ORT_FAIL => OrtError.Fail,
            c.ORT_INVALID_ARGUMENT => OrtError.InvalidArgument,
            c.ORT_NO_SUCHFILE => OrtError.NoSuchFile,
            c.ORT_NO_MODEL => OrtError.NoModel,
            c.ORT_ENGINE_ERROR => OrtError.EngineError,
            c.ORT_RUNTIME_EXCEPTION => OrtError.RuntimeException,
            c.ORT_INVALID_PROTOBUF => OrtError.InvalidProtobuf,
            c.ORT_MODEL_LOADED => OrtError.ModelLoaded,
            c.ORT_NOT_IMPLEMENTED => OrtError.NotImplemented,
            c.ORT_INVALID_GRAPH => OrtError.InvalidGraph,
            c.ORT_EP_FAIL => OrtError.ExecutionProviderFail,
            else => OrtError.Fail,
        };
    }
}

/// Get error message from status (caller must not release status yet)
pub fn getStatusMessage(api: *const OrtApi, status: *const OrtStatus) []const u8 {
    const msg = api.GetErrorMessage.?(status);
    return std.mem.span(msg);
}

// =============================================================================
// Type Conversion Helpers
// =============================================================================

/// Convert Zig type to ONNX tensor element type
pub fn zigTypeToOnnx(comptime T: type) c_int {
    return switch (T) {
        f32 => c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        f64 => c.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
        i8 => c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
        i16 => c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
        i32 => c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
        i64 => c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        u8 => c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
        u16 => c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
        u32 => c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
        u64 => c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
        bool => c.ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
        else => c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,
    };
}

/// Get byte size for ONNX tensor element type
pub fn onnxTypeByteSize(dtype: c_int) usize {
    return switch (dtype) {
        c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => 4,
        c.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => 8,
        c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 => 1,
        c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => 2,
        c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => 4,
        c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => 8,
        c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => 1,
        c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 => 2,
        c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 => 4,
        c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 => 8,
        c.ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => 1,
        c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 => 2,
        c.ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 => 2,
        else => 0,
    };
}

// =============================================================================
// Tests
// =============================================================================

test "zigTypeToOnnx conversion" {
    try std.testing.expectEqual(c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, zigTypeToOnnx(f32));
    try std.testing.expectEqual(c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, zigTypeToOnnx(i64));
    try std.testing.expectEqual(c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, zigTypeToOnnx(u8));
}

test "onnxTypeByteSize" {
    try std.testing.expectEqual(@as(usize, 4), onnxTypeByteSize(c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
    try std.testing.expectEqual(@as(usize, 8), onnxTypeByteSize(c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));
    try std.testing.expectEqual(@as(usize, 1), onnxTypeByteSize(c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8));
}
