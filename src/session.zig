//! # High-level ONNX Inference Session
//!
//! This module provides an idiomatic Zig API for ONNX model inference,
//! wrapping the ONNX Runtime C API with automatic resource management.
//!
//! ## Features
//!
//! - Model loading from file path
//! - Automatic memory management with deferred cleanup
//! - Input/output tensor conversion (Zig Tensor <-> ONNX Runtime)
//! - Model introspection (input/output names, shapes, types)
//! - Comprehensive error handling
//!
//! ## Example Usage
//!
//! ```zig
//! const allocator = std.heap.page_allocator;
//!
//! // Load model
//! var session = try Session.init(allocator, "model.onnx");
//! defer session.deinit();
//!
//! // Check model metadata
//! std.debug.print("Inputs: {d}\n", .{session.getInputCount()});
//! std.debug.print("Outputs: {d}\n", .{session.getOutputCount()});
//!
//! // Run inference with f32 data
//! const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
//! const shape = [_]i64{ 1, 4 };
//!
//! const outputs = try session.runF32(
//!     &[_][]const f32{&input},
//!     &[_][]const i64{&shape},
//! );
//! defer session.freeOutputs(outputs);
//!
//! // Access output tensor data
//! const result = outputs[0];
//! for (result.data) |val| {
//!     std.debug.print("{d:.4} ", .{val});
//! }
//! ```
//!
//! ## Error Handling
//!
//! All operations that can fail return `SessionError`, which includes
//! both ONNX Runtime errors and session-specific errors:
//!
//! - `OrtNoSuchFile`: Model file not found
//! - `OrtInvalidProtobuf`: Invalid ONNX model format
//! - `InvalidInputCount`: Wrong number of inputs provided
//! - `ShapeMismatch`: Input tensor shape doesn't match model expectations

const std = @import("std");
const Allocator = std.mem.Allocator;
const ort = @import("onnxruntime.zig");
const tensor_mod = @import("tensor.zig");
const arena_mod = @import("arena.zig");
const tensor_pool_mod = @import("tensor_pool.zig");

const Shape = tensor_mod.Shape;
const Tensor = tensor_mod.Tensor;
const TensorF32 = tensor_mod.TensorF32;
const MAX_DIMS = tensor_mod.MAX_DIMS;
const ScratchAllocator = arena_mod.ScratchAllocator;
const TensorPoolF32 = tensor_pool_mod.TensorPoolF32;

// =============================================================================
// Execution Providers
// =============================================================================

/// Available execution providers for ONNX Runtime
pub const ExecutionProvider = enum {
    /// CPU execution provider (default, always available)
    cpu,
    /// NVIDIA CUDA execution provider (requires CUDA toolkit)
    cuda,
    /// NVIDIA TensorRT execution provider (requires TensorRT)
    tensorrt,
    /// Apple CoreML execution provider (macOS/iOS only)
    coreml,
    /// Microsoft DirectML execution provider (Windows only)
    directml,
    /// AMD ROCm execution provider (Linux only)
    rocm,
    /// Intel OpenVINO execution provider
    openvino,
    /// ARM Neural Network SDK execution provider
    nnapi,
    /// Intel oneDNN execution provider
    dnnl,
    /// XNNPACK execution provider (mobile-optimized)
    xnnpack,
};

// =============================================================================
// Session Options
// =============================================================================

/// Graph optimization level
pub const GraphOptLevel = enum(c_uint) {
    /// Disable all optimizations
    disable = 0,
    /// Basic optimizations (constant folding, redundant node elimination)
    basic = 1,
    /// Extended optimizations (includes basic + complex graph rewrites)
    extended = 2,
    /// All optimizations enabled (default)
    all = 99,
};

/// Execution mode for session
pub const ExecMode = enum(c_uint) {
    /// Sequential execution (default)
    sequential = 0,
    /// Parallel execution of independent operators
    parallel = 1,
};

/// Log severity level
pub const LogLevel = enum(c_uint) {
    verbose = 0,
    info = 1,
    warning = 2,
    err = 3,
    fatal = 4,
};

/// Configuration options for creating a Session
pub const SessionOptions = struct {
    /// Execution providers to use, in order of preference.
    /// Default is CPU only.
    execution_providers: []const ExecutionProvider = &[_]ExecutionProvider{.cpu},

    /// Number of threads for intra-op parallelism (0 = auto)
    intra_op_num_threads: u32 = 0,

    /// Number of threads for inter-op parallelism (0 = auto)
    inter_op_num_threads: u32 = 0,

    /// Graph optimization level
    graph_optimization_level: GraphOptLevel = .all,

    /// Execution mode
    execution_mode: ExecMode = .sequential,

    /// Enable/disable memory pattern optimization
    enable_mem_pattern: bool = true,

    /// Enable/disable memory arena
    enable_mem_arena: bool = true,

    /// Enable/disable CPU memory arena
    enable_cpu_mem_arena: bool = true,

    /// Log severity level
    log_severity_level: LogLevel = .warning,

    /// Log identifier for this session
    log_id: ?[]const u8 = null,

    /// Path to save optimized model (null = don't save)
    optimized_model_filepath: ?[]const u8 = null,

    /// Enable/disable profiling
    enable_profiling: bool = false,

    /// Profile output file path
    profile_file_prefix: ?[]const u8 = null,
};

// =============================================================================
// CUDA Provider Options
// =============================================================================

/// Options for CUDA execution provider
pub const CudaProviderOptions = struct {
    /// CUDA device ID (0 = first GPU)
    device_id: i32 = 0,

    /// GPU memory limit in bytes (0 = no limit)
    gpu_mem_limit: usize = 0,

    /// Memory arena extend strategy (0 = next power of 2, 1 = same as requested)
    arena_extend_strategy: i32 = 0,

    /// CUDA compute stream (null = create new)
    cuda_stream: ?*anyopaque = null,

    /// Enable/disable CUDA graph capture
    enable_cuda_graph: bool = false,

    /// cuDNN convolution algorithm search mode
    cudnn_conv_algo_search: i32 = 0, // EXHAUSTIVE=0, HEURISTIC=1, DEFAULT=2
};

/// Options for TensorRT execution provider
pub const TensorRTProviderOptions = struct {
    /// Device ID
    device_id: i32 = 0,

    /// Enable FP16 mode
    fp16_enable: bool = false,

    /// Enable INT8 mode
    int8_enable: bool = false,

    /// Maximum workspace size
    max_workspace_size: usize = 1 << 30, // 1 GB default

    /// Maximum partition iterations
    max_partition_iterations: i32 = 1000,

    /// Minimum subgraph size
    min_subgraph_size: i32 = 1,
};

/// Options for CoreML execution provider
pub const CoreMLProviderOptions = struct {
    /// Use only CPU for CoreML
    coreml_flags: u32 = 0, // 0=default, 1=CPU_ONLY, 2=ENABLE_ON_SUBGRAPH
};

// =============================================================================
// Model Metadata
// =============================================================================

/// Metadata about an ONNX model
pub const ModelMetadata = struct {
    /// Producer name (e.g., "pytorch", "tensorflow")
    producer_name: ?[]const u8 = null,
    /// Producer version
    producer_version: ?[]const u8 = null,
    /// Domain
    domain: ?[]const u8 = null,
    /// Model description
    description: ?[]const u8 = null,
    /// Graph name
    graph_name: ?[]const u8 = null,
    /// Graph description
    graph_description: ?[]const u8 = null,
    /// Model version
    version: i64 = 0,
    /// Custom metadata key-value pairs
    custom_metadata_keys: []const []const u8 = &[_][]const u8{},
    custom_metadata_values: []const []const u8 = &[_][]const u8{},

    allocator: ?Allocator = null,

    pub fn deinit(self: *ModelMetadata) void {
        if (self.allocator) |alloc| {
            if (self.producer_name) |s| alloc.free(s);
            if (self.producer_version) |s| alloc.free(s);
            if (self.domain) |s| alloc.free(s);
            if (self.description) |s| alloc.free(s);
            if (self.graph_name) |s| alloc.free(s);
            if (self.graph_description) |s| alloc.free(s);
            for (self.custom_metadata_keys) |k| alloc.free(k);
            for (self.custom_metadata_values) |v| alloc.free(v);
            if (self.custom_metadata_keys.len > 0) alloc.free(self.custom_metadata_keys);
            if (self.custom_metadata_values.len > 0) alloc.free(self.custom_metadata_values);
        }
    }
};

// =============================================================================
// Error Types
// =============================================================================

pub const SessionError = error{
    // ONNX Runtime errors
    OrtFail,
    OrtInvalidArgument,
    OrtNoSuchFile,
    OrtNoModel,
    OrtEngineError,
    OrtRuntimeException,
    OrtInvalidProtobuf,
    OrtModelLoaded,
    OrtNotImplemented,
    OrtInvalidGraph,
    OrtExecutionProviderFail,
    OrtNullApi,
    OrtNullResult,
    // Session-specific errors
    SessionNotInitialized,
    InvalidInputCount,
    InvalidOutputCount,
    TensorCreationFailed,
    ShapeMismatch,
    TypeMismatch,
    AllocationFailed,
};

/// Convert OrtError to SessionError
fn fromOrtError(err: ort.OrtError) SessionError {
    return switch (err) {
        ort.OrtError.Fail => SessionError.OrtFail,
        ort.OrtError.InvalidArgument => SessionError.OrtInvalidArgument,
        ort.OrtError.NoSuchFile => SessionError.OrtNoSuchFile,
        ort.OrtError.NoModel => SessionError.OrtNoModel,
        ort.OrtError.EngineError => SessionError.OrtEngineError,
        ort.OrtError.RuntimeException => SessionError.OrtRuntimeException,
        ort.OrtError.InvalidProtobuf => SessionError.OrtInvalidProtobuf,
        ort.OrtError.ModelLoaded => SessionError.OrtModelLoaded,
        ort.OrtError.NotImplemented => SessionError.OrtNotImplemented,
        ort.OrtError.InvalidGraph => SessionError.OrtInvalidGraph,
        ort.OrtError.ExecutionProviderFail => SessionError.OrtExecutionProviderFail,
        ort.OrtError.NullApi => SessionError.OrtNullApi,
        ort.OrtError.NullResult => SessionError.OrtNullResult,
    };
}

// =============================================================================
// Tensor Info
// =============================================================================

/// Information about a model input or output tensor
pub const TensorInfo = struct {
    name: []const u8,
    shape: Shape,
    dtype: c_uint,

    pub fn deinit(self: *TensorInfo, allocator: Allocator) void {
        allocator.free(self.name);
    }
};

// =============================================================================
// Named Tensor
// =============================================================================

/// A tensor with an associated name for input binding
pub const NamedTensor = struct {
    name: []const u8,
    data: []const u8, // Raw bytes
    shape: []const i64,
    dtype: c_uint,
};

// =============================================================================
// Session
// =============================================================================

/// High-level ONNX inference session
pub const Session = struct {
    const Self = @This();

    allocator: Allocator,
    api: *const ort.OrtApi,
    env: *ort.OrtEnv,
    session: *ort.OrtSession,
    session_options: *ort.OrtSessionOptions,

    // Cached metadata
    input_count: usize,
    output_count: usize,
    input_names: [][:0]const u8,
    output_names: [][:0]const u8,

    /// Create a new inference session from an ONNX model file with default options
    pub fn init(allocator: Allocator, model_path: []const u8) !Self {
        return initWithOptions(allocator, model_path, .{});
    }

    /// Create a new inference session from an ONNX model file with custom options
    pub fn initWithOptions(allocator: Allocator, model_path: []const u8, options: SessionOptions) !Self {
        // Get the API
        const api = ort.getApi() catch |err| return fromOrtError(err);

        // Create environment with configured log level
        var env: ?*ort.OrtEnv = null;
        const log_level: c_uint = @intFromEnum(options.log_severity_level);
        const log_id = if (options.log_id) |id| blk: {
            const id_z = try allocator.allocSentinel(u8, id.len, 0);
            @memcpy(id_z, id);
            break :blk id_z.ptr;
        } else "onnx_zig";
        defer if (options.log_id != null) allocator.free(@as([:0]u8, @ptrCast(log_id[0..std.mem.len(log_id)])));

        try checkOrtStatus(api, api.CreateEnv.?(log_level, log_id, &env));

        errdefer if (env) |e| api.ReleaseEnv.?(e);

        // Create session options
        var session_options: ?*ort.OrtSessionOptions = null;
        try checkOrtStatus(api, api.CreateSessionOptions.?(&session_options));

        errdefer if (session_options) |so| api.ReleaseSessionOptions.?(so);

        // Configure session options
        try configureSessionOptions(api, session_options.?, options, allocator);

        // Create null-terminated path
        const path_z = try allocator.allocSentinel(u8, model_path.len, 0);
        defer allocator.free(path_z);
        @memcpy(path_z, model_path);

        // Create session from model file
        var session: ?*ort.OrtSession = null;
        try checkOrtStatus(api, api.CreateSession.?(env.?, path_z.ptr, session_options.?, &session));

        errdefer if (session) |s| api.ReleaseSession.?(s);

        // Get input/output counts
        var input_count: usize = 0;
        var output_count: usize = 0;
        try checkOrtStatus(api, api.SessionGetInputCount.?(session.?, &input_count));
        try checkOrtStatus(api, api.SessionGetOutputCount.?(session.?, &output_count));

        // Get default allocator for name retrieval
        var ort_allocator: ?*ort.OrtAllocator = null;
        try checkOrtStatus(api, api.GetAllocatorWithDefaultOptions.?(&ort_allocator));
        if (ort_allocator == null) return SessionError.AllocationFailed;

        // Get input names
        const input_names = try allocator.alloc([:0]const u8, input_count);
        errdefer allocator.free(input_names);

        for (0..input_count) |i| {
            var name_ptr: [*c]u8 = undefined;
            try checkOrtStatus(api, api.SessionGetInputName.?(session.?, i, ort_allocator.?, &name_ptr));

            const name_len = std.mem.len(name_ptr);
            const name_copy = try allocator.allocSentinel(u8, name_len, 0);
            @memcpy(name_copy, name_ptr[0..name_len]);
            input_names[i] = name_copy;

            // Free the ORT-allocated name
            try checkOrtStatus(api, api.AllocatorFree.?(ort_allocator.?, name_ptr));
        }

        // Get output names
        const output_names = try allocator.alloc([:0]const u8, output_count);
        errdefer allocator.free(output_names);

        for (0..output_count) |i| {
            var name_ptr: [*c]u8 = undefined;
            try checkOrtStatus(api, api.SessionGetOutputName.?(session.?, i, ort_allocator.?, &name_ptr));

            const name_len = std.mem.len(name_ptr);
            const name_copy = try allocator.allocSentinel(u8, name_len, 0);
            @memcpy(name_copy, name_ptr[0..name_len]);
            output_names[i] = name_copy;

            // Free the ORT-allocated name
            try checkOrtStatus(api, api.AllocatorFree.?(ort_allocator.?, name_ptr));
        }

        return Self{
            .allocator = allocator,
            .api = api,
            .env = env.?,
            .session = session.?,
            .session_options = session_options.?,
            .input_count = input_count,
            .output_count = output_count,
            .input_names = input_names,
            .output_names = output_names,
        };
    }

    /// Release all resources
    pub fn deinit(self: *Self) void {
        // Free cached names
        for (self.input_names) |name| {
            self.allocator.free(name);
        }
        self.allocator.free(self.input_names);

        for (self.output_names) |name| {
            self.allocator.free(name);
        }
        self.allocator.free(self.output_names);

        // Release ORT resources
        self.api.ReleaseSession.?(self.session);
        self.api.ReleaseSessionOptions.?(self.session_options);
        self.api.ReleaseEnv.?(self.env);
    }

    /// Get input tensor names
    pub fn getInputNames(self: *const Self) []const [:0]const u8 {
        return self.input_names;
    }

    /// Get output tensor names
    pub fn getOutputNames(self: *const Self) []const [:0]const u8 {
        return self.output_names;
    }

    /// Get number of model inputs
    pub fn getInputCount(self: *const Self) usize {
        return self.input_count;
    }

    /// Get number of model outputs
    pub fn getOutputCount(self: *const Self) usize {
        return self.output_count;
    }

    /// Get information about an input tensor
    pub fn getInputInfo(self: *const Self, index: usize) !TensorInfo {
        if (index >= self.input_count) return SessionError.InvalidInputCount;

        var type_info: ?*ort.OrtTypeInfo = null;
        try checkOrtStatus(self.api, self.api.SessionGetInputTypeInfo.?(self.session, index, &type_info));
        defer self.api.ReleaseTypeInfo.?(type_info.?);

        return self.extractTensorInfo(type_info.?, self.input_names[index]);
    }

    /// Get information about an output tensor
    pub fn getOutputInfo(self: *const Self, index: usize) !TensorInfo {
        if (index >= self.output_count) return SessionError.InvalidOutputCount;

        var type_info: ?*ort.OrtTypeInfo = null;
        try checkOrtStatus(self.api, self.api.SessionGetOutputTypeInfo.?(self.session, index, &type_info));
        defer self.api.ReleaseTypeInfo.?(type_info.?);

        return self.extractTensorInfo(type_info.?, self.output_names[index]);
    }

    /// Get model metadata (producer, version, description, custom properties)
    pub fn getModelMetadata(self: *const Self) !ModelMetadata {
        var metadata: ?*ort.c.OrtModelMetadata = null;
        try checkOrtStatus(self.api, self.api.SessionGetModelMetadata.?(self.session, &metadata));
        defer self.api.ReleaseModelMetadata.?(metadata.?);

        var result = ModelMetadata{
            .allocator = self.allocator,
        };

        // Get default allocator for string retrieval
        var ort_allocator: ?*ort.OrtAllocator = null;
        try checkOrtStatus(self.api, self.api.GetAllocatorWithDefaultOptions.?(&ort_allocator));

        // Producer name
        var producer_name_ptr: [*c]u8 = null;
        const pn_status = self.api.ModelMetadataGetProducerName.?(metadata.?, ort_allocator.?, &producer_name_ptr);
        if (pn_status == null and producer_name_ptr != null) {
            const len = std.mem.len(producer_name_ptr);
            if (len > 0) {
                const name_copy = try self.allocator.alloc(u8, len);
                @memcpy(name_copy, producer_name_ptr[0..len]);
                result.producer_name = name_copy;
            }
            _ = self.api.AllocatorFree.?(ort_allocator.?, producer_name_ptr);
        } else if (pn_status != null) {
            self.api.ReleaseStatus.?(pn_status);
        }

        // Graph name
        var graph_name_ptr: [*c]u8 = null;
        const gn_status = self.api.ModelMetadataGetGraphName.?(metadata.?, ort_allocator.?, &graph_name_ptr);
        if (gn_status == null and graph_name_ptr != null) {
            const len = std.mem.len(graph_name_ptr);
            if (len > 0) {
                const name_copy = try self.allocator.alloc(u8, len);
                @memcpy(name_copy, graph_name_ptr[0..len]);
                result.graph_name = name_copy;
            }
            _ = self.api.AllocatorFree.?(ort_allocator.?, graph_name_ptr);
        } else if (gn_status != null) {
            self.api.ReleaseStatus.?(gn_status);
        }

        // Domain
        var domain_ptr: [*c]u8 = null;
        const dm_status = self.api.ModelMetadataGetDomain.?(metadata.?, ort_allocator.?, &domain_ptr);
        if (dm_status == null and domain_ptr != null) {
            const len = std.mem.len(domain_ptr);
            if (len > 0) {
                const name_copy = try self.allocator.alloc(u8, len);
                @memcpy(name_copy, domain_ptr[0..len]);
                result.domain = name_copy;
            }
            _ = self.api.AllocatorFree.?(ort_allocator.?, domain_ptr);
        } else if (dm_status != null) {
            self.api.ReleaseStatus.?(dm_status);
        }

        // Description
        var desc_ptr: [*c]u8 = null;
        const desc_status = self.api.ModelMetadataGetDescription.?(metadata.?, ort_allocator.?, &desc_ptr);
        if (desc_status == null and desc_ptr != null) {
            const len = std.mem.len(desc_ptr);
            if (len > 0) {
                const desc_copy = try self.allocator.alloc(u8, len);
                @memcpy(desc_copy, desc_ptr[0..len]);
                result.description = desc_copy;
            }
            _ = self.api.AllocatorFree.?(ort_allocator.?, desc_ptr);
        } else if (desc_status != null) {
            self.api.ReleaseStatus.?(desc_status);
        }

        // Version
        var version: i64 = 0;
        const ver_status = self.api.ModelMetadataGetVersion.?(metadata.?, &version);
        if (ver_status == null) {
            result.version = version;
        } else {
            self.api.ReleaseStatus.?(ver_status);
        }

        // Custom metadata keys
        var keys_ptr: [*c][*c]u8 = null;
        var num_keys: i64 = 0;
        const keys_status = self.api.ModelMetadataGetCustomMetadataMapKeys.?(metadata.?, ort_allocator.?, &keys_ptr, &num_keys);
        if (keys_status == null and num_keys > 0 and keys_ptr != null) {
            const key_count: usize = @intCast(num_keys);
            const keys = try self.allocator.alloc([]const u8, key_count);
            const values = try self.allocator.alloc([]const u8, key_count);

            for (0..key_count) |i| {
                const key_ptr = keys_ptr[i];
                const key_len = std.mem.len(key_ptr);
                const key_copy = try self.allocator.alloc(u8, key_len);
                @memcpy(key_copy, key_ptr[0..key_len]);
                keys[i] = key_copy;

                // Get value for this key
                var value_ptr: [*c]u8 = null;
                const val_status = self.api.ModelMetadataLookupCustomMetadataMap.?(metadata.?, ort_allocator.?, key_ptr, &value_ptr);
                if (val_status == null and value_ptr != null) {
                    const val_len = std.mem.len(value_ptr);
                    const val_copy = try self.allocator.alloc(u8, val_len);
                    @memcpy(val_copy, value_ptr[0..val_len]);
                    values[i] = val_copy;
                    _ = self.api.AllocatorFree.?(ort_allocator.?, value_ptr);
                } else {
                    values[i] = "";
                    if (val_status != null) self.api.ReleaseStatus.?(val_status);
                }

                _ = self.api.AllocatorFree.?(ort_allocator.?, key_ptr);
            }

            result.custom_metadata_keys = keys;
            result.custom_metadata_values = values;
            _ = self.api.AllocatorFree.?(ort_allocator.?, keys_ptr);
        } else if (keys_status != null) {
            self.api.ReleaseStatus.?(keys_status);
        }

        return result;
    }

    /// Get the opset version used by the model
    pub fn getOpsetVersion(self: *const Self) !i64 {
        var metadata: ?*ort.c.OrtModelMetadata = null;
        try checkOrtStatus(self.api, self.api.SessionGetModelMetadata.?(self.session, &metadata));
        defer self.api.ReleaseModelMetadata.?(metadata.?);

        var version: i64 = 0;
        try checkOrtStatus(self.api, self.api.ModelMetadataGetVersion.?(metadata.?, &version));
        return version;
    }

    /// Extract tensor info from ORT type info
    fn extractTensorInfo(self: *const Self, type_info: *ort.OrtTypeInfo, name: [:0]const u8) !TensorInfo {
        var tensor_info: ?*const ort.OrtTensorTypeAndShapeInfo = null;
        try checkOrtStatus(self.api, self.api.CastTypeInfoToTensorInfo.?(type_info, &tensor_info));

        // Get element type
        var dtype: c_uint = ort.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        try checkOrtStatus(self.api, self.api.GetTensorElementType.?(tensor_info.?, &dtype));

        // Get dimensions
        var dim_count: usize = 0;
        try checkOrtStatus(self.api, self.api.GetDimensionsCount.?(tensor_info.?, &dim_count));

        var shape = Shape{};
        if (dim_count > 0 and dim_count <= MAX_DIMS) {
            var dims: [MAX_DIMS]i64 = [_]i64{0} ** MAX_DIMS;
            try checkOrtStatus(self.api, self.api.GetDimensions.?(tensor_info.?, &dims, dim_count));

            shape.ndim = dim_count;
            for (0..dim_count) |i| {
                // Dynamic dimensions are -1, treat as 0 for now
                shape.dims[i] = if (dims[i] > 0) @intCast(dims[i]) else 0;
            }
        }

        // Copy name
        const name_copy = try self.allocator.alloc(u8, name.len);
        @memcpy(name_copy, name);

        return TensorInfo{
            .name = name_copy,
            .shape = shape,
            .dtype = dtype,
        };
    }

    /// Run inference with f32 tensors
    /// input_data: slice of input tensor data (must match model input count and shapes)
    /// input_shapes: shapes for each input tensor
    /// Returns allocated output tensors (caller must free)
    pub fn runF32(
        self: *Self,
        input_data: []const []const f32,
        input_shapes: []const []const i64,
    ) ![]TensorF32 {
        if (input_data.len != self.input_count) return SessionError.InvalidInputCount;
        if (input_shapes.len != self.input_count) return SessionError.InvalidInputCount;

        // Create memory info for CPU
        var memory_info: ?*ort.OrtMemoryInfo = null;
        try checkOrtStatus(self.api, self.api.CreateCpuMemoryInfo.?(ort.c.OrtArenaAllocator, ort.c.OrtMemTypeDefault, &memory_info));
        defer self.api.ReleaseMemoryInfo.?(memory_info.?);

        // Create input OrtValues
        var input_values = try self.allocator.alloc(?*ort.OrtValue, self.input_count);
        defer self.allocator.free(input_values);

        for (0..self.input_count) |i| {
            const data_ptr: ?*anyopaque = @constCast(@ptrCast(input_data[i].ptr));
            const data_len = input_data[i].len * @sizeOf(f32);

            var value: ?*ort.OrtValue = null;
            try checkOrtStatus(self.api, self.api.CreateTensorWithDataAsOrtValue.?(
                memory_info.?,
                data_ptr,
                data_len,
                input_shapes[i].ptr,
                input_shapes[i].len,
                ort.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                &value,
            ));
            input_values[i] = value;
        }

        // Ensure input values are released on error or completion
        defer {
            for (input_values) |val| {
                if (val) |v| self.api.ReleaseValue.?(v);
            }
        }

        // Allocate output value pointers
        const output_values = try self.allocator.alloc(?*ort.OrtValue, self.output_count);
        defer self.allocator.free(output_values);
        @memset(output_values, null);

        // Convert names to C pointers
        var input_name_ptrs = try self.allocator.alloc([*c]const u8, self.input_count);
        defer self.allocator.free(input_name_ptrs);
        for (0..self.input_count) |i| {
            input_name_ptrs[i] = self.input_names[i].ptr;
        }

        var output_name_ptrs = try self.allocator.alloc([*c]const u8, self.output_count);
        defer self.allocator.free(output_name_ptrs);
        for (0..self.output_count) |i| {
            output_name_ptrs[i] = self.output_names[i].ptr;
        }

        // Run inference
        try checkOrtStatus(self.api, self.api.Run.?(
            self.session,
            null, // run options
            input_name_ptrs.ptr,
            @ptrCast(input_values.ptr),
            self.input_count,
            output_name_ptrs.ptr,
            self.output_count,
            output_values.ptr,
        ));

        // Convert outputs to Zig tensors
        var result_tensors = try self.allocator.alloc(TensorF32, self.output_count);
        errdefer self.allocator.free(result_tensors);

        for (0..self.output_count) |i| {
            if (output_values[i]) |ort_value| {
                defer self.api.ReleaseValue.?(ort_value);

                // Get tensor info
                var tensor_info: ?*ort.OrtTensorTypeAndShapeInfo = null;
                try checkOrtStatus(self.api, self.api.GetTensorTypeAndShape.?(ort_value, &tensor_info));
                defer self.api.ReleaseTensorTypeAndShapeInfo.?(tensor_info.?);

                // Get dimensions
                var dim_count: usize = 0;
                try checkOrtStatus(self.api, self.api.GetDimensionsCount.?(tensor_info.?, &dim_count));

                var dims: [MAX_DIMS]i64 = [_]i64{0} ** MAX_DIMS;
                if (dim_count > 0) {
                    try checkOrtStatus(self.api, self.api.GetDimensions.?(tensor_info.?, &dims, dim_count));
                }

                // Convert to usize shape
                var shape_usize: [MAX_DIMS]usize = [_]usize{0} ** MAX_DIMS;
                for (0..dim_count) |j| {
                    shape_usize[j] = @intCast(dims[j]);
                }

                // Get data pointer
                var data_ptr: ?*anyopaque = null;
                try checkOrtStatus(self.api, self.api.GetTensorMutableData.?(ort_value, &data_ptr));

                // Get element count
                var elem_count: usize = 0;
                try checkOrtStatus(self.api, self.api.GetTensorShapeElementCount.?(tensor_info.?, &elem_count));

                // Create Zig tensor and copy data
                const out_tensor = try TensorF32.init(self.allocator, shape_usize[0..dim_count]);
                const src_slice: [*]const f32 = @ptrCast(@alignCast(data_ptr.?));
                @memcpy(out_tensor.data, src_slice[0..elem_count]);

                result_tensors[i] = out_tensor;
            } else {
                return SessionError.OrtNullResult;
            }
        }

        return result_tensors;
    }

    /// Run inference with int64 inputs (e.g., for BERT token IDs)
    /// Returns f32 output tensors (common for embedding models)
    pub fn runI64(
        self: *Self,
        input_data: []const []const i64,
        input_shapes: []const []const i64,
    ) ![]TensorF32 {
        if (input_data.len != self.input_count) return SessionError.InvalidInputCount;
        if (input_shapes.len != self.input_count) return SessionError.InvalidInputCount;

        // Create memory info for CPU
        var memory_info: ?*ort.OrtMemoryInfo = null;
        try checkOrtStatus(self.api, self.api.CreateCpuMemoryInfo.?(ort.c.OrtArenaAllocator, ort.c.OrtMemTypeDefault, &memory_info));
        defer self.api.ReleaseMemoryInfo.?(memory_info.?);

        // Create input OrtValues
        var input_values = try self.allocator.alloc(?*ort.OrtValue, self.input_count);
        defer self.allocator.free(input_values);

        for (0..self.input_count) |i| {
            const data_ptr: ?*anyopaque = @constCast(@ptrCast(input_data[i].ptr));
            const data_len = input_data[i].len * @sizeOf(i64);

            var value: ?*ort.OrtValue = null;
            try checkOrtStatus(self.api, self.api.CreateTensorWithDataAsOrtValue.?(
                memory_info.?,
                data_ptr,
                data_len,
                input_shapes[i].ptr,
                input_shapes[i].len,
                ort.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                &value,
            ));
            input_values[i] = value;
        }

        // Ensure input values are released on error or completion
        defer {
            for (input_values) |val| {
                if (val) |v| self.api.ReleaseValue.?(v);
            }
        }

        // Allocate output value pointers
        const output_values = try self.allocator.alloc(?*ort.OrtValue, self.output_count);
        defer self.allocator.free(output_values);
        @memset(output_values, null);

        // Convert names to C pointers
        var input_name_ptrs = try self.allocator.alloc([*c]const u8, self.input_count);
        defer self.allocator.free(input_name_ptrs);
        for (0..self.input_count) |i| {
            input_name_ptrs[i] = self.input_names[i].ptr;
        }

        var output_name_ptrs = try self.allocator.alloc([*c]const u8, self.output_count);
        defer self.allocator.free(output_name_ptrs);
        for (0..self.output_count) |i| {
            output_name_ptrs[i] = self.output_names[i].ptr;
        }

        // Run inference
        try checkOrtStatus(self.api, self.api.Run.?(
            self.session,
            null, // run options
            input_name_ptrs.ptr,
            @ptrCast(input_values.ptr),
            self.input_count,
            output_name_ptrs.ptr,
            self.output_count,
            output_values.ptr,
        ));

        // Convert outputs to Zig tensors (outputs are typically f32 for embedding models)
        var result_tensors = try self.allocator.alloc(TensorF32, self.output_count);
        errdefer self.allocator.free(result_tensors);

        for (0..self.output_count) |i| {
            if (output_values[i]) |ort_value| {
                defer self.api.ReleaseValue.?(ort_value);

                // Get tensor info
                var tensor_info: ?*ort.OrtTensorTypeAndShapeInfo = null;
                try checkOrtStatus(self.api, self.api.GetTensorTypeAndShape.?(ort_value, &tensor_info));
                defer self.api.ReleaseTensorTypeAndShapeInfo.?(tensor_info.?);

                // Get dimensions
                var dim_count: usize = 0;
                try checkOrtStatus(self.api, self.api.GetDimensionsCount.?(tensor_info.?, &dim_count));

                var dims: [MAX_DIMS]i64 = [_]i64{0} ** MAX_DIMS;
                if (dim_count > 0) {
                    try checkOrtStatus(self.api, self.api.GetDimensions.?(tensor_info.?, &dims, dim_count));
                }

                // Convert to usize shape
                var shape_usize: [MAX_DIMS]usize = [_]usize{0} ** MAX_DIMS;
                for (0..dim_count) |j| {
                    shape_usize[j] = @intCast(dims[j]);
                }

                // Get data pointer
                var data_ptr: ?*anyopaque = null;
                try checkOrtStatus(self.api, self.api.GetTensorMutableData.?(ort_value, &data_ptr));

                // Get element count
                var elem_count: usize = 0;
                try checkOrtStatus(self.api, self.api.GetTensorShapeElementCount.?(tensor_info.?, &elem_count));

                // Create Zig tensor and copy data
                const out_tensor = try TensorF32.init(self.allocator, shape_usize[0..dim_count]);
                const src_slice: [*]const f32 = @ptrCast(@alignCast(data_ptr.?));
                @memcpy(out_tensor.data, src_slice[0..elem_count]);

                result_tensors[i] = out_tensor;
            } else {
                return SessionError.OrtNullResult;
            }
        }

        return result_tensors;
    }

    /// Free output tensors returned by run methods
    pub fn freeOutputs(self: *Self, outputs: []TensorF32) void {
        for (outputs) |*t| {
            t.deinit();
        }
        self.allocator.free(outputs);
    }
};

// =============================================================================
// Optimized Session
// =============================================================================

/// Memory-optimized inference session with built-in arena allocator and tensor pool.
/// Use this for repeated inference where memory reuse is important.
///
/// ## Example
///
/// ```zig
/// var opt_session = try OptimizedSession.init(allocator, "model.onnx");
/// defer opt_session.deinit();
///
/// // Run multiple inferences efficiently
/// for (0..100) |_| {
///     const outputs = try opt_session.runF32(&inputs, &shapes);
///     // ... process outputs ...
///     opt_session.releaseOutputs(outputs);
///     opt_session.resetScratch(); // Reuse scratch memory
/// }
///
/// // Check statistics
/// const stats = opt_session.getPoolStats();
/// std.debug.print("Hit rate: {d:.1}%\n", .{stats.hit_rate * 100});
/// ```
pub const OptimizedSession = struct {
    const Self = @This();

    session: Session,
    scratch: ScratchAllocator,
    tensor_pool: TensorPoolF32,

    /// Create an optimized session from an ONNX model file
    pub fn init(backing_allocator: Allocator, model_path: []const u8) !Self {
        return Self{
            .session = try Session.init(backing_allocator, model_path),
            .scratch = ScratchAllocator.init(backing_allocator),
            .tensor_pool = TensorPoolF32.init(backing_allocator),
        };
    }

    /// Release all resources
    pub fn deinit(self: *Self) void {
        self.tensor_pool.deinit();
        self.scratch.deinit();
        self.session.deinit();
    }

    /// Run inference and return pooled output tensors
    pub fn runF32(
        self: *Self,
        input_data: []const []const f32,
        input_shapes: []const []const i64,
    ) ![]TensorF32 {
        // Run inference using session's standard allocator
        const outputs = try self.session.runF32(input_data, input_shapes);

        // Convert to pooled tensors
        const pooled = try self.session.allocator.alloc(TensorF32, outputs.len);
        for (outputs, 0..) |output, i| {
            var pooled_tensor = try self.tensor_pool.acquire(output.shape.slice());
            @memcpy(pooled_tensor.data[0..output.numel()], output.data[0..output.numel()]);
            pooled_tensor.shape = output.shape;
            pooled[i] = pooled_tensor;
        }

        // Free original outputs
        self.session.freeOutputs(outputs);

        return pooled;
    }

    /// Release output tensors back to pool
    pub fn releaseOutputs(self: *Self, outputs: []TensorF32) void {
        for (outputs) |t| {
            self.tensor_pool.release(t);
        }
        self.session.allocator.free(outputs);
    }

    /// Reset scratch allocator for next inference
    pub fn resetScratch(self: *Self) void {
        self.scratch.reset();
    }

    /// Trim tensor pool to reduce memory usage
    pub fn trimPool(self: *Self, max_per_bucket: usize) void {
        self.tensor_pool.trim(max_per_bucket);
    }

    /// Get tensor pool statistics
    pub fn getPoolStats(self: *const Self) tensor_pool_mod.PoolStats {
        return self.tensor_pool.getStats();
    }

    /// Get scratch allocator statistics
    pub fn getScratchStats(self: *const Self) struct { bytes: usize, peak: usize } {
        return .{
            .bytes = self.scratch.getBytesAllocated(),
            .peak = self.scratch.getPeakBytes(),
        };
    }

    // Delegate common methods to underlying session
    pub fn getInputNames(self: *const Self) []const [:0]const u8 {
        return self.session.getInputNames();
    }

    pub fn getOutputNames(self: *const Self) []const [:0]const u8 {
        return self.session.getOutputNames();
    }

    pub fn getInputCount(self: *const Self) usize {
        return self.session.getInputCount();
    }

    pub fn getOutputCount(self: *const Self) usize {
        return self.session.getOutputCount();
    }

    pub fn getInputInfo(self: *const Self, index: usize) !TensorInfo {
        return self.session.getInputInfo(index);
    }

    pub fn getOutputInfo(self: *const Self, index: usize) !TensorInfo {
        return self.session.getOutputInfo(index);
    }
};

// =============================================================================
// Helper Functions
// =============================================================================

fn checkOrtStatus(api: *const ort.OrtApi, status: ?*ort.OrtStatus) SessionError!void {
    ort.checkStatus(api, status) catch |err| return fromOrtError(err);
}

/// Configure session options based on SessionOptions struct
fn configureSessionOptions(api: *const ort.OrtApi, session_options: *ort.OrtSessionOptions, options: SessionOptions, allocator: Allocator) SessionError!void {
    // Set graph optimization level
    const opt_level: c_uint = @intFromEnum(options.graph_optimization_level);
    try checkOrtStatus(api, api.SetSessionGraphOptimizationLevel.?(session_options, opt_level));

    // Set execution mode
    const exec_mode: c_uint = @intFromEnum(options.execution_mode);
    try checkOrtStatus(api, api.SetSessionExecutionMode.?(session_options, exec_mode));

    // Set thread counts
    if (options.intra_op_num_threads > 0) {
        try checkOrtStatus(api, api.SetIntraOpNumThreads.?(session_options, @intCast(options.intra_op_num_threads)));
    }
    if (options.inter_op_num_threads > 0) {
        try checkOrtStatus(api, api.SetInterOpNumThreads.?(session_options, @intCast(options.inter_op_num_threads)));
    }

    // Memory pattern and arena options
    if (!options.enable_mem_pattern) {
        try checkOrtStatus(api, api.DisableMemPattern.?(session_options));
    }
    if (!options.enable_cpu_mem_arena) {
        try checkOrtStatus(api, api.DisableCpuMemArena.?(session_options));
    }

    // Profiling
    if (options.enable_profiling) {
        if (options.profile_file_prefix) |prefix| {
            const prefix_z = try allocator.allocSentinel(u8, prefix.len, 0);
            defer allocator.free(prefix_z);
            @memcpy(prefix_z, prefix);
            try checkOrtStatus(api, api.EnableProfiling.?(session_options, prefix_z.ptr));
        } else {
            try checkOrtStatus(api, api.EnableProfiling.?(session_options, "onnx_zig_profile"));
        }
    }

    // Save optimized model
    if (options.optimized_model_filepath) |path| {
        const path_z = try allocator.allocSentinel(u8, path.len, 0);
        defer allocator.free(path_z);
        @memcpy(path_z, path);
        try checkOrtStatus(api, api.SetOptimizedModelFilePath.?(session_options, path_z.ptr));
    }

    // Append execution providers (in reverse order since ORT prepends)
    var i: usize = options.execution_providers.len;
    while (i > 0) {
        i -= 1;
        const provider = options.execution_providers[i];
        try appendExecutionProvider(api, session_options, provider);
    }
}

/// Append an execution provider to session options
fn appendExecutionProvider(api: *const ort.OrtApi, session_options: *ort.OrtSessionOptions, provider: ExecutionProvider) SessionError!void {
    switch (provider) {
        .cpu => {
            // CPU is always available and doesn't need explicit registration
        },
        .cuda => {
            // Try to append CUDA provider with default options
            var cuda_options: ort.c.OrtCUDAProviderOptions = std.mem.zeroes(ort.c.OrtCUDAProviderOptions);
            cuda_options.device_id = 0;
            const status = api.SessionOptionsAppendExecutionProvider_CUDA.?(session_options, &cuda_options);
            // Don't fail if CUDA isn't available, just skip
            if (status != null) {
                api.ReleaseStatus.?(status);
                // CUDA not available, continue with next provider
            }
        },
        .tensorrt => {
            // TensorRT requires CUDA
            var trt_options: ort.c.OrtTensorRTProviderOptions = std.mem.zeroes(ort.c.OrtTensorRTProviderOptions);
            trt_options.device_id = 0;
            const status = api.SessionOptionsAppendExecutionProvider_TensorRT.?(session_options, &trt_options);
            if (status != null) {
                api.ReleaseStatus.?(status);
            }
        },
        .coreml => {
            // CoreML for Apple platforms
            const status = api.SessionOptionsAppendExecutionProvider_CoreML.?(session_options, 0);
            if (status != null) {
                api.ReleaseStatus.?(status);
            }
        },
        .directml => {
            // DirectML for Windows
            const status = api.SessionOptionsAppendExecutionProvider_DML.?(session_options, 0);
            if (status != null) {
                api.ReleaseStatus.?(status);
            }
        },
        .rocm => {
            // ROCm for AMD GPUs
            var rocm_options: ort.c.OrtROCMProviderOptions = std.mem.zeroes(ort.c.OrtROCMProviderOptions);
            rocm_options.device_id = 0;
            const status = api.SessionOptionsAppendExecutionProvider_ROCM.?(session_options, &rocm_options);
            if (status != null) {
                api.ReleaseStatus.?(status);
            }
        },
        .openvino => {
            // OpenVINO provider
            const status = api.SessionOptionsAppendExecutionProvider_OpenVINO.?(session_options, null);
            if (status != null) {
                api.ReleaseStatus.?(status);
            }
        },
        .nnapi => {
            // Android NNAPI
            const status = api.SessionOptionsAppendExecutionProvider_Nnapi.?(session_options, 0);
            if (status != null) {
                api.ReleaseStatus.?(status);
            }
        },
        .dnnl => {
            // Intel oneDNN
            const status = api.SessionOptionsAppendExecutionProvider_Dnnl.?(session_options, 1);
            if (status != null) {
                api.ReleaseStatus.?(status);
            }
        },
        .xnnpack => {
            // XNNPACK
            const status = api.SessionOptionsAppendExecutionProvider.?(session_options, "XNNPACK", null, null, 0);
            if (status != null) {
                api.ReleaseStatus.?(status);
            }
        },
    }
}

// =============================================================================
// Tests
// =============================================================================

test "Session error conversion" {
    const err = fromOrtError(ort.OrtError.NoSuchFile);
    try std.testing.expectEqual(SessionError.OrtNoSuchFile, err);
}

test "TensorInfo shape" {
    var info = TensorInfo{
        .name = "test",
        .shape = Shape.init(&[_]usize{ 1, 3, 224, 224 }),
        .dtype = @intCast(ort.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
    };
    _ = &info;

    try std.testing.expectEqual(@as(usize, 4), info.shape.ndim);
    try std.testing.expectEqual(@as(usize, 1), info.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 224), info.shape.dims[3]);
}

// =============================================================================
// Integration Tests with ONNX Models
// =============================================================================

const test_models_path = "models/test/";

fn getTestModelPath(comptime filename: []const u8) []const u8 {
    return test_models_path ++ filename;
}

test "Session - load identity model" {
    const allocator = std.testing.allocator;

    var session = Session.init(allocator, getTestModelPath("identity.onnx")) catch |err| {
        std.debug.print("Skipping test: Could not load model (error: {s})\n", .{@errorName(err)});
        return;
    };
    defer session.deinit();

    // Verify model metadata
    try std.testing.expectEqual(@as(usize, 1), session.getInputCount());
    try std.testing.expectEqual(@as(usize, 1), session.getOutputCount());

    const input_names = session.getInputNames();
    try std.testing.expectEqualStrings("X", input_names[0]);

    const output_names = session.getOutputNames();
    try std.testing.expectEqualStrings("Y", output_names[0]);
}

test "Session - identity model inference" {
    const allocator = std.testing.allocator;

    var session = Session.init(allocator, getTestModelPath("identity.onnx")) catch |err| {
        std.debug.print("Skipping test: Could not load model (error: {s})\n", .{@errorName(err)});
        return;
    };
    defer session.deinit();

    // Input: [1, 4] tensor with values [1.0, 2.0, 3.0, 4.0]
    const input_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const input_shape = [_]i64{ 1, 4 };

    const outputs = try session.runF32(
        &[_][]const f32{&input_data},
        &[_][]const i64{&input_shape},
    );
    defer session.freeOutputs(outputs);

    // Identity should return same values
    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    try std.testing.expectEqual(@as(usize, 4), outputs[0].numel());

    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(input_data[i], outputs[0].data[i], 1e-6);
    }
}

test "Session - add model inference" {
    const allocator = std.testing.allocator;

    var session = Session.init(allocator, getTestModelPath("add.onnx")) catch |err| {
        std.debug.print("Skipping test: Could not load model (error: {s})\n", .{@errorName(err)});
        return;
    };
    defer session.deinit();

    // Verify two inputs
    try std.testing.expectEqual(@as(usize, 2), session.getInputCount());
    try std.testing.expectEqual(@as(usize, 1), session.getOutputCount());

    // A: [2, 3] tensor
    const a_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    // B: [2, 3] tensor
    const b_data = [_]f32{ 0.5, 1.5, 2.5, 3.5, 4.5, 5.5 };
    const shape = [_]i64{ 2, 3 };

    const outputs = try session.runF32(
        &[_][]const f32{ &a_data, &b_data },
        &[_][]const i64{ &shape, &shape },
    );
    defer session.freeOutputs(outputs);

    // C = A + B
    const expected = [_]f32{ 1.5, 3.5, 5.5, 7.5, 9.5, 11.5 };

    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    try std.testing.expectEqual(@as(usize, 6), outputs[0].numel());

    for (0..6) |i| {
        try std.testing.expectApproxEqAbs(expected[i], outputs[0].data[i], 1e-6);
    }
}

test "Session - relu model inference" {
    const allocator = std.testing.allocator;

    var session = Session.init(allocator, getTestModelPath("relu.onnx")) catch |err| {
        std.debug.print("Skipping test: Could not load model (error: {s})\n", .{@errorName(err)});
        return;
    };
    defer session.deinit();

    // Input with positive and negative values
    const input_data = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -0.5, 0.5 };
    const input_shape = [_]i64{ 1, 8 };

    const outputs = try session.runF32(
        &[_][]const f32{&input_data},
        &[_][]const i64{&input_shape},
    );
    defer session.freeOutputs(outputs);

    // ReLU: max(0, x)
    const expected = [_]f32{ 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.5 };

    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    try std.testing.expectEqual(@as(usize, 8), outputs[0].numel());

    for (0..8) |i| {
        try std.testing.expectApproxEqAbs(expected[i], outputs[0].data[i], 1e-6);
    }
}

test "Session - matmul model inference" {
    const allocator = std.testing.allocator;

    var session = Session.init(allocator, getTestModelPath("matmul.onnx")) catch |err| {
        std.debug.print("Skipping test: Could not load model (error: {s})\n", .{@errorName(err)});
        return;
    };
    defer session.deinit();

    // Verify shapes
    try std.testing.expectEqual(@as(usize, 2), session.getInputCount());
    try std.testing.expectEqual(@as(usize, 1), session.getOutputCount());

    // A: [2, 3] matrix
    const a_data = [_]f32{
        1.0, 2.0, 3.0, // row 0
        4.0, 5.0, 6.0, // row 1
    };
    const a_shape = [_]i64{ 2, 3 };

    // B: [3, 4] matrix
    const b_data = [_]f32{
        1.0, 0.0, 0.0, 1.0, // row 0
        0.0, 1.0, 0.0, 1.0, // row 1
        0.0, 0.0, 1.0, 1.0, // row 2
    };
    const b_shape = [_]i64{ 3, 4 };

    const outputs = try session.runF32(
        &[_][]const f32{ &a_data, &b_data },
        &[_][]const i64{ &a_shape, &b_shape },
    );
    defer session.freeOutputs(outputs);

    // Y = A @ B results in [2, 4]
    // Row 0: [1*1+2*0+3*0, 1*0+2*1+3*0, 1*0+2*0+3*1, 1*1+2*1+3*1] = [1, 2, 3, 6]
    // Row 1: [4*1+5*0+6*0, 4*0+5*1+6*0, 4*0+5*0+6*1, 4*1+5*1+6*1] = [4, 5, 6, 15]
    const expected = [_]f32{
        1.0, 2.0, 3.0,  6.0,
        4.0, 5.0, 6.0, 15.0,
    };

    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    try std.testing.expectEqual(@as(usize, 8), outputs[0].numel());
    try std.testing.expectEqual(@as(usize, 2), outputs[0].shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 4), outputs[0].shape.dims[1]);

    for (0..8) |i| {
        try std.testing.expectApproxEqAbs(expected[i], outputs[0].data[i], 1e-6);
    }
}

test "Session - get input/output info" {
    const allocator = std.testing.allocator;

    var session = Session.init(allocator, getTestModelPath("add.onnx")) catch |err| {
        std.debug.print("Skipping test: Could not load model (error: {s})\n", .{@errorName(err)});
        return;
    };
    defer session.deinit();

    // Get first input info
    var input_info = try session.getInputInfo(0);
    defer input_info.deinit(allocator);

    try std.testing.expectEqualStrings("A", input_info.name);
    try std.testing.expectEqual(@as(usize, 2), input_info.shape.ndim);
    try std.testing.expectEqual(@as(usize, 2), input_info.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), input_info.shape.dims[1]);
    try std.testing.expectEqual(@as(c_uint, @intCast(ort.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)), input_info.dtype);

    // Get output info
    var output_info = try session.getOutputInfo(0);
    defer output_info.deinit(allocator);

    try std.testing.expectEqualStrings("C", output_info.name);
    try std.testing.expectEqual(@as(c_uint, @intCast(ort.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)), output_info.dtype);
}
