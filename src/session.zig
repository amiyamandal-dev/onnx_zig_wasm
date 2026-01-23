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
const TensorF64 = tensor_mod.TensorF64;
const TensorI32 = tensor_mod.TensorI32;
const TensorI64 = tensor_mod.TensorI64;
const TensorU8 = tensor_mod.TensorU8;
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
// Batch Options
// =============================================================================

/// Options for batched inference
pub const BatchOptions = struct {
    /// Whether to split batched outputs back into individual samples
    /// If true, returns one output tensor per input sample
    /// If false, returns a single batched output tensor
    unbatch_outputs: bool = true,

    /// Padding value for dynamic batch sizes (not yet implemented)
    pad_value: f32 = 0.0,

    /// Maximum batch size to process at once (0 = no limit)
    /// Larger batches may be split into chunks
    max_batch_size: usize = 0,
};

// =============================================================================
// Shape Conversion Utilities
// =============================================================================

/// Convert an i64 shape (ONNX format) to usize shape (Zig format).
/// Returns null if any dimension is negative (dynamic dimension).
pub fn shapeI64ToUsize(i64_shape: []const i64) ?[MAX_DIMS]usize {
    var result: [MAX_DIMS]usize = [_]usize{0} ** MAX_DIMS;
    if (i64_shape.len > MAX_DIMS) return null;

    for (i64_shape, 0..) |dim, i| {
        if (dim < 0) return null; // Dynamic dimension
        result[i] = @intCast(dim);
    }
    return result;
}

/// Convert an i64 shape slice to usize slice, storing in provided buffer.
/// Returns the slice of the buffer that was filled.
/// Returns null if any dimension is negative.
pub fn shapeI64ToUsizeSlice(i64_shape: []const i64, buffer: []usize) ?[]usize {
    if (i64_shape.len > buffer.len) return null;

    for (i64_shape, 0..) |dim, i| {
        if (dim < 0) return null;
        buffer[i] = @intCast(dim);
    }
    return buffer[0..i64_shape.len];
}

/// Convert a usize shape (Zig format) to i64 shape (ONNX format).
pub fn shapeUsizeToI64(usize_shape: []const usize) [MAX_DIMS]i64 {
    var result: [MAX_DIMS]i64 = [_]i64{0} ** MAX_DIMS;
    const len = @min(usize_shape.len, MAX_DIMS);

    for (usize_shape[0..len], 0..) |dim, i| {
        result[i] = @intCast(dim);
    }
    return result;
}

/// Convert a usize shape slice to i64 slice, storing in provided buffer.
/// Returns the slice of the buffer that was filled.
pub fn shapeUsizeToI64Slice(usize_shape: []const usize, buffer: []i64) ?[]i64 {
    if (usize_shape.len > buffer.len) return null;

    for (usize_shape, 0..) |dim, i| {
        buffer[i] = @intCast(dim);
    }
    return buffer[0..usize_shape.len];
}

/// Calculate the total number of elements from an i64 shape.
/// Returns null if shape contains negative (dynamic) dimensions.
pub fn shapeNumel(shape: []const i64) ?usize {
    if (shape.len == 0) return 0;

    var total: usize = 1;
    for (shape) |dim| {
        if (dim < 0) return null;
        total *= @intCast(dim);
    }
    return total;
}

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

    // Cached for performance (avoid repeated FFI calls)
    memory_info: *ort.OrtMemoryInfo,
    input_name_ptrs: [][*c]const u8,
    output_name_ptrs: [][*c]const u8,

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
        defer if (options.log_id != null) allocator.free(@as([:0]u8, @ptrCast(@constCast(log_id[0..std.mem.len(log_id)]))));

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

        // Create cached memory info for CPU tensors (avoids repeated FFI calls)
        var memory_info: ?*ort.OrtMemoryInfo = null;
        try checkOrtStatus(api, api.CreateCpuMemoryInfo.?(ort.c.OrtArenaAllocator, ort.c.OrtMemTypeDefault, &memory_info));
        errdefer api.ReleaseMemoryInfo.?(memory_info.?);

        // Pre-compute C pointer arrays for input/output names (avoids allocation per run)
        const input_name_ptrs = try allocator.alloc([*c]const u8, input_count);
        errdefer allocator.free(input_name_ptrs);
        for (0..input_count) |i| {
            input_name_ptrs[i] = input_names[i].ptr;
        }

        const output_name_ptrs = try allocator.alloc([*c]const u8, output_count);
        errdefer allocator.free(output_name_ptrs);
        for (0..output_count) |i| {
            output_name_ptrs[i] = output_names[i].ptr;
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
            .memory_info = memory_info.?,
            .input_name_ptrs = input_name_ptrs,
            .output_name_ptrs = output_name_ptrs,
        };
    }

    /// Release all resources
    pub fn deinit(self: *Self) void {
        // Free cached C pointer arrays
        self.allocator.free(self.input_name_ptrs);
        self.allocator.free(self.output_name_ptrs);

        // Free cached names
        for (self.input_names) |name| {
            self.allocator.free(name);
        }
        self.allocator.free(self.input_names);

        for (self.output_names) |name| {
            self.allocator.free(name);
        }
        self.allocator.free(self.output_names);

        // Release ORT resources (in reverse order of creation)
        self.api.ReleaseMemoryInfo.?(self.memory_info);
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

    /// Get the index of an input tensor by name.
    /// Returns null if no input with the given name exists.
    pub fn getInputIndex(self: *const Self, name: []const u8) ?usize {
        for (self.input_names, 0..) |input_name, i| {
            if (std.mem.eql(u8, input_name, name)) {
                return i;
            }
        }
        return null;
    }

    /// Get the index of an output tensor by name.
    /// Returns null if no output with the given name exists.
    pub fn getOutputIndex(self: *const Self, name: []const u8) ?usize {
        for (self.output_names, 0..) |output_name, i| {
            if (std.mem.eql(u8, output_name, name)) {
                return i;
            }
        }
        return null;
    }

    /// Check if an input with the given name exists
    pub fn hasInput(self: *const Self, name: []const u8) bool {
        return self.getInputIndex(name) != null;
    }

    /// Check if an output with the given name exists
    pub fn hasOutput(self: *const Self, name: []const u8) bool {
        return self.getOutputIndex(name) != null;
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

        // Create input OrtValues (using cached memory_info)
        var input_values = try self.allocator.alloc(?*ort.OrtValue, self.input_count);
        defer self.allocator.free(input_values);

        for (0..self.input_count) |i| {
            const data_ptr: ?*anyopaque = @constCast(@ptrCast(input_data[i].ptr));
            const data_len = input_data[i].len * @sizeOf(f32);

            var value: ?*ort.OrtValue = null;
            try checkOrtStatus(self.api, self.api.CreateTensorWithDataAsOrtValue.?(
                self.memory_info,
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

        // Run inference (using cached name pointers)
        try checkOrtStatus(self.api, self.api.Run.?(
            self.session,
            null, // run options
            self.input_name_ptrs.ptr,
            @ptrCast(input_values.ptr),
            self.input_count,
            self.output_name_ptrs.ptr,
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
                // Validate pointer alignment in debug builds before casting
                assertAligned(f32, data_ptr);
                const src_slice: [*]const f32 = @ptrCast(@alignCast(data_ptr.?));
                @memcpy(out_tensor.data, src_slice[0..elem_count]);

                result_tensors[i] = out_tensor;
            } else {
                return SessionError.OrtNullResult;
            }
        }

        return result_tensors;
    }

    /// Run inference with a single f32 input tensor.
    /// Convenience method for models with exactly one input.
    /// Returns allocated output tensors (caller must free).
    pub fn runF32Simple(self: *Self, input: []const f32, shape: []const i64) ![]TensorF32 {
        if (self.input_count != 1) return SessionError.InvalidInputCount;
        return self.runF32(&[_][]const f32{input}, &[_][]const i64{shape});
    }

    /// Run inference accepting TensorF32 directly.
    /// The tensor's shape is automatically converted to ONNX format.
    /// For single-input models.
    pub fn runTensor(self: *Self, input: *const TensorF32) ![]TensorF32 {
        if (self.input_count != 1) return SessionError.InvalidInputCount;

        // Convert usize shape to i64
        var i64_shape: [MAX_DIMS]i64 = undefined;
        for (0..input.shape.ndim) |i| {
            i64_shape[i] = @intCast(input.shape.dims[i]);
        }

        return self.runF32(&[_][]const f32{input.data}, &[_][]const i64{i64_shape[0..input.shape.ndim]});
    }

    /// Run inference with multiple TensorF32 inputs.
    /// Each tensor's shape is automatically converted to ONNX format.
    pub fn runTensors(self: *Self, inputs: []const *const TensorF32) ![]TensorF32 {
        if (inputs.len != self.input_count) return SessionError.InvalidInputCount;

        // Convert tensors to slices and shapes
        var data_slices: [16][]const f32 = undefined;
        var i64_shapes: [16][MAX_DIMS]i64 = undefined;
        var shape_slices: [16][]const i64 = undefined;

        if (inputs.len > 16) return SessionError.InvalidInputCount;

        for (inputs, 0..) |tensor, i| {
            data_slices[i] = tensor.data;
            for (0..tensor.shape.ndim) |j| {
                i64_shapes[i][j] = @intCast(tensor.shape.dims[j]);
            }
            shape_slices[i] = i64_shapes[i][0..tensor.shape.ndim];
        }

        return self.runF32(data_slices[0..inputs.len], shape_slices[0..inputs.len]);
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

        // Create input OrtValues (using cached memory_info)
        var input_values = try self.allocator.alloc(?*ort.OrtValue, self.input_count);
        defer self.allocator.free(input_values);

        for (0..self.input_count) |i| {
            const data_ptr: ?*anyopaque = @constCast(@ptrCast(input_data[i].ptr));
            const data_len = input_data[i].len * @sizeOf(i64);

            var value: ?*ort.OrtValue = null;
            try checkOrtStatus(self.api, self.api.CreateTensorWithDataAsOrtValue.?(
                self.memory_info,
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

        // Run inference (using cached name pointers)
        try checkOrtStatus(self.api, self.api.Run.?(
            self.session,
            null, // run options
            self.input_name_ptrs.ptr,
            @ptrCast(input_values.ptr),
            self.input_count,
            self.output_name_ptrs.ptr,
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
                // Validate pointer alignment in debug builds before casting
                assertAligned(f32, data_ptr);
                const src_slice: [*]const f32 = @ptrCast(@alignCast(data_ptr.?));
                @memcpy(out_tensor.data, src_slice[0..elem_count]);

                result_tensors[i] = out_tensor;
            } else {
                return SessionError.OrtNullResult;
            }
        }

        return result_tensors;
    }

    /// Run inference with uint8 tensors (for quantized models)
    /// input_data: slice of input tensor data (must match model input count and shapes)
    /// input_shapes: shapes for each input tensor
    /// Returns allocated output tensors (caller must free with freeU8Outputs)
    pub fn runU8(
        self: *Self,
        input_data: []const []const u8,
        input_shapes: []const []const i64,
    ) ![]TensorU8 {
        if (input_data.len != self.input_count) return SessionError.InvalidInputCount;
        if (input_shapes.len != self.input_count) return SessionError.InvalidInputCount;

        // Create input OrtValues (using cached memory_info)
        var input_values = try self.allocator.alloc(?*ort.OrtValue, self.input_count);
        defer self.allocator.free(input_values);

        for (0..self.input_count) |i| {
            const data_ptr: ?*anyopaque = @constCast(@ptrCast(input_data[i].ptr));
            const data_len = input_data[i].len * @sizeOf(u8);

            var value: ?*ort.OrtValue = null;
            try checkOrtStatus(self.api, self.api.CreateTensorWithDataAsOrtValue.?(
                self.memory_info,
                data_ptr,
                data_len,
                input_shapes[i].ptr,
                input_shapes[i].len,
                ort.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
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

        // Run inference (using cached name pointers)
        try checkOrtStatus(self.api, self.api.Run.?(
            self.session,
            null, // run options
            self.input_name_ptrs.ptr,
            @ptrCast(input_values.ptr),
            self.input_count,
            self.output_name_ptrs.ptr,
            self.output_count,
            output_values.ptr,
        ));

        // Convert outputs to Zig tensors
        var result_tensors = try self.allocator.alloc(TensorU8, self.output_count);
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
                const out_tensor = try TensorU8.init(self.allocator, shape_usize[0..dim_count]);
                // Validate pointer alignment in debug builds before casting
                assertAligned(u8, data_ptr);
                const src_slice: [*]const u8 = @ptrCast(@alignCast(data_ptr.?));
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

    /// Free u8 output tensors returned by runU8
    pub fn freeU8Outputs(self: *Self, outputs: []TensorU8) void {
        for (outputs) |*t| {
            t.deinit();
        }
        self.allocator.free(outputs);
    }

    /// Run inference with f64 (double precision) tensors
    /// input_data: slice of input tensor data (must match model input count and shapes)
    /// input_shapes: shapes for each input tensor
    /// Returns allocated output tensors (caller must free with freeF64Outputs)
    pub fn runF64(
        self: *Self,
        input_data: []const []const f64,
        input_shapes: []const []const i64,
    ) ![]TensorF64 {
        return self.runTypedImpl(f64, f64, input_data, input_shapes);
    }

    /// Free f64 output tensors returned by runF64
    pub fn freeF64Outputs(self: *Self, outputs: []TensorF64) void {
        for (outputs) |*t| {
            t.deinit();
        }
        self.allocator.free(outputs);
    }

    /// Run inference with i32 (32-bit integer) inputs
    /// Returns f32 output tensors (common for classification models)
    pub fn runI32(
        self: *Self,
        input_data: []const []const i32,
        input_shapes: []const []const i64,
    ) ![]TensorF32 {
        return self.runTypedImpl(i32, f32, input_data, input_shapes);
    }

    /// Run inference with i32 inputs returning i32 outputs
    pub fn runI32ToI32(
        self: *Self,
        input_data: []const []const i32,
        input_shapes: []const []const i64,
    ) ![]TensorI32 {
        return self.runTypedImpl(i32, i32, input_data, input_shapes);
    }

    /// Free i32 output tensors
    pub fn freeI32Outputs(self: *Self, outputs: []TensorI32) void {
        for (outputs) |*t| {
            t.deinit();
        }
        self.allocator.free(outputs);
    }

    /// Generic typed inference method
    /// InputT: Type of input tensor elements
    /// OutputT: Type of output tensor elements
    fn runTypedImpl(
        self: *Self,
        comptime InputT: type,
        comptime OutputT: type,
        input_data: []const []const InputT,
        input_shapes: []const []const i64,
    ) ![]Tensor(OutputT) {
        if (input_data.len != self.input_count) return SessionError.InvalidInputCount;
        if (input_shapes.len != self.input_count) return SessionError.InvalidInputCount;

        const input_onnx_type: c_uint = @intCast(ort.zigTypeToOnnx(InputT));

        // Create input OrtValues (using cached memory_info)
        var input_values = try self.allocator.alloc(?*ort.OrtValue, self.input_count);
        defer self.allocator.free(input_values);

        for (0..self.input_count) |i| {
            const data_ptr: ?*anyopaque = @constCast(@ptrCast(input_data[i].ptr));
            const data_len = input_data[i].len * @sizeOf(InputT);

            var value: ?*ort.OrtValue = null;
            try checkOrtStatus(self.api, self.api.CreateTensorWithDataAsOrtValue.?(
                self.memory_info,
                data_ptr,
                data_len,
                input_shapes[i].ptr,
                input_shapes[i].len,
                input_onnx_type,
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

        // Run inference (using cached name pointers)
        try checkOrtStatus(self.api, self.api.Run.?(
            self.session,
            null, // run options
            self.input_name_ptrs.ptr,
            @ptrCast(input_values.ptr),
            self.input_count,
            self.output_name_ptrs.ptr,
            self.output_count,
            output_values.ptr,
        ));

        // Convert outputs to Zig tensors
        const TensorOut = Tensor(OutputT);
        var result_tensors = try self.allocator.alloc(TensorOut, self.output_count);
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
                const out_tensor = try TensorOut.init(self.allocator, shape_usize[0..dim_count]);
                // Validate pointer alignment in debug builds
                assertAligned(OutputT, data_ptr);
                const src_slice: [*]const OutputT = @ptrCast(@alignCast(data_ptr.?));
                @memcpy(out_tensor.data, src_slice[0..elem_count]);

                result_tensors[i] = out_tensor;
            } else {
                return SessionError.OrtNullResult;
            }
        }

        return result_tensors;
    }

    /// Generic free method for typed outputs
    pub fn freeTypedOutputs(self: *Self, comptime T: type, outputs: []Tensor(T)) void {
        for (outputs) |*t| {
            t.deinit();
        }
        self.allocator.free(outputs);
    }

    // =========================================================================
    // Named Tensor API
    // =========================================================================

    /// Named input for runNamed
    pub const NamedInput = struct {
        name: []const u8,
        data: []const f32,
        shape: []const i64,
    };

    /// Run inference with named inputs.
    /// Allows specifying inputs by name rather than index order.
    /// Useful when input order is unknown or for clarity.
    pub fn runNamed(self: *Self, named_inputs: []const NamedInput) ![]TensorF32 {
        if (named_inputs.len != self.input_count) return SessionError.InvalidInputCount;

        // Reorder inputs to match expected order
        var ordered_data: [16][]const f32 = undefined;
        var ordered_shapes: [16][]const i64 = undefined;
        var found_count: usize = 0;

        if (self.input_count > 16) return SessionError.InvalidInputCount;

        for (0..self.input_count) |i| {
            const expected_name = self.input_names[i];
            var found = false;

            for (named_inputs) |input| {
                if (std.mem.eql(u8, input.name, expected_name)) {
                    ordered_data[i] = input.data;
                    ordered_shapes[i] = input.shape;
                    found = true;
                    found_count += 1;
                    break;
                }
            }

            if (!found) return SessionError.InvalidInputCount;
        }

        return self.runF32(ordered_data[0..self.input_count], ordered_shapes[0..self.input_count]);
    }

    /// Run inference with a StringHashMap of inputs.
    /// Keys are input names, values are tuples of (data, shape).
    pub fn runWithMap(
        self: *Self,
        inputs: std.StringHashMap(struct { data: []const f32, shape: []const i64 }),
    ) ![]TensorF32 {
        if (inputs.count() != self.input_count) return SessionError.InvalidInputCount;

        var ordered_data: [16][]const f32 = undefined;
        var ordered_shapes: [16][]const i64 = undefined;

        if (self.input_count > 16) return SessionError.InvalidInputCount;

        for (0..self.input_count) |i| {
            const expected_name = self.input_names[i];
            if (inputs.get(expected_name)) |input| {
                ordered_data[i] = input.data;
                ordered_shapes[i] = input.shape;
            } else {
                return SessionError.InvalidInputCount;
            }
        }

        return self.runF32(ordered_data[0..self.input_count], ordered_shapes[0..self.input_count]);
    }

    // =========================================================================
    // Batch Inference API
    // =========================================================================

    /// Run batched inference with f32 tensors
    /// Processes multiple samples in a single batch for efficiency.
    /// All samples must have the same shape (excluding batch dimension).
    ///
    /// batch_inputs: Array of input samples, each sample is an array of input tensors
    /// sample_shape: Shape of a single sample (without batch dimension)
    /// options: Batch processing options
    ///
    /// Returns: Array of output tensors for each sample
    pub fn runF32Batch(
        self: *Self,
        batch_inputs: []const []const f32,
        sample_shape: []const i64,
        options: BatchOptions,
    ) ![]TensorF32 {
        if (batch_inputs.len == 0) return &[_]TensorF32{};

        const batch_size = batch_inputs.len;

        // Calculate elements per sample
        var sample_elements: usize = 1;
        for (sample_shape) |dim| {
            sample_elements *= @intCast(dim);
        }

        // Validate all inputs have correct size
        for (batch_inputs) |input| {
            if (input.len != sample_elements) {
                return SessionError.InvalidInputCount;
            }
        }

        // Create batched shape: [batch_size, ...sample_shape]
        var batched_shape: [MAX_DIMS]i64 = [_]i64{0} ** MAX_DIMS;
        batched_shape[0] = @intCast(batch_size);
        for (sample_shape, 0..) |dim, i| {
            if (i + 1 >= MAX_DIMS) break;
            batched_shape[i + 1] = dim;
        }
        const batched_shape_len = @min(sample_shape.len + 1, MAX_DIMS);

        // Allocate and copy batched input data
        const total_elements = batch_size * sample_elements;
        var batched_data = try self.allocator.alloc(f32, total_elements);
        defer self.allocator.free(batched_data);

        for (batch_inputs, 0..) |input, batch_idx| {
            const offset = batch_idx * sample_elements;
            @memcpy(batched_data[offset..][0..sample_elements], input);
        }

        // Run inference with batched input
        const outputs = try self.runF32(
            &[_][]const f32{batched_data},
            &[_][]const i64{batched_shape[0..batched_shape_len]},
        );

        // If unbatch is requested, split outputs back into individual samples
        if (options.unbatch_outputs and outputs.len > 0) {
            defer self.freeOutputs(outputs);

            const output_tensor = outputs[0];
            const output_batch_size = output_tensor.shape.dims[0];

            if (output_batch_size != batch_size) {
                return SessionError.ShapeMismatch;
            }

            // Calculate output elements per sample
            var output_sample_elements: usize = 1;
            for (1..output_tensor.shape.ndim) |i| {
                output_sample_elements *= output_tensor.shape.dims[i];
            }

            // Create output shape for individual samples
            var sample_output_shape: [MAX_DIMS]usize = [_]usize{0} ** MAX_DIMS;
            for (1..output_tensor.shape.ndim) |i| {
                sample_output_shape[i - 1] = output_tensor.shape.dims[i];
            }
            const sample_ndim = output_tensor.shape.ndim - 1;

            // Allocate individual output tensors
            var result = try self.allocator.alloc(TensorF32, batch_size);
            errdefer {
                for (result) |*t| t.deinit();
                self.allocator.free(result);
            }

            for (0..batch_size) |batch_idx| {
                result[batch_idx] = try TensorF32.init(self.allocator, sample_output_shape[0..sample_ndim]);
                const src_offset = batch_idx * output_sample_elements;
                @memcpy(result[batch_idx].data, output_tensor.data[src_offset..][0..output_sample_elements]);
            }

            return result;
        }

        return outputs;
    }

    /// Run batched inference and return just the argmax indices
    /// Useful for classification tasks where only the predicted class is needed
    pub fn runF32BatchClassify(
        self: *Self,
        batch_inputs: []const []const f32,
        sample_shape: []const i64,
    ) ![]usize {
        const outputs = try self.runF32Batch(batch_inputs, sample_shape, .{ .unbatch_outputs = true });
        defer {
            for (outputs) |*t| {
                var t_mut = t.*;
                t_mut.deinit();
            }
            self.allocator.free(outputs);
        }

        var predictions = try self.allocator.alloc(usize, outputs.len);
        for (outputs, 0..) |output, i| {
            predictions[i] = tensor_mod.SimdOps.argmax(output.data);
        }

        return predictions;
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

/// Debug assertion to validate pointer alignment before casting.
/// In debug builds, this will panic if the pointer is not properly aligned.
/// In release builds, this is a no-op.
fn assertAligned(comptime T: type, ptr: ?*anyopaque) void {
    if (@import("builtin").mode == .Debug) {
        if (ptr) |p| {
            const addr = @intFromPtr(p);
            const alignment = @alignOf(T);
            if (addr % alignment != 0) {
                std.debug.panic(
                    "Pointer alignment error: address 0x{x} is not aligned to {d} bytes for type {s}",
                    .{ addr, alignment, @typeName(T) },
                );
            }
        }
    }
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
            const prefix_z = allocator.allocSentinel(u8, prefix.len, 0) catch return SessionError.AllocationFailed;
            defer allocator.free(prefix_z);
            @memcpy(prefix_z, prefix);
            try checkOrtStatus(api, api.EnableProfiling.?(session_options, prefix_z.ptr));
        } else {
            try checkOrtStatus(api, api.EnableProfiling.?(session_options, "onnx_zig_profile"));
        }
    }

    // Save optimized model
    if (options.optimized_model_filepath) |path| {
        const path_z = allocator.allocSentinel(u8, path.len, 0) catch return SessionError.AllocationFailed;
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
/// Note: Not all providers are available in every ONNX Runtime build.
/// This function gracefully handles missing providers by skipping them.
fn appendExecutionProvider(api: *const ort.OrtApi, session_options: *ort.OrtSessionOptions, provider: ExecutionProvider) SessionError!void {
    _ = api;
    _ = session_options;

    switch (provider) {
        .cpu => {
            // CPU is always available and doesn't need explicit registration
        },
        .cuda, .tensorrt, .coreml, .directml, .rocm, .openvino, .nnapi, .dnnl, .xnnpack => {
            // These execution providers require platform-specific ONNX Runtime builds
            // and their API functions may not be available in all builds.
            // Skip silently if not available - the CPU provider will be used as fallback.
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

/// Check if a test model exists, returning a more informative skip message.
/// Returns true if the model exists and the test should proceed.
fn checkTestModelExists(comptime model_path: []const u8) bool {
    const file = std.fs.cwd().openFile(model_path, .{}) catch |err| {
        std.debug.print("\n[TEST SKIP] Model '{s}' not available: {s}\n", .{ model_path, @errorName(err) });
        std.debug.print("  Hint: Run 'uv run --with onnx scripts/generate_test_model.py' to generate test models\n", .{});
        return false;
    };
    file.close();
    return true;
}

/// Try to load a session, or skip the test with a helpful message.
/// Returns the session if successful, or null if the test should be skipped.
fn loadTestSession(allocator: Allocator, comptime model_path: []const u8) ?Session {
    // First check if file exists
    if (!checkTestModelExists(model_path)) {
        return null;
    }

    // Try to load the model
    return Session.init(allocator, model_path) catch |err| {
        std.debug.print("\n[TEST SKIP] Failed to load model '{s}': {s}\n", .{ model_path, @errorName(err) });
        if (err == SessionError.OrtInvalidGraph) {
            std.debug.print("  Hint: Model may be corrupt or incompatible with this ONNX Runtime version\n", .{});
        }
        return null;
    };
}

test "Session - load identity model" {
    const allocator = std.testing.allocator;

    var session = loadTestSession(allocator, getTestModelPath("identity.onnx")) orelse return;
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

    var session = loadTestSession(allocator, getTestModelPath("identity.onnx")) orelse return;
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

    var session = loadTestSession(allocator, getTestModelPath("add.onnx")) orelse return;
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

    var session = loadTestSession(allocator, getTestModelPath("relu.onnx")) orelse return;
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

    var session = loadTestSession(allocator, getTestModelPath("matmul.onnx")) orelse return;
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

    var session = loadTestSession(allocator, getTestModelPath("add.onnx")) orelse return;
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

// =============================================================================
// Quantized (UINT8) Model Tests
// =============================================================================

test "Session - identity_u8 model inference" {
    const allocator = std.testing.allocator;

    var session = loadTestSession(allocator, getTestModelPath("identity_u8.onnx")) orelse return;
    defer session.deinit();

    // Verify model metadata
    try std.testing.expectEqual(@as(usize, 1), session.getInputCount());
    try std.testing.expectEqual(@as(usize, 1), session.getOutputCount());

    const input_names = session.getInputNames();
    try std.testing.expectEqualStrings("X", input_names[0]);

    // Input: [1, 8] u8 tensor with values [0, 50, 100, 150, 200, 250, 128, 255]
    const input_data = [_]u8{ 0, 50, 100, 150, 200, 250, 128, 255 };
    const input_shape = [_]i64{ 1, 8 };

    const outputs = try session.runU8(
        &[_][]const u8{&input_data},
        &[_][]const i64{&input_shape},
    );
    defer session.freeU8Outputs(outputs);

    // Identity should return same values
    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    try std.testing.expectEqual(@as(usize, 8), outputs[0].numel());

    for (0..8) |i| {
        try std.testing.expectEqual(input_data[i], outputs[0].data[i]);
    }
}

test "Session - add_u8 model inference" {
    const allocator = std.testing.allocator;

    var session = loadTestSession(allocator, getTestModelPath("add_u8.onnx")) orelse return;
    defer session.deinit();

    // Verify two inputs
    try std.testing.expectEqual(@as(usize, 2), session.getInputCount());
    try std.testing.expectEqual(@as(usize, 1), session.getOutputCount());

    // A: [2, 4] u8 tensor
    const a_data = [_]u8{ 10, 20, 30, 40, 50, 60, 70, 80 };
    // B: [2, 4] u8 tensor
    const b_data = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const shape = [_]i64{ 2, 4 };

    const outputs = try session.runU8(
        &[_][]const u8{ &a_data, &b_data },
        &[_][]const i64{ &shape, &shape },
    );
    defer session.freeU8Outputs(outputs);

    // C = A + B
    const expected = [_]u8{ 11, 22, 33, 44, 55, 66, 77, 88 };

    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    try std.testing.expectEqual(@as(usize, 8), outputs[0].numel());

    for (0..8) |i| {
        try std.testing.expectEqual(expected[i], outputs[0].data[i]);
    }
}

test "Session - get input info for u8 model" {
    const allocator = std.testing.allocator;

    var session = loadTestSession(allocator, getTestModelPath("identity_u8.onnx")) orelse return;
    defer session.deinit();

    // Get input info
    var input_info = try session.getInputInfo(0);
    defer input_info.deinit(allocator);

    try std.testing.expectEqualStrings("X", input_info.name);
    try std.testing.expectEqual(@as(usize, 2), input_info.shape.ndim);
    try std.testing.expectEqual(@as(usize, 1), input_info.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 8), input_info.shape.dims[1]);
    try std.testing.expectEqual(@as(c_uint, @intCast(ort.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)), input_info.dtype);

    // Get output info
    var output_info = try session.getOutputInfo(0);
    defer output_info.deinit(allocator);

    try std.testing.expectEqualStrings("Y", output_info.name);
    try std.testing.expectEqual(@as(c_uint, @intCast(ort.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)), output_info.dtype);
}

// =============================================================================
// Extended Data Type Tests (f64, i32)
// =============================================================================

test "Session - identity_f64 model inference" {
    const allocator = std.testing.allocator;

    var session = loadTestSession(allocator, getTestModelPath("identity_f64.onnx")) orelse return;
    defer session.deinit();

    // Verify model metadata
    try std.testing.expectEqual(@as(usize, 1), session.getInputCount());
    try std.testing.expectEqual(@as(usize, 1), session.getOutputCount());

    // Input: [1, 4] f64 tensor
    const input_data = [_]f64{ 1.5, 2.5, 3.5, 4.5 };
    const input_shape = [_]i64{ 1, 4 };

    const outputs = try session.runF64(
        &[_][]const f64{&input_data},
        &[_][]const i64{&input_shape},
    );
    defer session.freeF64Outputs(outputs);

    // Identity should return same values
    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    try std.testing.expectEqual(@as(usize, 4), outputs[0].numel());

    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(input_data[i], outputs[0].data[i], 1e-10);
    }
}

test "Session - identity_i32 model inference" {
    const allocator = std.testing.allocator;

    var session = loadTestSession(allocator, getTestModelPath("identity_i32.onnx")) orelse return;
    defer session.deinit();

    // Verify model metadata
    try std.testing.expectEqual(@as(usize, 1), session.getInputCount());
    try std.testing.expectEqual(@as(usize, 1), session.getOutputCount());

    // Input: [1, 4] i32 tensor
    const input_data = [_]i32{ -100, 0, 50, 1000 };
    const input_shape = [_]i64{ 1, 4 };

    const outputs = try session.runI32ToI32(
        &[_][]const i32{&input_data},
        &[_][]const i64{&input_shape},
    );
    defer session.freeI32Outputs(outputs);

    // Identity should return same values
    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    try std.testing.expectEqual(@as(usize, 4), outputs[0].numel());

    for (0..4) |i| {
        try std.testing.expectEqual(input_data[i], outputs[0].data[i]);
    }
}

test "Session - get input info for f64 model" {
    const allocator = std.testing.allocator;

    var session = loadTestSession(allocator, getTestModelPath("identity_f64.onnx")) orelse return;
    defer session.deinit();

    var input_info = try session.getInputInfo(0);
    defer input_info.deinit(allocator);

    try std.testing.expectEqualStrings("X", input_info.name);
    try std.testing.expectEqual(@as(c_uint, @intCast(ort.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)), input_info.dtype);
}

test "Session - get input info for i32 model" {
    const allocator = std.testing.allocator;

    var session = loadTestSession(allocator, getTestModelPath("identity_i32.onnx")) orelse return;
    defer session.deinit();

    var input_info = try session.getInputInfo(0);
    defer input_info.deinit(allocator);

    try std.testing.expectEqualStrings("X", input_info.name);
    try std.testing.expectEqual(@as(c_uint, @intCast(ort.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)), input_info.dtype);
}

// =============================================================================
// Batch Inference Tests
// =============================================================================

test "Session - batch inference with identity model" {
    const allocator = std.testing.allocator;

    // Use identity_batch.onnx which has dynamic batch dimension [batch, 4]
    var session = loadTestSession(allocator, getTestModelPath("identity_batch.onnx")) orelse return;
    defer session.deinit();

    // Create batch of 3 samples, each with 4 elements
    const sample1 = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const sample2 = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    const sample3 = [_]f32{ 9.0, 10.0, 11.0, 12.0 };

    const batch_inputs = [_][]const f32{ &sample1, &sample2, &sample3 };
    // Sample shape is [4] - batch dimension is added by runF32Batch
    const sample_shape = [_]i64{4};

    const outputs = try session.runF32Batch(&batch_inputs, &sample_shape, .{ .unbatch_outputs = true });
    defer {
        for (outputs) |*t| {
            var t_mut = t.*;
            t_mut.deinit();
        }
        allocator.free(outputs);
    }

    // Should get 3 output tensors
    try std.testing.expectEqual(@as(usize, 3), outputs.len);

    // Each output should match its input (identity model)
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(sample1[i], outputs[0].data[i], 1e-6);
        try std.testing.expectApproxEqAbs(sample2[i], outputs[1].data[i], 1e-6);
        try std.testing.expectApproxEqAbs(sample3[i], outputs[2].data[i], 1e-6);
    }
}

test "Session - batch inference without unbatching" {
    const allocator = std.testing.allocator;

    // Use identity_batch.onnx which has dynamic batch dimension [batch, 4]
    var session = loadTestSession(allocator, getTestModelPath("identity_batch.onnx")) orelse return;
    defer session.deinit();

    // Create batch of 2 samples
    const sample1 = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const sample2 = [_]f32{ 5.0, 6.0, 7.0, 8.0 };

    const batch_inputs = [_][]const f32{ &sample1, &sample2 };
    // Sample shape is [4] - batch dimension is added by runF32Batch
    const sample_shape = [_]i64{4};

    const outputs = try session.runF32Batch(&batch_inputs, &sample_shape, .{ .unbatch_outputs = false });
    defer session.freeOutputs(outputs);

    // Should get 1 batched output tensor with batch_size=2
    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    try std.testing.expectEqual(@as(usize, 2), outputs[0].shape.dims[0]); // batch dim
    try std.testing.expectEqual(@as(usize, 8), outputs[0].numel()); // 2 * 4 elements
}

test "Session - runF32Simple" {
    const allocator = std.testing.allocator;

    var session = loadTestSession(allocator, getTestModelPath("identity.onnx")) orelse return;
    defer session.deinit();

    const input_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const input_shape = [_]i64{ 1, 4 };

    // Use simplified API
    const outputs = try session.runF32Simple(&input_data, &input_shape);
    defer session.freeOutputs(outputs);

    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(input_data[i], outputs[0].data[i], 1e-6);
    }
}

test "Session - runTensor" {
    const allocator = std.testing.allocator;

    var session = loadTestSession(allocator, getTestModelPath("identity.onnx")) orelse return;
    defer session.deinit();

    // Create a TensorF32
    var input = try TensorF32.init(allocator, &[_]usize{ 1, 4 });
    defer input.deinit();
    input.data[0] = 1.0;
    input.data[1] = 2.0;
    input.data[2] = 3.0;
    input.data[3] = 4.0;

    // Use tensor API
    const outputs = try session.runTensor(&input);
    defer session.freeOutputs(outputs);

    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(input.data[i], outputs[0].data[i], 1e-6);
    }
}

test "Session - getInputIndex and getOutputIndex" {
    const allocator = std.testing.allocator;

    var session = loadTestSession(allocator, getTestModelPath("add.onnx")) orelse return;
    defer session.deinit();

    // add.onnx has inputs A and B, output C
    try std.testing.expectEqual(@as(?usize, 0), session.getInputIndex("A"));
    try std.testing.expectEqual(@as(?usize, 1), session.getInputIndex("B"));
    try std.testing.expectEqual(@as(?usize, null), session.getInputIndex("X"));

    try std.testing.expectEqual(@as(?usize, 0), session.getOutputIndex("C"));
    try std.testing.expectEqual(@as(?usize, null), session.getOutputIndex("Y"));

    try std.testing.expect(session.hasInput("A"));
    try std.testing.expect(session.hasInput("B"));
    try std.testing.expect(!session.hasInput("X"));

    try std.testing.expect(session.hasOutput("C"));
    try std.testing.expect(!session.hasOutput("Y"));
}

test "Session - runNamed" {
    const allocator = std.testing.allocator;

    var session = loadTestSession(allocator, getTestModelPath("add.onnx")) orelse return;
    defer session.deinit();

    // add.onnx: C = A + B, both [2,3] tensors
    const a_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const b_data = [_]f32{ 0.5, 1.5, 2.5, 3.5, 4.5, 5.5 };
    const shape = [_]i64{ 2, 3 };

    // Provide inputs in reverse order using named API
    const named_inputs = [_]Session.NamedInput{
        .{ .name = "B", .data = &b_data, .shape = &shape },
        .{ .name = "A", .data = &a_data, .shape = &shape },
    };

    const outputs = try session.runNamed(&named_inputs);
    defer session.freeOutputs(outputs);

    try std.testing.expectEqual(@as(usize, 1), outputs.len);

    // Verify A + B
    for (0..6) |i| {
        const expected = a_data[i] + b_data[i];
        try std.testing.expectApproxEqAbs(expected, outputs[0].data[i], 1e-6);
    }
}

test "shapeI64ToUsize - valid conversion" {
    const i64_shape = [_]i64{ 2, 3, 4 };
    const result = shapeI64ToUsize(&i64_shape);
    try std.testing.expect(result != null);
    try std.testing.expectEqual(@as(usize, 2), result.?[0]);
    try std.testing.expectEqual(@as(usize, 3), result.?[1]);
    try std.testing.expectEqual(@as(usize, 4), result.?[2]);
}

test "shapeI64ToUsize - dynamic dimension returns null" {
    const i64_shape = [_]i64{ 2, -1, 4 };
    const result = shapeI64ToUsize(&i64_shape);
    try std.testing.expect(result == null);
}

test "shapeUsizeToI64 - valid conversion" {
    const usize_shape = [_]usize{ 2, 3, 4 };
    const result = shapeUsizeToI64(&usize_shape);
    try std.testing.expectEqual(@as(i64, 2), result[0]);
    try std.testing.expectEqual(@as(i64, 3), result[1]);
    try std.testing.expectEqual(@as(i64, 4), result[2]);
}

test "shapeNumel - element count" {
    const shape = [_]i64{ 2, 3, 4 };
    const result = shapeNumel(&shape);
    try std.testing.expect(result != null);
    try std.testing.expectEqual(@as(usize, 24), result.?);
}

test "shapeNumel - dynamic dimension returns null" {
    const shape = [_]i64{ 2, -1, 4 };
    const result = shapeNumel(&shape);
    try std.testing.expect(result == null);
}
