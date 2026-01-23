//! WebAssembly exports for browser-based ONNX inference.
//!
//! This module provides exported functions that JavaScript can call to:
//! - Allocate and free memory
//! - Create and manipulate tensors
//! - Perform SIMD-optimized tensor operations
//!
//! The inference itself is handled by onnxruntime-web on the JavaScript side,
//! while this module provides efficient tensor preprocessing and postprocessing.

const std = @import("std");
const tensor_mod = @import("tensor.zig");

const TensorF32 = tensor_mod.TensorF32;
const Shape = tensor_mod.Shape;
const SimdOps = tensor_mod.SimdOps;
const MAX_DIMS = tensor_mod.MAX_DIMS;

// =============================================================================
// WASM Allocator
// =============================================================================

/// WebAssembly page allocator for memory management
const wasm_allocator = std.heap.wasm_allocator;

// =============================================================================
// Memory Management Exports
// =============================================================================

/// Allocate a block of memory. Returns pointer (0 on failure).
export fn wasm_alloc(size: usize) usize {
    const slice = wasm_allocator.alloc(u8, size) catch return 0;
    return @intFromPtr(slice.ptr);
}

/// Free a previously allocated block of memory.
export fn wasm_free(ptr: usize, size: usize) void {
    if (ptr == 0) return;
    const p: [*]u8 = @ptrFromInt(ptr);
    wasm_allocator.free(p[0..size]);
}

/// Allocate aligned memory for f32 data. Returns pointer (0 on failure).
export fn wasm_alloc_f32(count: usize) usize {
    const slice = wasm_allocator.alloc(f32, count) catch return 0;
    return @intFromPtr(slice.ptr);
}

/// Free f32 array.
export fn wasm_free_f32(ptr: usize, count: usize) void {
    if (ptr == 0) return;
    const p: [*]f32 = @ptrFromInt(ptr);
    wasm_allocator.free(p[0..count]);
}

// =============================================================================
// Error Codes (accessible via wasm_get_last_error)
// =============================================================================

pub const WasmError = enum(i32) {
    ok = 0,
    out_of_memory = -1,
    invalid_handle = -2,
    invalid_shape = -3,
    dimension_mismatch = -4,
    out_of_bounds = -5,
    no_free_slots = -6,
    null_pointer = -7,
};

var last_error: WasmError = .ok;

/// Get the last error code
export fn wasm_get_last_error() i32 {
    return @intFromEnum(last_error);
}

/// Clear the last error
export fn wasm_clear_error() void {
    last_error = .ok;
}

// =============================================================================
// Dynamic Tensor Handle System
// =============================================================================

/// Tensor entry with reference count
const TensorEntry = struct {
    tensor: ?TensorF32 = null,
    ref_count: u32 = 0,
    generation: u32 = 0, // Incremented on each allocation to detect stale handles
};

/// Initial tensor pool size
const INITIAL_POOL_SIZE = 64;

/// Maximum tensor pool size
const MAX_POOL_SIZE = 4096;

/// Current pool capacity
var pool_capacity: usize = 0;

/// Tensor handle storage (dynamically allocated)
var tensor_pool: ?[]TensorEntry = null;

/// Statistics
var stats = struct {
    total_allocations: u64 = 0,
    total_frees: u64 = 0,
    peak_usage: usize = 0,
    current_usage: usize = 0,
}{};

/// Initialize the tensor pool
fn initPool() bool {
    if (tensor_pool != null) return true;

    tensor_pool = wasm_allocator.alloc(TensorEntry, INITIAL_POOL_SIZE) catch {
        last_error = .out_of_memory;
        return false;
    };
    pool_capacity = INITIAL_POOL_SIZE;

    // Initialize all entries
    for (tensor_pool.?) |*entry| {
        entry.* = TensorEntry{};
    }

    return true;
}

/// Grow the tensor pool
fn growPool() bool {
    if (pool_capacity >= MAX_POOL_SIZE) {
        last_error = .out_of_memory;
        return false;
    }

    const new_capacity = @min(pool_capacity * 2, MAX_POOL_SIZE);
    const new_pool = wasm_allocator.alloc(TensorEntry, new_capacity) catch {
        last_error = .out_of_memory;
        return false;
    };

    // Copy existing entries
    if (tensor_pool) |old_pool| {
        @memcpy(new_pool[0..pool_capacity], old_pool);
        wasm_allocator.free(old_pool);
    }

    // Initialize new entries
    for (new_pool[pool_capacity..]) |*entry| {
        entry.* = TensorEntry{};
    }

    tensor_pool = new_pool;
    pool_capacity = new_capacity;
    return true;
}

/// Find a free tensor slot, growing pool if needed
fn findFreeSlot() ?usize {
    if (!initPool()) return null;

    // First pass: find existing free slot
    for (tensor_pool.?, 0..) |entry, i| {
        if (entry.tensor == null) return i;
    }

    // Pool is full, try to grow
    const old_capacity = pool_capacity;
    if (!growPool()) return null;

    // Return first slot in new space
    return old_capacity;
}

/// Validate a tensor handle and return the index
fn validateHandle(handle: i32) ?usize {
    if (handle < 0) return null;
    const idx: usize = @intCast(handle);
    if (idx >= pool_capacity) return null;
    if (tensor_pool == null) return null;
    return idx;
}

/// Get tensor from handle
fn getTensor(handle: i32) ?TensorF32 {
    const idx = validateHandle(handle) orelse {
        last_error = .invalid_handle;
        return null;
    };
    return tensor_pool.?[idx].tensor;
}

/// Get mutable tensor pointer from handle
fn getTensorPtr(handle: i32) ?*TensorF32 {
    const idx = validateHandle(handle) orelse {
        last_error = .invalid_handle;
        return null;
    };
    const entry = &tensor_pool.?[idx];
    return if (entry.tensor != null) &entry.tensor.? else null;
}

// =============================================================================
// Handle Statistics Exports
// =============================================================================

/// Get current number of active tensors
export fn wasm_tensor_count() usize {
    return stats.current_usage;
}

/// Get peak tensor usage
export fn wasm_tensor_peak() usize {
    return stats.peak_usage;
}

/// Get total allocations
export fn wasm_tensor_total_allocs() u64 {
    return stats.total_allocations;
}

/// Get pool capacity
export fn wasm_tensor_capacity() usize {
    return pool_capacity;
}

/// Retain a tensor handle (increment reference count)
export fn tensor_retain(handle: i32) i32 {
    const idx = validateHandle(handle) orelse return -1;
    var entry = &tensor_pool.?[idx];
    if (entry.tensor == null) return -1;
    entry.ref_count += 1;
    return handle;
}

/// Release a tensor handle (decrement reference count, free if zero)
export fn tensor_release(handle: i32) void {
    const idx = validateHandle(handle) orelse return;
    var entry = &tensor_pool.?[idx];
    if (entry.tensor == null) return;

    if (entry.ref_count > 0) {
        entry.ref_count -= 1;
    }

    if (entry.ref_count == 0) {
        entry.tensor.?.deinit();
        entry.tensor = null;
        entry.generation += 1;
        stats.total_frees += 1;
        stats.current_usage -= 1;
    }
}

// =============================================================================
// Tensor Creation Exports
// =============================================================================

/// Create a new f32 tensor with given dimensions.
/// Returns tensor handle (index) or -1 on failure.
/// shape_ptr: pointer to dimension array
/// ndim: number of dimensions (max 8)
export fn tensor_create(shape_ptr: [*]const usize, ndim: usize) i32 {
    if (ndim == 0 or ndim > MAX_DIMS) {
        last_error = .invalid_shape;
        return -1;
    }

    const slot = findFreeSlot() orelse return -1;
    const shape_slice = shape_ptr[0..ndim];

    const tensor = TensorF32.init(wasm_allocator, shape_slice) catch {
        last_error = .out_of_memory;
        return -1;
    };

    tensor_pool.?[slot] = TensorEntry{
        .tensor = tensor,
        .ref_count = 1,
        .generation = tensor_pool.?[slot].generation,
    };

    stats.total_allocations += 1;
    stats.current_usage += 1;
    if (stats.current_usage > stats.peak_usage) {
        stats.peak_usage = stats.current_usage;
    }

    return @intCast(slot);
}

/// Create a tensor filled with zeros.
export fn tensor_zeros(shape_ptr: [*]const usize, ndim: usize) i32 {
    return tensor_create(shape_ptr, ndim);
}

/// Create a tensor filled with ones.
export fn tensor_ones(shape_ptr: [*]const usize, ndim: usize) i32 {
    if (ndim == 0 or ndim > MAX_DIMS) {
        last_error = .invalid_shape;
        return -1;
    }

    const slot = findFreeSlot() orelse return -1;
    const shape_slice = shape_ptr[0..ndim];

    const tensor = TensorF32.ones(wasm_allocator, shape_slice) catch {
        last_error = .out_of_memory;
        return -1;
    };

    tensor_pool.?[slot] = TensorEntry{
        .tensor = tensor,
        .ref_count = 1,
        .generation = tensor_pool.?[slot].generation,
    };

    stats.total_allocations += 1;
    stats.current_usage += 1;
    if (stats.current_usage > stats.peak_usage) {
        stats.peak_usage = stats.current_usage;
    }

    return @intCast(slot);
}

/// Create a tensor filled with a specific value.
export fn tensor_full(shape_ptr: [*]const usize, ndim: usize, value: f32) i32 {
    if (ndim == 0 or ndim > MAX_DIMS) {
        last_error = .invalid_shape;
        return -1;
    }

    const slot = findFreeSlot() orelse return -1;
    const shape_slice = shape_ptr[0..ndim];

    const tensor = TensorF32.full(wasm_allocator, shape_slice, value) catch {
        last_error = .out_of_memory;
        return -1;
    };

    tensor_pool.?[slot] = TensorEntry{
        .tensor = tensor,
        .ref_count = 1,
        .generation = tensor_pool.?[slot].generation,
    };

    stats.total_allocations += 1;
    stats.current_usage += 1;
    if (stats.current_usage > stats.peak_usage) {
        stats.peak_usage = stats.current_usage;
    }

    return @intCast(slot);
}

/// Create a tensor from existing data (copies the data).
export fn tensor_from_data(data_ptr: [*]const f32, data_len: usize, shape_ptr: [*]const usize, ndim: usize) i32 {
    if (ndim == 0 or ndim > MAX_DIMS) {
        last_error = .invalid_shape;
        return -1;
    }

    const slot = findFreeSlot() orelse return -1;
    const shape_slice = shape_ptr[0..ndim];

    var tensor = TensorF32.init(wasm_allocator, shape_slice) catch {
        last_error = .out_of_memory;
        return -1;
    };
    const copy_len = @min(data_len, tensor.numel());
    @memcpy(tensor.data[0..copy_len], data_ptr[0..copy_len]);

    tensor_pool.?[slot] = TensorEntry{
        .tensor = tensor,
        .ref_count = 1,
        .generation = tensor_pool.?[slot].generation,
    };

    stats.total_allocations += 1;
    stats.current_usage += 1;
    if (stats.current_usage > stats.peak_usage) {
        stats.peak_usage = stats.current_usage;
    }

    return @intCast(slot);
}

/// Release a tensor and free its memory.
export fn tensor_free(handle: i32) void {
    // Just call release - with initial ref_count of 1, it will free
    tensor_release(handle);
}

// =============================================================================
// Tensor Accessors
// =============================================================================

/// Get pointer to tensor's raw data buffer (returns address as usize, 0 on error).
export fn tensor_data_ptr(handle: i32) usize {
    const t = getTensor(handle) orelse return 0;
    return @intFromPtr(t.data.ptr);
}

/// Get total number of elements in tensor.
export fn tensor_numel(handle: i32) usize {
    const t = getTensor(handle) orelse return 0;
    return t.numel();
}

/// Get number of dimensions.
export fn tensor_ndim(handle: i32) usize {
    const t = getTensor(handle) orelse return 0;
    return t.ndim();
}

/// Get dimension at index.
export fn tensor_dim(handle: i32, dim_idx: usize) usize {
    const t = getTensor(handle) orelse return 0;
    return if (dim_idx < t.ndim()) t.shape.dims[dim_idx] else 0;
}

/// Get element at flat index.
export fn tensor_get(handle: i32, index: usize) f32 {
    const t = getTensor(handle) orelse return 0.0;
    return if (index < t.numel()) t.data[index] else 0.0;
}

/// Set element at flat index.
export fn tensor_set(handle: i32, index: usize, value: f32) void {
    const t = getTensorPtr(handle) orelse return;
    if (index < t.numel()) {
        t.data[index] = value;
    }
}

// =============================================================================
// SIMD-Optimized Operations (operate on raw arrays)
// =============================================================================

/// Element-wise add: dst = a + b
export fn simd_add(dst: [*]f32, a: [*]const f32, b: [*]const f32, len: usize) void {
    SimdOps.add(dst[0..len], a[0..len], b[0..len]);
}

/// Element-wise subtract: dst = a - b
export fn simd_sub(dst: [*]f32, a: [*]const f32, b: [*]const f32, len: usize) void {
    SimdOps.sub(dst[0..len], a[0..len], b[0..len]);
}

/// Element-wise multiply: dst = a * b
export fn simd_mul(dst: [*]f32, a: [*]const f32, b: [*]const f32, len: usize) void {
    SimdOps.mul(dst[0..len], a[0..len], b[0..len]);
}

/// Element-wise divide: dst = a / b
export fn simd_div(dst: [*]f32, a: [*]const f32, b: [*]const f32, len: usize) void {
    SimdOps.div(dst[0..len], a[0..len], b[0..len]);
}

/// Scale by constant: dst = a * scalar
export fn simd_scale(dst: [*]f32, a: [*]const f32, scalar: f32, len: usize) void {
    SimdOps.scale(dst[0..len], a[0..len], scalar);
}

/// Add constant: dst = a + scalar
export fn simd_add_scalar(dst: [*]f32, a: [*]const f32, scalar: f32, len: usize) void {
    SimdOps.addScalar(dst[0..len], a[0..len], scalar);
}

/// ReLU activation: dst = max(0, a)
export fn simd_relu(dst: [*]f32, a: [*]const f32, len: usize) void {
    SimdOps.relu(dst[0..len], a[0..len]);
}

/// Sum all elements
export fn simd_sum(a: [*]const f32, len: usize) f32 {
    return SimdOps.sum(a[0..len]);
}

/// Dot product of two vectors
export fn simd_dot(a: [*]const f32, b: [*]const f32, len: usize) f32 {
    return SimdOps.dot(a[0..len], b[0..len]);
}

/// Find maximum value
export fn simd_max(a: [*]const f32, len: usize) f32 {
    return SimdOps.max(a[0..len]);
}

/// Find minimum value
export fn simd_min(a: [*]const f32, len: usize) f32 {
    return SimdOps.min(a[0..len]);
}

/// Fused multiply-add: dst = a * b + c
export fn simd_fma(dst: [*]f32, a: [*]const f32, b: [*]const f32, c: [*]const f32, len: usize) void {
    SimdOps.fma(dst[0..len], a[0..len], b[0..len], c[0..len]);
}

/// Copy data
export fn simd_copy(dst: [*]f32, src: [*]const f32, len: usize) void {
    SimdOps.copy(dst[0..len], src[0..len]);
}

/// Fill with value
export fn simd_fill(dst: [*]f32, value: f32, len: usize) void {
    SimdOps.fill(dst[0..len], value);
}

// =============================================================================
// Tensor Operations (operate on tensor handles)
// =============================================================================

/// Helper to store a tensor in a slot with proper stats tracking
fn storeTensorInSlot(slot: usize, tensor: TensorF32) void {
    tensor_pool.?[slot] = TensorEntry{
        .tensor = tensor,
        .ref_count = 1,
        .generation = tensor_pool.?[slot].generation,
    };
    stats.total_allocations += 1;
    stats.current_usage += 1;
    if (stats.current_usage > stats.peak_usage) {
        stats.peak_usage = stats.current_usage;
    }
}

/// Add two tensors element-wise: result = a + b
/// Creates a new tensor for the result.
export fn tensor_add(a_handle: i32, b_handle: i32) i32 {
    const a = getTensor(a_handle) orelse return -1;
    const b = getTensor(b_handle) orelse return -1;
    if (a.numel() != b.numel()) {
        last_error = .dimension_mismatch;
        return -1;
    }

    const slot = findFreeSlot() orelse return -1;
    const result = TensorF32.initUninit(wasm_allocator, a.shape.slice()) catch {
        last_error = .out_of_memory;
        return -1;
    };
    SimdOps.add(result.data, a.data, b.data);
    storeTensorInSlot(slot, result);
    return @intCast(slot);
}

/// Multiply two tensors element-wise: result = a * b
export fn tensor_mul(a_handle: i32, b_handle: i32) i32 {
    const a = getTensor(a_handle) orelse return -1;
    const b = getTensor(b_handle) orelse return -1;
    if (a.numel() != b.numel()) {
        last_error = .dimension_mismatch;
        return -1;
    }

    const slot = findFreeSlot() orelse return -1;
    const result = TensorF32.initUninit(wasm_allocator, a.shape.slice()) catch {
        last_error = .out_of_memory;
        return -1;
    };
    SimdOps.mul(result.data, a.data, b.data);
    storeTensorInSlot(slot, result);
    return @intCast(slot);
}

/// Apply ReLU to tensor: result = max(0, input)
export fn tensor_relu(input_handle: i32) i32 {
    const input = getTensor(input_handle) orelse return -1;
    const slot = findFreeSlot() orelse return -1;
    const result = TensorF32.initUninit(wasm_allocator, input.shape.slice()) catch {
        last_error = .out_of_memory;
        return -1;
    };
    SimdOps.relu(result.data, input.data);
    storeTensorInSlot(slot, result);
    return @intCast(slot);
}

/// Scale tensor by constant: result = input * scalar
export fn tensor_scale(input_handle: i32, scalar: f32) i32 {
    const input = getTensor(input_handle) orelse return -1;
    const slot = findFreeSlot() orelse return -1;
    const result = TensorF32.initUninit(wasm_allocator, input.shape.slice()) catch {
        last_error = .out_of_memory;
        return -1;
    };
    SimdOps.scale(result.data, input.data, scalar);
    storeTensorInSlot(slot, result);
    return @intCast(slot);
}

/// Clone a tensor (deep copy)
export fn tensor_clone(input_handle: i32) i32 {
    const input = getTensor(input_handle) orelse return -1;
    const slot = findFreeSlot() orelse return -1;
    const cloned = input.clone(wasm_allocator) catch {
        last_error = .out_of_memory;
        return -1;
    };
    storeTensorInSlot(slot, cloned);
    return @intCast(slot);
}

// =============================================================================
// Softmax Implementation
// =============================================================================

/// Apply softmax to tensor along the last dimension.
/// Creates a new tensor with softmax applied.
export fn tensor_softmax(input_handle: i32) i32 {
    const input = getTensor(input_handle) orelse return -1;
    const slot = findFreeSlot() orelse return -1;
    const result = TensorF32.initUninit(wasm_allocator, input.shape.slice()) catch {
        last_error = .out_of_memory;
        return -1;
    };

    const n = input.numel();
    if (n == 0) {
        storeTensorInSlot(slot, result);
        return @intCast(slot);
    }

    // Find max for numerical stability
    const max_val = SimdOps.max(input.data);

    // Compute exp(x - max) and sum
    var sum: f32 = 0.0;
    for (input.data, 0..) |val, i| {
        const exp_val = @exp(val - max_val);
        result.data[i] = exp_val;
        sum += exp_val;
    }

    // Normalize
    if (sum > 0) {
        SimdOps.scale(result.data, result.data, 1.0 / sum);
    }

    storeTensorInSlot(slot, result);
    return @intCast(slot);
}

// =============================================================================
// Argmax Implementation
// =============================================================================

/// Find index of maximum value in tensor.
export fn tensor_argmax(input_handle: i32) i32 {
    const input = getTensor(input_handle) orelse return -1;
    if (input.numel() == 0) return -1;

    var max_idx: usize = 0;
    var max_val = input.data[0];
    for (input.data[1..], 1..) |val, i| {
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }
    return @intCast(max_idx);
}

// =============================================================================
// Image Preprocessing Utilities
// =============================================================================

/// Normalize image data: output[i] = (input[i] - mean) / std
/// Useful for preprocessing images before inference.
export fn normalize_image(
    dst: [*]f32,
    src: [*]const f32,
    len: usize,
    mean: f32,
    std_val: f32,
) void {
    const inv_std = 1.0 / std_val;
    var i: usize = 0;
    while (i < len) : (i += 1) {
        dst[i] = (src[i] - mean) * inv_std;
    }
}

/// Convert uint8 image data to f32 (0-255 -> 0.0-1.0)
export fn uint8_to_f32(dst: [*]f32, src: [*]const u8, len: usize) void {
    const scale: f32 = 1.0 / 255.0;
    var i: usize = 0;
    while (i < len) : (i += 1) {
        dst[i] = @as(f32, @floatFromInt(src[i])) * scale;
    }
}

/// Convert f32 to uint8 (0.0-1.0 -> 0-255, clamped)
export fn f32_to_uint8(dst: [*]u8, src: [*]const f32, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const clamped = @max(0.0, @min(1.0, src[i]));
        dst[i] = @intFromFloat(clamped * 255.0);
    }
}

// =============================================================================
// Memory Info Export
// =============================================================================

/// Get current WASM memory size in pages (64KB each)
export fn wasm_memory_pages() usize {
    return @wasmMemorySize(0);
}

/// Grow WASM memory by N pages. Returns previous size or -1 on failure.
export fn wasm_memory_grow(pages: usize) isize {
    return @wasmMemoryGrow(0, pages);
}
