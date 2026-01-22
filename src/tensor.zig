//! Multi-dimensional Tensor implementation for ONNX inference.
//! Supports various data types and memory layouts for edge computing.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Maximum number of dimensions supported
pub const MAX_DIMS: usize = 8;

/// Calculate total number of elements from a shape slice
/// WARNING: This function can overflow silently. For checked arithmetic, use `calcNumelChecked`.
pub inline fn calcNumel(shape: []const usize) usize {
    var n: usize = 1;
    for (shape) |d| n *= d;
    return n;
}

/// Error type for checked arithmetic operations
pub const NumelError = error{
    /// The number of elements would overflow usize
    Overflow,
    /// Zero dimension in shape (results in zero elements)
    ZeroDimension,
};

/// Calculate total number of elements with overflow detection.
/// Returns an error if the multiplication would overflow or if any dimension is zero.
pub fn calcNumelChecked(shape: []const usize) NumelError!usize {
    if (shape.len == 0) return 0;

    var n: usize = 1;
    for (shape) |d| {
        if (d == 0) return NumelError.ZeroDimension;
        const result = @mulWithOverflow(n, d);
        if (result[1] != 0) return NumelError.Overflow;
        n = result[0];
    }
    return n;
}

/// Calculate total number of elements, returning null on overflow or zero dimension.
/// Useful for cases where you want to silently handle edge cases.
pub fn calcNumelOrNull(shape: []const usize) ?usize {
    return calcNumelChecked(shape) catch null;
}

/// Supported tensor data types (aligned with ONNX spec)
pub const DataType = enum(u8) {
    f32 = 1,
    f64 = 11,
    f16 = 10,
    i8 = 3,
    i16 = 5,
    i32 = 6,
    i64 = 7,
    u8 = 2,
    u16 = 4,
    u32 = 12,
    u64 = 13,
    bool_ = 9,

    pub fn byteSize(self: DataType) usize {
        return switch (self) {
            .f32, .i32, .u32 => 4,
            .f64, .i64, .u64 => 8,
            .f16, .i16, .u16 => 2,
            .i8, .u8, .bool_ => 1,
        };
    }
};

/// Shape representation with fixed maximum dimensions
pub const Shape = struct {
    dims: [MAX_DIMS]usize = [_]usize{0} ** MAX_DIMS,
    ndim: usize = 0,

    pub fn init(dimensions: []const usize) Shape {
        var shape = Shape{};
        shape.ndim = @min(dimensions.len, MAX_DIMS);
        for (0..shape.ndim) |i| {
            shape.dims[i] = dimensions[i];
        }
        return shape;
    }

    pub fn fromSlice(dimensions: []const usize) Shape {
        return init(dimensions);
    }

    /// Returns the total number of elements
    pub fn numel(self: Shape) usize {
        if (self.ndim == 0) return 0;
        var total: usize = 1;
        for (0..self.ndim) |i| {
            total *= self.dims[i];
        }
        return total;
    }

    /// Get dimension at index
    pub fn dim(self: Shape, index: usize) usize {
        if (index >= self.ndim) return 1; // Broadcasting behavior
        return self.dims[index];
    }

    /// Returns slice of valid dimensions
    pub fn slice(self: *const Shape) []const usize {
        return self.dims[0..self.ndim];
    }

    /// Check if two shapes are equal
    pub fn eql(self: Shape, other: Shape) bool {
        if (self.ndim != other.ndim) return false;
        for (0..self.ndim) |i| {
            if (self.dims[i] != other.dims[i]) return false;
        }
        return true;
    }

    /// Check if shapes are broadcastable
    pub fn broadcastable(self: Shape, other: Shape) bool {
        const max_ndim = @max(self.ndim, other.ndim);
        for (0..max_ndim) |i| {
            const d1 = if (i < self.ndim) self.dims[self.ndim - 1 - i] else 1;
            const d2 = if (i < other.ndim) other.dims[other.ndim - 1 - i] else 1;
            if (d1 != d2 and d1 != 1 and d2 != 1) return false;
        }
        return true;
    }

    /// Compute broadcast result shape
    pub fn broadcast(self: Shape, other: Shape) ?Shape {
        if (!self.broadcastable(other)) return null;

        var result = Shape{};
        result.ndim = @max(self.ndim, other.ndim);

        for (0..result.ndim) |i| {
            const idx = result.ndim - 1 - i;
            const d1 = if (i < self.ndim) self.dims[self.ndim - 1 - i] else 1;
            const d2 = if (i < other.ndim) other.dims[other.ndim - 1 - i] else 1;
            result.dims[idx] = @max(d1, d2);
        }

        return result;
    }
};

/// Strides for memory layout
pub const Strides = struct {
    data: [MAX_DIMS]usize = [_]usize{0} ** MAX_DIMS,
    ndim: usize = 0,

    /// Compute contiguous (row-major) strides for a shape
    pub fn contiguous(shape: Shape) Strides {
        var strides = Strides{};
        strides.ndim = shape.ndim;

        if (shape.ndim == 0) return strides;

        strides.data[shape.ndim - 1] = 1;
        var i: usize = shape.ndim - 1;
        while (i > 0) {
            i -= 1;
            strides.data[i] = strides.data[i + 1] * shape.dims[i + 1];
        }

        return strides;
    }

    /// Get stride at index
    pub fn get(self: Strides, index: usize) usize {
        if (index >= self.ndim) return 0;
        return self.data[index];
    }

    /// Returns slice of valid strides
    pub fn slice(self: *const Strides) []const usize {
        return self.data[0..self.ndim];
    }
};

/// Generic Tensor type
pub fn Tensor(comptime T: type) type {
    return struct {
        const Self = @This();

        data: []T,
        shape: Shape,
        strides: Strides,
        allocator: ?Allocator,
        owns_data: bool,

        /// Create a new tensor with allocated storage (zeroed)
        pub fn init(allocator: Allocator, shape_dims: []const usize) !Self {
            const tensor = try initUninit(allocator, shape_dims);
            @memset(tensor.data, 0);
            return tensor;
        }

        /// Create a new tensor with allocated but uninitialized storage.
        /// Use this for performance when you'll immediately overwrite all values.
        pub fn initUninit(allocator: Allocator, shape_dims: []const usize) !Self {
            const shape = Shape.init(shape_dims);
            const num_elements = calcNumel(shape_dims);
            const data = try allocator.alloc(T, num_elements);

            return Self{
                .data = data,
                .shape = shape,
                .strides = Strides.contiguous(shape),
                .allocator = allocator,
                .owns_data = true,
            };
        }

        /// Create tensor from existing data (no copy, no ownership)
        pub fn fromSlice(data: []T, shape_dims: []const usize) Self {
            const shape = Shape.init(shape_dims);
            return Self{
                .data = data,
                .shape = shape,
                .strides = Strides.contiguous(shape),
                .allocator = null,
                .owns_data = false,
            };
        }

        /// Create tensor filled with a specific value
        pub fn full(allocator: Allocator, shape_dims: []const usize, value: T) !Self {
            const tensor = try init(allocator, shape_dims);
            @memset(tensor.data, value);
            return tensor;
        }

        /// Create tensor filled with zeros
        pub fn zeros(allocator: Allocator, shape_dims: []const usize) !Self {
            return init(allocator, shape_dims);
        }

        /// Create tensor filled with ones
        pub fn ones(allocator: Allocator, shape_dims: []const usize) !Self {
            return full(allocator, shape_dims, 1);
        }

        /// Free tensor memory if owned
        pub fn deinit(self: *Self) void {
            if (self.owns_data) {
                if (self.allocator) |alloc| {
                    alloc.free(self.data);
                }
            }
            self.data = &[_]T{};
            self.owns_data = false;
        }

        /// Get total number of elements
        pub fn numel(self: Self) usize {
            return self.shape.numel();
        }

        /// Get number of dimensions
        pub fn ndim(self: Self) usize {
            return self.shape.ndim;
        }

        /// Convert multi-dimensional indices to flat offset
        pub fn flatIndex(self: Self, indices: []const usize) usize {
            var offset: usize = 0;
            for (0..@min(indices.len, self.shape.ndim)) |i| {
                offset += indices[i] * self.strides.data[i];
            }
            return offset;
        }

        /// Get element at indices
        pub fn get(self: Self, indices: []const usize) T {
            const offset = self.flatIndex(indices);
            return self.data[offset];
        }

        /// Set element at indices
        pub fn set(self: *Self, indices: []const usize, value: T) void {
            const offset = self.flatIndex(indices);
            self.data[offset] = value;
        }

        /// Get element at flat index
        pub fn getFlatIndex(self: Self, index: usize) T {
            return self.data[index];
        }

        /// Set element at flat index
        pub fn setFlatIndex(self: *Self, index: usize, value: T) void {
            self.data[index] = value;
        }

        /// Convert flat index to multi-dimensional indices
        pub fn unravelIndex(self: Self, flat_idx: usize) [MAX_DIMS]usize {
            var indices: [MAX_DIMS]usize = [_]usize{0} ** MAX_DIMS;
            var remaining = flat_idx;

            for (0..self.shape.ndim) |i| {
                indices[i] = remaining / self.strides.data[i];
                remaining = remaining % self.strides.data[i];
            }

            return indices;
        }

        /// Check if tensor is contiguous in memory
        pub fn isContiguous(self: Self) bool {
            const expected = Strides.contiguous(self.shape);
            for (0..self.shape.ndim) |i| {
                if (self.strides.data[i] != expected.data[i]) return false;
            }
            return true;
        }

        /// Reshape tensor (returns new view if contiguous, error otherwise)
        pub fn reshape(self: Self, new_shape: []const usize) !Self {
            if (calcNumel(new_shape) != self.numel()) {
                return error.ShapeMismatch;
            }

            if (!self.isContiguous()) {
                return error.NotContiguous;
            }

            var result = self;
            result.shape = Shape.init(new_shape);
            result.strides = Strides.contiguous(result.shape);
            result.owns_data = false; // View doesn't own data
            return result;
        }

        /// Clone tensor with new allocation
        pub fn clone(self: Self, allocator: Allocator) !Self {
            const new_tensor = try init(allocator, self.shape.slice());
            @memcpy(new_tensor.data, self.data);
            return new_tensor;
        }

        /// Create a zero-copy view of external data.
        /// The caller must ensure the data outlives the view.
        /// Useful for wrapping ONNX Runtime tensor data without copying.
        pub fn view(data: []T, shape_dims: []const usize) Self {
            const shape = Shape.init(shape_dims);
            return Self{
                .data = data,
                .shape = shape,
                .strides = Strides.contiguous(shape),
                .allocator = null,
                .owns_data = false,
            };
        }

        /// Create a zero-copy view from a raw pointer and element count.
        /// Useful for interfacing with C APIs.
        pub fn viewFromPtr(ptr: [*]T, num_elements: usize, shape_dims: []const usize) Self {
            return view(ptr[0..num_elements], shape_dims);
        }

        /// Create a sub-view (slice) of an existing tensor.
        /// Returns a view into a contiguous region starting at `start_idx`.
        pub fn subview(self: Self, start_idx: usize, length: usize, new_shape: []const usize) !Self {
            if (start_idx + length > self.data.len) {
                return error.OutOfBounds;
            }
            if (length < calcNumel(new_shape)) {
                return error.ShapeMismatch;
            }

            return Self{
                .data = self.data[start_idx..][0..length],
                .shape = Shape.init(new_shape),
                .strides = Strides.contiguous(Shape.init(new_shape)),
                .allocator = null,
                .owns_data = false,
            };
        }

        /// Check if this tensor is a view (doesn't own its data)
        pub fn isView(self: Self) bool {
            return !self.owns_data;
        }

        /// Get raw data pointer for C interop
        pub fn dataPtr(self: Self) [*]T {
            return self.data.ptr;
        }

        /// Get raw data pointer as anyopaque for generic C interop
        pub fn dataOpaque(self: Self) *anyopaque {
            return @ptrCast(self.data.ptr);
        }

        /// Iterator for traversing tensor elements
        pub const Iterator = struct {
            tensor: *const Self,
            index: usize,

            pub fn next(self: *Iterator) ?T {
                if (self.index >= self.tensor.numel()) return null;
                const value = self.tensor.data[self.index];
                self.index += 1;
                return value;
            }

            pub fn reset(self: *Iterator) void {
                self.index = 0;
            }
        };

        /// Get iterator over tensor elements
        pub fn iterator(self: *const Self) Iterator {
            return Iterator{ .tensor = self, .index = 0 };
        }
    };
}

// Convenience type aliases
pub const TensorF32 = Tensor(f32);
pub const TensorF64 = Tensor(f64);
pub const TensorI32 = Tensor(i32);
pub const TensorI64 = Tensor(i64);
pub const TensorU8 = Tensor(u8);

// =============================================================================
// SIMD-optimized Operations
// =============================================================================

/// SIMD vector width for f32 operations (128-bit = 4 floats)
const SIMD_F32_WIDTH = 4;
const SimdF32 = @Vector(SIMD_F32_WIDTH, f32);

/// SIMD-optimized tensor operations for f32 tensors
pub const SimdOps = struct {
    /// Element-wise addition: dst = a + b
    pub fn add(dst: []f32, a: []const f32, b: []const f32) void {
        const n = @min(dst.len, @min(a.len, b.len));
        const simd_end = n - (n % SIMD_F32_WIDTH);

        // SIMD loop
        var i: usize = 0;
        while (i < simd_end) : (i += SIMD_F32_WIDTH) {
            const va: SimdF32 = a[i..][0..SIMD_F32_WIDTH].*;
            const vb: SimdF32 = b[i..][0..SIMD_F32_WIDTH].*;
            dst[i..][0..SIMD_F32_WIDTH].* = va + vb;
        }

        // Scalar remainder
        while (i < n) : (i += 1) {
            dst[i] = a[i] + b[i];
        }
    }

    /// Element-wise subtraction: dst = a - b
    pub fn sub(dst: []f32, a: []const f32, b: []const f32) void {
        const n = @min(dst.len, @min(a.len, b.len));
        const simd_end = n - (n % SIMD_F32_WIDTH);

        var i: usize = 0;
        while (i < simd_end) : (i += SIMD_F32_WIDTH) {
            const va: SimdF32 = a[i..][0..SIMD_F32_WIDTH].*;
            const vb: SimdF32 = b[i..][0..SIMD_F32_WIDTH].*;
            dst[i..][0..SIMD_F32_WIDTH].* = va - vb;
        }

        while (i < n) : (i += 1) {
            dst[i] = a[i] - b[i];
        }
    }

    /// Element-wise multiplication: dst = a * b
    pub fn mul(dst: []f32, a: []const f32, b: []const f32) void {
        const n = @min(dst.len, @min(a.len, b.len));
        const simd_end = n - (n % SIMD_F32_WIDTH);

        var i: usize = 0;
        while (i < simd_end) : (i += SIMD_F32_WIDTH) {
            const va: SimdF32 = a[i..][0..SIMD_F32_WIDTH].*;
            const vb: SimdF32 = b[i..][0..SIMD_F32_WIDTH].*;
            dst[i..][0..SIMD_F32_WIDTH].* = va * vb;
        }

        while (i < n) : (i += 1) {
            dst[i] = a[i] * b[i];
        }
    }

    /// Element-wise division: dst = a / b
    pub fn div(dst: []f32, a: []const f32, b: []const f32) void {
        const n = @min(dst.len, @min(a.len, b.len));
        const simd_end = n - (n % SIMD_F32_WIDTH);

        var i: usize = 0;
        while (i < simd_end) : (i += SIMD_F32_WIDTH) {
            const va: SimdF32 = a[i..][0..SIMD_F32_WIDTH].*;
            const vb: SimdF32 = b[i..][0..SIMD_F32_WIDTH].*;
            dst[i..][0..SIMD_F32_WIDTH].* = va / vb;
        }

        while (i < n) : (i += 1) {
            dst[i] = a[i] / b[i];
        }
    }

    /// Scale by constant: dst = a * scalar
    pub fn scale(dst: []f32, a: []const f32, scalar: f32) void {
        const n = @min(dst.len, a.len);
        const simd_end = n - (n % SIMD_F32_WIDTH);
        const vs: SimdF32 = @splat(scalar);

        var i: usize = 0;
        while (i < simd_end) : (i += SIMD_F32_WIDTH) {
            const va: SimdF32 = a[i..][0..SIMD_F32_WIDTH].*;
            dst[i..][0..SIMD_F32_WIDTH].* = va * vs;
        }

        while (i < n) : (i += 1) {
            dst[i] = a[i] * scalar;
        }
    }

    /// Add constant: dst = a + scalar
    pub fn addScalar(dst: []f32, a: []const f32, scalar: f32) void {
        const n = @min(dst.len, a.len);
        const simd_end = n - (n % SIMD_F32_WIDTH);
        const vs: SimdF32 = @splat(scalar);

        var i: usize = 0;
        while (i < simd_end) : (i += SIMD_F32_WIDTH) {
            const va: SimdF32 = a[i..][0..SIMD_F32_WIDTH].*;
            dst[i..][0..SIMD_F32_WIDTH].* = va + vs;
        }

        while (i < n) : (i += 1) {
            dst[i] = a[i] + scalar;
        }
    }

    /// ReLU activation: dst = max(0, a)
    pub fn relu(dst: []f32, a: []const f32) void {
        const n = @min(dst.len, a.len);
        const simd_end = n - (n % SIMD_F32_WIDTH);
        const zero: SimdF32 = @splat(0.0);

        var i: usize = 0;
        while (i < simd_end) : (i += SIMD_F32_WIDTH) {
            const va: SimdF32 = a[i..][0..SIMD_F32_WIDTH].*;
            dst[i..][0..SIMD_F32_WIDTH].* = @max(zero, va);
        }

        while (i < n) : (i += 1) {
            dst[i] = @max(0.0, a[i]);
        }
    }

    /// Sum all elements
    pub fn sum(a: []const f32) f32 {
        const n = a.len;
        const simd_end = n - (n % SIMD_F32_WIDTH);

        var acc: SimdF32 = @splat(0.0);
        var i: usize = 0;

        while (i < simd_end) : (i += SIMD_F32_WIDTH) {
            const va: SimdF32 = a[i..][0..SIMD_F32_WIDTH].*;
            acc += va;
        }

        // Reduce SIMD vector
        var total: f32 = @reduce(.Add, acc);

        // Scalar remainder
        while (i < n) : (i += 1) {
            total += a[i];
        }

        return total;
    }

    /// Dot product
    pub fn dot(a: []const f32, b: []const f32) f32 {
        const n = @min(a.len, b.len);
        const simd_end = n - (n % SIMD_F32_WIDTH);

        var acc: SimdF32 = @splat(0.0);
        var i: usize = 0;

        while (i < simd_end) : (i += SIMD_F32_WIDTH) {
            const va: SimdF32 = a[i..][0..SIMD_F32_WIDTH].*;
            const vb: SimdF32 = b[i..][0..SIMD_F32_WIDTH].*;
            acc += va * vb;
        }

        var total: f32 = @reduce(.Add, acc);

        while (i < n) : (i += 1) {
            total += a[i] * b[i];
        }

        return total;
    }

    /// Find maximum value
    pub fn max(a: []const f32) f32 {
        if (a.len == 0) return -std.math.inf(f32);

        const n = a.len;
        const simd_end = n - (n % SIMD_F32_WIDTH);

        var max_vec: SimdF32 = @splat(-std.math.inf(f32));
        var i: usize = 0;

        while (i < simd_end) : (i += SIMD_F32_WIDTH) {
            const va: SimdF32 = a[i..][0..SIMD_F32_WIDTH].*;
            max_vec = @max(max_vec, va);
        }

        var result: f32 = @reduce(.Max, max_vec);

        while (i < n) : (i += 1) {
            result = @max(result, a[i]);
        }

        return result;
    }

    /// Find minimum value
    pub fn min(a: []const f32) f32 {
        if (a.len == 0) return std.math.inf(f32);

        const n = a.len;
        const simd_end = n - (n % SIMD_F32_WIDTH);

        var min_vec: SimdF32 = @splat(std.math.inf(f32));
        var i: usize = 0;

        while (i < simd_end) : (i += SIMD_F32_WIDTH) {
            const va: SimdF32 = a[i..][0..SIMD_F32_WIDTH].*;
            min_vec = @min(min_vec, va);
        }

        var result: f32 = @reduce(.Min, min_vec);

        while (i < n) : (i += 1) {
            result = @min(result, a[i]);
        }

        return result;
    }

    /// Fused multiply-add: dst = a * b + c
    pub fn fma(dst: []f32, a: []const f32, b: []const f32, c: []const f32) void {
        const n = @min(dst.len, @min(a.len, @min(b.len, c.len)));
        const simd_end = n - (n % SIMD_F32_WIDTH);

        var i: usize = 0;
        while (i < simd_end) : (i += SIMD_F32_WIDTH) {
            const va: SimdF32 = a[i..][0..SIMD_F32_WIDTH].*;
            const vb: SimdF32 = b[i..][0..SIMD_F32_WIDTH].*;
            const vc: SimdF32 = c[i..][0..SIMD_F32_WIDTH].*;
            dst[i..][0..SIMD_F32_WIDTH].* = @mulAdd(SimdF32, va, vb, vc);
        }

        while (i < n) : (i += 1) {
            dst[i] = @mulAdd(f32, a[i], b[i], c[i]);
        }
    }

    /// Copy data
    pub fn copy(dst: []f32, src: []const f32) void {
        const n = @min(dst.len, src.len);
        @memcpy(dst[0..n], src[0..n]);
    }

    /// Fill with value
    pub fn fill(dst: []f32, value: f32) void {
        @memset(dst, value);
    }

    // =========================================================================
    // Activation Functions
    // =========================================================================

    /// Sigmoid activation: dst = 1 / (1 + exp(-x))
    /// SIMD-optimized with scalar fallback
    pub fn sigmoid(dst: []f32, a: []const f32) void {
        const n = @min(dst.len, a.len);

        // Scalar implementation (SIMD exp is complex, scalar is often faster for small arrays)
        for (0..n) |i| {
            dst[i] = 1.0 / (1.0 + @exp(-a[i]));
        }
    }

    /// Tanh activation: dst = tanh(x)
    /// Uses identity: tanh(x) = 2*sigmoid(2x) - 1
    pub fn tanh_(dst: []f32, a: []const f32) void {
        const n = @min(dst.len, a.len);

        for (0..n) |i| {
            const exp_2x = @exp(2.0 * a[i]);
            dst[i] = (exp_2x - 1.0) / (exp_2x + 1.0);
        }
    }

    /// GELU activation: dst = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    /// Gaussian Error Linear Unit - used in transformers
    pub fn gelu(dst: []f32, a: []const f32) void {
        const n = @min(dst.len, a.len);
        const sqrt_2_over_pi: f32 = 0.7978845608028654; // sqrt(2/pi)
        const coeff: f32 = 0.044715;

        for (0..n) |i| {
            const x = a[i];
            const x3 = x * x * x;
            const inner = sqrt_2_over_pi * (x + coeff * x3);
            const exp_2inner = @exp(2.0 * inner);
            const tanh_val = (exp_2inner - 1.0) / (exp_2inner + 1.0);
            dst[i] = x * 0.5 * (1.0 + tanh_val);
        }
    }

    /// Softmax activation: dst = exp(x) / sum(exp(x))
    /// Numerically stable version that subtracts max before exp
    pub fn softmax(dst: []f32, a: []const f32) void {
        const n = @min(dst.len, a.len);
        if (n == 0) return;

        // Find max for numerical stability
        const max_val = max(a[0..n]);

        // Compute exp(x - max) and sum
        var sum_exp: f32 = 0.0;
        for (0..n) |i| {
            dst[i] = @exp(a[i] - max_val);
            sum_exp += dst[i];
        }

        // Normalize
        if (sum_exp > 0.0) {
            const inv_sum = 1.0 / sum_exp;
            for (0..n) |i| {
                dst[i] *= inv_sum;
            }
        }
    }

    /// Log-Softmax activation: dst = log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))
    /// Numerically stable version
    pub fn logSoftmax(dst: []f32, a: []const f32) void {
        const n = @min(dst.len, a.len);
        if (n == 0) return;

        // Find max for numerical stability
        const max_val = max(a[0..n]);

        // Compute log(sum(exp(x - max)))
        var sum_exp: f32 = 0.0;
        for (0..n) |i| {
            sum_exp += @exp(a[i] - max_val);
        }
        const log_sum_exp = @log(sum_exp);

        // Compute log_softmax = x - max - log_sum_exp
        for (0..n) |i| {
            dst[i] = a[i] - max_val - log_sum_exp;
        }
    }

    /// Argmax: returns the index of the maximum value
    pub fn argmax(a: []const f32) usize {
        if (a.len == 0) return 0;

        var max_idx: usize = 0;
        var max_val: f32 = a[0];

        for (1..a.len) |i| {
            if (a[i] > max_val) {
                max_val = a[i];
                max_idx = i;
            }
        }

        return max_idx;
    }

    /// Argmin: returns the index of the minimum value
    pub fn argmin(a: []const f32) usize {
        if (a.len == 0) return 0;

        var min_idx: usize = 0;
        var min_val: f32 = a[0];

        for (1..a.len) |i| {
            if (a[i] < min_val) {
                min_val = a[i];
                min_idx = i;
            }
        }

        return min_idx;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "Shape - basic operations" {
    const shape = Shape.init(&[_]usize{ 2, 3, 4 });
    try std.testing.expectEqual(@as(usize, 3), shape.ndim);
    try std.testing.expectEqual(@as(usize, 24), shape.numel());
    try std.testing.expectEqual(@as(usize, 2), shape.dim(0));
    try std.testing.expectEqual(@as(usize, 3), shape.dim(1));
    try std.testing.expectEqual(@as(usize, 4), shape.dim(2));
}

test "Shape - equality" {
    const s1 = Shape.init(&[_]usize{ 2, 3, 4 });
    const s2 = Shape.init(&[_]usize{ 2, 3, 4 });
    const s3 = Shape.init(&[_]usize{ 2, 3, 5 });

    try std.testing.expect(s1.eql(s2));
    try std.testing.expect(!s1.eql(s3));
}

test "Shape - broadcasting" {
    const s1 = Shape.init(&[_]usize{ 3, 1 });
    const s2 = Shape.init(&[_]usize{ 1, 4 });

    try std.testing.expect(s1.broadcastable(s2));

    const result = s1.broadcast(s2).?;
    try std.testing.expectEqual(@as(usize, 3), result.dim(0));
    try std.testing.expectEqual(@as(usize, 4), result.dim(1));
}

test "Strides - contiguous" {
    const shape = Shape.init(&[_]usize{ 2, 3, 4 });
    const strides = Strides.contiguous(shape);

    try std.testing.expectEqual(@as(usize, 12), strides.get(0)); // 3*4
    try std.testing.expectEqual(@as(usize, 4), strides.get(1)); // 4
    try std.testing.expectEqual(@as(usize, 1), strides.get(2)); // 1
}

test "Tensor - init and deinit" {
    const allocator = std.testing.allocator;
    var tensor = try TensorF32.init(allocator, &[_]usize{ 2, 3 });
    defer tensor.deinit();

    try std.testing.expectEqual(@as(usize, 6), tensor.numel());
    try std.testing.expectEqual(@as(usize, 2), tensor.ndim());
}

test "Tensor - get and set" {
    const allocator = std.testing.allocator;
    var tensor = try TensorF32.init(allocator, &[_]usize{ 2, 3 });
    defer tensor.deinit();

    tensor.set(&[_]usize{ 0, 0 }, 1.0);
    tensor.set(&[_]usize{ 0, 1 }, 2.0);
    tensor.set(&[_]usize{ 1, 2 }, 5.0);

    try std.testing.expectEqual(@as(f32, 1.0), tensor.get(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 2.0), tensor.get(&[_]usize{ 0, 1 }));
    try std.testing.expectEqual(@as(f32, 5.0), tensor.get(&[_]usize{ 1, 2 }));
}

test "Tensor - zeros and ones" {
    const allocator = std.testing.allocator;

    var zeros_t = try TensorF32.zeros(allocator, &[_]usize{ 2, 2 });
    defer zeros_t.deinit();

    var ones_t = try TensorF32.ones(allocator, &[_]usize{ 2, 2 });
    defer ones_t.deinit();

    for (zeros_t.data) |v| {
        try std.testing.expectEqual(@as(f32, 0.0), v);
    }

    for (ones_t.data) |v| {
        try std.testing.expectEqual(@as(f32, 1.0), v);
    }
}

test "Tensor - fromSlice" {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const tensor = TensorF32.fromSlice(&data, &[_]usize{ 2, 3 });

    try std.testing.expectEqual(@as(f32, 1.0), tensor.get(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 4.0), tensor.get(&[_]usize{ 1, 0 }));
    try std.testing.expectEqual(@as(f32, 6.0), tensor.get(&[_]usize{ 1, 2 }));
}

test "Tensor - reshape" {
    const allocator = std.testing.allocator;
    var tensor = try TensorF32.init(allocator, &[_]usize{ 2, 3 });
    defer tensor.deinit();

    // Fill with sequential values
    for (0..6) |i| {
        tensor.data[i] = @floatFromInt(i);
    }

    const reshaped = try tensor.reshape(&[_]usize{ 3, 2 });

    try std.testing.expectEqual(@as(usize, 3), reshaped.shape.dim(0));
    try std.testing.expectEqual(@as(usize, 2), reshaped.shape.dim(1));
    try std.testing.expectEqual(@as(f32, 0.0), reshaped.get(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 5.0), reshaped.get(&[_]usize{ 2, 1 }));
}

test "Tensor - iterator" {
    const allocator = std.testing.allocator;
    var tensor = try TensorF32.init(allocator, &[_]usize{ 2, 2 });
    defer tensor.deinit();

    tensor.data[0] = 1.0;
    tensor.data[1] = 2.0;
    tensor.data[2] = 3.0;
    tensor.data[3] = 4.0;

    var iter = tensor.iterator();
    var sum: f32 = 0.0;
    while (iter.next()) |v| {
        sum += v;
    }

    try std.testing.expectEqual(@as(f32, 10.0), sum);
}

test "Tensor - clone" {
    const allocator = std.testing.allocator;
    var tensor = try TensorF32.init(allocator, &[_]usize{ 2, 2 });
    defer tensor.deinit();

    tensor.data[0] = 42.0;

    var cloned = try tensor.clone(allocator);
    defer cloned.deinit();

    try std.testing.expectEqual(@as(f32, 42.0), cloned.data[0]);

    // Modify original, cloned should be unchanged
    tensor.data[0] = 100.0;
    try std.testing.expectEqual(@as(f32, 42.0), cloned.data[0]);
}

test "Tensor - unravel index" {
    const allocator = std.testing.allocator;
    var tensor = try TensorF32.init(allocator, &[_]usize{ 2, 3, 4 });
    defer tensor.deinit();

    // flat index 0 -> (0, 0, 0)
    const idx0 = tensor.unravelIndex(0);
    try std.testing.expectEqual(@as(usize, 0), idx0[0]);
    try std.testing.expectEqual(@as(usize, 0), idx0[1]);
    try std.testing.expectEqual(@as(usize, 0), idx0[2]);

    // flat index 5 -> (0, 1, 1)
    const idx5 = tensor.unravelIndex(5);
    try std.testing.expectEqual(@as(usize, 0), idx5[0]);
    try std.testing.expectEqual(@as(usize, 1), idx5[1]);
    try std.testing.expectEqual(@as(usize, 1), idx5[2]);

    // flat index 23 -> (1, 2, 3) - last element
    const idx23 = tensor.unravelIndex(23);
    try std.testing.expectEqual(@as(usize, 1), idx23[0]);
    try std.testing.expectEqual(@as(usize, 2), idx23[1]);
    try std.testing.expectEqual(@as(usize, 3), idx23[2]);
}

test "Tensor - view" {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const tensor = TensorF32.view(&data, &[_]usize{ 2, 3 });

    try std.testing.expect(tensor.isView());
    try std.testing.expectEqual(@as(usize, 6), tensor.numel());
    try std.testing.expectEqual(@as(f32, 1.0), tensor.get(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 6.0), tensor.get(&[_]usize{ 1, 2 }));

    // Modifying view modifies original
    var tensor_mut = tensor;
    tensor_mut.owns_data = false; // Already false, but explicit
    tensor_mut.data[0] = 99.0;
    try std.testing.expectEqual(@as(f32, 99.0), data[0]);
}

test "Tensor - subview" {
    const allocator = std.testing.allocator;
    var tensor = try TensorF32.init(allocator, &[_]usize{24});
    defer tensor.deinit();

    for (0..24) |i| {
        tensor.data[i] = @floatFromInt(i);
    }

    const sub = try tensor.subview(6, 12, &[_]usize{ 3, 4 });
    try std.testing.expect(sub.isView());
    try std.testing.expectEqual(@as(usize, 12), sub.numel());
    try std.testing.expectEqual(@as(f32, 6.0), sub.data[0]);
    try std.testing.expectEqual(@as(f32, 17.0), sub.data[11]);
}

// =============================================================================
// SIMD Tests
// =============================================================================

test "SimdOps - add" {
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    var b = [_]f32{ 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5 };
    var dst: [9]f32 = undefined;

    SimdOps.add(&dst, &a, &b);

    try std.testing.expectApproxEqAbs(@as(f32, 1.5), dst[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.5), dst[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 17.5), dst[8], 1e-6);
}

test "SimdOps - mul" {
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var b = [_]f32{ 2.0, 2.0, 2.0, 2.0, 2.0 };
    var dst: [5]f32 = undefined;

    SimdOps.mul(&dst, &a, &b);

    try std.testing.expectApproxEqAbs(@as(f32, 2.0), dst[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), dst[4], 1e-6);
}

test "SimdOps - scale" {
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var dst: [6]f32 = undefined;

    SimdOps.scale(&dst, &a, 0.5);

    try std.testing.expectApproxEqAbs(@as(f32, 0.5), dst[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), dst[5], 1e-6);
}

test "SimdOps - relu" {
    var a = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0, -0.5, 0.5, 3.0 };
    var dst: [8]f32 = undefined;

    SimdOps.relu(&dst, &a);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), dst[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), dst[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), dst[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), dst[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), dst[4], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), dst[6], 1e-6);
}

test "SimdOps - sum" {
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    const result = SimdOps.sum(&a);
    try std.testing.expectApproxEqAbs(@as(f32, 45.0), result, 1e-6);
}

test "SimdOps - dot" {
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var b = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0 };
    const result = SimdOps.dot(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 15.0), result, 1e-6);
}

test "SimdOps - max/min" {
    var a = [_]f32{ 3.0, -1.0, 5.0, 2.0, -3.0, 4.0, 1.0, 0.0 };

    const max_val = SimdOps.max(&a);
    const min_val = SimdOps.min(&a);

    try std.testing.expectApproxEqAbs(@as(f32, 5.0), max_val, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -3.0), min_val, 1e-6);
}

test "SimdOps - fma" {
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var b = [_]f32{ 2.0, 2.0, 2.0, 2.0 };
    var c = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    var dst: [4]f32 = undefined;

    SimdOps.fma(&dst, &a, &b, &c);

    // a * b + c = 1*2+0.5, 2*2+0.5, 3*2+0.5, 4*2+0.5 = 2.5, 4.5, 6.5, 8.5
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), dst[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.5), dst[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.5), dst[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 8.5), dst[3], 1e-6);
}

// =============================================================================
// Checked Arithmetic Tests
// =============================================================================

test "calcNumelChecked - normal shapes" {
    // Empty shape
    try std.testing.expectEqual(@as(usize, 0), try calcNumelChecked(&[_]usize{}));

    // Single dimension
    try std.testing.expectEqual(@as(usize, 5), try calcNumelChecked(&[_]usize{5}));

    // Multiple dimensions
    try std.testing.expectEqual(@as(usize, 24), try calcNumelChecked(&[_]usize{ 2, 3, 4 }));

    // Large but valid shape
    try std.testing.expectEqual(@as(usize, 1000000), try calcNumelChecked(&[_]usize{ 1000, 1000 }));
}

test "calcNumelChecked - zero dimension" {
    // Zero in first position
    try std.testing.expectError(NumelError.ZeroDimension, calcNumelChecked(&[_]usize{ 0, 3, 4 }));

    // Zero in middle
    try std.testing.expectError(NumelError.ZeroDimension, calcNumelChecked(&[_]usize{ 2, 0, 4 }));

    // Zero at end
    try std.testing.expectError(NumelError.ZeroDimension, calcNumelChecked(&[_]usize{ 2, 3, 0 }));

    // Single zero
    try std.testing.expectError(NumelError.ZeroDimension, calcNumelChecked(&[_]usize{0}));
}

test "calcNumelChecked - overflow detection" {
    // This shape would overflow on any platform
    const max = std.math.maxInt(usize);
    try std.testing.expectError(NumelError.Overflow, calcNumelChecked(&[_]usize{ max, 2 }));

    // Large dimensions that overflow
    try std.testing.expectError(NumelError.Overflow, calcNumelChecked(&[_]usize{ 1 << 32, 1 << 32 }));
}

test "calcNumelOrNull - returns null on error" {
    // Valid shape
    try std.testing.expectEqual(@as(?usize, 24), calcNumelOrNull(&[_]usize{ 2, 3, 4 }));

    // Zero dimension returns null
    try std.testing.expectEqual(@as(?usize, null), calcNumelOrNull(&[_]usize{ 0, 3, 4 }));

    // Overflow returns null
    const max = std.math.maxInt(usize);
    try std.testing.expectEqual(@as(?usize, null), calcNumelOrNull(&[_]usize{ max, 2 }));
}

test "Shape - zero dimension numel" {
    // Shape with zero dimension should return 0 elements
    const shape = Shape.init(&[_]usize{ 2, 0, 4 });
    try std.testing.expectEqual(@as(usize, 0), shape.numel());
}

test "Tensor - empty shape" {
    const allocator = std.testing.allocator;

    // Creating a tensor with empty shape should work
    var tensor = try TensorF32.init(allocator, &[_]usize{});
    defer tensor.deinit();

    try std.testing.expectEqual(@as(usize, 0), tensor.numel());
    try std.testing.expectEqual(@as(usize, 0), tensor.ndim());
}

// =============================================================================
// Activation Function Tests
// =============================================================================

test "SimdOps - sigmoid" {
    var a = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    var dst: [5]f32 = undefined;

    SimdOps.sigmoid(&dst, &a);

    // sigmoid(0) = 0.5
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), dst[2], 1e-6);
    // sigmoid(-x) + sigmoid(x) = 1
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), dst[0] + dst[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), dst[1] + dst[3], 1e-5);
    // Values should be in (0, 1)
    for (dst) |v| {
        try std.testing.expect(v > 0.0 and v < 1.0);
    }
}

test "SimdOps - tanh" {
    var a = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    var dst: [5]f32 = undefined;

    SimdOps.tanh_(&dst, &a);

    // tanh(0) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), dst[2], 1e-6);
    // tanh(-x) = -tanh(x)
    try std.testing.expectApproxEqAbs(-dst[4], dst[0], 1e-5);
    try std.testing.expectApproxEqAbs(-dst[3], dst[1], 1e-5);
    // Values should be in (-1, 1)
    for (dst) |v| {
        try std.testing.expect(v > -1.0 and v < 1.0);
    }
}

test "SimdOps - gelu" {
    var a = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    var dst: [5]f32 = undefined;

    SimdOps.gelu(&dst, &a);

    // gelu(0) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), dst[2], 1e-6);
    // gelu(x) ≈ x for large positive x
    try std.testing.expect(dst[4] > 1.5); // gelu(2) ≈ 1.95
    // gelu(x) ≈ 0 for large negative x
    try std.testing.expect(dst[0] < 0.0 and dst[0] > -0.2); // gelu(-2) ≈ -0.045
}

test "SimdOps - softmax" {
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var dst: [4]f32 = undefined;

    SimdOps.softmax(&dst, &a);

    // Sum should be 1
    var sum: f32 = 0.0;
    for (dst) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);

    // Values should be in (0, 1)
    for (dst) |v| {
        try std.testing.expect(v > 0.0 and v < 1.0);
    }

    // Larger input should have larger output
    try std.testing.expect(dst[3] > dst[2]);
    try std.testing.expect(dst[2] > dst[1]);
    try std.testing.expect(dst[1] > dst[0]);
}

test "SimdOps - softmax numerical stability" {
    // Test with large values that would overflow naive exp
    var a = [_]f32{ 1000.0, 1001.0, 1002.0 };
    var dst: [3]f32 = undefined;

    SimdOps.softmax(&dst, &a);

    // Should still sum to 1 despite large input
    var sum: f32 = 0.0;
    for (dst) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);

    // No NaN or Inf
    for (dst) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }
}

test "SimdOps - logSoftmax" {
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var dst: [4]f32 = undefined;

    SimdOps.logSoftmax(&dst, &a);

    // exp(log_softmax) should give softmax
    var softmax_dst: [4]f32 = undefined;
    SimdOps.softmax(&softmax_dst, &a);

    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(softmax_dst[i], @exp(dst[i]), 1e-5);
    }

    // log_softmax values should be negative (log of values < 1)
    for (dst) |v| {
        try std.testing.expect(v < 0.0);
    }
}

test "SimdOps - argmax" {
    const a = [_]f32{ 1.0, 5.0, 3.0, 2.0, 4.0 };
    try std.testing.expectEqual(@as(usize, 1), SimdOps.argmax(&a));

    const b = [_]f32{ 10.0, 5.0, 3.0 };
    try std.testing.expectEqual(@as(usize, 0), SimdOps.argmax(&b));

    const c = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    try std.testing.expectEqual(@as(usize, 4), SimdOps.argmax(&c));
}

test "SimdOps - argmin" {
    const a = [_]f32{ 1.0, 5.0, 3.0, 2.0, 4.0 };
    try std.testing.expectEqual(@as(usize, 0), SimdOps.argmin(&a));

    const b = [_]f32{ 10.0, 5.0, 3.0 };
    try std.testing.expectEqual(@as(usize, 2), SimdOps.argmin(&b));

    const c = [_]f32{ 5.0, 4.0, 3.0, 2.0, 1.0 };
    try std.testing.expectEqual(@as(usize, 4), SimdOps.argmin(&c));
}
