//! Tensor pool for efficient reuse of intermediate tensor allocations.
//!
//! During inference, many intermediate tensors are created and destroyed.
//! This pool maintains pre-allocated tensors organized by size, allowing
//! fast acquisition and release without repeated allocations.
//!
//! ## Example
//!
//! ```zig
//! var pool = TensorPool(f32).init(allocator);
//! defer pool.deinit();
//!
//! // Acquire tensor for computation
//! var tensor = try pool.acquire(&[_]usize{1, 256, 256});
//! defer pool.release(tensor);
//!
//! // Use tensor for computation...
//! ```

const std = @import("std");
const Allocator = std.mem.Allocator;
const tensor_mod = @import("tensor.zig");
const Shape = tensor_mod.Shape;
const Tensor = tensor_mod.Tensor;
const MAX_DIMS = tensor_mod.MAX_DIMS;
const calcNumel = tensor_mod.calcNumel;

/// Size bucket thresholds for finer-grained tensor pooling.
/// Uses smaller increments for small tensors (where waste is most significant)
/// and larger increments for big tensors (where percentage waste is lower).
///
/// Bucket layout:
/// - Buckets 0-7: Fine granularity (16, 32, 48, 64, 96, 128, 192, 256)
/// - Buckets 8-15: Medium granularity (384, 512, 768, 1K, 1.5K, 2K, 3K, 4K)
/// - Buckets 16-23: Power-of-2 (8K, 16K, 32K, 64K, 128K, 256K, 512K, 1M)
/// - Bucket 24: Overflow (>1M elements)
const BUCKET_THRESHOLDS = [_]usize{
    16, // 0: 1-16
    32, // 1: 17-32
    48, // 2: 33-48
    64, // 3: 49-64
    96, // 4: 65-96
    128, // 5: 97-128
    192, // 6: 129-192
    256, // 7: 193-256
    384, // 8: 257-384
    512, // 9: 385-512
    768, // 10: 513-768
    1024, // 11: 769-1024 (1K)
    1536, // 12: 1025-1536 (1.5K)
    2048, // 13: 1537-2048 (2K)
    3072, // 14: 2049-3072 (3K)
    4096, // 15: 3073-4096 (4K)
    8192, // 16: 4097-8192 (8K)
    16384, // 17: 8193-16384 (16K)
    32768, // 18: 16385-32768 (32K)
    65536, // 19: 32769-65536 (64K)
    131072, // 20: 65537-131072 (128K)
    262144, // 21: 131073-262144 (256K)
    524288, // 22: 262145-524288 (512K)
    1048576, // 23: 524289-1048576 (1M)
    // 24+: >1M elements (overflow bucket)
};

/// Tensor pool for efficient intermediate tensor management.
pub fn TensorPool(comptime T: type) type {
    return struct {
        const Self = @This();
        const TensorT = Tensor(T);

        const NUM_BUCKETS = BUCKET_THRESHOLDS.len + 1; // +1 for overflow bucket
        const MIN_BUCKET_SIZE = BUCKET_THRESHOLDS[0];

        /// Entry in the free list
        const PoolEntry = struct {
            tensor: TensorT,
            num_elements: usize,
            next: ?*PoolEntry,
        };

        allocator: Allocator,
        buckets: [NUM_BUCKETS]?*PoolEntry,

        // Statistics
        total_tensors: usize,
        active_tensors: usize,
        hit_count: usize,
        miss_count: usize,
        total_bytes: usize,

        /// Initialize empty tensor pool
        pub fn init(allocator: Allocator) Self {
            return Self{
                .allocator = allocator,
                .buckets = [_]?*PoolEntry{null} ** NUM_BUCKETS,
                .total_tensors = 0,
                .active_tensors = 0,
                .hit_count = 0,
                .miss_count = 0,
                .total_bytes = 0,
            };
        }

        /// Free all pooled tensors
        pub fn deinit(self: *Self) void {
            for (&self.buckets) |*bucket| {
                var entry = bucket.*;
                while (entry) |e| {
                    const next = e.next;
                    var tensor_copy = e.tensor;
                    tensor_copy.deinit();
                    self.allocator.destroy(e);
                    entry = next;
                }
                bucket.* = null;
            }
            self.total_tensors = 0;
            self.active_tensors = 0;
        }

        /// Get bucket index for element count using threshold array
        pub fn getBucketIndex(num_elements: usize) usize {
            // Binary search through thresholds
            for (BUCKET_THRESHOLDS, 0..) |threshold, i| {
                if (num_elements <= threshold) {
                    return i;
                }
            }
            // Overflow bucket for very large tensors
            return NUM_BUCKETS - 1;
        }

        /// Get minimum element count for bucket (max size of previous bucket + 1)
        fn getBucketMinSize(idx: usize) usize {
            if (idx == 0) return 1;
            if (idx > BUCKET_THRESHOLDS.len) return BUCKET_THRESHOLDS[BUCKET_THRESHOLDS.len - 1] + 1;
            return BUCKET_THRESHOLDS[idx - 1] + 1;
        }

        /// Get maximum element count for bucket
        fn getBucketMaxSize(idx: usize) usize {
            if (idx >= BUCKET_THRESHOLDS.len) {
                return std.math.maxInt(usize);
            }
            return BUCKET_THRESHOLDS[idx];
        }

        /// Acquire a tensor with at least the required shape.
        /// The tensor data is NOT zeroed - caller must initialize.
        pub fn acquire(self: *Self, shape: []const usize) !TensorT {
            const required_elements = calcNumel(shape);
            const bucket_idx = getBucketIndex(required_elements);

            // Look for suitable tensor in bucket
            var prev: ?*PoolEntry = null;
            var entry = self.buckets[bucket_idx];

            while (entry) |e| {
                if (e.num_elements >= required_elements) {
                    // Found suitable tensor - remove from list
                    if (prev) |p| {
                        p.next = e.next;
                    } else {
                        self.buckets[bucket_idx] = e.next;
                    }

                    // Reshape to requested shape
                    var tensor = e.tensor;
                    tensor.shape = Shape.init(shape);
                    tensor.strides = tensor_mod.Strides.contiguous(tensor.shape);
                    tensor.owns_data = true;

                    self.allocator.destroy(e);
                    self.active_tensors += 1;
                    self.hit_count += 1;

                    return tensor;
                }
                prev = e;
                entry = e.next;
            }

            // No suitable tensor found - allocate new
            const bucket_min = getBucketMinSize(bucket_idx);
            const alloc_size = @max(required_elements, bucket_min);

            var tensor = try TensorT.init(self.allocator, &[_]usize{alloc_size});
            tensor.shape = Shape.init(shape);
            tensor.strides = tensor_mod.Strides.contiguous(tensor.shape);

            self.total_tensors += 1;
            self.active_tensors += 1;
            self.miss_count += 1;
            self.total_bytes += alloc_size * @sizeOf(T);

            return tensor;
        }

        /// Release a tensor back to the pool for reuse.
        /// The tensor should not be used after release.
        pub fn release(self: *Self, tensor: TensorT) void {
            if (!tensor.owns_data) {
                // View tensor - nothing to pool
                return;
            }

            const bucket_idx = getBucketIndex(tensor.data.len);

            // Create pool entry
            const entry = self.allocator.create(PoolEntry) catch {
                // If allocation fails, just free the tensor
                var t = tensor;
                t.deinit();
                self.total_tensors -= 1;
                self.active_tensors -= 1;
                return;
            };

            entry.tensor = tensor;
            entry.num_elements = tensor.data.len;
            entry.next = self.buckets[bucket_idx];
            self.buckets[bucket_idx] = entry;
            self.active_tensors -= 1;
        }

        /// Clear pool, keeping only tensors up to max_keep per bucket
        pub fn trim(self: *Self, max_keep: usize) void {
            for (&self.buckets) |*bucket| {
                var count: usize = 0;
                var prev: ?*PoolEntry = null;
                var entry = bucket.*;

                while (entry) |e| {
                    if (count >= max_keep) {
                        // Free excess entries
                        if (prev) |p| {
                            p.next = null;
                        } else {
                            bucket.* = null;
                        }

                        var to_free: ?*PoolEntry = e;
                        while (to_free) |f| {
                            const next = f.next;
                            var tensor_copy = f.tensor;
                            tensor_copy.deinit();
                            self.total_bytes -= f.num_elements * @sizeOf(T);
                            self.total_tensors -= 1;
                            self.allocator.destroy(f);
                            to_free = next;
                        }
                        break;
                    }
                    count += 1;
                    prev = e;
                    entry = e.next;
                }
            }
        }

        /// Get pool statistics
        pub fn getStats(self: *const Self) PoolStats {
            return PoolStats{
                .total_tensors = self.total_tensors,
                .active_tensors = self.active_tensors,
                .pooled_tensors = self.total_tensors - self.active_tensors,
                .hit_count = self.hit_count,
                .miss_count = self.miss_count,
                .hit_rate = if (self.hit_count + self.miss_count > 0)
                    @as(f32, @floatFromInt(self.hit_count)) /
                        @as(f32, @floatFromInt(self.hit_count + self.miss_count))
                else
                    0.0,
                .total_bytes = self.total_bytes,
            };
        }
    };
}

/// Statistics about tensor pool usage
pub const PoolStats = struct {
    total_tensors: usize,
    active_tensors: usize,
    pooled_tensors: usize,
    hit_count: usize,
    miss_count: usize,
    hit_rate: f32,
    total_bytes: usize,
};

/// Convenience type aliases
pub const TensorPoolF32 = TensorPool(f32);
pub const TensorPoolF64 = TensorPool(f64);
pub const TensorPoolI32 = TensorPool(i32);
pub const TensorPoolI64 = TensorPool(i64);
pub const TensorPoolU8 = TensorPool(u8);

// =============================================================================
// Tests
// =============================================================================

test "TensorPool - basic acquire and release" {
    var pool = TensorPoolF32.init(std.testing.allocator);
    defer pool.deinit();

    // Acquire tensor
    var t1 = try pool.acquire(&[_]usize{ 2, 3 });
    try std.testing.expectEqual(@as(usize, 6), t1.numel());
    try std.testing.expectEqual(@as(usize, 1), pool.getStats().active_tensors);

    // Fill with data
    t1.data[0] = 1.0;
    t1.data[5] = 6.0;

    // Release back to pool
    pool.release(t1);
    try std.testing.expectEqual(@as(usize, 0), pool.getStats().active_tensors);
    try std.testing.expectEqual(@as(usize, 1), pool.getStats().pooled_tensors);
}

test "TensorPool - reuse" {
    var pool = TensorPoolF32.init(std.testing.allocator);
    defer pool.deinit();

    // Acquire and release
    const t1 = try pool.acquire(&[_]usize{ 10, 10 });
    pool.release(t1);

    const stats_before = pool.getStats();
    try std.testing.expectEqual(@as(usize, 1), stats_before.miss_count);

    // Acquire same size - should hit pool
    const t2 = try pool.acquire(&[_]usize{ 10, 10 });
    defer pool.release(t2);

    const stats_after = pool.getStats();
    try std.testing.expectEqual(@as(usize, 1), stats_after.hit_count);
    try std.testing.expectEqual(@as(usize, 1), stats_after.miss_count);
}

test "TensorPool - different sizes" {
    var pool = TensorPoolF32.init(std.testing.allocator);
    defer pool.deinit();

    // Acquire different sizes
    const t1 = try pool.acquire(&[_]usize{ 4, 4 });
    const t2 = try pool.acquire(&[_]usize{ 32, 32 });
    const t3 = try pool.acquire(&[_]usize{ 128, 128 });

    try std.testing.expectEqual(@as(usize, 3), pool.getStats().active_tensors);

    pool.release(t1);
    pool.release(t2);
    pool.release(t3);

    try std.testing.expectEqual(@as(usize, 0), pool.getStats().active_tensors);
    try std.testing.expectEqual(@as(usize, 3), pool.getStats().pooled_tensors);
}

test "TensorPool - trim" {
    var pool = TensorPoolF32.init(std.testing.allocator);
    defer pool.deinit();

    // Create many tensors of similar size
    var tensors: [10]tensor_mod.TensorF32 = undefined;
    for (&tensors) |*t| {
        t.* = try pool.acquire(&[_]usize{ 100, 100 });
    }
    for (&tensors) |t| {
        pool.release(t);
    }

    try std.testing.expectEqual(@as(usize, 10), pool.getStats().pooled_tensors);

    // Trim to max 2 per bucket
    pool.trim(2);

    try std.testing.expect(pool.getStats().pooled_tensors <= 2);
}

test "TensorPool - bucket sizing" {
    // Test bucket index calculation with fine-grained buckets
    // Bucket 0: 1-16
    try std.testing.expectEqual(@as(usize, 0), TensorPoolF32.getBucketIndex(1));
    try std.testing.expectEqual(@as(usize, 0), TensorPoolF32.getBucketIndex(16));

    // Bucket 1: 17-32
    try std.testing.expectEqual(@as(usize, 1), TensorPoolF32.getBucketIndex(17));
    try std.testing.expectEqual(@as(usize, 1), TensorPoolF32.getBucketIndex(32));

    // Bucket 2: 33-48
    try std.testing.expectEqual(@as(usize, 2), TensorPoolF32.getBucketIndex(33));
    try std.testing.expectEqual(@as(usize, 2), TensorPoolF32.getBucketIndex(48));

    // Bucket 3: 49-64
    try std.testing.expectEqual(@as(usize, 3), TensorPoolF32.getBucketIndex(49));
    try std.testing.expectEqual(@as(usize, 3), TensorPoolF32.getBucketIndex(64));

    // Larger sizes (power-of-2 region)
    try std.testing.expectEqual(@as(usize, 11), TensorPoolF32.getBucketIndex(1024));
    try std.testing.expectEqual(@as(usize, 16), TensorPoolF32.getBucketIndex(8192));

    // Overflow bucket for very large tensors
    try std.testing.expectEqual(@as(usize, 24), TensorPoolF32.getBucketIndex(2_000_000));
}

test "TensorPool - fine-grained reuse" {
    var pool = TensorPoolF32.init(std.testing.allocator);
    defer pool.deinit();

    // Acquire a 40-element tensor (bucket 2: 33-48)
    const t1 = try pool.acquire(&[_]usize{ 8, 5 }); // 40 elements
    pool.release(t1);

    // Request 35 elements - should reuse the 40-element tensor from bucket 2
    const t2 = try pool.acquire(&[_]usize{ 7, 5 }); // 35 elements
    defer pool.release(t2);

    // Should have hit the pool
    const stats = pool.getStats();
    try std.testing.expectEqual(@as(usize, 1), stats.hit_count);
}
