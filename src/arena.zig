//! Memory arena and scratch allocator for session-scoped allocations.
//!
//! Provides efficient memory management for ONNX inference with:
//! - Session-scoped arena for lifetime-bound allocations
//! - Scratch space for temporary inference allocations
//! - Reset capability for reusing memory between inference runs
//!
//! ## Usage
//!
//! ```zig
//! var scratch = ScratchAllocator.init(std.heap.page_allocator);
//! defer scratch.deinit();
//!
//! // Use for inference
//! const allocator = scratch.allocator();
//! // ... allocate temporaries ...
//!
//! // Reset between inference runs (reuses memory)
//! scratch.reset();
//! ```

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Scratch allocator wrapper around Zig's ArenaAllocator.
/// Provides reset capability for reusing memory between inference runs.
pub const ScratchAllocator = struct {
    const Self = @This();

    arena: std.heap.ArenaAllocator,
    bytes_allocated: usize,
    peak_bytes: usize,
    allocation_count: usize,

    /// Initialize scratch allocator with backing allocator
    pub fn init(backing_allocator: Allocator) Self {
        return Self{
            .arena = std.heap.ArenaAllocator.init(backing_allocator),
            .bytes_allocated = 0,
            .peak_bytes = 0,
            .allocation_count = 0,
        };
    }

    /// Free all memory
    pub fn deinit(self: *Self) void {
        self.arena.deinit();
    }

    /// Get allocator interface
    pub fn allocator(self: *Self) Allocator {
        return self.arena.allocator();
    }

    /// Reset arena, freeing all allocations but keeping capacity.
    /// Call between inference runs for memory reuse.
    pub fn reset(self: *Self) void {
        // Track peak before reset
        if (self.bytes_allocated > self.peak_bytes) {
            self.peak_bytes = self.bytes_allocated;
        }

        _ = self.arena.reset(.retain_capacity);
        self.bytes_allocated = 0;
        self.allocation_count = 0;
    }

    /// Get current bytes allocated (approximate)
    pub fn getBytesAllocated(self: *const Self) usize {
        return self.bytes_allocated;
    }

    /// Get peak bytes ever allocated
    pub fn getPeakBytes(self: *const Self) usize {
        return self.peak_bytes;
    }
};

/// Fixed-size memory pool for same-sized allocations.
/// Useful for tensor metadata or small fixed structures.
pub fn Pool(comptime T: type) type {
    return struct {
        const Self = @This();
        const BLOCK_SIZE = 64; // Items per block

        const Block = struct {
            items: [BLOCK_SIZE]T,
            next: ?*Block,
        };

        backing_allocator: Allocator,
        free_list: ?*T,
        blocks: ?*Block,
        items_allocated: usize,
        items_in_use: usize,

        pub fn init(backing_allocator: Allocator) Self {
            return Self{
                .backing_allocator = backing_allocator,
                .free_list = null,
                .blocks = null,
                .items_allocated = 0,
                .items_in_use = 0,
            };
        }

        pub fn deinit(self: *Self) void {
            var block = self.blocks;
            while (block) |b| {
                const next = b.next;
                self.backing_allocator.destroy(b);
                block = next;
            }
            self.blocks = null;
            self.free_list = null;
        }

        /// Allocate an item from the pool
        pub fn alloc(self: *Self) !*T {
            // Try free list first
            if (self.free_list) |item| {
                self.free_list = @as(*?*T, @ptrCast(item)).*;
                self.items_in_use += 1;
                return item;
            }

            // Need new block?
            if (self.items_allocated % BLOCK_SIZE == 0 or self.blocks == null) {
                const new_block = try self.backing_allocator.create(Block);
                new_block.next = self.blocks;
                self.blocks = new_block;
            }

            // Allocate from current block
            const block = self.blocks.?;
            const idx = self.items_allocated % BLOCK_SIZE;
            self.items_allocated += 1;
            self.items_in_use += 1;
            return &block.items[idx];
        }

        /// Return item to pool
        pub fn free(self: *Self, item: *T) void {
            @as(*?*T, @ptrCast(item)).* = self.free_list;
            self.free_list = item;
            self.items_in_use -= 1;
        }

        /// Get number of items currently in use
        pub fn getInUseCount(self: *const Self) usize {
            return self.items_in_use;
        }

        /// Get total items allocated (including free list)
        pub fn getTotalAllocated(self: *const Self) usize {
            return self.items_allocated;
        }
    };
}

/// Buffer pool for variable-sized byte allocations.
/// Maintains buckets of different sizes for efficient reuse.
pub const BufferPool = struct {
    const Self = @This();
    const NUM_BUCKETS = 8;
    const MIN_SIZE = 64;

    const BufferHeader = struct {
        size: usize,
        next: ?*BufferHeader,
    };

    backing_allocator: Allocator,
    buckets: [NUM_BUCKETS]?*BufferHeader,
    total_bytes: usize,
    reuse_count: usize,

    pub fn init(backing_allocator: Allocator) Self {
        return Self{
            .backing_allocator = backing_allocator,
            .buckets = [_]?*BufferHeader{null} ** NUM_BUCKETS,
            .total_bytes = 0,
            .reuse_count = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        for (&self.buckets) |*bucket| {
            var header = bucket.*;
            while (header) |h| {
                const next = h.next;
                const full_size = h.size + @sizeOf(BufferHeader);
                const ptr: [*]u8 = @ptrCast(h);
                self.backing_allocator.free(ptr[0..full_size]);
                header = next;
            }
            bucket.* = null;
        }
    }

    /// Get bucket index for size (rounds up to power of 2)
    fn getBucketIndex(size: usize) usize {
        if (size <= MIN_SIZE) return 0;
        // Find the highest bit position
        const effective = @max(size, 1);
        const bits = @bitSizeOf(usize) - @clz(effective - 1);
        const min_bits = @bitSizeOf(usize) - @clz(@as(usize, MIN_SIZE - 1));
        const idx = if (bits > min_bits) bits - min_bits else 0;
        return @min(idx, NUM_BUCKETS - 1);
    }

    /// Get bucket size for index
    fn getBucketSize(idx: usize) usize {
        return @as(usize, MIN_SIZE) << @as(u6, @intCast(idx));
    }

    /// Allocate a buffer of at least `size` bytes
    pub fn alloc(self: *Self, size: usize) ![]u8 {
        const bucket_idx = getBucketIndex(size);
        const bucket_size = getBucketSize(bucket_idx);

        // Check free list
        if (self.buckets[bucket_idx]) |header| {
            self.buckets[bucket_idx] = header.next;
            self.reuse_count += 1;
            const ptr: [*]u8 = @ptrCast(header);
            return (ptr + @sizeOf(BufferHeader))[0..bucket_size];
        }

        // Allocate new buffer with header
        const full_size = bucket_size + @sizeOf(BufferHeader);
        const mem = try self.backing_allocator.alloc(u8, full_size);
        const header: *BufferHeader = @ptrCast(@alignCast(mem.ptr));
        header.size = bucket_size;
        header.next = null;
        self.total_bytes += full_size;

        return mem[@sizeOf(BufferHeader)..];
    }

    /// Return buffer to pool
    pub fn free(self: *Self, buf: []u8) void {
        const header: *BufferHeader = @ptrCast(@alignCast(buf.ptr - @sizeOf(BufferHeader)));
        const bucket_idx = getBucketIndex(header.size);

        header.next = self.buckets[bucket_idx];
        self.buckets[bucket_idx] = header;
    }

    /// Get total bytes allocated by this pool
    pub fn getTotalBytes(self: *const Self) usize {
        return self.total_bytes;
    }

    /// Get number of buffer reuses
    pub fn getReuseCount(self: *const Self) usize {
        return self.reuse_count;
    }
};

// =============================================================================
// Tests
// =============================================================================

test "ScratchAllocator - basic usage" {
    var scratch = ScratchAllocator.init(std.testing.allocator);
    defer scratch.deinit();

    const alloc = scratch.allocator();

    // Allocate some memory
    const buf1 = try alloc.alloc(u8, 1024);
    const buf2 = try alloc.alloc(u32, 256);

    // Use the buffers
    buf1[0] = 42;
    buf2[0] = 123;

    try std.testing.expectEqual(@as(u8, 42), buf1[0]);
    try std.testing.expectEqual(@as(u32, 123), buf2[0]);

    // Reset for next inference
    scratch.reset();

    // Allocate again (reuses memory)
    const buf3 = try alloc.alloc(u8, 512);
    buf3[0] = 99;
    try std.testing.expectEqual(@as(u8, 99), buf3[0]);
}

test "Pool - alloc and free" {
    const TestItem = struct {
        value: u64,
        data: [56]u8,
    };

    var pool = Pool(TestItem).init(std.testing.allocator);
    defer pool.deinit();

    // Allocate several items
    const item1 = try pool.alloc();
    const item2 = try pool.alloc();
    const item3 = try pool.alloc();

    item1.value = 100;
    item2.value = 200;
    item3.value = 300;

    try std.testing.expectEqual(@as(usize, 3), pool.getInUseCount());

    // Free one
    pool.free(item2);
    try std.testing.expectEqual(@as(usize, 2), pool.getInUseCount());

    // Allocate again (should reuse)
    const item4 = try pool.alloc();
    try std.testing.expectEqual(@as(usize, 3), pool.getInUseCount());
    try std.testing.expectEqual(@as(usize, 3), pool.getTotalAllocated());

    item4.value = 400;

    pool.free(item1);
    pool.free(item3);
    pool.free(item4);
    try std.testing.expectEqual(@as(usize, 0), pool.getInUseCount());
}

test "BufferPool - basic allocation" {
    var pool = BufferPool.init(std.testing.allocator);
    defer pool.deinit();

    // Allocate buffers of various sizes
    const buf1 = try pool.alloc(100);
    const buf2 = try pool.alloc(500);
    const buf3 = try pool.alloc(100);

    buf1[0] = 1;
    buf2[0] = 2;
    buf3[0] = 3;

    try std.testing.expectEqual(@as(u8, 1), buf1[0]);
    try std.testing.expectEqual(@as(u8, 2), buf2[0]);
    try std.testing.expectEqual(@as(u8, 3), buf3[0]);

    // Free and reallocate
    pool.free(buf1);
    const buf4 = try pool.alloc(100);
    buf4[0] = 4;

    try std.testing.expect(pool.getReuseCount() >= 1);

    pool.free(buf2);
    pool.free(buf3);
    pool.free(buf4);
}

test "BufferPool - bucket sizing" {
    // Test bucket index calculation
    try std.testing.expectEqual(@as(usize, 0), BufferPool.getBucketIndex(1));
    try std.testing.expectEqual(@as(usize, 0), BufferPool.getBucketIndex(64));
    try std.testing.expectEqual(@as(usize, 1), BufferPool.getBucketIndex(65));
    try std.testing.expectEqual(@as(usize, 1), BufferPool.getBucketIndex(128));
    try std.testing.expectEqual(@as(usize, 2), BufferPool.getBucketIndex(129));
    try std.testing.expectEqual(@as(usize, 2), BufferPool.getBucketIndex(256));
}
