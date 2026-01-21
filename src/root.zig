//! # ONNX Zig - ONNX Inference Engine for Edge Computing
//!
//! A Zig library providing high-level bindings to ONNX Runtime for efficient
//! machine learning inference on edge devices.
//!
//! ## Quick Start
//!
//! ```zig
//! const std = @import("std");
//! const onnx_zig = @import("onnx_zig");
//!
//! pub fn main() !void {
//!     const allocator = std.heap.page_allocator;
//!
//!     // Load an ONNX model
//!     var session = try onnx_zig.Session.init(allocator, "model.onnx");
//!     defer session.deinit();
//!
//!     // Prepare input tensor data
//!     const input_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
//!     const input_shape = [_]i64{ 1, 4 };
//!
//!     // Run inference
//!     const outputs = try session.runF32(
//!         &[_][]const f32{&input_data},
//!         &[_][]const i64{&input_shape},
//!     );
//!     defer session.freeOutputs(outputs);
//!
//!     // Use outputs
//!     for (outputs[0].data) |val| {
//!         std.debug.print("{d} ", .{val});
//!     }
//! }
//! ```
//!
//! ## Architecture
//!
//! The library is organized into three layers:
//!
//! - **Session API** (`Session`): High-level, idiomatic Zig interface for model loading and inference
//! - **ONNX Runtime FFI** (`onnxruntime`): Low-level C API bindings via `@cImport`
//! - **Tensor Library** (`Tensor`): Multi-dimensional array implementation for input/output handling
//!
//! ## Building
//!
//! ```bash
//! # Build with ONNX Runtime from Homebrew
//! zig build -Donnxruntime_path=/opt/homebrew/Cellar/onnxruntime/1.23.2_2
//!
//! # Run tests
//! zig build test -Donnxruntime_path=/opt/homebrew/Cellar/onnxruntime/1.23.2_2
//! ```
//!
//! ## Key Types
//!
//! - `Session`: Main inference session for loading and running ONNX models
//! - `TensorF32`: 32-bit floating point tensor for input/output data
//! - `Shape`: Tensor dimension specification
//! - `TensorInfo`: Metadata about model inputs/outputs
//!

const std = @import("std");

// Core modules
pub const tensor = @import("tensor.zig");
pub const onnxruntime = @import("onnxruntime.zig");
pub const session = @import("session.zig");

// Memory management modules
pub const arena = @import("arena.zig");
pub const tensor_pool = @import("tensor_pool.zig");

// NLP modules
pub const tokenizer = @import("tokenizer.zig");

// Tensor type aliases for convenient access
pub const Tensor = tensor.Tensor;
pub const TensorF32 = tensor.TensorF32;
pub const TensorF64 = tensor.TensorF64;
pub const TensorI32 = tensor.TensorI32;
pub const TensorI64 = tensor.TensorI64;
pub const TensorU8 = tensor.TensorU8;
pub const Shape = tensor.Shape;
pub const Strides = tensor.Strides;
pub const DataType = tensor.DataType;
pub const MAX_DIMS = tensor.MAX_DIMS;
pub const SimdOps = tensor.SimdOps;
pub const calcNumel = tensor.calcNumel;

// ONNX Runtime type aliases
pub const OrtApi = onnxruntime.OrtApi;
pub const OrtEnv = onnxruntime.OrtEnv;
pub const OrtSession = onnxruntime.OrtSession;
pub const OrtSessionOptions = onnxruntime.OrtSessionOptions;
pub const OrtValue = onnxruntime.OrtValue;
pub const OrtError = onnxruntime.OrtError;
// C API constants available via onnxruntime.c
pub const c = onnxruntime.c;

// Session API
pub const Session = session.Session;
pub const OptimizedSession = session.OptimizedSession;
pub const SessionError = session.SessionError;
pub const TensorInfo = session.TensorInfo;
pub const NamedTensor = session.NamedTensor;

// Memory management
pub const ScratchAllocator = arena.ScratchAllocator;
pub const Pool = arena.Pool;
pub const BufferPool = arena.BufferPool;
pub const TensorPool = tensor_pool.TensorPool;
pub const TensorPoolF32 = tensor_pool.TensorPoolF32;
pub const TensorPoolF64 = tensor_pool.TensorPoolF64;
pub const PoolStats = tensor_pool.PoolStats;

// Tokenizer
pub const WordPieceTokenizer = tokenizer.WordPieceTokenizer;
pub const SpecialTokens = tokenizer.SpecialTokens;
pub const EncodedInput = tokenizer.EncodedInput;

// Convenience functions
pub const getApi = onnxruntime.getApi;
pub const checkStatus = onnxruntime.checkStatus;

// Re-run all tests
test {
    std.testing.refAllDecls(@This());
}
