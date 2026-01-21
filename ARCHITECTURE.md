# ONNX Zig Architecture

A comprehensive guide to the ONNX Zig codebase structure, design decisions, and component interactions.

## Table of Contents

- [Overview](#overview)
- [Architecture Diagram](#architecture-diagram)
- [Core Components](#core-components)
- [Module Reference](#module-reference)
- [Data Flow](#data-flow)
- [Memory Management](#memory-management)
- [Platform Support](#platform-support)
- [Build System](#build-system)
- [Examples & Tools](#examples--tools)

---

## Overview

ONNX Zig is a high-performance ONNX inference engine written in Zig, providing:

- **Native Performance**: Direct FFI bindings to ONNX Runtime C API
- **WebAssembly Support**: Browser-based inference with SIMD optimizations
- **Memory Efficiency**: Arena allocators and tensor pooling
- **Type Safety**: Compile-time verified tensor operations
- **NLP Support**: Built-in WordPiece tokenizer for transformer models

### Design Principles

1. **Zero-Cost Abstractions**: High-level API with no runtime overhead
2. **Explicit Memory Management**: No hidden allocations
3. **Layered Architecture**: Clear separation between FFI, session, and tensor layers
4. **Platform Agnostic**: Single codebase for native and WASM targets

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER APPLICATION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐│
│  │   Session   │  │  Tokenizer  │  │   Tensor    │  │   OptimizedSession  ││
│  │   (High)    │  │   (NLP)     │  │   (Data)    │  │   (Pooled Memory)   ││
│  └──────┬──────┘  └─────────────┘  └──────┬──────┘  └──────────┬──────────┘│
│         │                                  │                    │           │
│         ▼                                  ▼                    ▼           │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         PUBLIC API (root.zig)                           ││
│  │  Session, TensorF32, WordPieceTokenizer, ScratchAllocator, TensorPool   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                              INTERNAL LAYER                                  │
│                                                                             │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐ │
│  │  session.zig  │  │  tensor.zig   │  │   arena.zig   │  │tensor_pool.zig││
│  │               │  │               │  │               │  │              │ │
│  │ • Session     │  │ • Tensor(T)   │  │ • Scratch     │  │ • TensorPool │ │
│  │ • TensorInfo  │  │ • Shape       │  │   Allocator   │  │ • Bucketing  │ │
│  │ • runF32()    │  │ • SimdOps     │  │ • Pool(T)     │  │ • Free Lists │ │
│  │ • runI64()    │  │ • Strides     │  │ • BufferPool  │  │              │ │
│  └───────┬───────┘  └───────────────┘  └───────────────┘  └──────────────┘ │
│          │                                                                  │
│          ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                       onnxruntime.zig (FFI Layer)                       ││
│  │                                                                         ││
│  │  @cImport("onnxruntime_c_api.h")                                        ││
│  │  • OrtApi, OrtEnv, OrtSession, OrtValue                                 ││
│  │  • getApi(), checkStatus(), zigTypeToOnnx()                             ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                              NATIVE RUNTIME                                  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    ONNX Runtime C Library                               ││
│  │                    (libonnxruntime.dylib/.so/.dll)                      ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           WASM ARCHITECTURE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────────┐   │
│  │   Browser JS    │     │   loader.js     │     │  ONNX Runtime Web   │   │
│  │                 │────▶│                 │────▶│  (ort.min.js)       │   │
│  │  Application    │     │  OnnxZigModule  │     │                     │   │
│  └─────────────────┘     │  TensorAPI      │     │  InferenceSession   │   │
│                          │  SimdAPI        │     │                     │   │
│                          │  OrtAPI         │     └─────────────────────┘   │
│                          └────────┬────────┘                               │
│                                   │                                        │
│                                   ▼                                        │
│                          ┌─────────────────┐                               │
│                          │  onnx_zig.wasm  │                               │
│                          │                 │                               │
│                          │  wasm_exports:  │                               │
│                          │  • tensor_*     │                               │
│                          │  • simd_*       │                               │
│                          │  • wasm_alloc   │                               │
│                          └─────────────────┘                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. FFI Layer (`onnxruntime.zig`)

The foundation layer providing direct C API bindings:

```zig
// C API import via @cImport
pub const c = @cImport({
    @cInclude("onnxruntime/onnxruntime_c_api.h");
});

// Type re-exports
pub const OrtApi = c.OrtApi;
pub const OrtSession = c.OrtSession;
pub const OrtValue = c.OrtValue;

// Helper functions
pub fn getApi() OrtError!*const OrtApi
pub fn checkStatus(api: *const OrtApi, status: ?*OrtStatus) OrtError!void
pub fn zigTypeToOnnx(comptime T: type) c_int
```

**Responsibilities:**
- Import ONNX Runtime C headers
- Re-export C types with Zig-friendly names
- Convert between C and Zig error types
- Provide type mapping utilities

### 2. Session Layer (`session.zig`)

High-level inference API wrapping ONNX Runtime:

```zig
pub const Session = struct {
    allocator: Allocator,
    api: *const ort.OrtApi,
    env: *ort.OrtEnv,
    session: *ort.OrtSession,

    pub fn init(allocator: Allocator, model_path: []const u8) !Self
    pub fn runF32(self: *Self, inputs: []const []const f32, shapes: []const []const i64) ![]TensorF32
    pub fn runI64(self: *Self, inputs: []const []const i64, shapes: []const []const i64) ![]TensorF32
    pub fn getInputInfo(self: *const Self, index: usize) !TensorInfo
    pub fn getOutputInfo(self: *const Self, index: usize) !TensorInfo
};
```

**Responsibilities:**
- Model loading and validation
- Input/output metadata extraction
- Inference execution with type safety
- Resource lifecycle management

### 3. Tensor Layer (`tensor.zig`)

Multi-dimensional array implementation:

```zig
pub fn Tensor(comptime T: type) type {
    return struct {
        data: []T,
        shape: Shape,
        strides: Strides,
        allocator: ?Allocator,

        pub fn init(allocator: Allocator, shape_dims: []const usize) !Self
        pub fn get(self: *const Self, indices: []const usize) T
        pub fn set(self: *Self, indices: []const usize, value: T) void
        pub fn reshape(self: *Self, new_shape: []const usize) !void
    };
}

// SIMD-optimized operations
pub const SimdOps = struct {
    pub fn add(dst: []f32, a: []const f32, b: []const f32) void
    pub fn mul(dst: []f32, a: []const f32, b: []const f32) void
    pub fn relu(dst: []f32, src: []const f32) void
    pub fn dot(a: []const f32, b: []const f32) f32
};
```

**Responsibilities:**
- Shape and stride management
- Element access and iteration
- Memory layout optimization
- SIMD-accelerated operations

### 4. Memory Management (`arena.zig`, `tensor_pool.zig`)

Efficient allocation strategies:

```zig
// Arena allocator for scratch memory
pub const ScratchAllocator = struct {
    pub fn init(backing: Allocator) Self
    pub fn reset(self: *Self) void
    pub fn allocator(self: *Self) Allocator
};

// Tensor pooling with size buckets
pub fn TensorPool(comptime T: type) type {
    return struct {
        pub fn acquire(self: *Self, shape: []const usize) !Tensor(T)
        pub fn release(self: *Self, tensor: *Tensor(T)) void
        pub fn trim(self: *Self) void
    };
}
```

**Responsibilities:**
- Minimize allocation overhead
- Enable tensor reuse
- Track memory statistics
- Support batch inference patterns

### 5. Tokenizer (`tokenizer.zig`)

WordPiece tokenization for transformer models:

```zig
pub const WordPieceTokenizer = struct {
    vocab: std.StringHashMap(i64),
    vocab_list: [][]const u8,

    pub fn initFromFile(allocator: Allocator, path: []const u8) !Self
    pub fn encode(self: *const Self, text: []const u8, max_length: usize) !EncodedInput
    pub fn tokenToId(self: *const Self, token: []const u8) i64
    pub fn idToToken(self: *const Self, id: i64) ?[]const u8
};
```

**Responsibilities:**
- Vocabulary loading
- Text tokenization with WordPiece algorithm
- Special token handling (CLS, SEP, PAD)
- BERT-compatible encoding

---

## Module Reference

### Source Files

| File | Lines | Purpose |
|------|-------|---------|
| `root.zig` | 130 | Public API exports and documentation |
| `onnxruntime.zig` | 190 | ONNX Runtime C API FFI bindings |
| `session.zig` | 967 | High-level inference session API |
| `tensor.zig` | 945 | Multi-dimensional tensor + SIMD |
| `arena.zig` | 150+ | Arena and pool allocators |
| `tensor_pool.zig` | 150+ | Tensor reuse pooling |
| `tokenizer.zig` | 300+ | WordPiece tokenizer |
| `wasm_exports.zig` | 450+ | WebAssembly export functions |
| `main.zig` | 52 | Executable entry point |

### Type Hierarchy

```
Tensor(T)
├── TensorF32  (f32)
├── TensorF64  (f64)
├── TensorI32  (i32)
├── TensorI64  (i64)
└── TensorU8   (u8)

Session
└── OptimizedSession (with TensorPool + ScratchAllocator)

WordPieceTokenizer
└── EncodedInput (input_ids, attention_mask, token_type_ids)
```

---

## Data Flow

### Native Inference Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  User Input  │────▶│   Session    │────▶│  ORT C API   │
│  ([]f32)     │     │   .runF32()  │     │  CreateValue │
└──────────────┘     └──────────────┘     └──────────────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Output     │◀────│   Session    │◀────│  ORT C API   │
│  (TensorF32) │     │  extract     │     │  Run()       │
└──────────────┘     └──────────────┘     └──────────────┘
```

### BERT Inference Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Raw Text    │────▶│  Tokenizer   │────▶│ EncodedInput │
│  "Hello"     │     │  .encode()   │     │ [101,7592,..]│
└──────────────┘     └──────────────┘     └──────────────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Embeddings  │◀────│   Session    │◀────│  int64 IDs   │
│  [1,768]     │     │  .runI64()   │     │  + shapes    │
└──────────────┘     └──────────────┘     └──────────────┘
```

### WASM Data Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  JavaScript  │────▶│  loader.js   │────▶│  WASM Memory │
│  Float32Array│     │  fromArray() │     │  tensor_*    │
└──────────────┘     └──────────────┘     └──────────────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│    Result    │◀────│  loader.js   │◀────│  ORT Web     │
│  Float32Array│     │  toArray()   │     │  .run()      │
└──────────────┘     └──────────────┘     └──────────────┘
```

---

## Memory Management

### Allocation Strategies

| Strategy | Use Case | Overhead |
|----------|----------|----------|
| Direct Allocator | One-off operations | Per-allocation |
| ScratchAllocator | Batch processing | Reset between batches |
| TensorPool | Repeated inference | Near-zero after warmup |
| OptimizedSession | Production workloads | Combines both |

### TensorPool Bucketing

```
Bucket  │  Size Range      │  Typical Use
────────┼──────────────────┼─────────────────
   0    │  16 elements     │  Small metadata
   1    │  32 elements     │  Tiny tensors
   2    │  64 elements     │  Small vectors
   ...  │  ...             │  ...
   10   │  16K elements    │  Images (128x128)
   11   │  32K elements    │  Larger images
   ...  │  ...             │  ...
   15   │  ~32M elements   │  Large batches
```

### Memory Layout

```
TensorF32 [2, 3, 4] (24 elements)
├── data: []f32 (96 bytes, contiguous)
├── shape: Shape { dims: [2, 3, 4], ndim: 3 }
└── strides: Strides { values: [12, 4, 1] }  // Row-major

Memory: [e00 e01 e02 e03 | e10 e11 e12 e13 | e20 e21 e22 e23 | ...]
         └── row 0 ────────┘ └── row 1 ────────┘ └── row 2 ────────┘
```

---

## Platform Support

### Native Targets

| Platform | Architecture | ONNX Runtime |
|----------|--------------|--------------|
| macOS | arm64 (Apple Silicon) | Homebrew/Manual |
| macOS | x86_64 | Homebrew/Manual |
| Linux | x86_64, arm64 | Package manager |
| Windows | x86_64 | NuGet/Manual |

### WebAssembly

| Feature | Support |
|---------|---------|
| wasm32-freestanding | Primary target |
| SIMD | 128-bit vectors |
| Threads | Via onnxruntime-web |
| Memory | Linear (growable) |

### Build Configuration

```bash
# Native with ONNX Runtime
zig build -Donnxruntime_path=/opt/homebrew/Cellar/onnxruntime/1.23.2_2

# WebAssembly
zig build wasm -Dtarget=wasm32-freestanding

# Run tests
zig build test -Donnxruntime_path=/path/to/onnxruntime
```

---

## Build System

### Build Targets

| Target | Command | Output |
|--------|---------|--------|
| Library | `zig build` | `zig-out/lib/libonnx_zig.a` |
| Executable | `zig build` | `zig-out/bin/onnx_zig` |
| CLI | `zig build cli` | `zig-out/bin/onnx-zig` |
| WASM | `zig build wasm -Dtarget=wasm32-freestanding` | `zig-out/wasm/onnx_zig.wasm` |
| Tests | `zig build test` | Run all tests |
| Examples | `zig build example-basic` | `zig-out/bin/example-basic` |

### Module Graph

```
build.zig
├── mod (onnx_zig module)
│   └── root_source: src/root.zig
│       ├── tensor.zig
│       ├── onnxruntime.zig
│       ├── session.zig
│       ├── arena.zig
│       ├── tensor_pool.zig
│       └── tokenizer.zig
├── exe (main executable)
│   └── root_source: src/main.zig
│       └── imports: onnx_zig
├── cli (CLI tool)
│   └── root_source: tools/cli.zig
│       └── imports: onnx_zig
├── examples (example-basic, example-mnist, example-bert)
│   └── root_source: examples/zig/*.zig
│       └── imports: onnx_zig
└── wasm (WebAssembly module)
    └── root_source: src/wasm_exports.zig
        └── tensor.zig (subset)
```

---

## Examples & Tools

### Examples

| Example | Model | Input | Output |
|---------|-------|-------|--------|
| `example-basic` | Any ONNX | f32 tensors | f32 tensors |
| `example-bert` | BERT | Text string | 768-dim embeddings |
| `example-mnist` | CNN | 28x28 grayscale | Digit 0-9 |

### CLI Commands

```bash
# Model information
onnx-zig info model.onnx

# Benchmark (100 iterations)
onnx-zig bench model.onnx 100

# Single inference
onnx-zig run model.onnx

# Validate model
onnx-zig validate model.onnx

# Version info
onnx-zig version
```

### Scripts

| Script | Purpose |
|--------|---------|
| `export_bert.py` | Export BERT from HuggingFace to ONNX |
| `generate_mnist_model.py` | Train and export MNIST CNN |
| `generate_test_model.py` | Create minimal test models |

---

## Directory Structure

```
onnx_zig/
├── src/                          # Core library source
│   ├── root.zig                 # Public API (entry point)
│   ├── onnxruntime.zig          # FFI bindings
│   ├── session.zig              # Inference session
│   ├── tensor.zig               # Tensor implementation
│   ├── arena.zig                # Memory allocators
│   ├── tensor_pool.zig          # Tensor pooling
│   ├── tokenizer.zig            # WordPiece tokenizer
│   ├── wasm_exports.zig         # WASM exports
│   └── main.zig                 # Executable main
├── examples/zig/                 # Example programs
│   ├── basic_inference.zig      # General inference
│   ├── bert_embeddings.zig      # BERT example
│   └── mnist_classifier.zig     # Image classification
├── tools/                        # CLI tools
│   └── cli.zig                  # Model inspection/benchmark
├── scripts/                      # Utility scripts
│   ├── export_bert.py           # BERT export
│   ├── generate_mnist_model.py  # MNIST training
│   └── generate_test_model.py   # Test model generation
├── wasm/                         # WebAssembly assets
│   ├── loader.js                # JS WASM loader
│   ├── index.html               # Main demo
│   ├── embedding_demo.html      # Embedding demo
│   └── mnist_demo.html          # MNIST demo
├── models/                       # Model storage
│   ├── test/                    # Test models
│   ├── bert/                    # BERT model + vocab
│   └── mnist/                   # MNIST model
├── build.zig                     # Build configuration
├── build.zig.zon                 # Package manifest
├── README.md                     # User documentation
└── ARCHITECTURE.md               # This file
```

---

## Performance Considerations

### SIMD Optimization

The `SimdOps` module uses 128-bit SIMD vectors for:
- 4x f32 parallel operations
- Automatic fallback for non-aligned data
- Operations: add, sub, mul, div, relu, dot, sum, max, min

### Memory Efficiency

1. **Zero-copy views**: `Tensor.view()` wraps external data
2. **Pooled allocation**: `TensorPool` eliminates repeated malloc/free
3. **Arena reset**: `ScratchAllocator.reset()` for batch processing
4. **Uninit allocation**: `initUninit()` skips zeroing when data will be overwritten

### Inference Optimization

1. **Graph optimization**: ONNX Runtime's `ORT_ENABLE_ALL`
2. **Thread pool**: Configurable via ONNX Runtime options
3. **Cached metadata**: Input/output info cached after first query
4. **Batch support**: Efficient batch dimension handling

---

## Error Handling

### Error Types

```zig
// FFI errors
pub const OrtError = error{
    Fail, InvalidArgument, NoSuchFile, NoModel,
    EngineError, RuntimeException, InvalidProtobuf,
    ModelLoaded, NotImplemented, InvalidGraph,
    ExecutionProviderFail, NullApi, NullResult,
};

// Session errors
pub const SessionError = error{
    OrtFail, OrtInvalidArgument, OrtNoSuchFile, ...
    AllocationFailed, InvalidInput, ShapeMismatch,
};
```

### Error Propagation

```zig
// Zig error unions propagate naturally
const session = try Session.init(allocator, "model.onnx");
defer session.deinit();

const outputs = try session.runF32(&inputs, &shapes);
defer session.freeOutputs(outputs);
```

---

## Future Considerations

- [ ] GPU execution provider support (CUDA, DirectML)
- [ ] Quantization helpers (INT8, FP16)
- [ ] Dynamic shape inference
- [ ] Model optimization utilities
- [ ] Additional tokenizers (BPE, SentencePiece)
- [ ] Streaming inference API
- [ ] Model caching and warm-start
