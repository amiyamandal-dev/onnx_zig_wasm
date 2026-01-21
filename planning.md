# Technical Execution Strategy

## Project: Zig ONNX WASM Inference Engine for Edge Computing

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ONNX Zig Inference Engine                     │
├─────────────────────────────────────────────────────────────────┤
│  API Layer (Public Interface)                                    │
│  - Model loading                                                 │
│  - Inference execution                                           │
│  - Input/Output handling                                         │
├─────────────────────────────────────────────────────────────────┤
│  Inference Engine                                                │
│  - Graph execution (topological order)                          │
│  - Memory planning & optimization                               │
│  - Operator dispatch                                            │
├─────────────────────────────────────────────────────────────────┤
│  Operator Library                                                │
│  - Math ops (Add, Mul, MatMul, Gemm)                           │
│  - Activation (ReLU, Sigmoid, Softmax)                         │
│  - Convolution (Conv, MaxPool, AveragePool)                    │
│  - Normalization (BatchNorm, LayerNorm)                        │
│  - Reshape ops (Reshape, Flatten, Squeeze, Unsqueeze)          │
├─────────────────────────────────────────────────────────────────┤
│  Tensor Library                                                  │
│  - Multi-dimensional arrays                                     │
│  - Memory layouts (NCHW, NHWC)                                 │
│  - Broadcasting support                                         │
│  - Type support (f32, f16, i32, i64, u8)                       │
├─────────────────────────────────────────────────────────────────┤
│  ONNX Parser                                                     │
│  - Protobuf decoding (custom implementation)                   │
│  - Model graph construction                                     │
│  - Initializer/weight loading                                   │
├─────────────────────────────────────────────────────────────────┤
│  Platform Layer (WASM / Native)                                  │
│  - Memory allocation                                            │
│  - SIMD optimizations (where available)                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Core Infrastructure

### 1.1 Tensor Library
**Priority:** CRITICAL (Foundation for everything)

**Components:**
- `src/tensor.zig` - Multi-dimensional tensor struct
  - Shape handling (dimensions, strides)
  - Data storage (backed by slice)
  - Element access (indexing, iteration)
  - Memory layout support

**Key Features:**
- Generic over element type (f32, f16, i32, etc.)
- Contiguous and strided memory support
- Broadcasting semantics
- Zero-copy slicing

### 1.2 Memory Management
**Priority:** HIGH (Critical for edge computing)

**Strategy:**
- Arena allocator for inference session
- Pre-allocated tensor pools
- Memory reuse between operations
- No dynamic allocation during inference (goal)

---

## Phase 2: ONNX Runtime FFI Integration

### 2.1 ONNX Runtime C API Bindings
**Priority:** CRITICAL

**Approach:** FFI bindings to Microsoft ONNX Runtime C API
- Leverage battle-tested, optimized inference engine
- All ONNX operators already implemented
- WASM support via onnxruntime-web
- Reduces development effort significantly

**Files:**
- `src/onnxruntime.zig` - Low-level C API type declarations and bindings

**Key C API Types:**
```
OrtApi          - Function pointer table (main entry point)
OrtEnv          - Runtime environment (logging, threading)
OrtSession      - Loaded model + inference context
OrtSessionOptions - Session configuration
OrtValue        - Tensor/sequence/map values
OrtAllocator    - Memory allocation interface
OrtStatus       - Error handling
OrtMemoryInfo   - Memory location metadata
OrtRunOptions   - Per-inference options
```

### 2.2 High-Level Zig Wrapper
**Priority:** CRITICAL

**Components:**
- `src/session.zig` - Idiomatic Zig inference API
  - Session type wrapping OrtSession
  - Input/Output tensor conversion
  - Error handling with Zig error unions
  - Memory safety guarantees

---

## Phase 3: High-Level Inference API

### 3.1 Session API Design
**Priority:** HIGH

**Public Interface:**
```zig
const Session = struct {
    // Create session from ONNX model file
    pub fn init(allocator: Allocator, model_path: []const u8) !Session

    // Run inference with inputs, get outputs
    pub fn run(self: *Session, inputs: []const NamedTensor) ![]Tensor

    // Get model input/output metadata
    pub fn getInputInfo(self: *Session) []TensorInfo
    pub fn getOutputInfo(self: *Session) []TensorInfo

    // Clean up
    pub fn deinit(self: *Session) void
};
```

### 3.2 Tensor Conversion
**Priority:** HIGH

**Zig Tensor → OrtValue:**
- Get raw data pointer from Zig tensor
- Create OrtValue with matching shape and type
- Handle memory ownership carefully

**OrtValue → Zig Tensor:**
- Extract data pointer from OrtValue
- Create Zig tensor view (or copy if needed)
- Map ONNX data types to Zig types

### 3.3 Error Handling
**Priority:** HIGH

**Strategy:**
- Translate OrtStatus to Zig errors
- Custom error set for ONNX-specific failures
- Informative error messages via logging

---

## Phase 4: Memory & Performance

### 4.1 Memory Optimization
**Priority:** MEDIUM

**Strategies:**
- Arena allocator for session-scoped tensors
- Reuse intermediate buffers
- Zero-copy where ONNX Runtime supports it

### 4.2 Performance Tuning
**Priority:** MEDIUM

**Options:**
- Thread pool configuration
- Execution provider selection (CPU, GPU)
- Graph optimization level

---

## Phase 5: WASM Target

### 5.1 Build Configuration
**Priority:** HIGH

**Tasks:**
- Add WASM target to build.zig
- Configure freestanding/WASI support
- Memory import/export setup
- JavaScript interop layer

### 5.2 Optimizations
**Priority:** MEDIUM

**Targets:**
- SIMD128 (WebAssembly SIMD)
- Memory access patterns
- Code size optimization

---

## Phase 6: Testing & Validation

### 6.1 Unit Tests
- Tensor operations
- Protobuf parsing
- Individual operators

### 6.2 Integration Tests
- Full model inference
- ONNX test suite compatibility

### 6.3 Benchmark Suite
- Performance measurement
- Memory usage tracking

---

## Implementation Order (Milestones)

### Milestone 1: Tensor Foundation
1. Implement basic Tensor struct
2. Shape and stride handling
3. Element access and iteration
4. Basic arithmetic operations
5. Unit tests for tensor ops

### Milestone 2: ONNX Parsing
1. Protobuf wire format decoder
2. ONNX message types
3. Model file loading
4. Graph construction
5. Test with simple models

### Milestone 3: Basic Operators
1. Element-wise ops (Add, Mul, etc.)
2. MatMul implementation
3. ReLU activation
4. Reshape operations
5. Test operator correctness

### Milestone 4: Inference Engine
1. Graph traversal
2. Operator dispatch
3. Memory management
4. Session API
5. End-to-end inference test

### Milestone 5: WASM Build
1. WASM target configuration
2. JavaScript bindings
3. Browser testing
4. Size optimization

### Milestone 6: Extended Operators
1. Convolution operations
2. Pooling layers
3. Normalization layers
4. Softmax and other activations

---

## Technical Decisions

### Why ONNX Runtime FFI?
- Battle-tested inference engine (used in production at Microsoft, etc.)
- All 150+ ONNX operators already implemented and optimized
- Hardware acceleration support (CPU SIMD, GPU, NPU)
- Reduces development time from months to days
- WASM build available (onnxruntime-web)
- Active maintenance and community

### Why Zig Wrapper?
- Memory safety guarantees over raw C pointers
- Idiomatic Zig API for better developer experience
- Integration with Zig's allocator system
- Can still use Zig tensor library for pre/post-processing
- Single codebase builds to native and WASM

### Memory Strategy
- Arena allocator for session lifetime
- Pre-computed tensor sizes where possible
- Tensor pooling for intermediate results
- Zero-copy for model weights

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Protobuf complexity | Medium | Start with minimal subset |
| Operator coverage | High | Prioritize common ops |
| WASM performance | Medium | Profile early, optimize late |
| Memory constraints | High | Design for bounded memory |

---

## Success Criteria

1. **Parse ONNX models** - Load and interpret .onnx files
2. **Run inference** - Execute simple models (e.g., MNIST)
3. **WASM build** - Compile to WebAssembly successfully
4. **Edge-ready** - Work within memory constraints
5. **Correct results** - Match reference implementations

---

## Future Improvements Roadmap

### Phase 7: Code Quality & Robustness

#### 7.1 Error Handling Improvements
**Priority:** HIGH

**Issues Identified:**
- `session.zig:210,227` - Uninitialized pointers could crash if C API fails
- `session.zig:371,494` - Unsafe pointer casts without alignment checks
- `session.zig:462,585` - `@alignCast` assumes proper alignment without validation
- `tokenizer.zig:184-212` - Potential memory leak on partial tokenization failure

**Tasks:**
- [ ] Add debug assertions for pointer alignment validation
- [ ] Improve error context (which allocator failed, which input caused error)
- [ ] Add cleanup paths for partial failures in tokenizer
- [ ] Replace silent test skips with proper test prerequisites

#### 7.2 Integer Overflow Protection
**Priority:** MEDIUM

**Issue:** `tensor.zig:11-15` - `calcNumel()` has no overflow checking
```zig
pub inline fn calcNumel(shape: []const usize) usize {
    var n: usize = 1;
    for (shape) |d| n *= d;  // Can overflow silently
    return n;
}
```

**Tasks:**
- [ ] Add `calcNumelChecked()` with overflow detection
- [ ] Add tests for edge cases: zero dimensions, overflow scenarios
- [ ] Document expected behavior for degenerate shapes

---

### Phase 8: Performance Optimizations

#### 8.1 FFI Call Reduction
**Priority:** HIGH

**Issue:** `session.zig:353-472` - `CreateCpuMemoryInfo` called every inference

**Tasks:**
- [ ] Cache `memory_info` in Session struct during `init()`
- [ ] Cache allocator pointers
- [ ] Benchmark: measure FFI overhead vs computation time

#### 8.2 Extended SIMD Operations
**Priority:** HIGH

**Missing from SimdOps:**
- [ ] `softmax()` - exp + sum reduction + divide
- [ ] `sigmoid()` - 1 / (1 + exp(-x))
- [ ] `tanh()` - RNN activations
- [ ] `log_softmax()` - numerically stable for loss
- [ ] `argmax()` - index tracking with SIMD compare
- [ ] `gelu()` - transformer activation

**Implementation Notes:**
- Use SIMD exp approximation (polynomial or lookup)
- Implement horizontal reduction for sum
- ARM NEON variants for mobile

#### 8.3 Tensor Pool Optimization
**Priority:** MEDIUM

**Issue:** `tensor_pool.zig:92-99` - Logarithmic bucketing wastes memory

**Tasks:**
- [ ] Implement finer granularity buckets for common sizes
- [ ] Add bucket statistics to identify hot sizes
- [ ] Consider adaptive bucket sizing based on usage patterns
- [ ] Add `trimToSize(max_bytes)` for memory pressure scenarios

#### 8.4 Allocation Overhead Reduction
**Priority:** LOW

**Issue:** `session.zig:400-410` - Allocates array just for pointer conversion

**Tasks:**
- [ ] Pre-compute C pointer arrays during session init
- [ ] Use stack allocation for small fixed-size arrays
- [ ] Profile allocation hotspots with `std.heap.GeneralPurposeAllocator`

---

### Phase 9: Feature Additions

#### 9.1 Extended Data Type Support
**Priority:** HIGH

**Currently Missing:**
```zig
pub fn runF64(self: *Self, inputs: ...) ![]TensorF64  // Double precision
pub fn runI32(self: *Self, inputs: ...) ![]TensorI32  // 32-bit int
pub fn runU8(self: *Self, inputs: ...) ![]TensorU8    // Quantized models
```

**Tasks:**
- [ ] Add `runF64()` method
- [ ] Add `runI32()` method
- [ ] Add `runU8()` for quantized model support
- [ ] Generic `runTyped(comptime T: type)` method
- [ ] Add type conversion utilities

#### 9.2 Batch Processing API
**Priority:** HIGH

**Current Pain Point:** Users must loop for batch inference

**New API:**
```zig
pub fn runF32Batch(
    self: *Self,
    batch_data: []const []const f32,
    batch_shapes: []const []const i64,
    options: BatchOptions,
) !BatchOutput
```

**Tasks:**
- [ ] Design `BatchOptions` struct (max_batch, dynamic padding)
- [ ] Implement batched tensor creation
- [ ] Add batch dimension handling
- [ ] Benchmark batch vs sequential performance

#### 9.3 Execution Provider Configuration
**Priority:** HIGH

**Currently Missing:** No way to configure providers

**New API:**
```zig
pub const SessionOptions = struct {
    execution_providers: []const ExecutionProvider = &.{.cpu},
    num_threads: ?u32 = null,
    graph_optimization: GraphOptLevel = .all,
    memory_pattern: bool = true,
    log_level: LogLevel = .warning,
};

pub const ExecutionProvider = enum {
    cpu,
    cuda,
    tensorrt,
    coreml,
    directml,
    rocm,
};

pub fn initWithOptions(allocator: Allocator, path: []const u8, options: SessionOptions) !Session
```

**Tasks:**
- [ ] Define `SessionOptions` struct
- [ ] Implement provider selection via ORT API
- [ ] Add thread pool configuration
- [ ] Document provider requirements (CUDA toolkit, etc.)

#### 9.4 Model Introspection API
**Priority:** MEDIUM

**New API:**
```zig
pub fn getOpsetVersion(self: *const Self) u32
pub fn getModelMetadata(self: *const Self) ModelMetadata
pub fn getDynamicDimensions(self: *const Self, input_idx: usize) []DynamicDim
pub fn getQuantizationParams(self: *const Self, tensor_name: []const u8) ?QuantParams
```

**Tasks:**
- [ ] Expose opset version
- [ ] Extract custom metadata from model
- [ ] Identify dynamic axes
- [ ] Query quantization scales/zero points

#### 9.5 Streaming Model Loading
**Priority:** LOW

**Use Case:** Load large models progressively

```zig
pub fn initStreaming(allocator: Allocator, reader: anytype) !Session
pub fn loadWeightsAsync(self: *Self, progress_callback: fn(f32) void) !void
```

---

### Phase 10: API Ergonomics

#### 10.1 Shape Type Consistency
**Priority:** MEDIUM

**Issue:** Mixing `i64` (C API) and `usize` (Zig idiom)

**Tasks:**
- [ ] Add shape conversion utilities
- [ ] Create `Shape` wrapper that handles both
- [ ] Overload run methods to accept `TensorF32` directly:
  ```zig
  pub fn run(self: *Self, inputs: []const TensorF32) ![]TensorF32
  ```

#### 10.2 Convenience Constructors
**Priority:** MEDIUM

**New API:**
```zig
// Simple single-input inference
pub fn runF32Simple(self: *Self, data: []const f32, shape: []const usize) !TensorF32

// Named input inference
pub fn runNamed(self: *Self, inputs: std.StringHashMap(TensorF32)) !std.StringHashMap(TensorF32)
```

#### 10.3 Index Lookup Helpers
**Priority:** LOW

**New API:**
```zig
pub fn getInputIndex(self: *const Self, name: []const u8) ?usize
pub fn getOutputIndex(self: *const Self, name: []const u8) ?usize
pub fn getInputByName(self: *const Self, name: []const u8) ?TensorInfo
```

#### 10.4 Additional Tensor Operations
**Priority:** MEDIUM

**Missing Operations:**
- [ ] `matmul(a, b)` - Matrix multiplication
- [ ] `transpose(dims)` - Dimension permutation
- [ ] `reshape(shape)` - With copy when needed
- [ ] `concat(tensors, axis)` - Concatenation
- [ ] `slice(start, end)` - Slicing with ranges
- [ ] `pad(padding, value)` - Padding

---

### Phase 11: Platform Support

#### 11.1 Mobile Platforms
**Priority:** MEDIUM

**iOS Support:**
- [ ] Create `.framework` build target
- [ ] Swift bindings via C ABI
- [ ] CoreML execution provider integration
- [ ] ARM NEON SIMD specialization

**Android Support:**
- [ ] JNI bindings
- [ ] Android NDK build configuration
- [ ] NNAPI execution provider
- [ ] ARM NEON optimizations

#### 11.2 Node.js Native Module
**Priority:** MEDIUM

```javascript
const onnxZig = require('@anthropic/onnx-zig-native');
const session = await onnxZig.createSession('model.onnx');
const output = await session.run({ input: Float32Array.from([...]) });
```

**Tasks:**
- [ ] N-API wrapper for Session
- [ ] Async inference with libuv
- [ ] TypedArray zero-copy transfer
- [ ] Prebuilt binaries for major platforms

#### 11.3 Enhanced WASM Support
**Priority:** MEDIUM

**Current Limitations:**
- Max 256 tensors hardcoded
- Basic ops only, no full inference
- No SharedArrayBuffer support

**Tasks:**
- [ ] Dynamic tensor handle allocation
- [ ] WASM SIMD utilization (wasm-simd128)
- [ ] SharedArrayBuffer for multi-threading
- [ ] Streaming model loading via fetch
- [ ] WebGPU execution provider integration

#### 11.4 Container Support
**Priority:** LOW

**Tasks:**
- [ ] Add `Dockerfile` for reproducible builds
- [ ] Multi-stage build (Zig + ONNX Runtime)
- [ ] GitHub Actions CI/CD
- [ ] Published Docker images

---

### Phase 12: Testing & Quality

#### 12.1 Test Coverage Improvements
**Priority:** HIGH

**Current Gaps:**
- Tests skip silently if models don't exist
- No overflow/edge case tests
- No memory leak detection
- No concurrency tests

**Tasks:**
- [ ] Add test model generation in build step
- [ ] Property-based testing with `std.testing.fuzz`
- [ ] Memory leak tests with allocation tracking
- [ ] Thread safety tests for Session and TensorPool
- [ ] ONNX conformance test suite integration

#### 12.2 Benchmark Suite
**Priority:** MEDIUM

**Tasks:**
- [ ] Tensor operation benchmarks (add, mul, matmul)
- [ ] Inference latency benchmarks (MNIST, BERT, ResNet)
- [ ] Memory usage tracking
- [ ] Comparison with PyTorch, ONNX Runtime Python
- [ ] Performance regression CI checks

#### 12.3 Fuzzing
**Priority:** LOW

**Tasks:**
- [ ] Fuzz tensor shape inputs
- [ ] Fuzz tokenizer with random strings
- [ ] Fuzz model loading with corrupted files
- [ ] AFL/libFuzzer integration

---

### Phase 13: Documentation

#### 13.1 Performance Guide
**Priority:** MEDIUM

**Topics:**
- [ ] When to use TensorPool vs standard allocation
- [ ] SIMD operation characteristics
- [ ] Memory usage per inference
- [ ] Batch size tuning
- [ ] Thread count optimization

#### 13.2 Error Handling Guide
**Priority:** MEDIUM

**Topics:**
- [ ] Common error scenarios and solutions
- [ ] Debugging shape mismatches
- [ ] Model compatibility troubleshooting
- [ ] Memory debugging techniques

#### 13.3 Quantization Guide
**Priority:** LOW

**Topics:**
- [ ] Loading quantized models
- [ ] Dynamic quantization workflow
- [ ] Calibration data requirements
- [ ] Accuracy vs performance tradeoffs

#### 13.4 Concurrency Guide
**Priority:** LOW

**Topics:**
- [ ] Thread safety guarantees
- [ ] Session sharing between threads
- [ ] TensorPool thread safety
- [ ] WASM single-threaded constraints

---

### Phase 14: Extended NLP Support

#### 14.1 Additional Tokenizers
**Priority:** MEDIUM

**Tasks:**
- [ ] BPE (Byte Pair Encoding) tokenizer - GPT models
- [ ] SentencePiece tokenizer - T5, multilingual models
- [ ] Unigram tokenizer
- [ ] Tokenizer configuration loading from JSON

#### 14.2 Text Processing Utilities
**Priority:** LOW

**Tasks:**
- [ ] Unicode normalization (NFC, NFD)
- [ ] Whitespace handling options
- [ ] Special token configuration
- [ ] Vocabulary merging

---

## Priority Matrix

| Priority | Phase | Key Items |
|----------|-------|-----------|
| **P0** | 8.1, 9.1, 12.1 | Cache FFI calls, Data type support, Test fixes |
| **P1** | 7.1, 8.2, 9.2, 9.3 | Error handling, SIMD ops, Batch API, Providers |
| **P2** | 10.1-10.4, 11.3, 13.1 | API ergonomics, WASM, Performance docs |
| **P3** | 11.1, 11.2, 12.2, 14.1 | Mobile, Node.js, Benchmarks, Tokenizers |
| **P4** | 11.4, 12.3, 13.2-13.4 | Docker, Fuzzing, Guides |

---

## Specific Code Issues Tracker

| File | Line | Issue | Severity | Status |
|------|------|-------|----------|--------|
| `session.zig` | 462 | Unsafe pointer alignment | HIGH | TODO |
| `tensor.zig` | 11 | Integer overflow in calcNumel | MEDIUM | TODO |
| `tokenizer.zig` | 184 | Memory leak on partial failure | MEDIUM | TODO |
| `session.zig` | 778 | Silent test skip | MEDIUM | TODO |
| `session.zig` | 353 | Repeated FFI memory_info creation | MEDIUM | TODO |
| `wasm_exports.zig` | 288 | No error context in ops | LOW | TODO |
| `tensor_pool.zig` | 92 | Suboptimal bucket sizing | LOW | TODO |

---

## Completed Milestones

- [x] Milestone 1: Tensor Foundation
- [x] Milestone 2: ONNX Runtime FFI Integration
- [x] Milestone 3: Session API
- [x] Milestone 4: Memory Management (Arena, Pool)
- [x] Milestone 5: WASM Build
- [x] Milestone 6: CLI Tool
- [x] Milestone 7: Examples (Basic, MNIST, BERT)
- [x] Milestone 8: WordPiece Tokenizer
- [x] Milestone 9: Browser Demos (MNIST, Embeddings)
