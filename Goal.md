MAIN GOAL :- build zig onnx wasm inference engine for edge computing.

---

## Milestone 1: Tensor Foundation [COMPLETE]
- [x] 1.1 Create `src/tensor.zig` with basic Tensor struct
- [x] 1.2 Implement shape and dimension handling
- [x] 1.3 Implement stride calculation and memory layout
- [x] 1.4 Add element access (get/set by indices)
- [x] 1.5 Implement tensor iteration
- [x] 1.6 Add tensor creation utilities (zeros, ones, from_slice)
- [x] 1.7 Write unit tests for tensor operations (12 tests passing)

## Milestone 2: ONNX Runtime FFI Bindings [COMPLETE]
- [x] 2.1 Create `src/onnxruntime.zig` with C API type declarations
- [x] 2.2 Define OrtApi struct with core function pointers
- [x] 2.3 Implement OrtEnv wrapper (environment management)
- [x] 2.4 Implement OrtSessionOptions wrapper
- [x] 2.5 Implement OrtSession wrapper (model loading)
- [x] 2.6 Implement OrtValue wrapper (tensor I/O)
- [x] 2.7 Update build.zig to link ONNX Runtime library
- [x] 2.8 Test FFI bindings with simple model inference
- [x] 2.9 Add error handling and status code translation

## Milestone 3: High-Level Inference API [COMPLETE]
- [x] 3.1 Create `src/session.zig` with idiomatic Zig Session type
- [x] 3.2 Implement model loading from file path
- [x] 3.3 Implement input tensor binding (Zig Tensor -> OrtValue)
- [x] 3.4 Implement inference execution (run method)
- [x] 3.5 Implement output tensor retrieval (OrtValue -> Zig Tensor)
- [x] 3.6 Add input/output name and shape introspection
- [x] 3.7 Write comprehensive tests with MNIST or similar model
- [x] 3.8 Document public API

## Milestone 4: Memory & Performance Optimization [COMPLETE]
- [x] 4.1 Implement memory arena for session-scoped allocations
- [x] 4.2 Add tensor pool for intermediate results
- [x] 4.3 Implement zero-copy tensor views where possible
- [x] 4.4 Profile and optimize hot paths
- [x] 4.5 Add SIMD hints for Zig tensor operations

## Milestone 5: WASM Build [COMPLETE]
- [x] 5.1 Add WASM target to build.zig
- [x] 5.2 Configure ONNX Runtime WASM build (onnxruntime-web integration)
- [x] 5.3 Create JavaScript interop layer
- [x] 5.4 Handle WASM memory constraints
- [x] 5.5 Test in browser environment
- [x] 5.6 Optimize binary size (~11KB)

## Milestone 6: CLI Tool [COMPLETE]
- [x] 6.1 Create `tools/cli.zig` with command interface
- [x] 6.2 Implement `info` command for model inspection
- [x] 6.3 Implement `bench` command for performance benchmarking
- [x] 6.4 Implement `run` command for inference execution
- [x] 6.5 Implement `validate` command for model verification

## Milestone 7: Examples & Demos [COMPLETE]
- [x] 7.1 Create `examples/zig/basic_inference.zig`
- [x] 7.2 Create `examples/zig/mnist_classifier.zig`
- [x] 7.3 Create MNIST browser demo
- [x] 7.4 Create JavaScript/TypeScript SDK (`js/src/index.ts`)
- [x] 7.5 Add comprehensive README documentation

## Milestone 8: WordPiece Tokenizer [COMPLETE]
- [x] 8.1 Implement WordPiece tokenization algorithm
- [x] 8.2 Add vocabulary loading from file
- [x] 8.3 Implement BERT-style tokenization (CLS, SEP, PAD)
- [x] 8.4 Add attention mask generation

## Milestone 9: Browser Demos [COMPLETE]
- [x] 9.1 MNIST digit recognition demo (97% accuracy)
- [x] 9.2 Embeddings demo with onnxruntime-web integration
- [x] 9.3 Real-time inference in browser

---

## Milestone 10: Extended Features [COMPLETE]
- [x] 10.1 Add execution provider selection (CPU, CUDA, CoreML, etc.)
- [x] 10.2 Implement session configuration options (SessionOptions struct)
- [x] 10.3 Add model metadata introspection (getModelMetadata)
- [x] 10.4 Implement multi-threaded inference options (intra/inter op threads)
- [x] 10.5 Add quantized model support (INT8/U8)
- [x] 10.6 Add runU8() method for quantized inference

## Milestone 11: Code Quality & Robustness [COMPLETE]
- [x] 11.1 Add debug assertions for pointer alignment validation
- [x] 11.2 Improve error context (allocator failures, input errors)
- [x] 11.3 Add cleanup paths for partial failures in tokenizer
- [x] 11.4 Replace silent test skips with proper test prerequisites
- [x] 11.5 Add `calcNumelChecked()` with overflow detection
- [x] 11.6 Add tests for edge cases (zero dimensions, overflow)

## Milestone 12: Performance Optimizations [COMPLETE]
- [x] 12.1 Cache `memory_info` in Session struct during init
- [x] 12.2 Cache allocator pointers to reduce FFI calls
- [x] 12.3 Add SIMD `softmax()`, `sigmoid()`, `tanh()`, `gelu()`
- [x] 12.4 Add SIMD `log_softmax()`, `argmax()`
- [x] 12.5 Implement finer granularity tensor pool buckets
- [x] 12.6 Pre-compute C pointer arrays during session init

## Milestone 13: Extended Data Types & Batch API [COMPLETE]
- [x] 13.1 Add `runF64()` method for double precision
- [x] 13.2 Add `runI32()` method for 32-bit integers
- [x] 13.3 Add generic `runTypedImpl()` internal method
- [x] 13.4 Design and implement `BatchOptions` struct
- [x] 13.5 Implement `runF32Batch()` for batched inference
- [x] 13.6 Implement `runF32BatchClassify()` for classification
- [x] 13.7 Create identity_batch.onnx with dynamic batch dimension

## Milestone 14: API Ergonomics [COMPLETE]
- [x] 14.1 Add shape conversion utilities (i64 <-> usize)
- [x] 14.2 Add `runTensor()`, `runTensors()` accepting TensorF32 directly
- [x] 14.3 Add `runF32Simple()` for single-input inference
- [x] 14.4 Add `runNamed()` and `runWithMap()` for named inputs
- [x] 14.5 Add `getInputIndex()`, `getOutputIndex()`, `hasInput()`, `hasOutput()` helpers
- [x] 14.6 Add tensor ops: matmul, transpose, concat, sliceAxis0, pad

## Milestone 15: Platform Support [IN PROGRESS]
- [x] 15.1 iOS: C header and shared/static library targets
- [x] 15.2 iOS: Swift bindings via C ABI (`swift/OnnxZig.swift`)
- [ ] 15.3 Android: JNI bindings
- [ ] 15.4 Android: NDK build configuration
- [x] 15.5 Node.js: N-API wrapper for Session (`node/`)
- [x] 15.6 WASM: Dynamic tensor handle allocation with stats
- [ ] 15.7 WASM: SharedArrayBuffer for multi-threading
- [ ] 15.8 WASM: WebGPU execution provider integration

## Milestone 16: Testing & Quality
- [ ] 16.1 Add test model generation in build step
- [ ] 16.2 Property-based testing with `std.testing.fuzz`
- [ ] 16.3 Memory leak tests with allocation tracking
- [ ] 16.4 Thread safety tests for Session and TensorPool
- [ ] 16.5 ONNX conformance test suite integration
- [ ] 16.6 Benchmark suite (tensor ops, inference latency)
- [ ] 16.7 Performance regression CI checks

## Milestone 17: Documentation
- [ ] 17.1 Performance guide (TensorPool, SIMD, memory, threading)
- [ ] 17.2 Error handling guide (common errors, debugging)
- [ ] 17.3 Quantization guide (loading, calibration, tradeoffs)
- [ ] 17.4 Concurrency guide (thread safety, WASM constraints)

## Milestone 18: Extended NLP Support
- [ ] 18.1 BPE (Byte Pair Encoding) tokenizer for GPT models
- [ ] 18.2 SentencePiece tokenizer for T5/multilingual
- [ ] 18.3 Unigram tokenizer
- [ ] 18.4 Tokenizer configuration loading from JSON
- [ ] 18.5 Unicode normalization (NFC, NFD)

## Milestone 19: DevOps & Distribution
- [ ] 19.1 Add Dockerfile for reproducible builds
- [ ] 19.2 Multi-stage Docker build (Zig + ONNX Runtime)
- [ ] 19.3 GitHub Actions CI/CD pipeline
- [ ] 19.4 Published Docker images
- [ ] 19.5 npm package publishing workflow

---

## Current Focus: Milestone 15 - Platform Support (In Progress)

Milestones 1-14 are complete! The inference engine can now:
- Load ONNX models from file (native)
- Run inference with f32, f64, i32, i64, and u8 tensors
- Introspect model inputs/outputs
- Handle errors gracefully with improved context
- Reuse memory efficiently (arena allocator, optimized tensor pool)
- Perform SIMD-optimized tensor operations
- **Build to WASM (~11KB) for browser deployment**
- **Integrate with onnxruntime-web via JavaScript bridge**
- **MNIST digit recognition demo (97% accuracy)**
- **CLI tools for model inspection and benchmarking**
- **WordPiece tokenizer for BERT models**
- **Quantized model support (INT8/U8)**
- **Session configuration options**
- **Multi-threaded inference options**
- **Debug assertions for pointer alignment**
- **Overflow-checked arithmetic (calcNumelChecked)**
- **Improved test skip messages with prerequisites**
- **Cached memory_info and C pointer arrays**
- **SIMD activation functions (softmax, sigmoid, tanh, gelu, logSoftmax, argmax)**
- **Fine-grained tensor pool buckets (25 size classes)**
- **Extended data types: runF64(), runI32(), runI32ToI32()**
- **Batched inference: runF32Batch(), runF32BatchClassify()**
- **Dynamic batch dimension support**
- **Shape conversion utilities (shapeI64ToUsize, shapeUsizeToI64)**
- **Simplified inference: runF32Simple(), runTensor(), runTensors()**
- **Named tensor API: runNamed(), runWithMap()**
- **Input/output helpers: getInputIndex(), getOutputIndex(), hasInput(), hasOutput()**
- **Tensor operations: matmul, transpose, concat, sliceAxis0, pad**
- **C ABI exports for FFI integration**
- **Swift bindings for iOS/macOS**
- **Node.js N-API addon**
- **WASM dynamic tensor pool with stats**

Next priorities:
- Android JNI bindings and NDK build
- WASM SharedArrayBuffer support
- WebGPU execution provider

## Demo

Run the MNIST demo:
```bash
cd zig-out/wasm
python3 -m http.server 8080
# Open http://localhost:8080/mnist_demo.html
```

---

## Code Issues Tracker

| File | Line | Issue | Severity | Status |
|------|------|-------|----------|--------|
| `session.zig` | 462 | Unsafe pointer alignment | HIGH | FIXED (assertAligned) |
| `tensor.zig` | 11 | Integer overflow in calcNumel | MEDIUM | FIXED (calcNumelChecked) |
| `tokenizer.zig` | 184 | Memory leak on partial failure | MEDIUM | FIXED (errdefer added) |
| `session.zig` | 778 | Silent test skip | MEDIUM | FIXED (loadTestSession helper) |
| `session.zig` | 353 | Repeated FFI memory_info creation | MEDIUM | FIXED (cached in Session struct) |
| `wasm_exports.zig` | 288 | No error context in ops | LOW | TODO |
| `tensor_pool.zig` | 92 | Suboptimal bucket sizing | LOW | FIXED (25 fine-grained buckets) |

---

## Priority Matrix

| Priority | Milestone | Key Items |
|----------|-----------|-----------|
| **P0** | 10, 11 | Quantized support, Error handling, Test fixes |
| **P1** | 12, 13 | SIMD ops, Batch API, Data types |
| **P2** | 14, 15 | API ergonomics, Platform support |
| **P3** | 16, 17 | Testing, Documentation |
| **P4** | 18, 19 | NLP tokenizers, DevOps |
