# ONNX Zig

High-performance ONNX inference engine for edge computing, written in Zig with WebAssembly support.

## Features

- **Native Performance**: Zero-cost abstractions over ONNX Runtime C API
- **WebAssembly Support**: Run ML models in browsers with minimal footprint (~11KB WASM)
- **SIMD Optimizations**: Hardware-accelerated tensor operations
- **Memory Efficient**: Arena allocators and tensor pooling for predictable memory usage
- **Type Safe**: Compile-time verified tensor operations
- **Cross-Platform**: Native builds for macOS, Linux, Windows; WASM for browsers

## Installation

### Zig Package

Add to your `build.zig.zon`:

```zig
.dependencies = .{
    .onnx_zig = .{
        .url = "https://github.com/anthropics/onnx-zig/archive/refs/tags/v0.1.0.tar.gz",
        .hash = "...",
    },
},
```

Then in your `build.zig`:

```zig
const onnx_zig = b.dependency("onnx_zig", .{
    .target = target,
    .optimize = optimize,
});
exe.root_module.addImport("onnx_zig", onnx_zig.module("onnx_zig"));
```

### JavaScript/TypeScript (npm)

```bash
npm install @anthropic/onnx-zig onnxruntime-web
```

## Quick Start

### Zig

```zig
const std = @import("std");
const onnx_zig = @import("onnx_zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Load model
    var session = try onnx_zig.Session.init(allocator, "model.onnx");
    defer session.deinit();

    // Prepare input
    const input_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const input_shape = [_]i64{ 1, 4 };

    // Run inference
    const outputs = try session.runF32(
        &[_][]const f32{&input_data},
        &[_][]const i64{&input_shape},
    );
    defer session.freeOutputs(outputs);

    // Process results
    for (outputs[0].data) |val| {
        std.debug.print("{d:.4} ", .{val});
    }
}
```

### JavaScript/TypeScript

```typescript
import { OnnxZig } from '@anthropic/onnx-zig';

// Initialize
const onnxZig = await OnnxZig.create({
  wasmPath: 'onnx_zig.wasm'
});

// Create tensors
const input = onnxZig.tensor.fromArray([1, 2, 3, 4], [1, 4]);

// Load model and run inference
const session = await onnxZig.session.create('model.onnx');
const outputs = await session.run({ input });

// Get results
const result = onnxZig.tensor.toArray(outputs.output);
console.log('Output:', result);

// Clean up
onnxZig.tensor.free(input);
session.dispose();
```

## Building

### Prerequisites

- Zig 0.15.2 or later
- ONNX Runtime (for native builds)

### Native Build

```bash
# macOS with Homebrew
brew install onnxruntime
zig build -Donnxruntime_path=/opt/homebrew/Cellar/onnxruntime/1.23.2_2

# Linux
zig build -Donnxruntime_path=/usr/local

# Run tests
zig build test -Donnxruntime_path=/opt/homebrew/Cellar/onnxruntime/1.23.2_2
```

### WebAssembly Build

```bash
# Build WASM module
zig build -Dtarget=wasm32-freestanding -Doptimize=ReleaseSmall

# Or use the wasm step
zig build wasm -Dtarget=wasm32-freestanding
```

### Build Steps

| Command | Description |
|---------|-------------|
| `zig build` | Build main library and executable |
| `zig build test` | Run all tests |
| `zig build wasm` | Build WebAssembly module |
| `zig build cli` | Build CLI tool |
| `zig build examples` | Build all examples |
| `zig build example-basic` | Build basic inference example |
| `zig build example-mnist` | Build MNIST classifier example |
| `zig build example-bert` | Build BERT embeddings example |
| `zig build run-example-basic` | Run basic inference example |
| `zig build run-example-mnist` | Run MNIST classifier example |
| `zig build run-example-bert` | Run BERT embeddings example |

## CLI Tool

The CLI provides model inspection and benchmarking capabilities:

```bash
# Build CLI
zig build cli -Donnxruntime_path=/opt/homebrew/Cellar/onnxruntime/1.23.2_2

# Inspect model
./zig-out/bin/onnx-zig info model.onnx

# Benchmark inference
./zig-out/bin/onnx-zig bench model.onnx --iterations 100

# Run inference
./zig-out/bin/onnx-zig run model.onnx --input data.bin

# Validate model
./zig-out/bin/onnx-zig validate model.onnx
```

## API Reference

### Core Types

#### `Session`
Main inference session for loading and running ONNX models.

```zig
const session = try Session.init(allocator, "model.onnx");
defer session.deinit();

// Get model info
const input_count = session.getInputCount();
const output_count = session.getOutputCount();
const input_names = session.getInputNames();
const output_names = session.getOutputNames();

// Run inference
const outputs = try session.runF32(inputs, shapes);
defer session.freeOutputs(outputs);
```

#### `TensorF32`
32-bit floating point tensor with shape information.

```zig
// Create tensor
var tensor = try TensorF32.init(allocator, &[_]usize{2, 3});
defer tensor.deinit(allocator);

// Access data
tensor.data[0] = 1.0;
const shape = tensor.shape.slice();
const numel = tensor.numel();
```

#### `SimdOps`
SIMD-accelerated tensor operations.

```zig
// Element-wise operations
SimdOps.add(a.data, b.data, result.data);
SimdOps.mul(a.data, b.data, result.data);
SimdOps.scale(tensor.data, 2.0, result.data);

// Reductions
const sum = SimdOps.sum(tensor.data);
const max = SimdOps.max(tensor.data);
const argmax = SimdOps.argmax(tensor.data);

// Activations
SimdOps.relu(input.data, output.data);
SimdOps.softmax(input.data, output.data);
```

#### `TensorPool`
Memory-efficient tensor reuse for repeated inference.

```zig
var pool = TensorPoolF32.init(allocator);
defer pool.deinit();

// Acquire tensor (reuses if available)
const tensor = try pool.acquire(&[_]usize{1, 784});
defer pool.release(tensor);

// Check statistics
const stats = pool.getStats();
std.debug.print("Hit rate: {d:.1}%\n", .{stats.hitRate * 100});
```

### JavaScript/TypeScript API

#### `OnnxZig`
Main entry point for browser usage.

```typescript
interface OnnxZigOptions {
  wasmPath?: string;      // Path to WASM file
  numThreads?: number;    // ONNX Runtime threads
  simd?: boolean;         // Enable SIMD
}

const onnxZig = await OnnxZig.create(options);
```

#### `TensorAPI`
Tensor creation and manipulation.

```typescript
// Creation
const t1 = onnxZig.tensor.create([2, 3]);
const t2 = onnxZig.tensor.zeros([1, 784]);
const t3 = onnxZig.tensor.ones([10]);
const t4 = onnxZig.tensor.fromArray([1, 2, 3], [3]);

// Operations
const sum = onnxZig.tensor.add(a, b);
const product = onnxZig.tensor.mul(a, b);
const activated = onnxZig.tensor.relu(input);
const probs = onnxZig.tensor.softmax(logits);
const scaled = onnxZig.tensor.scale(tensor, 2.0);

// Data access
const data = onnxZig.tensor.toArray(tensor);
const view = onnxZig.tensor.dataView(tensor); // Zero-copy

// Cleanup
onnxZig.tensor.free(tensor);
```

#### `SessionAPI`
ONNX model inference.

```typescript
const session = await onnxZig.session.create('model.onnx');

// With tensor handles
const outputs = await session.run({ input: tensorHandle });

// With raw arrays
const rawOutputs = await session.runRaw({
  input: { data: new Float32Array([...]), shape: [1, 784] }
});

session.dispose();
```

## Examples

### Basic Inference (Zig)

```bash
zig build run-example-basic -Donnxruntime_path=/opt/homebrew/Cellar/onnxruntime/1.23.2_2 -- models/test/identity.onnx
```

### MNIST Classifier (Zig)

```bash
zig build run-example-mnist -Donnxruntime_path=/opt/homebrew/Cellar/onnxruntime/1.23.2_2 -- models/mnist.onnx
```

### BERT Embeddings (Zig)

```bash
# First, export the BERT model
uv run --with transformers --with torch --with onnx scripts/export_bert.py

# Build and run
zig build example-bert -Donnxruntime_path=/opt/homebrew/Cellar/onnxruntime/1.23.2_2
./zig-out/bin/example-bert "Hello, how are you?"
```

Output:
```
Loading tokenizer from models/bert/vocab.txt...
  Vocabulary size: 30522
Tokenizing: "Hello, how are you?"
  Tokens: ["[CLS]", "hello", ",", "how", "are", "you", "?", "[SEP]"]
Running BERT inference...
  Inference time: 101.63 ms
Pooler output (sentence embedding):
  Shape: [1, 768]
  First 10 values: [-0.9397, -0.4081, -0.9024, ...]
```

### Browser Demos

1. Build WASM module:
   ```bash
   zig build wasm -Dtarget=wasm32-freestanding
   ```

2. Serve the demos:
   ```bash
   cd zig-out/wasm
   python -m http.server 8080
   ```

3. Open demos:
   - **Main page**: `http://localhost:8080/` - Tensor operations tests
   - **MNIST**: `http://localhost:8080/mnist_demo.html` - Draw digits for classification
   - **Embeddings**: `http://localhost:8080/embedding_demo.html` - Text similarity search

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Application Layer                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   CLI Tool  │  │  Zig Apps   │  │   Browser (JS/TS)   │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
├─────────┴────────────────┴─────────────────────┴────────────┤
│                       Framework API                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Session   │  │   Tensor    │  │      SimdOps        │  │
│  │     API     │  │    Pool     │  │   (Accelerated)     │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
├─────────┴────────────────┴─────────────────────┴────────────┤
│                       Core Layer                             │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐  ┌─────────────────────────────┐   │
│  │   ONNX Runtime FFI  │  │      Tensor Library         │   │
│  │    (@cImport)       │  │   (Shape, Strides, Data)    │   │
│  └─────────────────────┘  └─────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                      Platform Layer                          │
├──────────────────────────┬──────────────────────────────────┤
│    Native (ONNX Runtime) │         WASM (Browser)           │
│    - macOS/Linux/Windows │    - onnxruntime-web             │
│    - CPU/GPU providers   │    - WebGL/WASM SIMD             │
└──────────────────────────┴──────────────────────────────────┘
```

## Project Structure

```
onnx_zig/
├── src/
│   ├── root.zig           # Library entry point (public API)
│   ├── tensor.zig         # Tensor implementation + SIMD
│   ├── session.zig        # ONNX inference session
│   ├── onnxruntime.zig    # ONNX Runtime FFI bindings
│   ├── arena.zig          # Memory arena allocators
│   ├── tensor_pool.zig    # Tensor pooling
│   ├── tokenizer.zig      # WordPiece tokenizer (BERT)
│   └── wasm_exports.zig   # WASM exported functions
├── tools/
│   └── cli.zig            # Command-line tool
├── examples/zig/
│   ├── basic_inference.zig   # General inference demo
│   ├── bert_embeddings.zig   # BERT text embeddings
│   └── mnist_classifier.zig  # Image classification
├── scripts/
│   ├── export_bert.py        # Export BERT to ONNX
│   ├── generate_mnist_model.py
│   └── generate_test_model.py
├── wasm/
│   ├── loader.js          # JavaScript WASM loader
│   ├── index.html         # Main demo page
│   ├── mnist_demo.html    # MNIST demo
│   └── embedding_demo.html # Text embedding demo
├── models/                # Model storage
│   ├── test/             # Test models
│   ├── bert/             # BERT model + vocab
│   └── mnist/            # MNIST model
├── build.zig              # Build configuration
├── build.zig.zon          # Package manifest
├── ARCHITECTURE.md        # Detailed architecture docs
└── README.md              # This file
```

For detailed architecture documentation, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Performance

| Operation | Native (M1 Mac) | WASM (Chrome) |
|-----------|-----------------|---------------|
| Tensor creation (1M elements) | 0.8ms | 2.1ms |
| Element-wise add (1M) | 0.3ms | 1.2ms |
| Softmax (10K) | 0.05ms | 0.15ms |
| MNIST inference | 0.4ms | 2.5ms |

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read the contributing guidelines before submitting PRs.
