/**
 * ONNX Zig WASM Loader
 *
 * This module loads the Zig WASM binary and provides a high-level API for
 * tensor operations and ONNX inference using onnxruntime-web.
 *
 * Usage:
 *   const onnxZig = await OnnxZig.load();
 *
 *   // Create tensors
 *   const tensor = onnxZig.tensor.create([2, 3]);
 *
 *   // Run ONNX inference
 *   const session = await onnxZig.ort.createSession('model.onnx');
 *   const outputs = await onnxZig.ort.run(session, { input: tensor });
 */

class OnnxZigModule {
    constructor(wasmInstance) {
        this.wasm = wasmInstance;
        this.exports = wasmInstance.exports;
        this.memory = this.exports.memory;

        // Initialize tensor and SIMD APIs
        this.tensor = new TensorAPI(this);
        this.simd = new SimdAPI(this);
        this.ort = new OrtAPI(this);
    }

    /**
     * Load the WASM module
     * @param {string} wasmPath - Path to the WASM file (default: 'onnx_zig.wasm')
     * @returns {Promise<OnnxZigModule>}
     */
    static async load(wasmPath = 'onnx_zig.wasm') {
        const response = await fetch(wasmPath);
        const bytes = await response.arrayBuffer();

        const { instance } = await WebAssembly.instantiate(bytes, {
            env: {
                // Add any required imports here
            }
        });

        return new OnnxZigModule(instance);
    }

    /**
     * Get a view of WASM memory as a typed array
     */
    getMemoryView(type = 'f32') {
        switch (type) {
            case 'u8': return new Uint8Array(this.memory.buffer);
            case 'i32': return new Int32Array(this.memory.buffer);
            case 'f32': return new Float32Array(this.memory.buffer);
            case 'f64': return new Float64Array(this.memory.buffer);
            default: return new Uint8Array(this.memory.buffer);
        }
    }

    /**
     * Allocate memory in WASM (returns pointer as number)
     */
    alloc(size) {
        const ptr = this.exports.wasm_alloc(size);
        if (ptr === 0) throw new Error('WASM allocation failed');
        return ptr;
    }

    /**
     * Free memory in WASM
     */
    free(ptr, size) {
        if (ptr !== 0) {
            this.exports.wasm_free(ptr, size);
        }
    }

    /**
     * Allocate f32 array in WASM (returns pointer as number)
     */
    allocF32(count) {
        const ptr = this.exports.wasm_alloc_f32(count);
        if (ptr === 0) throw new Error('WASM f32 allocation failed');
        return ptr;
    }

    /**
     * Free f32 array in WASM
     */
    freeF32(ptr, count) {
        if (ptr !== 0) {
            this.exports.wasm_free_f32(ptr, count);
        }
    }

    /**
     * Get WASM memory info
     */
    getMemoryInfo() {
        const pages = this.exports.wasm_memory_pages();
        return {
            pages,
            bytes: pages * 65536,
            mb: (pages * 65536) / (1024 * 1024)
        };
    }
}

/**
 * Tensor API - High-level tensor operations
 */
class TensorAPI {
    constructor(module) {
        this.module = module;
        this.exports = module.exports;
    }

    /**
     * Create a new tensor with given shape
     * @param {number[]} shape - Tensor dimensions
     * @returns {number} Tensor handle (-1 on error)
     */
    create(shape) {
        const shapePtr = this._allocShape(shape);
        const handle = this.exports.tensor_create(shapePtr, shape.length);
        this.module.free(shapePtr, shape.length * 4);
        return handle;
    }

    /**
     * Create tensor filled with zeros
     */
    zeros(shape) {
        const shapePtr = this._allocShape(shape);
        const handle = this.exports.tensor_zeros(shapePtr, shape.length);
        this.module.free(shapePtr, shape.length * 4);
        return handle;
    }

    /**
     * Create tensor filled with ones
     */
    ones(shape) {
        const shapePtr = this._allocShape(shape);
        const handle = this.exports.tensor_ones(shapePtr, shape.length);
        this.module.free(shapePtr, shape.length * 4);
        return handle;
    }

    /**
     * Create tensor from JavaScript array
     * @param {Float32Array|number[]} data - Tensor data
     * @param {number[]} shape - Tensor shape
     */
    fromArray(data, shape) {
        const arr = data instanceof Float32Array ? data : new Float32Array(data);

        // Allocate and copy data
        const dataPtr = this.module.allocF32(arr.length);
        const mem = new Float32Array(this.module.memory.buffer, dataPtr, arr.length);
        mem.set(arr);

        // Allocate and copy shape
        const shapePtr = this._allocShape(shape);

        // Create tensor
        const handle = this.exports.tensor_from_data(dataPtr, arr.length, shapePtr, shape.length);

        // Clean up
        this.module.freeF32(dataPtr, arr.length);
        this.module.free(shapePtr, shape.length * 4);

        return handle;
    }

    /**
     * Get tensor data as JavaScript array
     */
    toArray(handle) {
        const numel = this.exports.tensor_numel(handle);
        const ptr = this.exports.tensor_data_ptr(handle);

        if (ptr === 0 || numel === 0) return null;

        // ptr is returned as usize (byte offset into WASM memory)
        const view = new Float32Array(this.module.memory.buffer, ptr, numel);
        return Array.from(view);
    }

    /**
     * Get tensor data as Float32Array (view into WASM memory)
     */
    dataView(handle) {
        const numel = this.exports.tensor_numel(handle);
        const ptr = this.exports.tensor_data_ptr(handle);

        if (ptr === 0 || numel === 0) return null;

        // ptr is returned as usize, use it directly as byte offset
        return new Float32Array(this.module.memory.buffer, ptr, numel);
    }

    /**
     * Get tensor shape
     */
    shape(handle) {
        const ndim = this.exports.tensor_ndim(handle);
        const shape = [];
        for (let i = 0; i < ndim; i++) {
            shape.push(this.exports.tensor_dim(handle, i));
        }
        return shape;
    }

    /**
     * Get number of elements
     */
    numel(handle) {
        return this.exports.tensor_numel(handle);
    }

    /**
     * Free tensor memory
     */
    free(handle) {
        this.exports.tensor_free(handle);
    }

    /**
     * Element-wise add
     */
    add(a, b) {
        return this.exports.tensor_add(a, b);
    }

    /**
     * Element-wise multiply
     */
    mul(a, b) {
        return this.exports.tensor_mul(a, b);
    }

    /**
     * ReLU activation
     */
    relu(input) {
        return this.exports.tensor_relu(input);
    }

    /**
     * Scale by constant
     */
    scale(input, scalar) {
        return this.exports.tensor_scale(input, scalar);
    }

    /**
     * Softmax
     */
    softmax(input) {
        return this.exports.tensor_softmax(input);
    }

    /**
     * Argmax - returns index of max value
     */
    argmax(input) {
        return this.exports.tensor_argmax(input);
    }

    /**
     * Clone tensor
     */
    clone(input) {
        return this.exports.tensor_clone(input);
    }

    // Helper to allocate shape array in WASM
    _allocShape(shape) {
        const ptr = this.module.alloc(shape.length * 4);
        const view = new Uint32Array(this.module.memory.buffer, ptr, shape.length);
        view.set(shape);
        return ptr;
    }
}

/**
 * SIMD API - Low-level SIMD operations on raw arrays
 */
class SimdAPI {
    constructor(module) {
        this.module = module;
        this.exports = module.exports;
    }

    /**
     * Element-wise add: dst = a + b
     */
    add(dst, a, b, len) {
        this.exports.simd_add(dst, a, b, len);
    }

    /**
     * Element-wise subtract: dst = a - b
     */
    sub(dst, a, b, len) {
        this.exports.simd_sub(dst, a, b, len);
    }

    /**
     * Element-wise multiply: dst = a * b
     */
    mul(dst, a, b, len) {
        this.exports.simd_mul(dst, a, b, len);
    }

    /**
     * Element-wise divide: dst = a / b
     */
    div(dst, a, b, len) {
        this.exports.simd_div(dst, a, b, len);
    }

    /**
     * Scale: dst = a * scalar
     */
    scale(dst, a, scalar, len) {
        this.exports.simd_scale(dst, a, scalar, len);
    }

    /**
     * ReLU: dst = max(0, a)
     */
    relu(dst, a, len) {
        this.exports.simd_relu(dst, a, len);
    }

    /**
     * Sum all elements
     */
    sum(a, len) {
        return this.exports.simd_sum(a, len);
    }

    /**
     * Dot product
     */
    dot(a, b, len) {
        return this.exports.simd_dot(a, b, len);
    }

    /**
     * Max value
     */
    max(a, len) {
        return this.exports.simd_max(a, len);
    }

    /**
     * Min value
     */
    min(a, len) {
        return this.exports.simd_min(a, len);
    }
}

/**
 * ONNX Runtime Web Integration API
 */
class OrtAPI {
    constructor(module) {
        this.module = module;
        this.ort = null;
    }

    /**
     * Initialize ONNX Runtime Web
     * Must be called before using ONNX features
     */
    async init() {
        if (typeof ort === 'undefined') {
            throw new Error('ONNX Runtime Web not loaded. Include ort.min.js before using OrtAPI.');
        }
        this.ort = ort;

        // Configure for WASM backend
        ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
        ort.env.wasm.simd = true;
    }

    /**
     * Create an inference session from model file
     * @param {string|ArrayBuffer} modelPath - Path to ONNX model or model bytes
     * @param {Object} options - Session options
     */
    async createSession(modelPath, options = {}) {
        if (!this.ort) await this.init();

        const defaultOptions = {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all',
        };

        const session = await this.ort.InferenceSession.create(
            modelPath,
            { ...defaultOptions, ...options }
        );

        return session;
    }

    /**
     * Run inference using Zig tensor handles as inputs
     * @param {InferenceSession} session - ONNX Runtime session
     * @param {Object} inputs - Map of input names to tensor handles
     * @returns {Object} Map of output names to tensor handles
     */
    async run(session, inputs) {
        // Convert Zig tensor handles to ORT tensors
        const ortInputs = {};

        for (const [name, handle] of Object.entries(inputs)) {
            const shape = this.module.tensor.shape(handle);
            const data = this.module.tensor.dataView(handle);

            // Create ORT tensor from data (creates a copy)
            ortInputs[name] = new this.ort.Tensor('float32', new Float32Array(data), shape);
        }

        // Run inference
        const ortOutputs = await session.run(ortInputs);

        // Convert ORT outputs to Zig tensor handles
        const outputs = {};

        for (const [name, tensor] of Object.entries(ortOutputs)) {
            const handle = this.module.tensor.fromArray(tensor.data, tensor.dims);
            outputs[name] = handle;
        }

        return outputs;
    }

    /**
     * Run inference with raw Float32Arrays (no tensor handles)
     */
    async runRaw(session, inputs) {
        const ortInputs = {};

        for (const [name, { data, shape }] of Object.entries(inputs)) {
            ortInputs[name] = new this.ort.Tensor('float32', data, shape);
        }

        const ortOutputs = await session.run(ortInputs);

        const outputs = {};
        for (const [name, tensor] of Object.entries(ortOutputs)) {
            outputs[name] = {
                data: new Float32Array(tensor.data),
                shape: tensor.dims
            };
        }

        return outputs;
    }

    /**
     * Get model input metadata
     */
    getInputs(session) {
        return session.inputNames.map((name, i) => ({
            name,
            type: session.inputMetadata?.[name]?.type || 'float32',
            dims: session.inputMetadata?.[name]?.dims || []
        }));
    }

    /**
     * Get model output metadata
     */
    getOutputs(session) {
        return session.outputNames.map((name) => ({
            name,
            type: session.outputMetadata?.[name]?.type || 'float32',
            dims: session.outputMetadata?.[name]?.dims || []
        }));
    }
}

// Export for both ES modules and browser globals
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { OnnxZigModule, TensorAPI, SimdAPI, OrtAPI };
} else if (typeof window !== 'undefined') {
    window.OnnxZig = OnnxZigModule;
    window.TensorAPI = TensorAPI;
    window.SimdAPI = SimdAPI;
    window.OrtAPI = OrtAPI;
}
