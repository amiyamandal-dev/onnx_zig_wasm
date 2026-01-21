/**
 * ONNX Zig - High-performance ONNX inference for browsers
 *
 * @packageDocumentation
 *
 * @example
 * ```typescript
 * import { OnnxZig } from '@anthropic/onnx-zig';
 *
 * // Initialize the library
 * const onnxZig = await OnnxZig.create();
 *
 * // Create tensors
 * const input = onnxZig.tensor.fromArray([1, 2, 3, 4], [2, 2]);
 *
 * // Load and run ONNX model
 * const session = await onnxZig.session.create('model.onnx');
 * const outputs = await session.run({ input });
 *
 * // Clean up
 * onnxZig.tensor.free(input);
 * session.dispose();
 * ```
 */

// Type definitions
export interface TensorShape {
  dims: number[];
  numel: number;
}

export interface TensorHandle {
  readonly id: number;
  readonly shape: TensorShape;
}

export interface SessionHandle {
  readonly inputNames: string[];
  readonly outputNames: string[];
}

export interface InferenceResult {
  [outputName: string]: TensorHandle;
}

export interface PoolStats {
  hits: number;
  misses: number;
  hitRate: number;
}

export interface MemoryInfo {
  pages: number;
  bytes: number;
  mb: number;
}

/**
 * Configuration options for OnnxZig initialization
 */
export interface OnnxZigOptions {
  /** Path to the WASM file (default: 'onnx_zig.wasm') */
  wasmPath?: string;
  /** Number of threads for ONNX Runtime (default: navigator.hardwareConcurrency) */
  numThreads?: number;
  /** Enable SIMD (default: true) */
  simd?: boolean;
}

/**
 * Tensor API for creating and manipulating tensors
 */
export class TensorAPI {
  private module: OnnxZigModule;
  private exports: any;

  constructor(module: OnnxZigModule) {
    this.module = module;
    this.exports = module.exports;
  }

  /**
   * Create a new tensor with the given shape
   */
  create(shape: number[]): TensorHandle {
    const shapePtr = this.allocShape(shape);
    const id = this.exports.tensor_create(shapePtr, shape.length);
    this.module.free(shapePtr, shape.length * 4);

    if (id < 0) throw new Error('Failed to create tensor');

    return {
      id,
      shape: { dims: [...shape], numel: shape.reduce((a, b) => a * b, 1) }
    };
  }

  /**
   * Create a tensor filled with zeros
   */
  zeros(shape: number[]): TensorHandle {
    const shapePtr = this.allocShape(shape);
    const id = this.exports.tensor_zeros(shapePtr, shape.length);
    this.module.free(shapePtr, shape.length * 4);

    if (id < 0) throw new Error('Failed to create tensor');

    return {
      id,
      shape: { dims: [...shape], numel: shape.reduce((a, b) => a * b, 1) }
    };
  }

  /**
   * Create a tensor filled with ones
   */
  ones(shape: number[]): TensorHandle {
    const shapePtr = this.allocShape(shape);
    const id = this.exports.tensor_ones(shapePtr, shape.length);
    this.module.free(shapePtr, shape.length * 4);

    if (id < 0) throw new Error('Failed to create tensor');

    return {
      id,
      shape: { dims: [...shape], numel: shape.reduce((a, b) => a * b, 1) }
    };
  }

  /**
   * Create a tensor from a JavaScript array
   */
  fromArray(data: number[] | Float32Array, shape: number[]): TensorHandle {
    const arr = data instanceof Float32Array ? data : new Float32Array(data);
    const expectedNumel = shape.reduce((a, b) => a * b, 1);

    if (arr.length !== expectedNumel) {
      throw new Error(`Data length ${arr.length} does not match shape ${shape} (expected ${expectedNumel})`);
    }

    // Allocate and copy data
    const dataPtr = this.module.allocF32(arr.length);
    const mem = new Float32Array(this.module.memory.buffer, dataPtr, arr.length);
    mem.set(arr);

    // Allocate and copy shape
    const shapePtr = this.allocShape(shape);

    // Create tensor
    const id = this.exports.tensor_from_data(dataPtr, arr.length, shapePtr, shape.length);

    // Clean up
    this.module.freeF32(dataPtr, arr.length);
    this.module.free(shapePtr, shape.length * 4);

    if (id < 0) throw new Error('Failed to create tensor from data');

    return {
      id,
      shape: { dims: [...shape], numel: expectedNumel }
    };
  }

  /**
   * Get tensor data as a JavaScript array
   */
  toArray(tensor: TensorHandle): number[] {
    const numel = this.exports.tensor_numel(tensor.id);
    const ptr = this.exports.tensor_data_ptr(tensor.id);

    if (ptr === 0 || numel === 0) return [];

    const view = new Float32Array(this.module.memory.buffer, ptr, numel);
    return Array.from(view);
  }

  /**
   * Get a Float32Array view of tensor data (zero-copy)
   */
  dataView(tensor: TensorHandle): Float32Array | null {
    const numel = this.exports.tensor_numel(tensor.id);
    const ptr = this.exports.tensor_data_ptr(tensor.id);

    if (ptr === 0 || numel === 0) return null;

    return new Float32Array(this.module.memory.buffer, ptr, numel);
  }

  /**
   * Free tensor memory
   */
  free(tensor: TensorHandle): void {
    this.exports.tensor_free(tensor.id);
  }

  /**
   * Element-wise addition
   */
  add(a: TensorHandle, b: TensorHandle): TensorHandle {
    const id = this.exports.tensor_add(a.id, b.id);
    if (id < 0) throw new Error('Tensor add failed');

    return {
      id,
      shape: { ...a.shape }
    };
  }

  /**
   * Element-wise multiplication
   */
  mul(a: TensorHandle, b: TensorHandle): TensorHandle {
    const id = this.exports.tensor_mul(a.id, b.id);
    if (id < 0) throw new Error('Tensor mul failed');

    return {
      id,
      shape: { ...a.shape }
    };
  }

  /**
   * ReLU activation
   */
  relu(input: TensorHandle): TensorHandle {
    const id = this.exports.tensor_relu(input.id);
    if (id < 0) throw new Error('Tensor relu failed');

    return {
      id,
      shape: { ...input.shape }
    };
  }

  /**
   * Softmax activation
   */
  softmax(input: TensorHandle): TensorHandle {
    const id = this.exports.tensor_softmax(input.id);
    if (id < 0) throw new Error('Tensor softmax failed');

    return {
      id,
      shape: { ...input.shape }
    };
  }

  /**
   * Scale tensor by a constant
   */
  scale(input: TensorHandle, scalar: number): TensorHandle {
    const id = this.exports.tensor_scale(input.id, scalar);
    if (id < 0) throw new Error('Tensor scale failed');

    return {
      id,
      shape: { ...input.shape }
    };
  }

  /**
   * Find index of maximum value
   */
  argmax(input: TensorHandle): number {
    return this.exports.tensor_argmax(input.id);
  }

  /**
   * Clone a tensor (deep copy)
   */
  clone(input: TensorHandle): TensorHandle {
    const id = this.exports.tensor_clone(input.id);
    if (id < 0) throw new Error('Tensor clone failed');

    return {
      id,
      shape: { ...input.shape }
    };
  }

  private allocShape(shape: number[]): number {
    const ptr = this.module.alloc(shape.length * 4);
    const view = new Uint32Array(this.module.memory.buffer, ptr, shape.length);
    view.set(shape);
    return ptr;
  }
}

/**
 * Session API for ONNX model inference
 */
export class SessionAPI {
  private module: OnnxZigModule;
  private ort: any;

  constructor(module: OnnxZigModule) {
    this.module = module;
    this.ort = null;
  }

  /**
   * Initialize ONNX Runtime Web
   */
  async init(options: { numThreads?: number; simd?: boolean } = {}): Promise<void> {
    if (typeof ort === 'undefined') {
      throw new Error('ONNX Runtime Web not loaded. Include ort.min.js before using SessionAPI.');
    }
    this.ort = ort;

    ort.env.wasm.numThreads = options.numThreads ?? (navigator.hardwareConcurrency || 4);
    ort.env.wasm.simd = options.simd ?? true;
  }

  /**
   * Create an inference session from an ONNX model
   */
  async create(modelPath: string | ArrayBuffer): Promise<InferenceSession> {
    if (!this.ort) await this.init();

    const session = await this.ort.InferenceSession.create(modelPath, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    });

    return new InferenceSession(this.module, session);
  }
}

/**
 * ONNX Inference Session
 */
export class InferenceSession {
  private module: OnnxZigModule;
  private session: any;

  readonly inputNames: string[];
  readonly outputNames: string[];

  constructor(module: OnnxZigModule, session: any) {
    this.module = module;
    this.session = session;
    this.inputNames = session.inputNames;
    this.outputNames = session.outputNames;
  }

  /**
   * Run inference with tensor handles
   */
  async run(inputs: { [name: string]: TensorHandle }): Promise<InferenceResult> {
    const ortInputs: { [name: string]: any } = {};

    // Convert tensor handles to ORT tensors
    for (const [name, handle] of Object.entries(inputs)) {
      const data = this.module.tensor.dataView(handle);
      if (!data) throw new Error(`Failed to get data for tensor ${name}`);

      ortInputs[name] = new ort.Tensor('float32', new Float32Array(data), handle.shape.dims);
    }

    // Run inference
    const ortOutputs = await this.session.run(ortInputs);

    // Convert outputs to tensor handles
    const outputs: InferenceResult = {};
    for (const [name, tensor] of Object.entries(ortOutputs) as [string, any][]) {
      outputs[name] = this.module.tensor.fromArray(tensor.data, tensor.dims);
    }

    return outputs;
  }

  /**
   * Run inference with raw Float32Arrays
   */
  async runRaw(inputs: { [name: string]: { data: Float32Array; shape: number[] } }): Promise<{ [name: string]: { data: Float32Array; shape: number[] } }> {
    const ortInputs: { [name: string]: any } = {};

    for (const [name, { data, shape }] of Object.entries(inputs)) {
      ortInputs[name] = new ort.Tensor('float32', data, shape);
    }

    const ortOutputs = await this.session.run(ortInputs);

    const outputs: { [name: string]: { data: Float32Array; shape: number[] } } = {};
    for (const [name, tensor] of Object.entries(ortOutputs) as [string, any][]) {
      outputs[name] = {
        data: new Float32Array(tensor.data),
        shape: tensor.dims
      };
    }

    return outputs;
  }

  /**
   * Dispose the session
   */
  dispose(): void {
    // ORT sessions don't need explicit disposal in JS
  }
}

/**
 * Internal WASM module wrapper
 */
class OnnxZigModule {
  private instance: WebAssembly.Instance;
  readonly exports: any;
  readonly memory: WebAssembly.Memory;
  readonly tensor: TensorAPI;
  readonly session: SessionAPI;

  private constructor(instance: WebAssembly.Instance) {
    this.instance = instance;
    this.exports = instance.exports;
    this.memory = instance.exports.memory as WebAssembly.Memory;
    this.tensor = new TensorAPI(this);
    this.session = new SessionAPI(this);
  }

  static async load(wasmPath: string): Promise<OnnxZigModule> {
    const response = await fetch(wasmPath);
    const bytes = await response.arrayBuffer();
    const { instance } = await WebAssembly.instantiate(bytes, {});
    return new OnnxZigModule(instance);
  }

  alloc(size: number): number {
    const ptr = this.exports.wasm_alloc(size);
    if (ptr === 0) throw new Error('WASM allocation failed');
    return ptr;
  }

  free(ptr: number, size: number): void {
    if (ptr !== 0) this.exports.wasm_free(ptr, size);
  }

  allocF32(count: number): number {
    const ptr = this.exports.wasm_alloc_f32(count);
    if (ptr === 0) throw new Error('WASM f32 allocation failed');
    return ptr;
  }

  freeF32(ptr: number, count: number): void {
    if (ptr !== 0) this.exports.wasm_free_f32(ptr, count);
  }

  getMemoryInfo(): MemoryInfo {
    const pages = this.exports.wasm_memory_pages();
    return {
      pages,
      bytes: pages * 65536,
      mb: (pages * 65536) / (1024 * 1024)
    };
  }
}

/**
 * Main OnnxZig class - entry point for the library
 */
export class OnnxZig {
  private module: OnnxZigModule;

  readonly tensor: TensorAPI;
  readonly session: SessionAPI;

  private constructor(module: OnnxZigModule) {
    this.module = module;
    this.tensor = module.tensor;
    this.session = module.session;
  }

  /**
   * Create and initialize an OnnxZig instance
   *
   * @param options - Configuration options
   * @returns Initialized OnnxZig instance
   *
   * @example
   * ```typescript
   * const onnxZig = await OnnxZig.create();
   * ```
   */
  static async create(options: OnnxZigOptions = {}): Promise<OnnxZig> {
    const wasmPath = options.wasmPath ?? 'onnx_zig.wasm';
    const module = await OnnxZigModule.load(wasmPath);

    await module.session.init({
      numThreads: options.numThreads,
      simd: options.simd
    });

    return new OnnxZig(module);
  }

  /**
   * Get WASM memory information
   */
  getMemoryInfo(): MemoryInfo {
    return this.module.getMemoryInfo();
  }
}

// Default export
export default OnnxZig;
