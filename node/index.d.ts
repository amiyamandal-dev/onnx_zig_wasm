/**
 * ONNX Zig Node.js TypeScript Definitions
 */

/**
 * Get the library version
 */
export function version(): string;

/**
 * Tensor class for storing multi-dimensional arrays
 */
export class Tensor {
    /**
     * Create a new tensor
     * @param shape - Shape of the tensor
     * @param data - Optional initial data
     */
    constructor(shape: number[], data?: Float32Array);

    /**
     * Tensor data as Float32Array
     */
    data: Float32Array;

    /**
     * Tensor shape
     */
    readonly shape: number[];

    /**
     * Number of elements
     */
    readonly numel: number;

    /**
     * Number of dimensions
     */
    readonly ndim: number;

    /**
     * Apply softmax activation in-place
     */
    softmax(): Tensor;

    /**
     * Get index of maximum element
     */
    argmax(): number;

    /**
     * Convert to JavaScript array
     */
    toArray(): number[];

    /**
     * Create tensor from JavaScript array
     */
    static fromArray(shape: number[], data: number[]): Tensor;
}

/**
 * Session options
 */
export interface SessionOptions {
    /**
     * Number of threads for intra-op parallelism (0 = auto)
     */
    intraOpThreads?: number;

    /**
     * Number of threads for inter-op parallelism (0 = auto)
     */
    interOpThreads?: number;
}

/**
 * ONNX inference session
 */
export class Session {
    /**
     * Create a new inference session
     * @param modelPath - Path to ONNX model file
     * @param options - Session options
     */
    constructor(modelPath: string, options?: SessionOptions);

    /**
     * Number of model inputs
     */
    readonly inputCount: number;

    /**
     * Number of model outputs
     */
    readonly outputCount: number;

    /**
     * Input tensor names
     */
    readonly inputNames: string[];

    /**
     * Output tensor names
     */
    readonly outputNames: string[];

    /**
     * Run inference
     * @param data - Input data
     * @param shape - Input shape
     * @returns Output tensors
     */
    run(data: Float32Array | number[], shape: number[]): Tensor[];

    /**
     * Run inference with a Tensor input
     * @param input - Input tensor
     * @returns Output tensors
     */
    runTensor(input: Tensor): Tensor[];

    /**
     * Run inference and return only the predicted class index
     * @param data - Input data
     * @param shape - Input shape
     * @returns Predicted class index
     */
    classify(data: Float32Array | number[], shape: number[]): number;
}

/**
 * Native bindings (for advanced usage)
 */
export const native: {
    Tensor: new (shape: number[], data?: Float32Array) => any;
    Session: new (modelPath: string, options?: SessionOptions) => any;
    version: () => string;
};
