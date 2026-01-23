/**
 * ONNX Zig Node.js Module
 *
 * High-level JavaScript API for the ONNX Zig inference library.
 *
 * @example
 * const { Session, Tensor } = require('@anthropic/onnx-zig-node');
 *
 * const session = new Session('model.onnx');
 * const input = new Tensor([1, 4], new Float32Array([1, 2, 3, 4]));
 * const outputs = session.run(input.data, input.shape);
 * console.log(outputs[0].data);
 */

const native = require('./build/Release/onnx_zig.node');

/**
 * Get the library version
 * @returns {string} Version string
 */
function version() {
    return native.version();
}

/**
 * Tensor class for storing multi-dimensional arrays
 */
class Tensor {
    /**
     * Create a new tensor
     * @param {number[]} shape - Shape of the tensor
     * @param {Float32Array} [data] - Optional initial data
     */
    constructor(shape, data) {
        if (data instanceof native.Tensor) {
            // Wrap existing native tensor
            this._native = data;
        } else {
            this._native = new native.Tensor(shape, data);
        }
    }

    /**
     * Get tensor data as Float32Array
     * @returns {Float32Array}
     */
    get data() {
        return this._native.data;
    }

    /**
     * Set tensor data
     * @param {Float32Array} value
     */
    set data(value) {
        this._native.data = value;
    }

    /**
     * Get tensor shape
     * @returns {number[]}
     */
    get shape() {
        return Array.from(this._native.shape);
    }

    /**
     * Get number of elements
     * @returns {number}
     */
    get numel() {
        return this._native.numel;
    }

    /**
     * Get number of dimensions
     * @returns {number}
     */
    get ndim() {
        return this._native.ndim;
    }

    /**
     * Apply softmax activation in-place
     * @returns {Tensor} this
     */
    softmax() {
        this._native.softmax();
        return this;
    }

    /**
     * Get index of maximum element
     * @returns {number}
     */
    argmax() {
        return this._native.argmax();
    }

    /**
     * Convert to JavaScript array
     * @returns {number[]}
     */
    toArray() {
        return Array.from(this.data);
    }

    /**
     * Create tensor from JavaScript array
     * @param {number[]} shape - Shape of the tensor
     * @param {number[]} data - Flat array of numbers
     * @returns {Tensor}
     */
    static fromArray(shape, data) {
        return new Tensor(shape, new Float32Array(data));
    }
}

/**
 * ONNX inference session
 */
class Session {
    /**
     * Create a new inference session
     * @param {string} modelPath - Path to ONNX model file
     * @param {Object} [options] - Session options
     * @param {number} [options.intraOpThreads=0] - Number of threads for intra-op parallelism
     * @param {number} [options.interOpThreads=0] - Number of threads for inter-op parallelism
     */
    constructor(modelPath, options) {
        this._native = new native.Session(modelPath, options);
    }

    /**
     * Get number of model inputs
     * @returns {number}
     */
    get inputCount() {
        return this._native.inputCount;
    }

    /**
     * Get number of model outputs
     * @returns {number}
     */
    get outputCount() {
        return this._native.outputCount;
    }

    /**
     * Get input tensor names
     * @returns {string[]}
     */
    get inputNames() {
        return this._native.inputNames;
    }

    /**
     * Get output tensor names
     * @returns {string[]}
     */
    get outputNames() {
        return this._native.outputNames;
    }

    /**
     * Run inference
     * @param {Float32Array|number[]} data - Input data
     * @param {number[]} shape - Input shape
     * @returns {Tensor[]} Output tensors
     */
    run(data, shape) {
        const inputData = data instanceof Float32Array ? data : new Float32Array(data);
        const nativeOutputs = this._native.run(inputData, shape);
        return nativeOutputs.map(t => new Tensor(t.shape, t));
    }

    /**
     * Run inference with a Tensor input
     * @param {Tensor} input - Input tensor
     * @returns {Tensor[]} Output tensors
     */
    runTensor(input) {
        return this.run(input.data, input.shape);
    }

    /**
     * Run inference and return only the predicted class index (for classification)
     * @param {Float32Array|number[]} data - Input data
     * @param {number[]} shape - Input shape
     * @returns {number} Predicted class index
     */
    classify(data, shape) {
        const outputs = this.run(data, shape);
        if (outputs.length > 0) {
            return outputs[0].argmax();
        }
        return -1;
    }
}

module.exports = {
    version,
    Tensor,
    Session,
    native, // Export native bindings for advanced usage
};
