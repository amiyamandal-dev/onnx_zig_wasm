// OnnxZig.swift - Swift bindings for ONNX Zig library
//
// Usage:
//   1. Build the C library: zig build c-lib -Donnxruntime_path=/path/to/onnxruntime
//   2. Add libonnx_zig.dylib to your Xcode project
//   3. Add the bridging header or include this Swift file
//
// Example:
//   let session = try OnnxZigSession(modelPath: "model.onnx")
//   let input = OnnxZigTensor(shape: [1, 4], data: [1.0, 2.0, 3.0, 4.0])
//   let outputs = try session.run(input: input)
//   let result = outputs[0].data

import Foundation

// MARK: - Error Types

public enum OnnxZigError: Error {
    case allocationFailed
    case invalidInput
    case invalidOutput
    case modelNotFound
    case invalidModel
    case inferenceFailed
    case shapeMismatch
    case nullPointer
    case unknown(String)

    init(code: Int32, message: String? = nil) {
        switch code {
        case 0: self = .allocationFailed // Shouldn't happen for code 0
        case 1: self = .allocationFailed
        case 2: self = .invalidInput
        case 3: self = .invalidOutput
        case 4: self = .modelNotFound
        case 5: self = .invalidModel
        case 6: self = .inferenceFailed
        case 7: self = .shapeMismatch
        case 8: self = .nullPointer
        default: self = .unknown(message ?? "Unknown error code: \(code)")
        }
    }
}

// MARK: - Tensor

public class OnnxZigTensor {
    internal var handle: OpaquePointer?
    private var ownsHandle: Bool = true

    /// Create a tensor with the given shape (uninitialized data)
    public init(shape: [Int]) throws {
        var shapeArray = shape.map { UInt(bitPattern: $0) }
        var handlePtr: OpaquePointer?

        let result = shapeArray.withUnsafeMutableBufferPointer { shapePtr in
            onnx_zig_tensor_create_f32(shapePtr.baseAddress, UInt(shape.count), &handlePtr)
        }

        guard result == 0, let h = handlePtr else {
            throw OnnxZigError(code: result)
        }

        self.handle = h
    }

    /// Create a tensor from existing data
    public init(shape: [Int], data: [Float]) throws {
        var shapeArray = shape.map { UInt(bitPattern: $0) }
        var handlePtr: OpaquePointer?

        let result = data.withUnsafeBufferPointer { dataPtr in
            shapeArray.withUnsafeMutableBufferPointer { shapePtr in
                onnx_zig_tensor_create_f32_from_data(
                    dataPtr.baseAddress,
                    shapePtr.baseAddress,
                    UInt(shape.count),
                    &handlePtr
                )
            }
        }

        guard result == 0, let h = handlePtr else {
            throw OnnxZigError(code: result)
        }

        self.handle = h
    }

    /// Internal initializer for output tensors
    internal init(handle: OpaquePointer, ownsHandle: Bool = false) {
        self.handle = handle
        self.ownsHandle = ownsHandle
    }

    deinit {
        if ownsHandle, let h = handle {
            onnx_zig_tensor_destroy(h)
        }
    }

    /// Get tensor data as Float array
    public var data: [Float] {
        guard let h = handle else { return [] }
        guard let dataPtr = onnx_zig_tensor_get_data_f32(h) else { return [] }
        let count = Int(onnx_zig_tensor_get_numel(h))
        return Array(UnsafeBufferPointer(start: dataPtr, count: count))
    }

    /// Get mutable pointer to tensor data
    public var mutableData: UnsafeMutablePointer<Float>? {
        guard let h = handle else { return nil }
        return onnx_zig_tensor_get_data_f32_mut(h)
    }

    /// Total number of elements
    public var count: Int {
        guard let h = handle else { return 0 }
        return Int(onnx_zig_tensor_get_numel(h))
    }

    /// Number of dimensions
    public var ndim: Int {
        guard let h = handle else { return 0 }
        return Int(onnx_zig_tensor_get_ndim(h))
    }

    /// Get tensor shape
    public var shape: [Int] {
        guard let h = handle else { return [] }
        let ndim = Int(onnx_zig_tensor_get_ndim(h))
        var shapeArray = [UInt](repeating: 0, count: ndim)

        _ = shapeArray.withUnsafeMutableBufferPointer { ptr in
            onnx_zig_tensor_get_shape(h, ptr.baseAddress, UInt(ndim))
        }

        return shapeArray.map { Int(bitPattern: $0) }
    }

    /// Get size of a specific dimension
    public func dim(_ index: Int) -> Int {
        guard let h = handle else { return 0 }
        return Int(bitPattern: onnx_zig_tensor_get_dim(h, UInt(bitPattern: index)))
    }

    /// Apply softmax in-place
    public func softmax() {
        guard let h = handle else { return }
        onnx_zig_softmax_inplace(h)
    }

    /// Get argmax index
    public func argmax() -> Int {
        guard let h = handle else { return 0 }
        return Int(bitPattern: onnx_zig_argmax(h))
    }
}

// MARK: - Session

public class OnnxZigSession {
    private var handle: OpaquePointer?

    /// Create a session from an ONNX model file
    public init(modelPath: String) throws {
        var handlePtr: OpaquePointer?

        let result = modelPath.withCString { pathPtr in
            onnx_zig_session_create(pathPtr, &handlePtr)
        }

        guard result == 0, let h = handlePtr else {
            let errorMsg = String(cString: onnx_zig_get_last_error())
            throw OnnxZigError(code: result, message: errorMsg)
        }

        self.handle = h
    }

    /// Create a session with custom thread options
    public init(modelPath: String, intraOpThreads: UInt32 = 0, interOpThreads: UInt32 = 0) throws {
        var handlePtr: OpaquePointer?

        let result = modelPath.withCString { pathPtr in
            onnx_zig_session_create_with_options(pathPtr, intraOpThreads, interOpThreads, &handlePtr)
        }

        guard result == 0, let h = handlePtr else {
            let errorMsg = String(cString: onnx_zig_get_last_error())
            throw OnnxZigError(code: result, message: errorMsg)
        }

        self.handle = h
    }

    deinit {
        if let h = handle {
            onnx_zig_session_destroy(h)
        }
    }

    /// Number of model inputs
    public var inputCount: Int {
        guard let h = handle else { return 0 }
        return Int(onnx_zig_session_get_input_count(h))
    }

    /// Number of model outputs
    public var outputCount: Int {
        guard let h = handle else { return 0 }
        return Int(onnx_zig_session_get_output_count(h))
    }

    /// Get input tensor names
    public var inputNames: [String] {
        guard let h = handle else { return [] }
        var names: [String] = []
        for i in 0..<inputCount {
            if let namePtr = onnx_zig_session_get_input_name(h, UInt(i)) {
                names.append(String(cString: namePtr))
            }
        }
        return names
    }

    /// Get output tensor names
    public var outputNames: [String] {
        guard let h = handle else { return [] }
        var names: [String] = []
        for i in 0..<outputCount {
            if let namePtr = onnx_zig_session_get_output_name(h, UInt(i)) {
                names.append(String(cString: namePtr))
            }
        }
        return names
    }

    /// Run inference with a single input
    public func run(data: [Float], shape: [Int64]) throws -> [OnnxZigTensor] {
        guard let h = handle else {
            throw OnnxZigError.nullPointer
        }

        var outTensors: UnsafeMutablePointer<OpaquePointer?>?
        var outCount: UInt = 0

        let result = data.withUnsafeBufferPointer { dataPtr in
            shape.withUnsafeBufferPointer { shapePtr in
                onnx_zig_session_run_f32_simple(
                    h,
                    dataPtr.baseAddress,
                    shapePtr.baseAddress,
                    UInt(shape.count),
                    &outTensors,
                    &outCount
                )
            }
        }

        guard result == 0 else {
            let errorMsg = String(cString: onnx_zig_get_last_error())
            throw OnnxZigError(code: result, message: errorMsg)
        }

        // Wrap output handles in Swift tensors
        var outputs: [OnnxZigTensor] = []
        if let tensors = outTensors {
            for i in 0..<Int(outCount) {
                if let tensorHandle = tensors[i] {
                    outputs.append(OnnxZigTensor(handle: tensorHandle, ownsHandle: false))
                }
            }
        }

        return outputs
    }

    /// Run inference with an OnnxZigTensor input
    public func run(input: OnnxZigTensor) throws -> [OnnxZigTensor] {
        let shape = input.shape.map { Int64($0) }
        return try run(data: input.data, shape: shape)
    }
}

// MARK: - Tensor Operations

public extension OnnxZigTensor {
    /// Matrix multiplication
    static func matmul(_ a: OnnxZigTensor, _ b: OnnxZigTensor) throws -> OnnxZigTensor {
        guard let aHandle = a.handle, let bHandle = b.handle else {
            throw OnnxZigError.nullPointer
        }

        var resultHandle: OpaquePointer?
        let result = onnx_zig_matmul(aHandle, bHandle, &resultHandle)

        guard result == 0, let h = resultHandle else {
            throw OnnxZigError(code: result)
        }

        return OnnxZigTensor(handle: h, ownsHandle: true)
    }

    /// Transpose
    func transposed() throws -> OnnxZigTensor {
        guard let h = handle else {
            throw OnnxZigError.nullPointer
        }

        var resultHandle: OpaquePointer?
        let result = onnx_zig_transpose(h, &resultHandle)

        guard result == 0, let rh = resultHandle else {
            throw OnnxZigError(code: result)
        }

        return OnnxZigTensor(handle: rh, ownsHandle: true)
    }
}

// MARK: - Utility

public func onnxZigVersion() -> String {
    return String(cString: onnx_zig_version())
}

// MARK: - C Function Declarations (Bridging)

// These would typically be in a bridging header, but can be declared here for single-file usage
@_silgen_name("onnx_zig_session_create")
private func onnx_zig_session_create(_ modelPath: UnsafePointer<CChar>?, _ outSession: UnsafeMutablePointer<OpaquePointer?>?) -> Int32

@_silgen_name("onnx_zig_session_create_with_options")
private func onnx_zig_session_create_with_options(_ modelPath: UnsafePointer<CChar>?, _ intraOpThreads: UInt32, _ interOpThreads: UInt32, _ outSession: UnsafeMutablePointer<OpaquePointer?>?) -> Int32

@_silgen_name("onnx_zig_session_destroy")
private func onnx_zig_session_destroy(_ session: OpaquePointer?)

@_silgen_name("onnx_zig_session_get_input_count")
private func onnx_zig_session_get_input_count(_ session: OpaquePointer?) -> UInt

@_silgen_name("onnx_zig_session_get_output_count")
private func onnx_zig_session_get_output_count(_ session: OpaquePointer?) -> UInt

@_silgen_name("onnx_zig_session_get_input_name")
private func onnx_zig_session_get_input_name(_ session: OpaquePointer?, _ index: UInt) -> UnsafePointer<CChar>?

@_silgen_name("onnx_zig_session_get_output_name")
private func onnx_zig_session_get_output_name(_ session: OpaquePointer?, _ index: UInt) -> UnsafePointer<CChar>?

@_silgen_name("onnx_zig_session_run_f32_simple")
private func onnx_zig_session_run_f32_simple(_ session: OpaquePointer?, _ inputData: UnsafePointer<Float>?, _ inputShape: UnsafePointer<Int64>?, _ shapeDims: UInt, _ outTensors: UnsafeMutablePointer<UnsafeMutablePointer<OpaquePointer?>?>?, _ outCount: UnsafeMutablePointer<UInt>?) -> Int32

@_silgen_name("onnx_zig_tensor_create_f32")
private func onnx_zig_tensor_create_f32(_ shape: UnsafePointer<UInt>?, _ ndim: UInt, _ outTensor: UnsafeMutablePointer<OpaquePointer?>?) -> Int32

@_silgen_name("onnx_zig_tensor_create_f32_from_data")
private func onnx_zig_tensor_create_f32_from_data(_ data: UnsafePointer<Float>?, _ shape: UnsafePointer<UInt>?, _ ndim: UInt, _ outTensor: UnsafeMutablePointer<OpaquePointer?>?) -> Int32

@_silgen_name("onnx_zig_tensor_destroy")
private func onnx_zig_tensor_destroy(_ tensor: OpaquePointer?)

@_silgen_name("onnx_zig_tensor_get_data_f32")
private func onnx_zig_tensor_get_data_f32(_ tensor: OpaquePointer?) -> UnsafePointer<Float>?

@_silgen_name("onnx_zig_tensor_get_data_f32_mut")
private func onnx_zig_tensor_get_data_f32_mut(_ tensor: OpaquePointer?) -> UnsafeMutablePointer<Float>?

@_silgen_name("onnx_zig_tensor_get_numel")
private func onnx_zig_tensor_get_numel(_ tensor: OpaquePointer?) -> UInt

@_silgen_name("onnx_zig_tensor_get_ndim")
private func onnx_zig_tensor_get_ndim(_ tensor: OpaquePointer?) -> UInt

@_silgen_name("onnx_zig_tensor_get_dim")
private func onnx_zig_tensor_get_dim(_ tensor: OpaquePointer?, _ dim: UInt) -> UInt

@_silgen_name("onnx_zig_tensor_get_shape")
private func onnx_zig_tensor_get_shape(_ tensor: OpaquePointer?, _ outShape: UnsafeMutablePointer<UInt>?, _ maxDims: UInt) -> UInt

@_silgen_name("onnx_zig_matmul")
private func onnx_zig_matmul(_ a: OpaquePointer?, _ b: OpaquePointer?, _ outResult: UnsafeMutablePointer<OpaquePointer?>?) -> Int32

@_silgen_name("onnx_zig_transpose")
private func onnx_zig_transpose(_ tensor: OpaquePointer?, _ outResult: UnsafeMutablePointer<OpaquePointer?>?) -> Int32

@_silgen_name("onnx_zig_softmax_inplace")
private func onnx_zig_softmax_inplace(_ tensor: OpaquePointer?)

@_silgen_name("onnx_zig_argmax")
private func onnx_zig_argmax(_ tensor: OpaquePointer?) -> UInt

@_silgen_name("onnx_zig_version")
private func onnx_zig_version() -> UnsafePointer<CChar>

@_silgen_name("onnx_zig_get_last_error")
private func onnx_zig_get_last_error() -> UnsafePointer<CChar>
