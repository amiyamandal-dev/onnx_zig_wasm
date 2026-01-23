/**
 * ONNX Zig Node.js N-API Addon
 *
 * Provides Node.js bindings for the ONNX Zig inference library.
 */

#include <napi.h>
#include "onnx_zig.h"

// =============================================================================
// Tensor Wrapper Class
// =============================================================================

class TensorWrapper : public Napi::ObjectWrap<TensorWrapper> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    TensorWrapper(const Napi::CallbackInfo& info);
    ~TensorWrapper();

    static Napi::Object NewInstance(Napi::Env env, OnnxZigTensorHandle handle, bool ownsHandle = true);

    OnnxZigTensorHandle GetHandle() { return handle_; }

private:
    static Napi::FunctionReference constructor;
    OnnxZigTensorHandle handle_;
    bool ownsHandle_;

    Napi::Value GetData(const Napi::CallbackInfo& info);
    void SetData(const Napi::CallbackInfo& info, const Napi::Value& value);
    Napi::Value GetShape(const Napi::CallbackInfo& info);
    Napi::Value GetNumel(const Napi::CallbackInfo& info);
    Napi::Value GetNdim(const Napi::CallbackInfo& info);
    Napi::Value Softmax(const Napi::CallbackInfo& info);
    Napi::Value Argmax(const Napi::CallbackInfo& info);
};

Napi::FunctionReference TensorWrapper::constructor;

Napi::Object TensorWrapper::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "Tensor", {
        InstanceAccessor<&TensorWrapper::GetData, &TensorWrapper::SetData>("data"),
        InstanceAccessor<&TensorWrapper::GetShape>("shape"),
        InstanceAccessor<&TensorWrapper::GetNumel>("numel"),
        InstanceAccessor<&TensorWrapper::GetNdim>("ndim"),
        InstanceMethod<&TensorWrapper::Softmax>("softmax"),
        InstanceMethod<&TensorWrapper::Argmax>("argmax"),
    });

    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();

    exports.Set("Tensor", func);
    return exports;
}

TensorWrapper::TensorWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<TensorWrapper>(info), handle_(nullptr), ownsHandle_(true) {

    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsArray()) {
        Napi::TypeError::New(env, "Shape array expected").ThrowAsJavaScriptException();
        return;
    }

    Napi::Array shapeArray = info[0].As<Napi::Array>();
    std::vector<size_t> shape;
    for (uint32_t i = 0; i < shapeArray.Length(); i++) {
        shape.push_back(shapeArray.Get(i).As<Napi::Number>().Uint32Value());
    }

    OnnxZigError result;
    if (info.Length() >= 2 && info[1].IsTypedArray()) {
        // Create from data
        Napi::Float32Array dataArray = info[1].As<Napi::Float32Array>();
        result = onnx_zig_tensor_create_f32_from_data(
            dataArray.Data(),
            shape.data(),
            shape.size(),
            &handle_
        );
    } else {
        // Create empty tensor
        result = onnx_zig_tensor_create_f32(shape.data(), shape.size(), &handle_);
    }

    if (result != ONNX_ZIG_OK) {
        Napi::Error::New(env, onnx_zig_get_last_error()).ThrowAsJavaScriptException();
    }
}

TensorWrapper::~TensorWrapper() {
    if (ownsHandle_ && handle_) {
        onnx_zig_tensor_destroy(handle_);
    }
}

Napi::Object TensorWrapper::NewInstance(Napi::Env env, OnnxZigTensorHandle handle, bool ownsHandle) {
    Napi::Object obj = constructor.New({});
    TensorWrapper* wrapper = Napi::ObjectWrap<TensorWrapper>::Unwrap(obj);
    wrapper->handle_ = handle;
    wrapper->ownsHandle_ = ownsHandle;
    return obj;
}

Napi::Value TensorWrapper::GetData(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (!handle_) {
        return env.Null();
    }

    const float* data = onnx_zig_tensor_get_data_f32(handle_);
    size_t numel = onnx_zig_tensor_get_numel(handle_);

    Napi::Float32Array result = Napi::Float32Array::New(env, numel);
    for (size_t i = 0; i < numel; i++) {
        result[i] = data[i];
    }

    return result;
}

void TensorWrapper::SetData(const Napi::CallbackInfo& info, const Napi::Value& value) {
    Napi::Env env = info.Env();

    if (!handle_ || !value.IsTypedArray()) {
        return;
    }

    Napi::Float32Array dataArray = value.As<Napi::Float32Array>();
    float* data = onnx_zig_tensor_get_data_f32_mut(handle_);
    size_t numel = onnx_zig_tensor_get_numel(handle_);
    size_t copyLen = std::min(numel, dataArray.ElementLength());

    for (size_t i = 0; i < copyLen; i++) {
        data[i] = dataArray[i];
    }
}

Napi::Value TensorWrapper::GetShape(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (!handle_) {
        return Napi::Array::New(env);
    }

    size_t ndim = onnx_zig_tensor_get_ndim(handle_);
    std::vector<size_t> shape(ndim);
    onnx_zig_tensor_get_shape(handle_, shape.data(), ndim);

    Napi::Array result = Napi::Array::New(env, ndim);
    for (size_t i = 0; i < ndim; i++) {
        result.Set(i, Napi::Number::New(env, shape[i]));
    }

    return result;
}

Napi::Value TensorWrapper::GetNumel(const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), handle_ ? onnx_zig_tensor_get_numel(handle_) : 0);
}

Napi::Value TensorWrapper::GetNdim(const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), handle_ ? onnx_zig_tensor_get_ndim(handle_) : 0);
}

Napi::Value TensorWrapper::Softmax(const Napi::CallbackInfo& info) {
    if (handle_) {
        onnx_zig_softmax_inplace(handle_);
    }
    return info.This();
}

Napi::Value TensorWrapper::Argmax(const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), handle_ ? onnx_zig_argmax(handle_) : 0);
}

// =============================================================================
// Session Wrapper Class
// =============================================================================

class SessionWrapper : public Napi::ObjectWrap<SessionWrapper> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    SessionWrapper(const Napi::CallbackInfo& info);
    ~SessionWrapper();

private:
    static Napi::FunctionReference constructor;
    OnnxZigSessionHandle handle_;

    Napi::Value GetInputCount(const Napi::CallbackInfo& info);
    Napi::Value GetOutputCount(const Napi::CallbackInfo& info);
    Napi::Value GetInputNames(const Napi::CallbackInfo& info);
    Napi::Value GetOutputNames(const Napi::CallbackInfo& info);
    Napi::Value Run(const Napi::CallbackInfo& info);
};

Napi::FunctionReference SessionWrapper::constructor;

Napi::Object SessionWrapper::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "Session", {
        InstanceAccessor<&SessionWrapper::GetInputCount>("inputCount"),
        InstanceAccessor<&SessionWrapper::GetOutputCount>("outputCount"),
        InstanceAccessor<&SessionWrapper::GetInputNames>("inputNames"),
        InstanceAccessor<&SessionWrapper::GetOutputNames>("outputNames"),
        InstanceMethod<&SessionWrapper::Run>("run"),
    });

    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();

    exports.Set("Session", func);
    return exports;
}

SessionWrapper::SessionWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<SessionWrapper>(info), handle_(nullptr) {

    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Model path string expected").ThrowAsJavaScriptException();
        return;
    }

    std::string modelPath = info[0].As<Napi::String>().Utf8Value();

    OnnxZigError result;
    if (info.Length() >= 2 && info[1].IsObject()) {
        Napi::Object options = info[1].As<Napi::Object>();
        uint32_t intraOpThreads = 0;
        uint32_t interOpThreads = 0;

        if (options.Has("intraOpThreads")) {
            intraOpThreads = options.Get("intraOpThreads").As<Napi::Number>().Uint32Value();
        }
        if (options.Has("interOpThreads")) {
            interOpThreads = options.Get("interOpThreads").As<Napi::Number>().Uint32Value();
        }

        result = onnx_zig_session_create_with_options(
            modelPath.c_str(),
            intraOpThreads,
            interOpThreads,
            &handle_
        );
    } else {
        result = onnx_zig_session_create(modelPath.c_str(), &handle_);
    }

    if (result != ONNX_ZIG_OK) {
        Napi::Error::New(env, onnx_zig_get_last_error()).ThrowAsJavaScriptException();
    }
}

SessionWrapper::~SessionWrapper() {
    if (handle_) {
        onnx_zig_session_destroy(handle_);
    }
}

Napi::Value SessionWrapper::GetInputCount(const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), handle_ ? onnx_zig_session_get_input_count(handle_) : 0);
}

Napi::Value SessionWrapper::GetOutputCount(const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), handle_ ? onnx_zig_session_get_output_count(handle_) : 0);
}

Napi::Value SessionWrapper::GetInputNames(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Array result = Napi::Array::New(env);

    if (handle_) {
        size_t count = onnx_zig_session_get_input_count(handle_);
        for (size_t i = 0; i < count; i++) {
            const char* name = onnx_zig_session_get_input_name(handle_, i);
            if (name) {
                result.Set(i, Napi::String::New(env, name));
            }
        }
    }

    return result;
}

Napi::Value SessionWrapper::GetOutputNames(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Array result = Napi::Array::New(env);

    if (handle_) {
        size_t count = onnx_zig_session_get_output_count(handle_);
        for (size_t i = 0; i < count; i++) {
            const char* name = onnx_zig_session_get_output_name(handle_, i);
            if (name) {
                result.Set(i, Napi::String::New(env, name));
            }
        }
    }

    return result;
}

Napi::Value SessionWrapper::Run(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (!handle_) {
        Napi::Error::New(env, "Session is not initialized").ThrowAsJavaScriptException();
        return env.Null();
    }

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected data array and shape array").ThrowAsJavaScriptException();
        return env.Null();
    }

    // Get input data
    Napi::Float32Array dataArray = info[0].As<Napi::Float32Array>();
    Napi::Array shapeArray = info[1].As<Napi::Array>();

    std::vector<int64_t> shape;
    for (uint32_t i = 0; i < shapeArray.Length(); i++) {
        shape.push_back(shapeArray.Get(i).As<Napi::Number>().Int64Value());
    }

    // Run inference
    OnnxZigTensorHandle* outTensors = nullptr;
    size_t outCount = 0;

    OnnxZigError result = onnx_zig_session_run_f32_simple(
        handle_,
        dataArray.Data(),
        shape.data(),
        shape.size(),
        &outTensors,
        &outCount
    );

    if (result != ONNX_ZIG_OK) {
        Napi::Error::New(env, onnx_zig_get_last_error()).ThrowAsJavaScriptException();
        return env.Null();
    }

    // Wrap output tensors
    Napi::Array outputs = Napi::Array::New(env, outCount);
    for (size_t i = 0; i < outCount; i++) {
        outputs.Set(i, TensorWrapper::NewInstance(env, outTensors[i], false));
    }

    return outputs;
}

// =============================================================================
// Module Initialization
// =============================================================================

Napi::Value GetVersion(const Napi::CallbackInfo& info) {
    return Napi::String::New(info.Env(), onnx_zig_version());
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    TensorWrapper::Init(env, exports);
    SessionWrapper::Init(env, exports);
    exports.Set("version", Napi::Function::New(env, GetVersion));
    return exports;
}

NODE_API_MODULE(onnx_zig, Init)
