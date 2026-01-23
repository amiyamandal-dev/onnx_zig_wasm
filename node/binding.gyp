{
  "targets": [
    {
      "target_name": "onnx_zig",
      "cflags!": ["-fno-exceptions"],
      "cflags_cc!": ["-fno-exceptions"],
      "sources": ["src/addon.cc"],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "../include"
      ],
      "libraries": [
        "-L../zig-out/lib",
        "-lonnx_zig"
      ],
      "defines": ["NAPI_DISABLE_CPP_EXCEPTIONS"],
      "conditions": [
        ["OS=='mac'", {
          "xcode_settings": {
            "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
            "CLANG_CXX_LIBRARY": "libc++",
            "MACOSX_DEPLOYMENT_TARGET": "10.15"
          },
          "libraries": [
            "-Wl,-rpath,@loader_path/../zig-out/lib"
          ]
        }],
        ["OS=='linux'", {
          "libraries": [
            "-Wl,-rpath,'$$ORIGIN/../zig-out/lib'"
          ]
        }]
      ]
    }
  ]
}
