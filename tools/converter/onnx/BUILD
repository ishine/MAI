cc_binary(
    name = "onnx_converter",
    srcs = glob([
        "*.cpp",
        "*.h",
    ]),
    includes = [
        "protos",
    ],
    copts = ["-std=c++11"],
    deps = [
        "//tools/converter/onnx/protos:onnx_cc_proto",
        "//:mai"
    ],
)
