exports_files(["OnnxParser.h", "OnnxParser.cpp"],
        ["//visibility:public"])

cc_binary(
    name = "onnx_converter",
    srcs = glob([
        "Main.cpp",
    ]),
    includes = [
        "protos",
    ],
    copts = ["-std=c++11"],
    deps = [
        "//:mai",
        "//tools/converter/onnx/protos:onnx_cc_proto",
    ],
)
