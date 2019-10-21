exports_files(["TensorflowParser.h", "TensorflowParser.cpp"],
        ["//visibility:public"])

cc_binary(
    name = "tensorflow_converter",
    srcs = glob([
        "Main.cpp",
    ]),
    includes = [
        "protos",
    ],
    copts = ["-std=c++11"],
    deps = [
        "//tools/converter/tensorflow/protos:tensorflow_cc_proto",
        "//:mai"
    ],
)
