cc_binary(
    name = "tensorflow_converter",
    srcs = glob([
        "*.cpp",
        "*.h",
    ]),
    includes = [
    #    "tools/converter/tensorflow/",
        "protos",
    ],
    copts = ["-std=c++11"],
    deps = [
        "//tools/converter/tensorflow/protos:tensorflow_cc_proto",
        "//:mai"
    ],
)
