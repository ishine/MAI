config_setting(
    name = "android",
    values = {
        "crosstool_top": "//external:android/crosstool",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "neon_enabled",
    define_values = {
        #"crosstool_top": "//external:android/crosstool",
        "neon": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "tensorflow_enabled",
    define_values = {
        "tensorflow": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "onnx_enabled",
    define_values = {
        "onnx": "true",
    },
    visibility = ["//visibility:public"],
)

load("//:MAI.bzl", "if_android", "if_neon_enabled", "if_tensorflow_enabled", "if_onnx_enabled")

cc_library(
   name = "mai",
   srcs = glob([
       "source/core/*.cpp",
       "source/core/optimizers/*.cpp",
       "source/ops/cpu/*.cpp",
       "source/ops/cpu/runtime/*.cpp",
       "source/util/*.cpp",
   ]) + if_tensorflow_enabled(["//tools/converter/tensorflow:TensorflowParser.cpp"])
      + if_onnx_enabled(["//tools/converter/onnx:OnnxParser.cpp"]),
   hdrs = glob([
       "include/OperatorType.def",
       "include/*.h",
       "source/core/*.h",
       "source/core/optimizers/*.h",
       "source/util/*.h",
       "source/ops/cpu/*.h",
       "source/ops/cpu/runtime/*.h",
       "tools/profiling/*.h",
       "source/ops/cpu/ref/*.h",
       "source/ops/cpu/neon/*.h",

   ]) + ["//tools/converter/tensorflow:TensorflowParser.h"]
      + ["//tools/converter/onnx:OnnxParser.h"],
   copts = ["-Wall", "-Wextra", "-std=c++11", "-fopenmp", "-O3"]
        + if_neon_enabled(["-DMAI_NEON_ENABLED"])
        + if_tensorflow_enabled(["-DMAI_TENSORFLOW_ENABLED"])
        + if_onnx_enabled(["-DMAI_ONNX_ENABLED"]),
   linkopts = ["-fopenmp"],
   includes = ["source", "include"],
   visibility = ["//visibility:public"],
   alwayslink = 1,

   deps = [
        "//tools/profiling:mai_profiling",
        "//tools/converter/tensorflow/protos:tensorflow_cc_proto",
        "//tools/converter/onnx/protos:onnx_cc_proto",
   ]
)

cc_binary(
    name = "test",
    srcs = ["main.cpp"],
    includes = ["./source/"],
    deps = [
        "//:mai",
    ],
    linkopts = ["-lpthread"],
)
