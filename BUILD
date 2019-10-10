cc_library(
   name = "mai",
   srcs = glob([
       "source/core/*.cpp",
       "source/ops/cpu/*.cpp",
       "source/util/*.cpp",
   ]) + ["//tools/converter/tensorflow:TensorflowParser.cpp"]
      + ["//tools/converter/onnx:OnnxParser.cpp"],
   hdrs = glob([
       "include/OperatorType.def",
       "include/*.h",
       "source/core/*.h",
       "source/util/*.h",
       "source/ops/cpu/*.h",
       "tools/profiling/*.h",

   ]) + ["//tools/converter/tensorflow:TensorflowParser.h"]
      + ["//tools/converter/onnx:OnnxParser.h"],
   copts = ["-Wall", "-Wextra", "-std=c++11", "-fopenmp", "-O3"],
   linkopts = ["-fopenmp"],
   includes = ["source", "include"],
   visibility = ["//visibility:public"],

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
