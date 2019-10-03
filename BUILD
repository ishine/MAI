cc_library(
   name = "mai",
   srcs = glob([
       "source/core/*.cpp",
       "source/ops/cpu/*.cpp",
       "source/util/*.cpp",
       "tools/converter/tensorflow/TensorflowNetwork.cpp",
   ]),
   hdrs = glob([
       "include/OperatorType.def",
       "include/*.h",
       "source/core/*.h",
       "source/util/*.h",
       "source/ops/cpu/*.h",
       "tools/converter/tensorflow/*.h",
       "tools/profiling/*.h",

   ]),
   copts = ["-Wall", "-Wextra",],
   includes = ["source", "include"],
   visibility = ["//visibility:public"],

   deps = [
        "//tools/converter/tensorflow/protos:tensorflow_cc_proto",
        "//tools/profiling:mai_profiling",
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
