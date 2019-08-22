cc_library(
   name = "mai",
   srcs = glob([
       "source/core/*.cpp",
       "source/ops/cpu/*.cpp",
   ]),
   hdrs = glob([
       "include/*.h",
       "source/core/*.h",
       "source/util/*.h",
       "source/ops/cpu/*.h",
   ]),
   copts = ["-Wall", "-Wextra"],
   includes = ["source", "include"],
   visibility = ["//visibility:public"],
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
