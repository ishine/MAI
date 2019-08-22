config_setting(
    name = "windows",
    constraint_values = ["@bazel_tools//platforms:windows"],
)

cc_library(
    name = "gtest",
    srcs = glob(
        include = ["src/*.cc"],
        exclude = [
            "src/gtest_main.cc",
            "src/gtest-all.cc",
        ],
    ),
    hdrs = glob(
        include = [
            "src/*.h",
            "include/gtest/**/*.h",
        ],
    ),
    includes = ["include", "./"],
    visibility = ["//visibility:public"],
    linkopts = select({
        ":windows": [],
        "//conditions:default": ["-pthread"],
    }),
)
