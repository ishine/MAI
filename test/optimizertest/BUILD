cc_binary(
    name = "optimizer_test",
    srcs = glob(
        [
            "core/*.cpp",
            "units/*.cpp",
            "core/*.h",
        ]
    ),
    includes = ["./"],
    copts = ["-Wall", "-Wextra"],
    deps = [
        "//3rd_party/gtest:gtest",
        "//:mai",
    ],
)
