cc_binary(
    name = "optest",
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
