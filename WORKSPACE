#load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
#
#http_archive(
#    name = "com_google_protobuf",
#    sha256 = "3d4e589d81b2006ca603c1ab712c9715a76227293032d05b26fca603f90b3f5b",
#    strip_prefix = "protobuf-3.6.1",
#    urls = [
#        "https://mirror.bazel.build/github.com/google/protobuf/archive/v3.6.1.tar.gz",
#        "https://github.com/google/protobuf/archive/v3.6.1.tar.gz",
#    ],
#)

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

git_repository(
    name = "protobuf_bzl",
    # v3.6.1.3
    commit = "66dc42d891a4fc8e9190c524fd67961688a37bbe",
    remote = "https://github.com/google/protobuf.git",
)

bind(
    name = "protobuf",
    actual = "@protobuf_bzl//:protobuf",
)
bind(
    name = "protoc",
    actual = "@protobuf_bzl//:protoc",
)
# Using protobuf version 3.6.1.3
http_archive(
    name = "com_google_protobuf",
    strip_prefix = "protobuf-3.6.1.3",
    urls = [
        "https://mirror.bazel.build/github.com/google/protobuf/archive/v3.6.1.3.tar.gz",
        "https://github.com/google/protobuf/archive/v3.6.1.3.zip"],
)
