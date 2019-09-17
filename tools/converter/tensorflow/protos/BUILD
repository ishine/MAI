#load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library")

#proto_library(
#    name = "tensorflow_proto",
#    srcs = [
#        glob(["*.proto"])
#    ],
#)

cc_proto_library(
    name = "tensorflow_cc_proto",
    deps = [
        ":graph_proto"
    ],
    visibility = ["//visibility:public"],
)

proto_library(
    name = "graph_proto",
    srcs = glob(["*.proto"]),
)

#proto_library(
#    name = "graph_proto",
#    srcs = ["graph.proto", "node_def.proto"],
#    deps = [":node_def_proto", ":function_proto", ":versions_proto"],
#)
#
#proto_library(
#    name = "node_def_proto",
#    srcs = ["node_def.proto"],
#    deps = [":attr_value_proto"],
#)
#
#proto_library(
#    name = "function_proto",
#    srcs = ["function.proto"],
#    deps = [":node_def_proto", ":attr_value_proto", ":op_def_proto"],
#)
#
#proto_library(
#    name = "attr_value_proto",
#    srcs = ["attr_value.proto"],
#    deps = [":tensor_proto", ":tensor_shape_proto", ":types_proto"],
#)
#
#proto_library(
#    name = "op_def_proto",
#    srcs = ["op_def.proto"],
#    deps = [":attr_value_proto", ":types_proto"],
#)
#
#proto_library(
#    name = "tensor_proto",
#    srcs = ["tensor.proto"],
#    deps = [":resource_handle_proto", ":tensor_shape_proto", ":types_proto"]
#)
#
#proto_library(
#    name = "types_proto",
#    srcs = ["types.proto"],
#)
#
#proto_library(
#    name = "tensor_shape_proto",
#    srcs = ["tensor_shape.proto"],
#)
#
#proto_library(
#    name = "resource_handle_proto",
#    srcs = ["resource_handle.proto"],
#)
#
#proto_library(
#    name = "versions_proto",
#    srcs = ["versions.proto"],
#)
#
