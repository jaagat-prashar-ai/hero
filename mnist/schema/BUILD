load("//tools/build_rules:py.bzl", "py_library", "py_test")

py_library(
    name = "schema",
    srcs = ["schemas.py", "constants.py", "__init__.py"],
    deps = [
        "@research_core//research_core/schema:schema",
        "@research_core//research_core/proto:typed_tensors_proto_py",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "schemas_test",
    srcs = ["schemas_test.py"],
    deps = [":schema"],
)
