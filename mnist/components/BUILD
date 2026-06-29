load("//tools/build_rules:py.bzl", "py_library", "py_test")

py_library(
    name = "encoder",
    srcs = ["encoder.py", "__init__.py"],
    deps = [
        "//mnist/schema:schema",
        "@research_core//research_core/schema:schema",
        "@research_core//research_core/proto:typed_tensors_proto_py",
        "@python_deps//torch",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "classifier",
    srcs = ["classifier.py"],
    deps = [
        "//mnist/schema:schema",
        "@research_core//research_core/schema:schema",
        "@research_core//research_core/proto:typed_tensors_proto_py",
        "@python_deps//torch",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "components",
    deps = [":encoder", ":classifier"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "encoder_test",
    srcs = ["encoder_test.py"],
    deps = [
        ":encoder",
        "//mnist/schema:schema",
        "@python_deps//torch",
    ],
)

py_test(
    name = "classifier_test",
    srcs = ["classifier_test.py"],
    deps = [
        ":classifier",
        "@python_deps//torch",
    ],
)
