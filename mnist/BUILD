load("//tools/build_rules:py.bzl", "py_library")

py_library(
    name = "model",
    srcs = ["model.py", "__init__.py"],
    deps = [
        "//mnist/components:components",
        "//mnist/schema:schema",
        "@research_core//research_core/schema:schema",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "config",
    srcs = ["config.py"],
    deps = ["//mnist/schema:schema"],
    visibility = ["//visibility:public"],
)
