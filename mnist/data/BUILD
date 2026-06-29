load("//tools/build_rules:py.bzl", "py_library")

py_library(
    name = "datamodule",
    srcs = ["datamodule.py", "__init__.py"],
    deps = [
        "//mnist:config",
        "//mnist/schema:schema",
        "@research_core//research_core/schema:schema_data",
        "@python_deps//torch",
        "@python_deps//torchvision",
    ],
    visibility = ["//visibility:public"],
)
