load("//tools/build_rules:py.bzl", "py_library")

py_library(
    name = "callbacks",
    srcs = ["confusion_matrix_callback.py", "__init__.py"],
    deps = [
        "@python_deps//torch",
        "@python_deps//torchmetrics",
        "@python_deps//wandb",
        "@python_deps//lightning",
        "@python_deps//matplotlib",
    ],
    visibility = ["//visibility:public"],
)
