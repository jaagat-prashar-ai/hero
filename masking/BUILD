load("//tools/build_rules:py.bzl", "py_library")

py_library(
    name = "masked_model",
    srcs = ["masked_model.py", "__init__.py"],
    deps = [
        "@python_deps//einops",
        "@python_deps//numpy",
        "@python_deps//torch",
        "@python_deps//transformers",
        # alpamayo1_5 is installed from the nvidia/Alpamayo-1.5 venv;
        # declared as a data dep so the import resolves at runtime.
    ],
    visibility = ["//visibility:public"],
)
