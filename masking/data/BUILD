load("//tools/build_rules:py.bzl", "py_library")

py_library(
    name = "s3_download",
    srcs = ["s3_download.py", "__init__.py"],
    deps = [
        "@python_deps//boto3",
        "@python_deps//botocore",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "wds_dataset",
    srcs = ["wds_dataset.py"],
    deps = [
        "@python_deps//numpy",
        "@python_deps//torch",
        "@python_deps//webdataset",
    ],
    visibility = ["//visibility:public"],
)
