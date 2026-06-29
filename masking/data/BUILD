load("@rules_lilypad//lilypad/bazel:lilypad_workload_image.bzl", "lilypad_workload_image")
load("//tools/build_rules:py.bzl", "lilypad_py_binary", "py_library")

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

py_library(
    name = "build_wds_worker",
    srcs = ["build_wds_worker.py", "build_webdataset.py"],
    deps = [
        ":s3_download",
        "@python_deps//boto3",
        "@python_deps//botocore",
        "@python_deps//huggingface_hub",
        "@python_deps//numpy",
        "@python_deps//pandas",
        "@python_deps//pyarrow",
        "@python_deps//webdataset",
        "@python_deps//lilypad_py",
    ],
    visibility = ["//visibility:public"],
)

lilypad_workload_image(
    name = "lilypad_build_wds",
    lilypad = "@python_deps//lilypad_py",
    py_binary = lilypad_py_binary,
    workspace_name = "mnist_template",
    deps = [
        ":build_wds_worker",
        "@python_deps//boto3",
        "@python_deps//botocore",
        "@python_deps//huggingface_hub",
        "@python_deps//numpy",
        "@python_deps//pandas",
        "@python_deps//pyarrow",
        "@python_deps//webdataset",
        "@python_deps//aiohttp_cors",
        "@python_deps//colorful",
        "@python_deps//opencensus",
        "@python_deps//prometheus_client",
        "@python_deps//opentelemetry_exporter_prometheus",
        "@python_deps//virtualenv",
    ],
)
