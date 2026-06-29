load("@rules_lilypad//lilypad/bazel:lilypad_workload_image.bzl", "lilypad_workload_image")
load("//tools/build_rules:py.bzl", "lilypad_py_binary", "py_library")

py_library(
    name = "run_lib",
    srcs = ["run.py", "__init__.py"],
    deps = [
        "//masking:masked_model",
        "//masking/data:s3_download",
        "//masking/data:wds_dataset",
        "@python_deps//numpy",
        "@python_deps//torch",
        "@python_deps//lilypad_py",
    ],
    visibility = ["//visibility:public"],
)

lilypad_workload_image(
    name = "lilypad_masking",
    lilypad = "@python_deps//lilypad_py",
    py_binary = lilypad_py_binary,
    workspace_name = "mnist_template",
    deps = [
        ":run_lib",
        "@python_deps//boto3",
        "@python_deps//botocore",
        "@python_deps//einops",
        "@python_deps//numpy",
        "@python_deps//torch",
        "@python_deps//transformers",
        "@python_deps//webdataset",
        "@python_deps//pyarrow",
        "@python_deps//pandas",
        "@python_deps//aiohttp_cors",
        "@python_deps//colorful",
        "@python_deps//opencensus",
        "@python_deps//prometheus_client",
        "@python_deps//opentelemetry_exporter_prometheus",
        "@python_deps//virtualenv",
    ],
)
