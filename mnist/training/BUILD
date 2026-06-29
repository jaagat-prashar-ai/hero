load("@rules_lilypad//lilypad/bazel:lilypad_workload_image.bzl", "lilypad_workload_image")
load("//tools/build_rules:py.bzl", "lilypad_py_binary", "py_library", "py_test")

py_library(
    name = "lightning_module",
    srcs = ["lightning_module.py"],
    deps = [
        "//mnist/training/losses:losses",
        "@research_core//research_core/lightning:lightning_trainer",
        "@python_deps//torch",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "training_lib",
    srcs = ["train.py", "__init__.py"],
    deps = [
        ":lightning_module",
        "//mnist/data:datamodule",
        "//mnist:model",
        "//mnist:config",
        "//mnist/training/callbacks:callbacks",
        "@research_core//research_core/lightning:lightning_trainer",
        "@research_core//research_core/lightning:preemption",
        "@research_core//research_core/lightning:loss_logging_callback",
        "@research_core//research_core/lightning:classification_metrics_callback",
        "@research_core//research_core/lightning:wandb_log_image_callback",
        "@python_deps//torch",
        "@python_deps//wandb",
        "@python_deps//lilypad_py",
    ],
    visibility = ["//visibility:public"],
)

lilypad_workload_image(
    name = "lilypad_mnist_training",
    lilypad = "@python_deps//lilypad_py",
    py_binary = lilypad_py_binary,
    workspace_name = "mnist_template",
    deps = [
        ":training_lib",
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

py_test(
    name = "train_test",
    srcs = ["train_test.py"],
    deps = [
        "//mnist:config",
        ":training_lib",
    ],
)

py_test(
    name = "lightning_module_test",
    srcs = ["lightning_module_test.py"],
    deps = [
        ":lightning_module",
        "//mnist:model",
        "//mnist/components:components",
        "@python_deps//torch",
    ],
)
