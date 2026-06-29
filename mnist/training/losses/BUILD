load("//tools/build_rules:py.bzl", "py_library", "py_test")

py_library(
    name = "cross_entropy_loss",
    srcs = ["cross_entropy_loss.py", "__init__.py"],
    deps = [
        "//mnist/schema:schema",
        "@research_core//research_core/schema:schema",
        "@research_core//research_core/lightning:lightning_trainer",
        "@python_deps//torch",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "label_smoothing_loss",
    srcs = ["label_smoothing_loss.py"],
    deps = [
        "//mnist/schema:schema",
        "@research_core//research_core/schema:schema",
        "@research_core//research_core/lightning:lightning_trainer",
        "@python_deps//torch",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "losses",
    deps = [":cross_entropy_loss", ":label_smoothing_loss"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "cross_entropy_loss_test",
    srcs = ["cross_entropy_loss_test.py"],
    deps = [
        ":cross_entropy_loss",
        ":label_smoothing_loss",
        "@python_deps//torch",
    ],
)
