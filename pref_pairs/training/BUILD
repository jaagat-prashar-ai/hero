load("//tools/build_rules:py.bzl", "py_library", "py_test")

py_library(
    name = "run_lib",
    srcs = ["run.py", "__init__.py"],
    deps = [
        "//pref_pairs:action_space_variance",
        "//pref_pairs:classify_maneuvers",
        "//pref_pairs:rollout_harvester",
        "//pref_pairs:trajectory_features",
        "//masking/data:wds_dataset",
        "@python_deps//pandas",
        "@python_deps//torch",
        "@python_deps//yaml",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "run_test",
    srcs = ["run_test.py"],
    deps = [
        ":run_lib",
    ],
)
