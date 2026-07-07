load("//tools/build_rules:py.bzl", "py_library", "py_test")

py_library(
    name = "rollout_harvester",
    srcs = ["rollout_harvester.py", "__init__.py"],
    deps = [
        "@python_deps//einops",
        "@python_deps//numpy",
        "@python_deps//torch",
        "@python_deps//transformers",
        "@python_deps//yaml",
        # alpamayo1_5 is installed from the nvidia/Alpamayo-1.5 venv (same as
        # masking/BUILD's masked_model target) -- declared as a runtime dep,
        # not a Bazel py dep, so the import resolves at run time.
        # masking.data.wds_dataset is imported lazily (inside harvest_dataset,
        # not at module load time) so this target's own deps stay minimal;
        # //masking/data:wds_dataset is still listed here since harvest_dataset
        # is part of this library and bazel needs it on the runtime path.
        "//masking/data:wds_dataset",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "rollout_harvester_test",
    srcs = ["rollout_harvester_test.py"],
    deps = [
        ":rollout_harvester",
        "@python_deps//numpy",
        "@python_deps//torch",
    ],
)

py_library(
    name = "trajectory_features",
    srcs = ["trajectory_features.py"],
    deps = [
        "@python_deps//numpy",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "synthetic_trajectory_fixtures",
    srcs = ["synthetic_trajectory_fixtures.py"],
    deps = [
        "@python_deps//numpy",
    ],
    testonly = True,
)

py_test(
    name = "trajectory_features_test",
    srcs = ["trajectory_features_test.py"],
    deps = [
        ":synthetic_trajectory_fixtures",
        ":trajectory_features",
        "@python_deps//numpy",
    ],
)

py_library(
    name = "classify_maneuvers",
    srcs = ["classify_maneuvers.py"],
    deps = [
        ":trajectory_features",
        "@python_deps//pandas",
        "@python_deps//yaml",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "classify_maneuvers_test",
    srcs = ["classify_maneuvers_test.py"],
    deps = [
        ":classify_maneuvers",
        ":synthetic_trajectory_fixtures",
        ":trajectory_features",
    ],
)

py_library(
    name = "maneuver_report",
    srcs = ["maneuver_report.py"],
    deps = [
        ":classify_maneuvers",
        "@python_deps//matplotlib",
        "@python_deps//numpy",
        "@python_deps//pandas",
        "@python_deps//yaml",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "maneuver_report_test",
    srcs = ["maneuver_report_test.py"],
    deps = [
        ":classify_maneuvers",
        ":maneuver_report",
        ":synthetic_trajectory_fixtures",
    ],
)

py_library(
    name = "action_space_variance",
    srcs = ["action_space_variance.py"],
    deps = [
        "@python_deps//numpy",
        "@python_deps//pandas",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "action_space_variance_test",
    srcs = ["action_space_variance_test.py"],
    deps = [
        ":action_space_variance",
        "@python_deps//numpy",
        "@python_deps//pandas",
    ],
)
