workspace(name = "mnist_template")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# ---- bazel_skylib ----
http_archive(
    name = "bazel_skylib",
    sha256 = "f24ab666394232f834f74d19e2ff142b0af17466ea0c69a3f4c276ee75f6efce",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.4.0/bazel-skylib-1.4.0.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.4.0/bazel-skylib-1.4.0.tar.gz",
    ],
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

# ---- aspect_bazel_lib 2.16.0 ----
# Must be declared before aspect_rules_py and rules_lilypad to ensure the version
# with aspect_bazel_lib_register_toolchains is used (required by rules_lilypad setup.bzl).
http_archive(
    name = "aspect_bazel_lib",
    sha256 = "fc8fe1be58ae39f84a8613d554534760c7f0819d407afcc98bbcbd990523bfed",
    strip_prefix = "bazel-lib-2.16.0",
    urls = [
        "https://github.com/bazel-contrib/bazel-lib/releases/download/v2.16.0/bazel-lib-v2.16.0.tar.gz",
    ],
)

# ---- aspect_rules_py 0.7.4 ----
# Required for py_binary venv isolation (same as core-stack).
# rules_python py_binary does not propagate Python paths to Ray subprocesses (GCS, Raylet),
# causing GCS to fail to find packages. aspect_rules_py installs into a real venv that
# all subprocesses can discover.
http_archive(
    name = "aspect_rules_py",
    patches = [
        # Add external workspace directories to Python path so @python_deps packages
        # are visible inside the venv (required for Ray subprocess package discovery).
        "//third_party:aspect_rules_py.add_external_workspaces_to_python_path.patch",
        # Propagate target tags to the generated .venv target.
        "//third_party:aspect_rules_py.propagate_tags_to_venv.patch",
    ],
    sha256 = "04278ce23cc5c91a24b62ea00ac04c553fe40ca390943acf6684d367a681a871",
    strip_prefix = "rules_py-0.7.4",
    urls = [
        "https://github.com/aspect-build/rules_py/releases/download/v0.7.4/rules_py-v0.7.4.tar.gz",
    ],
)

load("@aspect_rules_py//py:repositories.bzl", "rules_py_dependencies")

rules_py_dependencies()

load("@aspect_rules_py//py:toolchains.bzl", "rules_py_toolchains")

rules_py_toolchains()

# ---- rules_python 0.31.0 ----
http_archive(
    name = "rules_python",
    sha256 = "c68bdc4fbec25de5b5493b8819cfc877c4ea299c0dcb15c244c5a00208cde311",
    strip_prefix = "rules_python-0.31.0",
    urls = [
        "https://github.com/bazelbuild/rules_python/releases/download/0.31.0/rules_python-0.31.0.tar.gz",
    ],
)

# ---- bazel_features 1.13.0 ----
# Must be declared before rules_proto_dependencies() which would otherwise pin 1.4.1
# (too old for rules_oci's permits_treeartifact_uplevel_symlinks check).
http_archive(
    name = "bazel_features",
    sha256 = "5d7e4eb0bb17aee392143cd667b67d9044c270a9345776a5e5a3cccbc44aa4b3",
    strip_prefix = "bazel_features-1.13.0",
    urls = [
        "https://github.com/bazel-contrib/bazel_features/releases/download/v1.13.0/bazel_features-v1.13.0.tar.gz",
    ],
)

# ---- rules_proto 6.0.2 ----
http_archive(
    name = "rules_proto",
    sha256 = "6fb6767d1bef535310547e03247f7518b03487740c11b6c6adb7952033fe1295",
    strip_prefix = "rules_proto-6.0.2",
    urls = [
        "https://github.com/bazelbuild/rules_proto/releases/download/6.0.2/rules_proto-6.0.2.tar.gz",
    ],
)

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies")

rules_proto_dependencies()

load("@rules_proto//proto:toolchains.bzl", "rules_proto_toolchains")

rules_proto_toolchains()

# ---- com_google_protobuf 27.0 ----
# Required by @rules_python//python/private/proto:py_proto_library.bzl
# (provides @com_google_protobuf//:protobuf_python runtime + protoc binary).
# Version 27.0 (Python pkg 5.27.0) is compatible with runtime protobuf==5.27.3
# since runtime >= gencode within the same major version is guaranteed.
http_archive(
    name = "com_google_protobuf",
    integrity = "sha256-2iiL8dqmwE0DqQUXgcqlKs65FjWGv/mqbPsS9puTlao=",
    strip_prefix = "protobuf-27.0",
    urls = [
        "https://github.com/protocolbuffers/protobuf/archive/refs/tags/v27.0.tar.gz",
    ],
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

# ---- local_repository for research-core ----
# research-core is a git submodule at research-core/. All downstream
# Bazel targets reference it via @research_core//.
local_repository(
    name = "research_core",
    path = "research-core",
)

# ---- rules_lilypad (via s3_archive) ----
load("//repository_rules:s3_archive.bzl", "s3_archive")

s3_archive(
    name = "rules_lilypad",
    bucket = "ursa-sdk-releases",
    file_path = "rules_lilypad-1.14.1.tar.gz",
    patches = ["//third_party:rules_lilypad.patch"],
    sha256 = "cdb78f1dde65ca0aba217841489df4707c7171c15430ef65ebb41198a8afecb7",
)

load("@rules_lilypad//lilypad/bazel:repositories.bzl", "repositories")

repositories()

load("@rules_lilypad//lilypad/bazel:setup.bzl", "setup")

setup()

load("@rules_lilypad//lilypad/bazel:resolve.bzl", "resolve")

resolve()

# ---- Python 3.10.11 hermetic toolchain ----
load("@rules_python//python:repositories.bzl", "py_repositories", "python_register_toolchains")

py_repositories()

python_register_toolchains(
    name = "python_3_10",
    python_version = "3.10.11",
    tool_versions = {
        "3.10.11": {
            "sha256": {
                "aarch64-apple-darwin": "8348bc3c2311f94ec63751fb71bd0108174be1c4def002773cf519ee1506f96f",
                "aarch64-unknown-linux-gnu": "c7573fdb00239f86b22ea0e8e926ca881d24fde5e5890851339911d76110bc35",
                "x86_64-apple-darwin": "bd3fc6e4da6f4033ebf19d66704e73b0804c22641ddae10bbe347c48f82374ad",
                # Applied Mosaic Python build for x86_64 Linux — matches what core-stack uses
                # and is known to work with rules_lilypad in the cluster environment.
                "x86_64-unknown-linux-gnu": "60f7b0f6c81552d2b1c8231cefd5db5f932bafa1fdad769259fe89998fea136b",
            },
            "strip_prefix": "python",
            "url": {
                "aarch64-apple-darwin": "https://github.com/indygreg/python-build-standalone/releases/download/20230507/cpython-3.10.11+20230507-aarch64-apple-darwin-install_only.tar.gz",
                "aarch64-unknown-linux-gnu": "https://github.com/indygreg/python-build-standalone/releases/download/20230507/cpython-3.10.11+20230507-aarch64-unknown-linux-gnu-install_only.tar.gz",
                "x86_64-apple-darwin": "https://github.com/indygreg/python-build-standalone/releases/download/20230507/cpython-3.10.11+20230507-x86_64-apple-darwin-install_only.tar.gz",
                "x86_64-unknown-linux-gnu": "https://applied-mosaic-public.s3.amazonaws.com/2023_11_28T20_06_27_mosaic_python310_with_pretty_print.tar.gz",
            },
        },
    },
)

load("@python_3_10//:defs.bzl", "interpreter")

# ---- pip_parse for Python dependencies ----
load("@rules_python//python:pip.bzl", "pip_parse")

load("//lockfiles:resolve_lockfile.bzl", "resolve_lockfile")

resolve_lockfile(
    name = "resolved_requirements",
    lockfile = "//lockfiles:requirements_lock.txt",
)

pip_parse(
    name = "python_deps",
    experimental_requirement_cycles = {
        "markdown": [
            "markdown_it_py",
            "mdit_py_plugins",
        ],
        "markdown-it-py": [
            "markdown-it-py",
            "mdit-py-plugins",
        ],
    },
    python_interpreter_target = interpreter,
    requirements_lock = "@resolved_requirements//:requirements_lock.txt",
)

load("@python_deps//:requirements.bzl", "install_deps")

install_deps()
