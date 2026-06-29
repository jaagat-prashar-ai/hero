"""Python build rules for mnist_template.

Re-exports the py_library, py_binary, and py_test wrappers from research-core so
experiment code can load //tools/build_rules:py.bzl without duplication.

Loaded symbols are private to the loading file by default, so they must be bound
to top-level names here to be re-exported to other BUILD/.bzl files.
"""

load(
    "@research_core//tools/build_rules:py.bzl",
    _py_binary = "py_binary",
    _py_library = "py_library",
    _py_test = "py_test",
)
load("@aspect_rules_py//py:defs.bzl", _aspect_py_binary = "py_binary")

py_binary = _py_binary
py_library = _py_library
py_test = _py_test

def lilypad_py_binary(package_collisions = "ignore", **kwargs):
    """aspect_rules_py py_binary for the Lilypad/Ray workload image.

    research-core's py_binary wraps rules_python, whose bootstrap sets the import
    path only for the launched process. Ray spawns worker *subprocesses* that do
    not inherit that path, so any package living in an external repo (notably
    @research_core's top-level `proto`) is invisible to workers and imports fail.

    aspect_rules_py (patched in WORKSPACE via
    aspect_rules_py.add_external_workspaces_to_python_path.patch) instead writes a
    venv `.pth` enumerating every external repo root into site-packages. site
    `.pth` files are processed on every interpreter startup, including Ray
    workers, so all external packages resolve there. This mirrors core-stack,
    whose //tools/build_rules:py.bzl wraps aspect_rules_py for the same reason.

    Use this only for the `py_binary` argument of lilypad_workload_image. Regular
    targets keep the rules_python-backed `py_binary` above. package_collisions
    defaults to "ignore" because the workload venv bundles wheels (torch, ray,
    ...) with overlapping files.
    """
    _aspect_py_binary(package_collisions = package_collisions, **kwargs)
