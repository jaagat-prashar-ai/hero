"""Repository rule that resolves relative file: URIs in a pip lockfile to absolute paths.

The committed lockfile uses workspace-relative paths (e.g. file:third_party/foo.whl)
so it is portable across different mount points. This rule rewrites them to absolute
file:/// URIs at build time so pip can locate the wheels.
"""

def _resolve_lockfile_impl(rctx):
    lockfile_path = rctx.path(rctx.attr.lockfile)
    lockfile_content = rctx.read(lockfile_path)

    workspace_root = str(lockfile_path.dirname.dirname)

    lines = []
    for line in lockfile_content.splitlines():
        if "@ file:" in line and "@ file://" not in line:
            line = line.replace("@ file:", "@ file://" + workspace_root + "/")
        lines.append(line)

    rctx.file("requirements_lock.txt", "\n".join(lines) + "\n")
    rctx.file("BUILD.bazel", 'exports_files(["requirements_lock.txt"])\n')

resolve_lockfile = repository_rule(
    implementation = _resolve_lockfile_impl,
    attrs = {
        "lockfile": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
    },
)
