"""Generate requirements_lock.txt from pyproject.toml using uv.

Usage:
    python3 lockfiles/generate_lockfiles.py pyproject.toml -o lockfiles/requirements_lock.txt
"""

import argparse
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List


def get_extra_index_urls(pyproject_data: Dict) -> List[str]:
    """Get the extra index URLs from the pyproject.toml file."""
    indices = []
    if "tool" in pyproject_data and "uv" in pyproject_data["tool"]:
        uv_config = pyproject_data["tool"]["uv"]
        if "index" in uv_config:
            for index in uv_config["index"]:
                if "url" in index:
                    indices.append(index["url"].strip())
    return indices


def get_local_file_packages(pyproject_data: Dict) -> Dict[str, str]:
    """Query all local files from source."""
    local_files = {}
    if "tool" in pyproject_data and "uv" in pyproject_data["tool"]:
        uv_config = pyproject_data["tool"]["uv"]
        if "sources" in uv_config:
            for name, source in uv_config["sources"].items():
                if "path" in source:
                    path = source["path"].strip()
                    if Path(path).is_absolute():
                        raise ValueError(
                            f"[tool.uv.sources] '{name}' uses absolute path '{path}'. "
                            "Use a path relative to pyproject.toml instead."
                        )
                    local_files[name] = path
    return local_files


def get_package_extras(pyproject_data: Dict) -> Dict[str, List[str]]:
    """Get the extras for each package from the pyproject.toml file."""
    extras = {}
    if "dependency-groups" in pyproject_data:
        for group in pyproject_data["dependency-groups"].values():
            for dep in group:
                if isinstance(dep, str) and "[" in dep and "]" in dep:
                    package = dep.split("[")[0].strip()
                    package_extras = dep.split("[")[1].split("]")[0].split(",")
                    extras[package] = package_extras
    return extras


def adapt_local_file_packages(
    pyproject_toml: Path, local_file_packages: Dict[str, str], req: str
) -> str:
    """Adapt the local file packages to use workspace-relative file: URIs."""
    workspace_root = pyproject_toml.parent.resolve()
    for name, path in local_file_packages.items():
        if path in req:
            req_path = req.split(";")[0].strip()
            abs_path = (pyproject_toml.parent / Path(path)).resolve()
            workspace_rel = abs_path.relative_to(workspace_root)
            return req.replace(req_path, f"{name} @ file:{workspace_rel}")
    return req


def add_extras_to_requirement(req: str, package_extras: Dict[str, List[str]]) -> str:
    """Add extras to a requirement if specified in pyproject.toml."""
    for package, extras in package_extras.items():
        if package in req:
            extras_str = ",".join(extras)
            return re.sub(rf"^{package}", f"{package}[{extras_str}]", req)
    return req


def generate_lockfiles(pyproject_toml: Path, output: Path, check_only: bool) -> None:
    """Generate requirements_lock.txt from pyproject.toml using uv."""
    import tomli

    pyproject_data = tomli.load(pyproject_toml.open("rb"))
    extra_indices = get_extra_index_urls(pyproject_data)
    path_sources = get_local_file_packages(pyproject_data)
    package_extras = get_package_extras(pyproject_data)

    # Classify [tool.uv.sources] path entries:
    #   - wheel files (foo.whl, or any path resolving to a file) are emitted into
    #     the lockfile rewritten to workspace-relative file: URIs.
    #   - directory entries (e.g. research-core) are editable local packages. We
    #     union their dependency closure into the lockfile but exclude the package
    #     itself via `uv export --no-emit-package`, because Bazel supplies their
    #     source via local_repository rather than installing them with pip.
    wheel_packages: Dict[str, str] = {}
    editable_packages: List[str] = []
    for name, path in path_sources.items():
        if (pyproject_toml.parent / Path(path)).is_dir():
            editable_packages.append(name)
        else:
            wheel_packages[name] = path

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt") as temp_file:
        temp_path = temp_file.name
        sync_cmd = ["uv", "lock", "--no-cache", "--index-strategy", "unsafe-best-match"]
        if not check_only:
            # Force re-resolution of all local sources so edits to their
            # pyproject.toml (added/removed deps) are picked up.
            for package in list(wheel_packages) + editable_packages:
                sync_cmd.extend(["--upgrade-package", package])
        export_cmd = [
            "uv", "export", "--no-emit-workspace",
            "--index-strategy", "unsafe-best-match",
            "-o", temp_path,
        ]
        # Emit each editable package's deps but not the package entry itself.
        for name in editable_packages:
            export_cmd.extend(["--no-emit-package", name])
        subprocess.run(sync_cmd, cwd=pyproject_toml.parent, check=True)
        subprocess.run(
            export_cmd, cwd=pyproject_toml.parent, check=True, capture_output=True
        )

        with open(temp_path, "r") as f:
            requirements = [
                line.rstrip() for line in f if line.strip() and not line.startswith("#")
            ]

        with open(output, "w") as f:
            f.write(
                "# This file was autogenerated by the following command:\n"
                f"#    python3 lockfiles/generate_lockfiles.py {pyproject_toml} -o {output}\n\n"
            )
            for index in extra_indices:
                f.write(f"--extra-index-url {index}\n")
            if extra_indices:
                f.write("\n")

            for req in requirements:
                req = adapt_local_file_packages(pyproject_toml, wheel_packages, req)
                req = add_extras_to_requirement(req, package_extras)
                if "file:///" in req:
                    raise ValueError(
                        f"Lockfile would contain an absolute file:// URI:\n  {req}\n"
                        "All local wheel paths must be relative to the workspace root."
                    )
                f.write(f"{req}\n")


def argparser() -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Generate lockfiles from pyproject.toml"
    )
    parser.add_argument(
        "pyproject_toml", type=Path, help="Path to the pyproject.toml file"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("requirements_lock.txt"),
        help="Path to the output requirements_lock.txt file",
    )
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="Check only if the output file is up to date.",
    )
    return vars(parser.parse_args())


if __name__ == "__main__":
    args = argparser()
    generate_lockfiles(**args)
