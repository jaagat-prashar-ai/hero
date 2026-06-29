"""Minimal s3_archive repository rule.

Downloads an archive from an S3 bucket using the AWS CLI, then extracts it.
Compatible with the AWS credentials available in the development environment.
"""

def _s3_archive_impl(rctx):
    aws = rctx.which("aws")
    if not aws:
        fail("aws CLI not found in PATH. Please run 'awslogin --sso' and ensure awscli is installed.")

    bucket = rctx.attr.bucket
    file_path = rctx.attr.file_path
    archive_name = file_path.split("/")[-1]

    # Try without explicit profile first (uses default credential chain),
    # then fall back to named profiles
    profiles = [None, "oci", "default", "applied-sso"]
    success = False
    last_stderr = ""
    for profile in profiles:
        cmd = [aws, "s3", "cp", "s3://{}/{}".format(bucket, file_path), archive_name]
        if profile:
            cmd += ["--profile", profile]
        result = rctx.execute(cmd, timeout = 600)
        if result.return_code == 0:
            success = True
            break
        last_stderr = result.stderr

    if not success:
        fail("Failed to download s3://{}/{} (tried all profiles)\nstderr: {}".format(
            bucket,
            file_path,
            last_stderr,
        ))

    # Extract the archive
    strip_prefix = rctx.attr.strip_prefix
    rctx.extract(archive_name, stripPrefix = strip_prefix)

    # Apply patches
    for patch in rctx.attr.patches:
        rctx.patch(patch, strip = 0)

    # Run patch_cmds
    for cmd in rctx.attr.patch_cmds:
        rctx.execute(["bash", "-c", cmd])

    # Create a root BUILD if not provided by the archive and no build_file specified
    if rctx.attr.build_file:
        rctx.file("BUILD.bazel", rctx.read(rctx.path(rctx.attr.build_file)))

s3_archive = repository_rule(
    implementation = _s3_archive_impl,
    attrs = {
        "bucket": attr.string(mandatory = True, doc = "S3 bucket name"),
        "file_path": attr.string(mandatory = True, doc = "Path within the S3 bucket"),
        "sha256": attr.string(doc = "Expected SHA256 hash of the archive (informational only)"),
        "strip_prefix": attr.string(default = "", doc = "Path prefix to strip when extracting"),
        "build_file": attr.label(allow_single_file = True, doc = "BUILD file to use for this repository"),
        "patches": attr.label_list(allow_files = True, doc = "Patch files to apply"),
        "patch_cmds": attr.string_list(doc = "Shell commands to run after extraction"),
    },
)
