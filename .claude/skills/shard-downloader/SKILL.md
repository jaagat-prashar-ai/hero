---
name: shard-downloader
description: Download dataset shard files (typically .tar WebDataset shards, but any object) from the OCI-backed research-datasets-chicago S3-compatible bucket to a local ~/Desktop/Shards folder. Use this skill whenever the user wants to pull down a shard, tarball, checkpoint, or any object from the research datasets bucket — phrases like "download shard", "grab that shard locally", "pull shard_000_00000", "get me shards from wds-smoke-av1-v2", "download from research-datasets-chicago", or "save that dataset file to my Desktop". Also use it when the user gives a full s3:// path into research-datasets-chicago and just wants it on disk, or when they want a whole prefix / range of shards mirrored locally.
---

# Shard Downloader

Download objects (usually WebDataset `.tar` shards) from the OCI object-storage bucket `research-datasets-chicago` into a local `~/Desktop/Shards` folder, using the AWS CLI with the OCI S3-compatible profile.

## Prerequisites (verify, don't assume)

This skill relies on an already-working AWS CLI + OCI setup on the user's Mac. Before downloading, confirm the pieces are in place:

1. **AWS CLI installed** — `aws --version`. If missing, tell the user to run `brew install awscli`.
2. **The `oci.chi` profile exists** in `~/.aws/config` with the Chicago region and the OCI S3-compatible endpoint, and matching credentials in `~/.aws/credentials`. Check with:
   ```bash
   aws configure list --profile oci.chi
   ```
   If the profile is missing or credentials are wrong, see `references/setup.md` for the exact config blocks to write. Do NOT paste real secret keys into the skill or into chat — point the user to their existing credential source (e.g. the `~/.aws/credentials` they copied from another host).
A quick liveness check that also confirms auth works:
```bash
aws --profile oci.chi s3 ls s3://research-datasets-chicago/ | head
```
If this returns a `SignatureDoesNotMatch` or "secret key ... could not be found" error, the credentials in `~/.aws/credentials` are placeholders or stale — the user needs to fill in the real `oci.chi` key/secret. See `references/setup.md`.

## The core workflow

Use the helper script for all downloads — it handles creating `~/Desktop/Shards`, resolving partial shard names, single-file vs prefix vs range, and reporting what landed on disk.

```bash
python3 scripts/download_shards.py <what-to-download> [options]
```

`<what-to-download>` can be any of:

- **A full S3 URI**: `s3://research-datasets-chicago/nvidia_physicalai_datasets/.../train/shard_000_00000.tar`
- **A bucket-relative key**: `nvidia_physicalai_datasets/PhysicalAI-Autonomous-Vehicles/wds-smoke-av1-v2/train/shard_000_00000.tar`
- **A prefix** (ends in `/`), to mirror everything under it: `.../wds-smoke-av1-v2/train/`
- **Just a shard filename**, when combined with `--prefix` (see below): `shard_000_00000.tar`
### Common invocations

Download one shard by full path:
```bash
python3 scripts/download_shards.py \
  s3://research-datasets-chicago/nvidia_physicalai_datasets/PhysicalAI-Autonomous-Vehicles/wds-smoke-av1-v2/train/shard_000_00000.tar
```

Download several shards from the same prefix by name:
```bash
python3 scripts/download_shards.py \
  --prefix nvidia_physicalai_datasets/PhysicalAI-Autonomous-Vehicles/wds-smoke-av1-v2/train/ \
  shard_000_00000.tar shard_000_00001.tar shard_000_00002.tar
```

Download a contiguous range (inclusive) of shards — great for WebDataset naming:
```bash
python3 scripts/download_shards.py \
  --prefix nvidia_physicalai_datasets/PhysicalAI-Autonomous-Vehicles/wds-smoke-av1-v2/train/ \
  --range shard_000_00000.tar shard_000_00009.tar
```

Mirror an entire prefix (all shards under it):
```bash
python3 scripts/download_shards.py \
  s3://research-datasets-chicago/nvidia_physicalai_datasets/PhysicalAI-Autonomous-Vehicles/wds-smoke-av1-v2/train/
```

### Useful options

- `--dest PATH` — override the destination (defaults to `~/Desktop/Shards`).
- `--flat` — save every file directly into the destination folder instead of recreating the S3 key's directory structure. Default is `--flat` for single files and a mirrored tree for prefixes; pass `--no-flat` / `--flat` to force either.
- `--dry-run` — list what *would* be downloaded (with sizes) without transferring. Use this first when the user asks for a prefix or range, so they can confirm the size before a large pull.
- `--profile NAME` — AWS profile (default `oci.chi`).
## Recommended sequence when helping a user

1. Figure out exactly what they want: a single named shard, a set of named shards, a numeric range, or a whole prefix.
2. If it's a prefix or range, run with `--dry-run` first and report the count + total size. Confirm before a large download.
3. Run the real download.
4. Report where the files landed (`~/Desktop/Shards/...`) and their sizes. If any object failed, surface the exact error rather than glossing over it.
## Notes and gotchas

- **This is OCI object storage behind an S3-compatible endpoint**, not real AWS S3. The `oci.chi` profile must carry both `region = us-chicago-1` and the `endpoint_url` (via the `services` block). If a command errors about region or signatures, that config is the first suspect. See `references/setup.md`.
- **Listing a prefix** uses `aws s3 ls`; large prefixes can contain thousands of objects (some of these dataset prefixes are huge). Prefer ranges or explicit names over blindly mirroring a top-level prefix.
- **Never hardcode credentials.** The script only ever references a named profile. If auth fails, the fix is in `~/.aws/credentials`, not in this skill.
- **`.tar` shards can be large** (hundreds of MB to GB each). For multi-file pulls, the script downloads sequentially and prints progress per file; warn the user about size before big transfers.
- The destination `~/Desktop/Shards` is created automatically if it doesn't exist.
## Reference files

- `references/setup.md` — exact `~/.aws/config` and `~/.aws/credentials` blocks for the `oci.chi` profile, and how to diagnose the common `SignatureDoesNotMatch` error.
