# Sample N random OOD/long-tail clips from reasoning/ood_reasoning.parquet and
# resolve each one's exact S3 WDS shard, using the same deterministic
# sort+partition reconstruction we validated in T1.3 (no clip->shard index
# exists upstream; we derive it from build_webdataset.py's own logic).

import hashlib
import json
import os
import random

import boto3
import pandas as pd
import physical_ai_av
from huggingface_hub import hf_hub_download

WORLD_SIZE = 100
CLIPS_PER_SHARD = 50
MIN_T0_US = 1_700_000  # needs > 1.6e6 (history window) with a small buffer
BUCKET = "research-datasets-chicago"
PREFIX = "nvidia_physicalai_datasets/PhysicalAI-Autonomous-Vehicles/wds"
S3_ENDPOINT = "https://idskhu5vqvtl.compat.objectstorage.us-chicago-1.oraclecloud.com"


def shard_exists(s3, key: str) -> bool:
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except Exception:
        return False


def _hash_split(clip_id: str, val_frac: float = 0.10) -> str:
    h = int(hashlib.md5(clip_id.encode()).hexdigest(), 16) % 1000
    return "val" if h < int(val_frac * 1000) else "train"


def build_splits_map(feature_df: pd.DataFrame, ood_df: pd.DataFrame) -> dict[str, str]:
    ood_split_map = {}
    for cid, row in ood_df.iterrows():
        cid = str(cid)
        ood_split_map[cid] = str(row.get("split", "train"))
    all_clips = feature_df.index.astype(str).tolist()
    return {cid: ood_split_map.get(cid, _hash_split(cid)) for cid in all_clips}


def resolve_shard(clip_id: str, chunk_of: dict, splits_map: dict, split: str) -> dict:
    work = [
        cid
        for cid, s in sorted(splits_map.items())
        if s == split and chunk_of.get(cid, -1) % WORLD_SIZE == chunk_of[clip_id] % WORLD_SIZE
    ]
    idx = work.index(clip_id)
    return {
        "rank": chunk_of[clip_id] % WORLD_SIZE,
        "shard_idx": idx // CLIPS_PER_SHARD,
        "position_in_shard": idx % CLIPS_PER_SHARD,
        "shard_key": f"{PREFIX}/{split}/shard_{chunk_of[clip_id] % WORLD_SIZE:03d}_{idx // CLIPS_PER_SHARD:05d}.tar",
    }


def main(n: int = 10, seed: int = 42) -> None:
    ood_path = hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", repo_type="dataset", filename="reasoning/ood_reasoning.parquet")
    ood_df = pd.read_parquet(ood_path)
    fp_path = hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", repo_type="dataset", filename="metadata/feature_presence.parquet")
    feature_df = pd.read_parquet(fp_path)

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    chunk_of = avdi.clip_index["chunk"].to_dict()
    splits_map = build_splits_map(feature_df, ood_df)

    session = boto3.Session(profile_name="oci.chi")
    s3 = session.client("s3", endpoint_url=S3_ENDPOINT)

    rng = random.Random(seed)
    pool = list(ood_df.index.astype(str))
    rng.shuffle(pool)

    picked = []
    skipped_no_shard = []
    for cid in pool:
        if len(picked) >= n:
            break
        row = ood_df.loc[cid]
        if pd.isna(row["events"]):
            continue  # ~9/1740 rows have no events (known upstream gap, not a bug)
        events = json.loads(row["events"]) if isinstance(row["events"], str) else row["events"]
        valid = [e for e in events if e["event_start_timestamp"] > MIN_T0_US]
        if not valid:
            continue
        shard_info = resolve_shard(cid, chunk_of, splits_map, splits_map[cid])
        if not shard_exists(s3, shard_info["shard_key"]):
            # The WDS build was stopped/relaunched many times (see memory) and
            # never finished every rank x split combo -- confirmed on rank 82's
            # val split, which has zero shards. Redraw rather than assume the
            # math is wrong.
            skipped_no_shard.append((cid, shard_info["shard_key"]))
            continue
        picked.append(
            {
                "clip_id": cid,
                "cluster": row["event_cluster"],
                "t0_us": int(valid[0]["event_start_timestamp"]),
                "coc": valid[0]["coc"],
                **shard_info,
            }
        )

    if skipped_no_shard:
        print(f"skipped {len(skipped_no_shard)} clips with no shard in S3 yet:")
        for cid, key in skipped_no_shard:
            print("  ", cid, "->", key)

    with open("ood_sample_manifest.json", "w") as f:
        json.dump(picked, f, indent=2)

    shards = sorted({p["shard_key"] for p in picked})
    print(f"picked {len(picked)} clips, {len(shards)} distinct shards needed:")
    for s in shards:
        print(" ", s)
    for p in picked:
        print(p["clip_id"], p["cluster"], "rank=", p["rank"], "shard_idx=", p["shard_idx"], "pos=", p["position_in_shard"])


if __name__ == "__main__":
    main()
