"""Open-loop faithfulness inference worker: one checkpoint, one GPU.

Runs inside simlingo/simlingo (PYTHONPATH + cwd set by eval_faith.py), mirrors
upstream simlingo_training/eval.py's commentary branch, but with a seeded
fixed-subset predict loop and JSONL output instead of the upstream save path.
All arms share the SAME base hydra config (the HF original's .hydra/config.yaml)
and the SAME subset indices, so the eval is paired across arms; the contrastive
arms' extra projection heads (traj_proj/text_proj/traj_state_proj/traj_encoder/query_state_proj) are inference-dead and are
the only state-dict keys allowed to mismatch.
"""
import argparse
import json
import random
from pathlib import Path

import hydra
import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from omegaconf import OmegaConf
from pytorch_lightning.utilities import move_data_to_device
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor, AutoTokenizer

from simlingo_training.models.driving import decode_uint8

# heads created only when contrastive_loss_weight > 0 at train time; harmless
# and unused in predict, so they may be absent from the eval-time model
ALLOWED_MISSING_PREFIXES = ("traj_proj.", "text_proj.", "traj_state_proj.", "traj_encoder.", "query_state_proj.")


def load_state(ckpt_path: Path) -> dict:
    if ckpt_path.is_dir():
        return get_fp32_state_dict_from_zero_checkpoint(str(ckpt_path))
    return torch.load(str(ckpt_path), map_location="cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="base .hydra/config.yaml (shared across arms)")
    ap.add_argument("--checkpoint", required=True, help="ZeRO ckpt dir or consolidated .pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-clips", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--mode", choices=["commentary", "dreamer"], default="commentary")
    args = ap.parse_args()

    torch.set_float32_matmul_precision("high")

    # dataset_base/driving call hydra's get_original_cwd(), which upstream
    # eval.py satisfies via @hydra.main; we run bare, so initialize a minimal
    # HydraConfig whose runtime.cwd is our cwd (= simlingo repo root)
    from hydra.conf import HydraConf  # noqa: PLC0415
    from hydra.core.hydra_config import HydraConfig  # noqa: PLC0415
    from omegaconf import open_dict  # noqa: PLC0415
    hydra_cfg = OmegaConf.create({"hydra": OmegaConf.structured(HydraConf)})
    with open_dict(hydra_cfg.hydra):
        hydra_cfg.hydra.runtime.cwd = str(Path.cwd())
        hydra_cfg.hydra.job.name = "eval_faith_worker"
    HydraConfig.instance().set_config(hydra_cfg)

    # dataset_base shuffles route_dirs with the GLOBAL random module before
    # scanning, and Eval_Dreamer.__getitem__ draws the counterfactual option,
    # instruction phrasing, and prompt template with the global RNG too;
    # identical seeds across arms => identical scan order AND identical
    # per-sample instruction draws (paired eval)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = OmegaConf.load(args.config)

    cfg.data_module.dreamer_dataset = None
    cfg.data_module.driving_dataset = None
    if args.mode == "dreamer":
        # instruction-following eval: Eval_Dreamer feeds counterfactual dreamer
        # instructions; datamodule's predict branch picks insteval_dataset when
        # qa_dataset is None
        cfg.data_module.qa_dataset = None
        # the HF original's 2025 config predates this fork's file split; the
        # class now lives in dataset_eval_dreamer (see config.py:140)
        cfg.data_module.insteval_dataset._target_ = (
            "simlingo_training.dataloader.dataset_eval_dreamer.Eval_Dreamer"
        )
        # no <SAFETY>/<INSTRUCTION_FOLLOWING> prefix tokens: the k4 arms were
        # trained with use_safety_flag=false and have never seen them; with the
        # flag off the GT is always the instructed trajectory
        cfg.data_module.base_dataset.use_safety_flag = False
        cfg.data_module.base_dataset.use_commentary = False
        cfg.data_module.base_dataset.use_qa = False
    else:
        # commentary predict mode, exactly as upstream eval.py's branch
        cfg.data_module.insteval_dataset = None
        cfg.data_module.base_dataset.use_commentary = True
        cfg.data_module.base_dataset.use_qa = False
    # keep all scanned routes: the eval mirror holds only the held-out split
    # (and in commentary mode the evalset intersection restricts frames)
    cfg.data_module.base_dataset.eval_use_all_routes = True
    cfg.data_module.base_dataset.img_augmentation = False
    cfg.data_module.base_dataset.img_shift_augmentation = False
    # data_path stays database/simlingo_v2_2025_01_10 (the HF config value):
    # BaseDataset string-matches scanned measurement paths against
    # data/evalset_commentary.json entries, which hardcode that prefix --
    # eval_faith.py extracts the mirror under exactly that directory name
    cfg.data_module.batch_size = args.batch_size
    # dreamer mode MUST load single-process: worker processes each reseed the
    # global RNG, which would decouple the per-sample instruction draws from
    # the main-process seed and break cross-arm pairing
    num_workers = 0 if args.mode == "dreamer" else 8
    cfg.data_module.num_workers = num_workers
    cfg.gpus = 1

    if "2B" in cfg.model.language_model.variant:
        processor = AutoTokenizer.from_pretrained(cfg.model.language_model.variant, trust_remote_code=True, use_fast=False)
    else:
        processor = AutoProcessor.from_pretrained(cfg.model.language_model.variant, trust_remote_code=True, use_fast=False)

    data_module = hydra.utils.instantiate(
        cfg.data_module,
        processor=processor,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant=cfg.model.language_model.variant,
        predict=True,
        _recursive_=False,
    )
    data_module.setup("predict")
    dataset = data_module.predict_dataset
    n = len(dataset)
    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(n), min(args.max_clips, n)))
    print(f"[worker] predict dataset {n} samples -> subset {len(indices)} (seed {args.seed})", flush=True)

    model_type_name = cfg.model.vision_model.variant.split("/")[1]
    model = hydra.utils.instantiate(
        cfg.model,
        cfg_data_module=cfg.data_module,
        processor=processor,
        cache_dir=f"pretrained/{model_type_name}",
        _recursive_=False,
    )
    state = load_state(Path(args.checkpoint))
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in list(missing) + list(unexpected) if not k.startswith(ALLOWED_MISSING_PREFIXES)]
    assert not bad, f"state-dict mismatch beyond contrastive heads: {bad[:10]}"
    print(f"[worker] loaded {args.checkpoint}; skipped keys: {len(missing) + len(unexpected)}", flush=True)

    model = model.cuda().eval()
    # model init consumed RNG (weight init of arm-specific heads differs);
    # re-pin the python RNG so dreamer-mode __getitem__ draws are identical
    # across arms from here on
    random.seed(args.seed + 1)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=data_module.dl_collate_fn,
        pin_memory=True,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w") as f, torch.no_grad():
        for bi, batch in enumerate(loader):
            batch = move_data_to_device(batch, torch.device("cuda"))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                speed_wps, route, language, speed_wps_gt, route_gt, language_gt = model.predict_step(batch, bi)
            prompts = batch.driving_input.prompt.language_string
            paths = decode_uint8(batch.run_id)
            infos = batch.driving_label.eval_infos
            for j in range(len(language)):
                row = {
                    "path": paths[j],
                    "prompt": prompts[j],
                    "language_pred": language[j],
                    "language_gt": language_gt[j],
                    "waypoints_pred": speed_wps[j].float().cpu().tolist(),
                    "route_pred": route[j].float().cpu().tolist(),
                    "waypoints_gt": speed_wps_gt[j].float().cpu().tolist(),
                    "route_gt": route_gt[j].float().cpu().tolist(),
                }
                if infos is not None and infos[j] is not None:
                    # dreamer mode: instructed (new) vs default (org) trajectories
                    row["eval_infos"] = {
                        k: (v.tolist() if hasattr(v, "tolist") else v)
                        for k, v in infos[j].items()
                    }
                f.write(json.dumps(row) + "\n")
                written += 1
            if bi % 10 == 0:
                print(f"[worker] batch {bi}, rows {written}", flush=True)
            # predict_step accumulates into model.prediction; drop it so a
            # 500-clip run doesn't hold every batch's tensors on GPU
            model.prediction = {}
    print(f"[worker] wrote {written} rows to {out_path}", flush=True)


if __name__ == "__main__":
    main()
