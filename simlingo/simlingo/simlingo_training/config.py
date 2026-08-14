from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union
import time

from hydra.core.config_store import ConfigStore

@dataclass
class VLMEncoderConfig:
    variant: str = 'OpenGVLab/InternVL2-1B'
    embed_dim: int = 512
    freeze: bool = False

    _target_: str = "simlingo_training.models.encoder.vlm.VLMEncoderModel"


@dataclass
class LanguageModelConfig:
    variant: str = 'OpenGVLab/InternVL2-1B'
    lora: bool = True
    lora_alpha: int = 64
    lora_r: int = 32
    lora_dropout: float = 0.1
    # trades ~20-30% slower steps for lower activation memory; needed to scale
    # dreamer_contrastive_k up without OOMing at a fixed batch_size
    gradient_checkpointing: bool = False

    _target_: str = "simlingo_training.models.language_model.llm.LLM"


@dataclass
class DrivingModelConfig:
    vision_model: Any
    language_model: Any

    lr: float = 5e-2

    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.999)
    pct_start: float = 0.05
    speed_wps_mode: str = '2d'
    predict_route_as_wps: bool = True

    # intra-scene counterfactual contrastive alignment (0.0 = disabled)
    contrastive_loss_weight: float = 0.0
    contrastive_embed_dim: int = 256
    contrastive_temperature: float = 0.07

    # trajectory->instruction grouped inverse-cycle consistency (0.0 = disabled).
    # cycle_detach=True trains only the explainer direction (trunk reads the
    # trajectory); False lets the ranking gradient reshape the trajectory itself.
    cycle_loss_weight: float = 0.0
    cycle_detach: bool = True
    cycle_temperature: float = 1.0
    # linear ramp of the cycle weight over the first N optimizer steps
    # (0 = no warmup); protects early waypoint regression
    cycle_warmup_steps: int = 0

    _target_: str = "simlingo_training.models.driving.DrivingModel"


@dataclass
class DatasetBaseConfig:
    data_path: str = "/home/katrinrenz/coding/wayve_carla/database/expertv3_2*"
    bucket_path: str = "data/buckets"

    cut_bottom_quarter: bool = False
    use_1d_wps: bool = False

    use_commentary: bool = False
    use_qa: bool = False
    qa_augmentation: bool = True
    commentary_augmentation: bool = True
    use_old_towns: bool = False
    use_only_old_towns: bool = False
    use_town13: bool = False

    skip_first_n_frames: int = 10
    pred_len: int = 11 # including the current time step
    hist_len: int = 1 # including the current time step
    hist_len_commentary: int = 5 # including the current time step
    
    img_augmentation: bool = True
    img_augmentation_prob: float = 0.5
    img_shift_augmentation: bool = True
    img_shift_augmentation_prob: float = 0.5
    
    use_safety_flag: bool = False

    # intra-scene counterfactual contrastive alignment:
    # each dreamer sample becomes a group of K counterfactuals of the same frame
    dreamer_contrastive: bool = False
    dreamer_contrastive_k: int = 4

    num_route_points: int = 20

    route_as: str = 'target_point_command' # target_point_command, target_point, command
    use_lmdrive_commands: bool = True

@dataclass
class DrivingDatasetConfig:
    # base: DatasetBaseConfig = field(default_factory=DatasetBaseConfig)
    _target_: str = "simlingo_training.dataloader.dataset_driving.Data_Driving"
    
@dataclass
class DreamerDatasetConfig:
    # base: DatasetBaseConfig = field(default_factory=DatasetBaseConfig)
    _target_: str = "simlingo_training.dataloader.dataset_dreamer.Data_Dreamer"
    
@dataclass
class QADatasetConfig:
    # base: DatasetBaseConfig = field(default_factory=DatasetBaseConfig)
    _target_: str = "simlingo_training.dataloader.dataset_eval_qa_comm.Data_Eval"
    
@dataclass
class InstEvalDatasetConfig:
    # base: DatasetBaseConfig = field(default_factory=DatasetBaseConfig)
    _target_: str = "simlingo_training.dataloader.dataset_eval_dreamer.Eval_Dreamer"

@dataclass
class DrivingDataModuleConfig:
    
    base_dataset: DatasetBaseConfig
    
    driving_dataset:Optional[ DrivingDatasetConfig] = field(default_factory=DrivingDatasetConfig)
    dreamer_dataset: Optional[DreamerDatasetConfig] = field(default_factory=DreamerDatasetConfig)
    qa_dataset: Optional[QADatasetConfig] = field(default_factory=QADatasetConfig)
    insteval_dataset: Optional[InstEvalDatasetConfig] = field(default_factory=InstEvalDatasetConfig)

    batch_size: int = 16
    num_workers: int = 10
    
    train_partitions: Optional[Dict[str, float]] = None
    train_partitions_dreamer: Optional[Dict[str, float]] = None
    use_global_img: bool = False
    
    _target_: str = "simlingo_training.dataloader.datamodule.DataModule"


@dataclass
class TrainConfig:
    model: DrivingModelConfig
    data_module: Any

    seed: int = 42
    gpus: int = 8
    # >1 when each DDP member is launched externally as its own "node"
    # (e.g. one process per Ray train worker on Lilypad)
    num_nodes: int = 1

    resume: bool = False
    resume_path: Optional[str] = None

    debug: bool = False
    overfit: int = 0
    fp16_loss_scale: float = 32.0 # 0.0 means dynamic loss scaling, only used with deepspeed

    enable_wandb: bool = True
    wandb_project: Optional[str] = "simlingo"
    if debug:
        wandb_name: Optional[str] = f"debug"
        gpus: int = 1
    else:
        # wandb_name: Optional[str] = f"debug"
        name: Optional[str] = 'test'
        wandb_name: Optional[str] = f"{time.strftime('%Y_%m_%d_%H_%M_%S')}"
    
    # max_steps: int = 100_000
    max_epochs: int = 20
    # optimizer batch = batch_size x accumulate_grad_batches x world_size items;
    # lets arms whose K fan-out forces a small forward microbatch keep the same
    # optimizer batch / LR schedule as the rest of a sweep
    accumulate_grad_batches: int = 1
    precision: str = "16-mixed"
    strategy: str = "deepspeed_stage_2" # deepspeed_stage_2 ddp
    # val_check_interval: int = 5000
    val_every_n_epochs: int = 1

    checkpoint: Optional[str] = None


def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="train_base", node=TrainConfig)
    cs.store(group="data_module", name="driving", node=DrivingDataModuleConfig)
    cs.store(group="data_module/base_dataset", name="dataset", node=DatasetBaseConfig)
    cs.store(group="model", name="driving", node=DrivingModelConfig)
    cs.store(group="model/vision_model", name="vlm", node=VLMEncoderConfig)
    cs.store(group="model/language_model", name="llm", node=LanguageModelConfig)


register_configs()
