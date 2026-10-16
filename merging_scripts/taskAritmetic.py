import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import torch

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

BASE_MODEL_PATH = "/home/ashley-bravo/outputs/slam_asr/100h_train_v4/asr_epoch_5_step_4680/model.pt"

CHECKPOINT_PATHS = [
"/home/ashley-bravo/outputs/slam_asr/Finetuning_FS/male/asr_epoch_1_step_200/model.pt",
"/home/ashley-bravo/outputs/slam_asr/Finetuning_FS/Latino/asr_epoch_1_step_200/model.pt",
]

def load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    logger.info(f"Loading checkpoint: {path}")
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if key in raw and isinstance(raw[key], dict):
                logger.info(f"  → using subkey '{key}'")
                return raw[key]
        if all(isinstance(v, torch.Tensor) for v in raw.values()):
            return raw
    raise ValueError(f"Could not extract state dict from {path}")


def task_arithmetic_merge(base_sd: Dict[str, torch.Tensor],finetuned_sds: List[Dict[str, torch.Tensor]],scaling_factor: float = 0.3,) -> Dict[str, torch.Tensor]:
    """
    Computes Task Arithmetic Merge (Addition):
        τ_t = θ_ft^t - θ_pre
        θ_merged = θ_pre + λ * Σ(τ_t)
    """
    merged_sd = {}
    num_models = len(finetuned_sds)
    for key in base_sd.keys():
        base_tensor = base_sd[key]
        if not base_tensor.dtype.is_floating_point:
            merged_sd[key] = base_tensor.clone()
            continue
        total_task_vector = torch.zeros_like(base_tensor, dtype=torch.float32)
        compatible_count = 0
        for i, sd in enumerate(finetuned_sds):
            if (key in sd and sd[key].shape == base_tensor.shape and sd[key].dtype.is_floating_point):
                # τ_t = θ_ft - θ_pre
                task_vector = sd[key].float() - base_tensor.float()
                total_task_vector += task_vector
                compatible_count += 1
            else:
                logger.warning(f"Model {i} misses or has incompatible tensor for key '{key}'")
        if compatible_count == 0:
            logger.warning(f"No compatible fine-tuned tensors for key '{key}'. Using base tensor.")
            merged_sd[key] = base_tensor.clone()
            continue
        if compatible_count < num_models:
            logger.warning(f"Only {compatible_count}/{num_models} models contributed to key '{key}'")
        # θ_merged = θ_pre + λ * Σ(τ_t)
        merged_tensor = base_tensor.float() + (scaling_factor * total_task_vector)
        merged_sd[key] = merged_tensor.to(base_tensor.dtype)

    return merged_sd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",type=str,default=BASE_MODEL_PATH,help="Path to the pre-trained/base model checkpoint.",)
    parser.add_argument("--output_dir",type=str,default="/home/ashley-bravo/EXPERIMENTATION/RANDOM_CONTROL/4th-Control/TaskArithmetic/Hispanic-Male")
    parser.add_argument("--scaling_factor",type=float,default=1/2,help="The scaling factor (lambda) for the combined task vectors. Usually between 0.1 and 0.7.",)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_path = Path(args.base_model)
    if not base_path.exists():
        raise FileNotFoundError(f"Base model checkpoint not found: {base_path}")
    
    base_sd = load_state_dict(base_path)

    checkpoint_paths = [Path(p) for p in CHECKPOINT_PATHS]
    logger.info("Fine-tuned checkpoints to extract task vectors from:")
    for p in checkpoint_paths:
        logger.info(f"  - {p}")

    finetuned_sds = []
    for ckpt in checkpoint_paths:
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        finetuned_sds.append(load_state_dict(ckpt))

    # Verificar llaves con el modelo base
    base_keys = set(base_sd.keys())
    for i, sd in enumerate(finetuned_sds):
        diff = base_keys.symmetric_difference(sd.keys())
        if diff:
            logger.warning(
                f"Model {i} has differences in keys compared to base: {list(diff)[:5]}"
            )

    logger.info(
        f"TASK ARITHMETIC MERGE (ADDITION) | models={len(finetuned_sds)} | "
        f"scaling_factor (λ)={args.scaling_factor}"
    )

    merged_sd = task_arithmetic_merge(
        base_sd=base_sd, 
        finetuned_sds=finetuned_sds, 
        scaling_factor=args.scaling_factor
    )

    out_path = output_dir / f"model.pt"
    torch.save(merged_sd, out_path)
    logger.info(f"Model saved to: {out_path}")

    n_params = sum(
        v.numel() for v in merged_sd.values() if isinstance(v, torch.Tensor)
    )
    logger.info(f"Total parameters: {n_params:,}")


if __name__ == "__main__":
    main()