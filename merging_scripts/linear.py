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


def linear_merge(
    finetuned_sds: List[Dict[str, torch.Tensor]],
    weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Weighted average of model parameters:
        θ_merged = Σ w_t * θ_t   where Σ w_t = 1
    
    If no weights are provided, uniform averaging is used.
    """
    n = len(finetuned_sds)

    # Uniform weights if none provided
    if weights is None:
        weights = [1.0 / n] * n
        logger.info(f"No weights provided — using uniform weights: {weights}")
    else:
        if len(weights) != n:
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of models ({n})."
            )
        total = sum(weights)
        weights = [w / total for w in weights]  # normalise to sum to 1
        logger.info(f"Normalised weights: {[round(w, 4) for w in weights]}")

    ref_sd = finetuned_sds[0]
    merged_sd = {}

    for key in ref_sd.keys():
        ref_tensor = ref_sd[key]

        if not ref_tensor.dtype.is_floating_point:
            merged_sd[key] = ref_tensor.clone()
            continue

        compatible = []
        compatible_weights = []
        for sd, w in zip(finetuned_sds, weights):
            if (key in sd and sd[key].shape == ref_tensor.shape and sd[key].dtype.is_floating_point):
                compatible.append(sd[key].float())
                compatible_weights.append(w)
        if not compatible:
            logger.warning(f"No compatible tensors for key '{key}', skipping.")
            merged_sd[key] = ref_tensor.clone()
            continue
        if len(compatible) < n:
            logger.warning(
                f"Only {len(compatible)}/{n} models have compatible "
                f"tensor for '{key}'. Re-normalising weights."
            )
            total = sum(compatible_weights)
            compatible_weights = [w / total for w in compatible_weights]

        stacked = torch.stack(compatible, dim=0)              
        w_tensor = torch.tensor(compatible_weights, dtype=torch.float32).view(-1, *([1] * ref_tensor.dim()))           
        merged_sd[key] = (stacked * w_tensor).sum(dim=0).to(ref_tensor.dtype)
    
    return merged_sd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/ashley-bravo/EXPERIMENTATION/RANDOM_CONTROL/4th-Control/Linear/Hispanic-Male",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Per-model weights (space-separated). "
            "Will be normalised to sum to 1. "
            "Defaults to uniform weighting."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = [Path(p) for p in CHECKPOINT_PATHS]
    logger.info("Checkpoints to merge:")
    for p in checkpoint_paths:
        logger.info(f"  - {p}")

    finetuned_sds = []
    for ckpt in checkpoint_paths:
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        finetuned_sds.append(load_state_dict(ckpt))

    # Sanity-check keys
    keys_0 = set(finetuned_sds[0].keys())
    for i, sd in enumerate(finetuned_sds[1:], 1):
        diff = keys_0.symmetric_difference(sd.keys())
        if diff:
            logger.warning(
                f"Model {i} has different keys: {list(diff)[:10]}"
            )

    logger.info(
        f"LINEAR MERGE | models={len(finetuned_sds)} | "
        f"weights={args.weights or 'uniform'}"
    )

    merged_sd = linear_merge(finetuned_sds, weights=args.weights)

    out_path = output_dir / "model.pt"
    torch.save(merged_sd, out_path)
    logger.info(f"Model saved to: {out_path}")

    n_params = sum(
        v.numel() for v in merged_sd.values() if isinstance(v, torch.Tensor)
    )
    logger.info(f"Total parameters: {n_params:,}")


if __name__ == "__main__":
    main()