import argparse
import logging
from pathlib import Path
from typing import Dict, List
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
                return raw[key]
        if all(isinstance(v, torch.Tensor) for v in raw.values()):
            return raw
    raise ValueError(f"Could not extract state dict from {path}")

def compute_breadcrumbs_mask(task_vector: torch.Tensor, beta: float, gamma: float) -> torch.Tensor:
    abs_vector = torch.abs(task_vector.float())
    flat_abs = abs_vector.view(-1)
    lower_threshold = torch.quantile(flat_abs, beta)
    upper_threshold = torch.quantile(flat_abs, gamma)
    mask = (abs_vector > lower_threshold) & (abs_vector < upper_threshold)
    return mask.to(task_vector.dtype)

def breadcrumbs_merge(base_sd,finetuned_sds,beta,gamma,alpha):
    merged_sd = {}

    for key in base_sd.keys():
        base_tensor = base_sd[key]
        if not base_tensor.dtype.is_floating_point:
            merged_sd[key] = base_tensor.clone()
            continue
        total_trajectory = torch.zeros_like(base_tensor, dtype=torch.float32)
        for sd in finetuned_sds:
            if key in sd and sd[key].shape == base_tensor.shape:
                task_vector = sd[key].float() - base_tensor.float()
                mask = compute_breadcrumbs_mask(task_vector, beta, gamma)
                total_trajectory += mask.float() * task_vector
        merged_tensor = base_tensor.float() +  total_trajectory * alpha
        merged_sd[key] = merged_tensor.to(base_tensor.dtype)

    return merged_sd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default=BASE_MODEL_PATH)
    parser.add_argument("--output_dir", type=str, default="/home/ashley-bravo/EXPERIMENTATION/RANDOM_CONTROL/4th-Control/Breadcrumbs/Hispanic-Male")
    parser.add_argument("--beta", type=float, default=0.90)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=1/17)
    return parser.parse_args()

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_sd = load_state_dict(Path(args.base_model))
    finetuned_sds = [load_state_dict(Path(p)) for p in CHECKPOINT_PATHS if Path(p).exists()]
    logger.info(f"Fusing {len(finetuned_sds)} models via BREADCRUMBS | β={args.beta}, γ={args.gamma}")
    merged_sd = breadcrumbs_merge(base_sd=base_sd,finetuned_sds=finetuned_sds,beta=args.beta,gamma=args.gamma,alpha=args.alpha)

    out_path = output_dir / f"model.pt"
    torch.save(merged_sd, out_path)
    logger.info(f"Merged model saved to: {out_path}")

if __name__ == "__main__":
    main()