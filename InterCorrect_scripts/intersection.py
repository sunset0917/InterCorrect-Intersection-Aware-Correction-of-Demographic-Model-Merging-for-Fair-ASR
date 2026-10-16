import argparse
import logging
from pathlib import Path
from typing import Dict
import torch

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


STEP_4_MODEL = "/home/ashley-bravo/EXPERIMENTATION/RANDOM_CONTROL/4th-Control/DARE-TIES/Hispanic-Male/FT/asr_epoch_2_step_79/model.pt" # FT MODEL ON BOTH DATA ON MERGED MODEL
STEP_3_MODEL = "/home/ashley-bravo/EXPERIMENTATION/RANDOM_CONTROL/4th-Control/DARE-TIES/Hispanic-Male/model.pt" # MERGED MODEL
DEFAULT_OUTPUT_DIR = "/home/ashley-bravo/EXPERIMENTATION/RANDOM_CONTROL/4th-Control/DARE-TIES/Hispanic-Male/Intersection"

def load_state_dict(path):

    logger.info(f"Loading checkpoint: {path}")
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict):

        for key in ("model", "model_state_dict", "state_dict"):
            if key in raw and isinstance(raw[key], dict):
                logger.info(f"  → using subkey '{key}'")
                return raw[key]

        if all(isinstance(v, torch.Tensor) for v in raw.values()):
            logger.info("  → checkpoint itself is a state_dict")
            return raw

    raise ValueError(f"Could not extract state_dict from checkpoint: {path}")


def compute_difference(step4_sd,step3_sd):
    if set(step4_sd.keys()) != set(step3_sd.keys()):
        only_step4 = set(step4_sd.keys()) - set(step3_sd.keys())
        only_step3 = set(step3_sd.keys()) - set(step4_sd.keys())
    difference_sd = {}
    num_subtracted = 0
    num_copied = 0
    for key, step4_tensor in step4_sd.items():
        if key not in step3_sd:
            logger.warning(
                f"Key '{key}' not found in Step 3 model. "
                f"Copying Step 4 tensor."
            )
            difference_sd[key] = step4_tensor.clone()
            num_copied += 1
            continue
        step3_tensor = step3_sd[key]
        difference_sd[key] = (step4_tensor.to(torch.float64) - step3_tensor.to(torch.float64))
        num_subtracted += 1

    return difference_sd
def compute_statistics(difference_sd: Dict[str, torch.Tensor]):
    total_elements = 0
    total_abs = 0.0
    max_abs = 0.0
    nonzero_elements = 0

    for tensor in difference_sd.values():
        if not isinstance(tensor, torch.Tensor):
            continue
        if not tensor.dtype.is_floating_point:
            continue
        t = tensor.to(torch.float64)
        total_elements += t.numel()
        total_abs += t.abs().sum().item()
        max_abs = max(max_abs, t.abs().max().item())
        nonzero_elements += torch.count_nonzero(t).item()

    if total_elements > 0:
        mean_abs = total_abs / total_elements
        sparsity = 1.0 - ( nonzero_elements / total_elements)

        logger.info("Correction statistics:")
        logger.info(
            f"  Total elements : {total_elements:,}"
        )
        logger.info(
            f"  Mean |Δ|       : {mean_abs:.8e}"
        )
        logger.info(
            f"  Max  |Δ|       : {max_abs:.8e}"
        )
        logger.info(
            f"  Non-zero       : {nonzero_elements:,}"
        )
        logger.info(
            f"  Sparsity       : {sparsity:.6f}"
        )

def parse_args():
    parser = argparse.ArgumentParser(description=("Compute intersection correction as "
            "Step 4 Model - Step 3 Model."))
    parser.add_argument("--step4",type=str,default=STEP_4_MODEL,help="Path to Step 4 model.",)
    parser.add_argument("--step3",type=str,default=STEP_3_MODEL,help="Path to Step 3 model.",)
    parser.add_argument("--output_dir",type=str,default=DEFAULT_OUTPUT_DIR,help="Directory where the correction is saved.",)
    return parser.parse_args()

def main():

    args = parse_args()
    step4_path = Path(args.step4)
    step3_path = Path(args.step3)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True,exist_ok=True,)
    if not step4_path.exists():
        raise FileNotFoundError(f"Step 4 checkpoint not found:\n{step4_path}")

    if not step3_path.exists():
        raise FileNotFoundError(f"Step 3 checkpoint not found:\n{step3_path}")
    logger.info("INTERSECTION CORRECTION")
    logger.info(f"Step 4 model: {step4_path}")
    logger.info(f"Step 3 model: {step3_path}")
    step4_sd = load_state_dict(step4_path)
    step3_sd = load_state_dict(step3_path)
    logger.info(f"Step 4 tensors: {len(step4_sd):,}")
    logger.info(f"Step 3 tensors: {len(step3_sd):,}")
    correction_sd = compute_difference(step4_sd=step4_sd,step3_sd=step3_sd,)
    compute_statistics(correction_sd)
    output_path = output_dir / "model.pt"
    torch.save(correction_sd,output_path,)
    logger.info(f"Intersection correction saved to:\n{output_path}")

if __name__ == "__main__":
    main()