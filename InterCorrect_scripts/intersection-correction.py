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
from fractions import Fraction

def parse_coef(value):
    try:
        return float(Fraction(value))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid coefficient: {value}"
        )

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
    raise ValueError(f"Could not extract state dict from {path}")
def apply_intersection_corrections(initial_sd,correction_sds,corr_coefs):

    if len(correction_sds) != len(corr_coefs):
        raise ValueError(
            f"Number of corrections ({len(correction_sds)}) "
            f"must match number of coefficients ({len(corr_coefs)})."
        )

    corrected_sd = {}

    for key, initial_tensor in initial_sd.items():

        # Non-floating-point tensors cannot be safely interpolated
        if not initial_tensor.dtype.is_floating_point:
            corrected_sd[key] = initial_tensor.clone()
            continue
        result = initial_tensor.to(torch.float64)
        for i, (correction_sd, coef) in enumerate(zip(correction_sds, corr_coefs), start=1):
            if key not in correction_sd:
                logger.warning(
                    f"Key '{key}' missing from correction {i}. "
                    f"Skipping correction {i} for this tensor."
                )
                continue

            correction_tensor = correction_sd[key]
            if correction_tensor.shape != initial_tensor.shape:
                logger.warning(
                    f"Shape mismatch for key '{key}' in correction {i}. "
                    f"Skipping correction {i}."
                )
                continue

            correction_float = correction_tensor.to(torch.float64)
            result = result + coef * correction_float
        corrected_sd[key] = result.to(initial_tensor.dtype)
    return corrected_sd

'''
python /home/ashley-bravo/EXPERIMENTATION/intersection-correction2.py \
    --initial_model \
    /home/ashley-bravo/EXPERIMENTATION/RQ2/TIES/model.pt\
    --correction_models \
    /home/ashley-bravo/EXPERIMENTATION/random_control/Linear/18-22-L1/Intersection/model.pt\
    /home/ashley-bravo/EXPERIMENTATION/random_control/Linear/18-22-Low/Intersection/model.pt\
    --corr_coefs\
    0.9 0.1\
    --output_dir\
    /home/ashley-bravo/EXPERIMENTATION/random_control/Linear/ICM/K2_M9_M1

'''
def parse_args():
    parser = argparse.ArgumentParser(description=(
            "Apply multiple intersection corrections jointly "
            "to a single initial merged model."))
    parser.add_argument("--initial_model",type=str,required=True)
    parser.add_argument("--correction_models",type=str,nargs='+',required=True)
    parser.add_argument("--corr_coefs",type=parse_coef,nargs='+',required=True)
    parser.add_argument("--output_dir",type=str,required=True)

    return parser.parse_args()
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_path = Path(args.initial_model)
    correction_paths = [Path(path) for path in args.correction_models]

    logger.info(f"Initial model: {initial_path}")

    for i, (path, coef) in enumerate(
        zip(correction_paths, args.corr_coefs),
        start=1,
    ):
        logger.info(f"Correction {i}: {path}")
        logger.info(f"Coefficient {i}: {coef}")
    initial_sd = load_state_dict(initial_path)
    correction_sds = []
    for i, correction_path in enumerate(correction_paths, start=1):
        if not correction_path.exists():
            raise FileNotFoundError(f"Correction {i} not found:\n{correction_path}")
        correction_sd = load_state_dict(correction_path)
        logger.info(f"Correction {i} tensors: {len(correction_sd):,}")
        correction_sds.append(correction_sd)

    corrected_sd = apply_intersection_corrections(initial_sd=initial_sd,correction_sds=correction_sds,corr_coefs=args.corr_coefs,)
    output_path = output_dir / "model.pt"
    torch.save(corrected_sd, output_path)
    logger.info(f"Final model saved to: {output_path}")
    n_params = sum(v.numel() for v in corrected_sd.values() if isinstance(v, torch.Tensor))
    logger.info(f"Total parameters: {n_params:,}")


if __name__ == "__main__":
    main()