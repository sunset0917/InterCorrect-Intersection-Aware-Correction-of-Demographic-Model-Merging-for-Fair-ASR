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

    logger.info(f"Cargando checkpoint: {path}")
    raw = torch.load(path,map_location="cpu")
    if isinstance(raw, dict):
        for key in ("model","model_state_dict","state_dict",):
            if key in raw and isinstance(raw[key], dict):
                logger.info(f"  → usando subclave '{key}'")
                return raw[key]
        if all(isinstance(v, torch.Tensor) for v in raw.values()):
            return raw

    raise ValueError(f"Unable to extract state_dict from: {path}")
def compute_task_vectors(finetuned_sds: List[Dict[str, torch.Tensor]],base_sd: Dict[str, torch.Tensor],) -> List[Dict[str, torch.Tensor]]:
    """
    τ_t = θ_t - θ_init
    """
    task_vectors = []
    for sd in finetuned_sds:
        tv = {}
        for k, v in sd.items():
            if (k in base_sd and v.dtype.is_floating_point and base_sd[k].dtype.is_floating_point and v.shape == base_sd[k].shape):
                delta = v.float() - base_sd[k].float()
                
                if delta.abs().mean() > 1e-8:
                    tv[k] = delta

        task_vectors.append(tv)

    return task_vectors


def trim(tv, k):
    trimmed = {}
    for key, tensor in tv.items():
        flat = tensor.abs().flatten()
        if flat.numel() == 0:
            trimmed[key] = tensor.clone()
            continue
        n_keep = max(1, int(flat.numel() * k / 100.0))

        threshold = torch.kthvalue(flat,flat.numel() - n_keep + 1).values
        mask = tensor.abs() >= threshold
        trimmed[key] = tensor * mask
    return trimmed


def elect_sign(trimmed_tvs):
    keys = trimmed_tvs[0].keys()
    gamma = {}
    for key in keys:
        total = sum(tv[key] for tv in trimmed_tvs)
        gamma[key] = torch.sign(total)
    return gamma

def disjoint_merge(trimmed_tvs, gamma):
    merged = {}
    for key in gamma.keys():
        g = gamma[key]
        stack = torch.stack([tv[key] for tv in trimmed_tvs],dim=0)
        sign_match = ((torch.sign(stack) == g.unsqueeze(0)) & (stack != 0))
        aligned_sum = (stack * sign_match.float()).sum(dim=0)
        count = (sign_match.float().sum(dim=0).clamp(min=1))
        merged[key] = aligned_sum / count

    return merged


def ties_merge(finetuned_sds,base_sd,k=20.0,lam=1.0,):
    task_vectors = compute_task_vectors(finetuned_sds,base_sd)
    logger.info(f"  {len(task_vectors)} task vectors created")
    logger.info(f"1. Trim (k={k}%)")
    trimmed_tvs = [trim(tv, k) for tv in task_vectors]
    logger.info("2. Elect Sign")
    gamma = elect_sign(trimmed_tvs)
    logger.info("3. Disjoint Merge")
    merged_tv = disjoint_merge(trimmed_tvs,gamma)
    logger.info(f"4. Reconstruct model (λ={lam})")
    merged_sd = {}

    for key, base_val in base_sd.items():
        if key in merged_tv:
            delta = merged_tv[key].to(base_val.dtype)
            merged_sd[key] = (base_val + lam * delta)
        else:
            merged_sd[key] = base_val.clone()
    return merged_sd

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",type=str,default=BASE_MODEL_PATH,help="Path to the pre-trained/base model checkpoint.",)
    parser.add_argument("--output_dir",type=str,default="/home/ashley-bravo/EXPERIMENTATION/RANDOM_CONTROL/4th-Control/TIES/Hispanic-Male")
    parser.add_argument("--k",type=float,default=20.0)
    parser.add_argument("--lam",type=float,default=1.0)
    return parser.parse_args()


def main():

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    base_path = Path(args.base_model)

    if not base_path.exists():
        raise FileNotFoundError(
            f"Base model checkpoint not found: {base_path}"
        )

    logger.info(f"Loading base model: {base_path}")
    base_sd = load_state_dict(base_path)
    checkpoint_paths = [Path(p) for p in CHECKPOINT_PATHS]

    logger.info("Fine-tuned models:")

    for p in checkpoint_paths:
        logger.info(f"  - {p}")

    finetuned_sds = []

    for ckpt in checkpoint_paths:

        if not ckpt.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt}"
            )

        finetuned_sds.append(
            load_state_dict(ckpt)
        )

    keys_0 = set(finetuned_sds[0].keys())

    for i, sd in enumerate(finetuned_sds[1:], start=1):

        diff = keys_0.symmetric_difference(sd.keys())

        if diff:
            logger.warning(
                f"Model {i} has different keys: "
                f"{list(diff)[:10]}"
            )

    logger.info(
        "Checking compatibility with base model..."
    )

    for i, sd in enumerate(finetuned_sds):

        for key in base_sd.keys():

            if key not in sd:
                raise ValueError(
                    f"Key '{key}' missing in model {i}"
                )

            if (
                isinstance(base_sd[key], torch.Tensor)
                and isinstance(sd[key], torch.Tensor)
                and base_sd[key].shape != sd[key].shape
            ):
                raise ValueError(
                    f"Shape mismatch for key '{key}' "
                    f"between base model and model {i}"
                )


    logger.info(
        f"TIES MERGING | "
        f"models={len(finetuned_sds)} | "
        f"k={args.k} | "
        f"λ={args.lam}"
    )

    merged_sd = ties_merge(
        finetuned_sds=finetuned_sds,
        base_sd=base_sd,
        k=args.k,
        lam=args.lam,
    )


    out_path = output_dir / "model.pt"

    torch.save(
        merged_sd,
        out_path
    )

    logger.info(
        f"\nModel saved to:\n{out_path}"
    )

    n_params = sum(
        v.numel()
        for v in merged_sd.values()
        if isinstance(v, torch.Tensor)
    )

    logger.info(
        f"Total parameters: {n_params:,}"
    )

if __name__ == "__main__":
    main()