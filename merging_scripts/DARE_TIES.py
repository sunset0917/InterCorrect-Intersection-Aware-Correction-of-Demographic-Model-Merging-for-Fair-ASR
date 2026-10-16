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

    raw = torch.load(
        path,
        map_location="cpu"
    )

    if isinstance(raw, dict):

        for key in (
            "model",
            "model_state_dict",
            "state_dict",
        ):
            if key in raw and isinstance(raw[key], dict):
                logger.info(f"  -> using subkey '{key}'")
                return raw[key]

        if all(
            isinstance(v, torch.Tensor)
            for v in raw.values()
        ):
            return raw

    raise ValueError(
        f"Unable to extract state_dict from {path}"
    )

def apply_dare(task_vector, drop_rate,):
    """
    DARE:
        Drop p% of parameters randomly
        Rescale surviving parameters by 1/(1-p)
    """

    if drop_rate <= 0.0:
        return task_vector
    survival_rate = 1.0 - drop_rate
    mask = (torch.rand_like(task_vector,dtype=torch.float32,) < survival_rate)
    scale = 1.0 / survival_rate

    return (
        task_vector.float()
        * mask.float()
        * scale
    ).to(task_vector.dtype)

def compute_dare_task_vectors(finetuned_sds,base_sd,drop_rate):
    logger.info(f"Computing task vectors + DARE (drop_rate={drop_rate})")
    task_vectors = []
    for model_idx, sd in enumerate(finetuned_sds):
        logger.info(
            f"Processing model "
            f"{model_idx + 1}/{len(finetuned_sds)}"
        )

        tv = {}
        for key, ft_tensor in sd.items():
            if key not in base_sd:
                continue
            base_tensor = base_sd[key]
            if (not ft_tensor.dtype.is_floating_point  or not base_tensor.dtype.is_floating_point):
                continue
            if ft_tensor.shape != base_tensor.shape:
                continue
            delta = (ft_tensor.float() - base_tensor.float())
            if delta.abs().mean() < 1e-8:
                continue
            delta = apply_dare(delta,drop_rate)
            tv[key] = delta
        task_vectors.append(tv)
    return task_vectors

def trim(task_vector: Dict[str, torch.Tensor],k: float):
    trimmed = {}
    for key, tensor in task_vector.items():
        flat = tensor.abs().flatten()
        if flat.numel() == 0:
            trimmed[key] = tensor.clone()
            continue
        n_keep = max(1,int(flat.numel() * k / 100.0))
        threshold = torch.kthvalue(flat,flat.numel() - n_keep + 1).values
        mask = tensor.abs() >= threshold
        trimmed[key] = tensor * mask

    return trimmed

def elect_sign(trimmed_tvs: List[Dict[str, torch.Tensor]]):
    gamma = {}
    keys = trimmed_tvs[0].keys()
    for key in keys:
        total = sum(tv[key] for tv in trimmed_tvs)
        gamma[key] = torch.where(total >= 0,torch.ones_like(total),-torch.ones_like(total))

    return gamma

def disjoint_merge(trimmed_tvs,gamma):

    merged = {}
    for key in gamma.keys():
        sign = gamma[key]
        stack = torch.stack([tv[key] for tv in trimmed_tvs],dim=0)
        sign_match = ((torch.sign(stack) == sign.unsqueeze(0))& (stack != 0))
        aligned_sum = (stack* sign_match.float()).sum(dim=0)
        count = (sign_match.float().sum(dim=0).clamp(min=1))
        merged[key] = aligned_sum / count

    return merged


def dare_ties_merge(finetuned_sds,base_sd,drop_rate=0.8,k=20.0,lam=1.0,):

    logger.info("DARE + TIES MERGING")
    task_vectors = compute_dare_task_vectors(finetuned_sds=finetuned_sds,base_sd=base_sd,drop_rate=drop_rate,)
    logger.info("Step 1: TRIM")
    trimmed_tvs = [trim(tv, k) for tv in task_vectors]
    logger.info("Step 2: ELECT SIGN")
    gamma = elect_sign(trimmed_tvs)
    logger.info("Step 3: DISJOINT MERGE")
    merged_tv = disjoint_merge(trimmed_tvs,gamma)
    logger.info("Step 4: RECONSTRUCT")
    merged_sd = {}

    for key, base_tensor in base_sd.items():
        if (key in merged_tv and base_tensor.dtype.is_floating_point):
            merged_sd[key] = (base_tensor.float()+ lam * merged_tv[key]).to(base_tensor.dtype)
        else:
            merged_sd[key] = base_tensor.clone()
    return merged_sd

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",type=str,default=BASE_MODEL_PATH,)
    parser.add_argument("--output_dir",type=str,default="/home/ashley-bravo/EXPERIMENTATION/RANDOM_CONTROL/4th-Control/DARE-TIES/Hispanic-Male")
    parser.add_argument("--drop_rate",type=float,default=0.2,help="DARE drop rate")
    parser.add_argument("--k",type=float,default=20.0,help="Top-k percentage retained by TIES",)
    parser.add_argument("--lam",type=float,default=1.0)
    return parser.parse_args()
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    logger.info(f"Loading base model: " f"{args.base_model}")
    base_sd = load_state_dict(Path(args.base_model))
    finetuned_sds = []
    for ckpt in CHECKPOINT_PATHS:
        ckpt = Path(ckpt)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        finetuned_sds.append(load_state_dict(ckpt))

    merged_sd = dare_ties_merge(finetuned_sds=finetuned_sds,base_sd=base_sd,drop_rate=args.drop_rate,k=args.k,lam=args.lam,)
    output_path = (output_dir/ "model.pt")
    torch.save(merged_sd,output_path)
    logger.info(f"Merged model saved to:\n" f"{output_path}")

if __name__ == "__main__":
    main()