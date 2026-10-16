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

# Rutas de ejemplo (reemplaza con tus paths reales)
BASE_MODEL_PATH = "/home/ashley-bravo/outputs/slam_asr/100h_train_v4/asr_epoch_5_step_4680/model.pt"
CHECKPOINT_PATHS = [
    "/home/ashley-bravo/outputs/slam_asr/Finetuning_FS/18-22/asr_epoch_1_step_200/model.pt",
    "/home/ashley-bravo/outputs/slam_asr/Finetuning_FS/23-30/asr_epoch_1_step_200/model.pt",
    "/home/ashley-bravo/outputs/slam_asr/Finetuning_FS/31-45/asr_epoch_1_step_200/model.pt",
    "/home/ashley-bravo/outputs/slam_asr/Finetuning_FS/46-65/asr_epoch_1_step_200/model.pt",
    "/home/ashley-bravo/outputs/slam_asr/Finetuning_FS/affluent/asr_epoch_2_step_153/model.pt",
    "/home/ashley-bravo/outputs/slam_asr/Finetuning_FS/Low-SES/asr_epoch_1_step_200/model.pt",
    "/home/ashley-bravo/outputs/slam_asr/Finetuning_FS/Medium-SES/asr_epoch_1_step_200/model.pt",
    "/home/ashley-bravo/outputs/slam_asr/Finetuning_FS/female/asr_epoch_2_step_127/model.pt",
    "/home/ashley-bravo/outputs/slam_asr/Finetuning_FS/male/asr_epoch_1_step_200/model.pt",
    "/home/ashley-bravo/outputs/slam_asr/Finetuning_FS/L2/asr_epoch_1_step_200/model.pt",
    "/home/ashley-bravo/outputs/slam_asr/Finetuning_FS/English-FL/asr_epoch_2_step_125/model.pt",
    "/home/ashley-bravo/outputs/slam_asr/Finetuning_FS/Asian/asr_epoch_1_step_200/model.pt",
    "/home/ashley-bravo/outputs/slam_asr/Finetuning_FS/Black/asr_epoch_1_step_200/model.pt",
    "/home/ashley-bravo/outputs/slam_asr/Finetuning_FS/Hawai/asr_epoch_6_step_80/model.pt",
    "/home/ashley-bravo/outputs/slam_asr/Finetuning_FS/Latino/asr_epoch_1_step_200/model.pt",
    "/home/ashley-bravo/outputs/slam_asr/Finetuning_FS/ME/asr_epoch_8_step_74/model.pt",
    "/home/ashley-bravo/outputs/slam_asr/Finetuning_FS/NA-AI-AN/asr_epoch_2_step_125/model.pt",
    "/home/ashley-bravo/outputs/slam_asr/Finetuning_FS/white/asr_epoch_2_step_128/model.pt"
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

def apply_dare_sparsification(task_vector: torch.Tensor, drop_rate: float) -> torch.Tensor:
    """
    Aplica el algoritmo DARE a un vector de tarea.
    drop_rate (p): Probabilidad de eliminar un peso (ej. 0.90 para eliminar el 90%)
    """
    if drop_rate <= 0.0:
        return task_vector
    
    # 1. Crear una máscara aleatoria de sobrevivientes (Bernoulli) con probabilidad (1 - p)
    # torch.rand_like genera valores aleatorios uniformes entre 0 y 1
    survival_rate = 1.0 - drop_rate
    mask = (torch.rand_like(task_vector, dtype=torch.float32) < survival_rate)
    
    # 2. Reescalar los pesos que sobrevivieron: 1 / (1 - p)
    scale_factor = 1.0 / survival_rate
    
    # 3. Combinar: poner a cero los eliminados y multiplicar por el factor los que quedaron
    sparsified_vector = task_vector.float() * mask.float() * scale_factor
    
    return sparsified_vector.to(task_vector.dtype)

def dare_linear_merge(base_sd: Dict[str, torch.Tensor],finetuned_sds: List[Dict[str, torch.Tensor]],drop_rate: float = 0.90,) -> Dict[str, torch.Tensor]:
    """
    DARE Model Fusion (Multi-task Addition con Drop & Rescale):
        τ_t = DARE(θ_ft^t - θ_base, drop_rate)
        θ* = θ_base + α * Σ(τ_t)
    """
    merged_sd = {}
    
    for key in base_sd.keys():
        base_tensor = base_sd[key]

        # Ignorar tensores enteros (conteo de pasos, tokens, etc.)
        if not base_tensor.dtype.is_floating_point:
            merged_sd[key] = base_tensor.clone()
            continue

        total_trajectory = torch.zeros_like(base_tensor, dtype=torch.float32)
        
        for sd in finetuned_sds:
            if key in sd and sd[key].shape == base_tensor.shape:
                # Calcular el delta original
                task_vector = sd[key].float() - base_tensor.float()
                
                # Aplicar Drop And Rescale de forma independiente a este vector de tarea
                dare_vector = apply_dare_sparsification(task_vector, drop_rate)
                
                # Acumular vectores dispersos
                total_trajectory += dare_vector
                
        merged_tensor = base_tensor.float() +  total_trajectory
        merged_sd[key] = merged_tensor.to(base_tensor.dtype)

    return merged_sd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default=BASE_MODEL_PATH)
    parser.add_argument("--output_dir", type=str, default="/home/ashley-bravo/EXPERIMENTATION/RQ2/DARE/5")
    parser.add_argument("--drop_rate", type=float, default=0.05, help="Fracción de deltas a eliminar (0.0 a 0.99)")
    return parser.parse_args()

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_sd = load_state_dict(Path(args.base_model))
    finetuned_sds = [load_state_dict(Path(p)) for p in CHECKPOINT_PATHS if Path(p).exists()]
    
    logger.info(f"Fusing {len(finetuned_sds)} models via DARE | drop_rate (p)={args.drop_rate}")
    
    merged_sd = dare_linear_merge(
        base_sd=base_sd,
        finetuned_sds=finetuned_sds,
        drop_rate=args.drop_rate,
    )

    out_path = output_dir / "model.pt"
    torch.save(merged_sd, out_path)
    logger.info(f"Merged model saved to: {out_path}")

if __name__ == "__main__":
    main()