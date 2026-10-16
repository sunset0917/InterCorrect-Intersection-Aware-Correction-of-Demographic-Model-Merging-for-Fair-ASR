from dataclasses import dataclass, field
from typing import Optional, List
from torch.distributed.fsdp import ShardingStrategy


@dataclass
class ModelConfig:
    file: str = "examples/asr_librispeech/model/slam_model_asr.py:model_factory"
    llm_name: str = "tinyllama"
    llm_path: str = "/home/ashley-bravo/models/tinyllama"
    llm_type: str = "decoder_only"
    llm_dim: int = 2048
    encoder_name: Optional[str] = None
    encoder_ds_rate: int = 5
    encoder_path: Optional[str] = None
    encoder_path_hf: Optional[str] = None
    encoder_dim: int = 1280
    encoder_projector: str = "linear"
    encoder_projector_ds_rate: int = 5
    modal: str = "audio"
    normalize: Optional[bool] = field(default=False, metadata={
        "help": "whether input is normalized, used for models such as wavlm"
    })
    encoder_type: str = field(default="finetune", metadata={
        "help": "whether model is only pretrained or finetuned, used for models such as hubert"
    })
    whisper_decode: bool = False  # 👈 AÑADIR ESTA LÍNEA


@dataclass
class PeftConfig:
    peft_method: str = "lora" # None , llama_adapter, prefix
    r: int = 16
    lora_alpha: int = 32
    target_modules: List = field(default_factory=lambda: [ "q_proj", "v_proj", "k_proj", "o_proj" ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.05
    inference_mode: bool = False

@dataclass
class TrainConfig:
    model_name:str = "/home/ashley-bravo/models/tinyllama"
    enable_ddp:bool = False
    enable_deepspeed:bool = False
    enable_fsdp:bool = False
    low_cpu_fsdp:bool = False
    run_validation:bool = True
    batch_size_training:int = 8
    cluster_mix_ratio: float = 0.7
    batching_strategy:str = field(default="packing", metadata={
        "help":"alternative: padding"
    }) #
    context_length:int = 1024
    gradient_accumulation_steps:int = 4
    num_epochs:int = 10
    num_workers_dataloader:int = 1
    warmup_steps:int = 200
    total_steps:int = -1
    validation_interval:int = 600
    lr:float = 1e-4
    weight_decay:float = 1e-4
    gamma:float = 0.85
    seed:int = 42
    use_fp16:bool = False
    mixed_precision:bool = True
    val_batch_size:int = 4
    early_stopping_patience: int = 10  # Número de validaciones sin mejora antes de parar
    optimizer: str = "AdamW"
    use_peft:bool = True
    peft_config:PeftConfig = field(default_factory=PeftConfig)
    output_dir:str = "/home/ashley-bravo/outputs/slam_asr/gemma"
    freeze_layers:bool = False
    num_freeze_layers:int = 1
    quantization:bool = False
    one_gpu:bool = True
    save_model:bool = True
    dist_checkpoint_root_folder:str = "PATH/to/save/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder:str = "fine-tuned" # will be used if using FSDP
    save_optimizer:bool = False # will be used if using FSDP
    use_fast_kernels:bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    run_test_during_validation:bool = False
    run_test_during_validation_file:str = "/home/ashley-bravo/datasets/common_voice_22_storage/validation/036a10312816b365eadbd31df6bcd24be82157c1e758ee67d"
    run_test_during_validation_prompt:str = "Transcribe speech to text."
    freeze_llm:bool = field(default=False, metadata={
        "help": "whether to freeze llm when finetuning, should be true when use peft finetuning"
    })
    freeze_encoder:bool = False

@dataclass
class DataConfig:
    dataset: str = "Fair-Speech"
    file: str = "src/slam_llm/datasets/speech_dataset.py:get_speech_dataset"
    train_data_path: Optional[str] = "/home/ashley-bravo/SLAM-LLM/examples/asr_librispeech/data/Fairspeech2/fairspeech_train.jsonl"
    val_data_path: Optional[str] = "/home/ashley-bravo/SLAM-LLM/examples/asr_librispeech/data/Fairspeech2/fairspeech_validation.jsonl"
    train_split: str = "train"
    test_split:str = "validation"
    prompt: Optional[str] = "Transcribe speech to text."
    data_path: Optional[str] = None
    max_words: Optional[int] = None
    max_mel: Optional[float] = None
    fix_length_audio: int = -1
    inference_mode:bool = False
    weighted_sampling: bool = False
    input_type: str = field(default="mel", metadata={
                                "help":"Use raw when input is wav, mel when for whisper"
                            })
    mel_size: int = field(default=128, metadata={
        "help": "80 for whisper large v1 and v2, 128 for v3"
    })

    normalize: Optional[bool] = field(default=False, metadata={
        "help": "whether input is normalized, used for models such as wavlm"
    })

@dataclass
class FSDPConfig:
    mixed_precision: bool = True
    use_fp16: bool = False
    # sharding_strategy = "FULL_SHARD" #ShardingStrategy = ShardingStrategy.FULL_SHARD
    sharding_strategy: ShardingStrategy = "NO_SHARD" #ShardingStrategy.NO_SHARD #MZY: set NO_SHARD when use DDP
    checkpoint_type: str = "SHARDED_STATE_DICT"  # alternatively can use SHARDED_STATE_DICT save one file per rank, and can resize the world-size.
    fsdp_activation_checkpointing: bool = True
    fsdp_cpu_offload: bool = False
    pure_bf16: bool = False
    optimizer: str = "AdamW"


@dataclass
class LogConfig:
    use_wandb: bool = False
    # Cambia /root/ por /home/ashley-bravo/
    wandb_dir: str = "/home/ashley-bravo/outputs/wandb" 
    wandb_entity_name: str = "project_name"
    wandb_project_name: str = "project_name"
    wandb_exp_name: str = "exp_name"
    # ¡ESTE ES EL CULPABLE! Cambia /root/test.log a tu home:
    log_file: str = "/home/ashley-bravo/outputs/slam_asr/tiny_large/test.log" 
    log_interval: int = 5
