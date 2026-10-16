#!/bin/bash
# export PYTHONPATH=/root/whisper:$PYTHONPATH
export PYTHONPATH=~/SLAM-LLM/src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO


speech_encoder_path=facebook/hubert-xlarge-ll60k
llm_path=/home/ashley-bravo/models/tinyllama

train_data_path=/home/ashley-bravo/SLAM-LLM/examples/asr_librispeech/data/cv_newdata/commonvoice_train.jsonl
val_data_path=/home/ashley-bravo/SLAM-LLM/examples/asr_librispeech/data/cv_newdata/commonvoice_validation.jsonl


output_dir=/home/ashley-bravo/outputs/slam_asr/hubert_100h

hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=TinyLlama-1.1B-Chat-v1.0 \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=2048 \
++model_config.encoder_name=hubert \
++model_config.encoder_projector_ds_rate=5 \
++model_config.encoder_path=null \
++model_config.encoder_path_hf=$speech_encoder_path \
++model_config.encoder_dim=1280 \
++model_config.encoder_projector=linear \
++model_config.whisper_decode=false \
++dataset_config.dataset=speech_dataset \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.input_type=raw \
++dataset_config.mel_size=128 \
++train_config.model_name=asr \
++train_config.num_epochs=10 \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=2500 \
++train_config.total_steps=-1 \
++train_config.lr=1e-4 \
++train_config.use_fp16=false \
++train_config.validation_interval=3000 \
++train_config.weight_decay=1e-6 \
++train_config.batch_size_training=4 \
++train_config.gradient_accumulation_steps=16 \
++train_config.val_batch_size=16 \
++train_config.dist_checkpoint_root_folder=/home/ashley-bravo/outputs/slam_asr \
++train_config.dist_checkpoint_folder=100h_train_v2 \
++train_config.num_workers_dataloader=4 \
++train_config.run_validation=true \
++train_config.save_model=true \
++train_config.output_dir=$output_dir \
++log_config.log_file=$output_dir/train.log \
++metric=acc \
"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( realpath "$SCRIPT_DIR/../../../" )"
TRAIN_PY="$ROOT_DIR/examples/asr_librispeech/finetune_asr.py"

if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
    # Quitamos -m debugpy y los flags de listen/wait
    python $TRAIN_PY \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        $hydra_args
else
    # El bloque del torchrun déjalo como está por si algún día usas 2 GPUs
    torchrun \
        --nnodes 1 \
        --nproc_per_node 2 \
        --master_port=29503 \
        $TRAIN_PY \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        ++train_config.enable_fsdp=false \
        ++train_config.enable_ddp=true \
        ++train_config.use_fp16=true \
        $hydra_args
fi
