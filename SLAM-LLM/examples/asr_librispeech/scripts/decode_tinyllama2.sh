#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/ashley-bravo/SLAM-LLM/src
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

run_dir=/home/ashley-bravo/SLAM-LLM
cd $run_dir
code_dir=examples/asr_librispeech

speech_encoder_path=large-v3
llm_path=/home/ashley-bravo/models/tinyllama
output_dir=/home/ashley-bravo/outputs/slam_asr/experimento3
ckpt_path=$output_dir/asr_epoch_5_step_6332
split=CV_full
val_data_path=$run_dir/examples/asr_librispeech/data/commonvoice/cv22_test_full.jsonl
decode_log=$ckpt_path/decode_experimento3

python $code_dir/inference_asr_batch.py \
    --config-path "conf" \
    --config-name "prompt.yaml" \
    hydra.run.dir=$ckpt_path \
    ++model_config.llm_name="TinyLlama-1.1B-Chat-v1.0" \
    ++model_config.llm_path=$llm_path \
    ++model_config.llm_dim=2048 \
    ++model_config.encoder_name=whisper \
    ++model_config.encoder_projector_ds_rate=5 \
    ++model_config.encoder_path=$speech_encoder_path \
    ++model_config.encoder_dim=1280 \
    ++model_config.encoder_projector=linear \
    ++dataset_config.dataset=speech_dataset \
    ++dataset_config.val_data_path=$val_data_path \
    ++dataset_config.input_type=mel \
    ++dataset_config.mel_size=128 \
    ++dataset_config.inference_mode=true \
    ++train_config.model_name=asr \
    ++train_config.freeze_encoder=true \
    ++train_config.freeze_llm=true \
    ++train_config.batching_strategy=custom \
    ++train_config.num_epochs=1 \
    ++train_config.val_batch_size=16 \
    ++train_config.num_workers_dataloader=4 \
    ++train_config.output_dir=$output_dir \
    ++decode_log=$decode_log \
    ++ckpt_path=$ckpt_path/model.pt \
    ++log_config.log_file=$ckpt_path/inference_experimento3.log
