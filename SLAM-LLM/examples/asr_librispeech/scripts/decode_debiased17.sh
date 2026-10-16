#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/ashley-bravo/SLAM-LLM/src
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false

run_dir=/home/ashley-bravo/SLAM-LLM
cd $run_dir
code_dir=examples/asr_librispeech
speech_encoder_path=large-v3
llm_path=/home/ashley-bravo/models/tinyllama

output_dir=/home/ashley-bravo/EXPERIMENTATION/Top-K/Linear/1st-Method/K2/
split=muestras
val_data_path=/home/ashley-bravo/SLAM-LLM/examples/asr_librispeech/data/Fairspeech2/fairspeech_test.jsonl

ckpt_path=$output_dir/K2_M1_M5_c0p5_0p5
decode_log=$ckpt_path/fair_test
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
    ++train_config.val_batch_size=32 \
    ++train_config.num_workers_dataloader=2 \
    ++train_config.output_dir=$output_dir \
    ++decode_log=$decode_log \
    ++ckpt_path=$ckpt_path/model.pt \
    ++log_config.log_file=$ckpt_path/merged.log
