#!/bin/bash
# Evaluate counting hallucinations of a trained unconditional model over
# several seeds. Paths come from config/eval/<dataset>.yaml.
#
# Usage:
#   bash scripts/evaluate.sh <dataset> <solver> <initial_noise> <num_gpus>
#   e.g. bash scripts/evaluate.sh toyshape ddpm normal 4
set -e

dataset_name=$1     # toyshape | simobject | realhand
sampling_solver=$2  # ddpm | dpm-1 | dpm-2 | dpm-plus | ddpm-gt | ddim-gt
initial_noise=$3    # normal | diffused
num_gpus=$4

seeds=(111 222 333)
multi_gpu=$([ "$num_gpus" -gt 1 ] && echo "--multi_gpu" || echo "")

# Sampling steps evaluated per solver.
declare -A inference_steps_map
inference_steps_map["dpm-1"]="15"
inference_steps_map["dpm-2"]="15"
inference_steps_map["dpm-plus"]="25 50 100 200 500 1000"
inference_steps_map["ddpm"]="1000"
inference_steps_map["ddpm-gt"]="1000"
inference_steps_map["ddim-gt"]="1000"
inference_steps_list=(${inference_steps_map["$sampling_solver"]})

for steps in "${inference_steps_list[@]}"; do
  for seed in "${seeds[@]}"; do
    accelerate launch --num_processes "$num_gpus" $multi_gpu \
      --main_process_port "$(shuf -i 2000-65000 -n 1)" \
      counthallu/eval/eval_uncond.py \
      --seed "$seed" \
      --sampling_solver "$sampling_solver" \
      --num_inference_timesteps "$steps" \
      --dataset_name "$dataset_name" \
      --initial_noise "$initial_noise"
  done
done
