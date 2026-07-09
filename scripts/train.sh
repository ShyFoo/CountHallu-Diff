#!/bin/bash
# Launch a training run with accelerate.
#
# Usage:
#   bash scripts/train.sh uncond config/train/toyshape.yaml
#   bash scripts/train.sh t2i    config/train/sd15-hands.yaml
set -e

trainer=$1      # "uncond" or "t2i"
yaml_path=$2    # training config

case "$trainer" in
  uncond) entry="counthallu/train/train_uncond.py" ;;
  t2i)    entry="counthallu/train/train_t2i.py" ;;
  *) echo "Usage: bash scripts/train.sh {uncond|t2i} <config.yaml>"; exit 1 ;;
esac

# Parallelism comes from the num_gpus field of the training yaml.
num_gpus=$(python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['num_gpus'])" "$yaml_path")
multi_gpu=$([ "$num_gpus" -gt 1 ] && echo "--multi_gpu" || echo "")

accelerate launch --num_processes "$num_gpus" $multi_gpu \
  --main_process_port "$(shuf -i 2000-65000 -n 1)" \
  "$entry" "$yaml_path"
