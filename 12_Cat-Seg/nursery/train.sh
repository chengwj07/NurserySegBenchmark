#!/bin/sh
# Four types of seedling segmentation fine-tuning training script (CAT-Seg ViT-B/16)
# Usage:
#   sh miaomu/train.sh [CONFIG] [NUM_GPUS] [OUTPUT_DIR] [OPTS...]
# Example:
#   sh miaomu/train.sh configs/plant_vitb_384.yaml 1 output/plant
#   sh miaomu/train.sh configs/plant_vitb_384.yaml 2 output/plant \
#       SOLVER.IMS_PER_BATCH 8 SOLVER.MAX_ITER 50000
# Note: The data root directory is specified through the environment variable PLANT_DATA_ROOT (default datasets/plant), for example:
#   PLANT_DATA_ROOT=/data/.../seg_exp/data sh miaomu/train.sh configs/plant_vitb_384.yaml 1 output/plant

config=$1
gpus=$2
output=$3

if [ -z $config ]
then
    echo "No config file found! Usage: sh miaomu/train.sh [CONFIG] [NUM_GPUS] [OUTPUT_DIR] [OPTS]"
    exit 0
fi
if [ -z $gpus ]
then
    echo "Number of gpus not specified! Usage: sh miaomu/train.sh [CONFIG] [NUM_GPUS] [OUTPUT_DIR] [OPTS]"
    exit 0
fi
if [ -z $output ]
then
    echo "No output directory found! Usage: sh miaomu/train.sh [CONFIG] [NUM_GPUS] [OUTPUT_DIR] [OPTS]"
    exit 0
fi

shift 3
opts=${@}

python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --resume \
 OUTPUT_DIR $output \
 $opts
