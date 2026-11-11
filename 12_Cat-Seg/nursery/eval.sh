#!/bin/sh
# Four types of seedling segmentation evaluation script (CAT-Seg ViT-B/16)
# Usage:
#   sh miaomu/eval.sh [CONFIG] [NUM_GPUS] [OUTPUT_DIR] [WEIGHTS] [OPTS...]
# Example:
#   sh miaomu/eval.sh configs/plant_vitb_384.yaml 1 output/plant output/plant/model_final.pth
#   sh miaomu/eval.sh configs/plant_vitb_384.yaml 1 output/plant_eval weights/model_base.pth

config=$1
gpus=$2
output=$3
weights=$4

if [ -z $config ]
then
    echo "No config file found! Usage: sh miaomu/eval.sh [CONFIG] [NUM_GPUS] [OUTPUT_DIR] [WEIGHTS] [OPTS]"
    exit 0
fi
if [ -z $gpus ]
then
    echo "Number of gpus not specified!"
    exit 0
fi
if [ -z $output ]
then
    echo "No output directory found!"
    exit 0
fi
if [ -z $weights ]
then
    echo "No weights specified!"
    exit 0
fi

shift 4
opts=${@}

python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output \
 MODEL.WEIGHTS $weights \
 $opts
