# -*- coding: utf-8 -*-
# mmseg style accuracy evaluation entrance (four types of seedling segmentation)
# Usage (run in 12_Cat-Seg root directory):
#   python miaomu/eval_mmseg.py --config-file configs/plant_vitb_384.yaml \
#       --eval-only --num-gpus 1 \
#       OUTPUT_DIR output/plant_eval MODEL.WEIGHTS output/plant/model_final.pth
# Output: Console print mmseg style table (IoU/Dice/F1/Precision/Recall/Acc + mean per category),
# And save the JSON to OUTPUT_DIR/sem_seg_evaluation.json.
import os
import sys

import torch

# fmt: off
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))
# fmt: on

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import DatasetEvaluators
from detectron2.utils.logger import setup_logger

from train_net import Trainer, setup  # Reuse official Trainer and setup (including add_deeplab/add_cat_seg_config)
from miaomu.mmseg_evaluator import MMSegStyleEvaluator


class MMSegTrainer(Trainer):
    'Same as the official Trainer, just change the evaluator to mmseg style.'

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return DatasetEvaluators(
            [MMSegStyleEvaluator(dataset_name, distributed=True, output_dir=output_folder)]
        )


def main(args):
    cfg = setup(args)
    torch.set_float32_matmul_precision("high")
    if args.eval_only:
        model = MMSegTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = MMSegTrainer.test(cfg, model)
        if comm_is_main_process():
            from detectron2.evaluation import verify_results

            verify_results(cfg, res)
        return res
    else:
        raise SystemExit("eval_mmseg.py 只用于评估 (请加 --eval-only)")


def comm_is_main_process():
    from detectron2.utils.comm import is_main_process

    return is_main_process()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
