# SAM 3 Nursery Stock Segmentation

This experiment adapts [Meta SAM 3](https://github.com/facebookresearch/sam3) to nursery stock segmentation in UAV imagery. The training entry point is `sam3/train/train.py`. Hydra configurations are in `sam3/train/configs/`. See the [training guide](README_TRAIN.md) for framework usage.

From `10_SAM3/`, run the following command to compute pixel metrics from the exported COCO segmentation predictions and annotations for the same validation set.

```bash
python eval_plant_metrics.py --pred-json /path/to/coco_predictions_segm.json --coco-json /path/to/validation_annotations.json
```

To run the shared boundary evaluation, use `common/convert_sam3_predictions.py` to convert predictions into class-index masks. See the [repository README](../README.md) for the evaluation workflow.
