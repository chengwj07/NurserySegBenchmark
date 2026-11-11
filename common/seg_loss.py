"""
common/seg_loss.py

SafeCrossEntropyLoss: Fix for mmseg CrossEntropyLoss when class_weight is enabled.

Background: The training pipeline RandomResize(ratio_range=(0.5, 2.0)) will randomly shrink the 1024x1024 image to
512~2048, about half of the samples are reduced to less than 1024; RandomCrop(crop_size=1024) returns the small image if it cannot crop 1024;
When making a batch, data_preprocessor uses seg_pad_val=255 to pad the small image to 1024x1024, so 255 appears in the label.

mmseg original cross_entropy will be executed in class_weight mode
    torch.stack([class_weight[cls] for cls in label])
Index all tags (including 255), 255 is out of bounds -> IndexKernel assertion crashes.
When class_weight is not added, 255 will be silently ignored by F.cross_entropy(ignore_index=255), so it will not crash before.

Repair of this type: First temporarily set the ignore_index(255) pixel label to 0 (safe index), and generate a pixel-by-pixel weight
Multiply the loss of ignore pixels by 0 (maintain the \"ignore\" semantics and do not use void as background training), and then hand it over to the parent class.

In Usage:config, change the type of loss_decode to \"SafeCrossEntropyLoss\", and the other parameters remain unchanged.
"""
import torch

from mmseg.registry import MODELS
from mmseg.models.losses.cross_entropy_loss import CrossEntropyLoss


@MODELS.register_module()
class SafeCrossEntropyLoss(CrossEntropyLoss):
    'Safe version of CrossEntropyLoss: class_weight does not crash when used with ignore_index(255).'

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-100,
                **kwargs):
        if ignore_index is not None:
            ignore = (label == ignore_index)
            if ignore.any():
                # Set the ignore pixel label to 0 temporarily (class_weight index safety),
                # And use weight=0 to exclude their contribution to loss (do not use void as background training).
                label = torch.where(ignore, torch.zeros_like(label), label)
                mask = (~ignore).to(cls_score.dtype)
                if weight is None:
                    weight = mask
                else:
                    weight = weight * mask
        return super().forward(
            cls_score,
            label,
            weight=weight,
            avg_factor=avg_factor,
            reduction_override=reduction_override,
            ignore_index=ignore_index,
            **kwargs)
