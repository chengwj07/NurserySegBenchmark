"""
Self-contained implementation of Grounding DINO training loss (refer to the official GroundingDINOCriterion design).

The official warehouse currently does not release the training code with the source code. Here it is implemented according to the official loss design:
  1. HungarianMatcher: Match 900 queries to GT box.
     Classification cost = query and the expected logit of the GT category token are negative (positive_map dot product),
     bbox cost = L1, giou cost = -GIoU。
  2. GroundingDINOCriterion:
     - loss_ce: perform sigmoid focal loss on the matched pairs of pred_logits and positive_map
                      (This is the text-box alignment loss of GroundingDINO)
     - loss_bbox: L1 regression
     - loss_giou:    GIoU
  forward returns {loss_ce, loss_bbox, loss_giou, loss_total}.

Depends on: util.box_ops / models.GroundingDINO.utils.sigmoid_focal_loss in the local groundingdino package,
      scipy's linear_sum_assignment does Hungarian matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from groundingdino.util import box_ops
from groundingdino.models.GroundingDINO.utils import sigmoid_focal_loss


def _get_src_permutation_idx(indices):
    # Concatenate the (matched_query_idx, matched_gt_idx) of each image into the global (batch_idx, src_idx)
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


class HungarianMatcher(nn.Module):
    'Perform binary matching on query and GT according to DETR method, cost consists of classification (text alignment) + L1 + GIoU'

    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        'Return (src_idx, tgt_idx) of each picture: query subscript -> GT subscript'
        num_queries = outputs["pred_logits"].shape[1]
        indices = []
        for i in range(len(targets)):
            src_logits = outputs["pred_logits"][i].sigmoid()  # [nq, 256]
            src_boxes = outputs["pred_boxes"][i]              # [nq, 4] cxcywh normalization
            tgt_map = targets[i]["positive_map"]              # [ng, 256] row normalization
            tgt_boxes = targets[i]["boxes"]                   # [ng, 4] cxcywh normalization
            ng = tgt_boxes.shape[0]

            if ng == 0:
                indices.append(
                    (torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64))
                )
                continue

            # Classification cost: The expected logit of the GT category token is negative
            cost_class = -(src_logits @ tgt_map.t())          # [nq, ng]
            cost_bbox = torch.cdist(src_boxes, tgt_boxes, p=1)  # [nq, ng]
            cost_giou = -box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(tgt_boxes),
            )                                                 # [nq, ng]
            cost = (self.cost_class * cost_class
                    + self.cost_bbox * cost_bbox
                    + self.cost_giou * cost_giou)

            # Most of the 900 queries are background. We first take the top-k candidates of each GT and then run them in Hungary, which greatly speeds up the process.
            k = min(num_queries, max(64, ng * 10))
            candidates = torch.topk(cost, k, dim=0, largest=False).indices.unique()
            sub_cost = cost[candidates]                        # [n_cand, ng]
            row, col = linear_sum_assignment(
                sub_cost.detach().cpu().numpy()
            )
            src_idx = candidates[torch.as_tensor(row, dtype=torch.int64)]
            tgt_idx = torch.as_tensor(col, dtype=torch.int64)
            indices.append((src_idx, tgt_idx))
        return indices


class GroundingDINOCriterion(nn.Module):
    def __init__(self, matcher, weight_dict=None, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict or {
            "loss_ce": 2.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
        }
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def loss_ce(self, outputs, targets, indices, num_boxes):
        'Text alignment loss: Matching pairs of pred_logits using positive_map as sigmoid focal'
        src_logits = outputs["pred_logits"]
        idx = _get_src_permutation_idx(indices)
        target_classes = torch.cat(
            [t["positive_map"][j] for t, (_, j) in zip(targets, indices)]
        ).to(src_logits.dtype)
        loss = sigmoid_focal_loss(
            src_logits[idx], target_classes, num_boxes,
            alpha=self.focal_alpha, gamma=self.focal_gamma,
        ) * src_logits.shape[1]  # Official scaling based on query number
        return {"loss_ce": loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = _get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="sum") / num_boxes

        ious = torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes),
        )).clamp(min=-1.0, max=1.0)
        loss_giou = (1 - ious).sum() / num_boxes
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = max(num_boxes, 1)  # Defense: Theoretically the data has been filtered into empty images

        losses = {}
        if "loss_ce" in self.weight_dict:
            losses.update(self.loss_ce(outputs, targets, indices, num_boxes))
        if "loss_bbox" in self.weight_dict or "loss_giou" in self.weight_dict:
            losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))

        losses["loss_total"] = sum(
            self.weight_dict[k] * losses[k] for k in self.weight_dict
        )
        return losses
