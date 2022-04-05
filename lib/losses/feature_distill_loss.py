import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torch import einsum

def compute_imitation_loss(input, target, weights):
    target = torch.where(torch.isnan(target), input, target)  # ignore nan targets
    diff = input - target
    loss = 0.5 * diff ** 2

    assert weights.shape == loss.shape[:-1]
    weights = weights.unsqueeze(-1)
    assert len(loss.shape) == len(weights.shape)
    loss = loss * weights
    return loss

def compute_backbone_l1_loss(features_preds, features_targets, target):
    feature_ditill_loss = 0.0
    if isinstance(features_preds, list):
        for i in range(len(features_preds)):
            downsample_ratio = 2 ** (i + 3)

            feature_target = features_targets[i].detach()
            feature_pred = features_preds[i]

            mask = calculate_box_mask(feature_pred, target, downsample_ratio)  # mask [B, H, W]

            feature_pred = feature_pred.permute(0, *range(2, len(feature_pred.shape)), 1)
            feature_target = feature_target.permute(0, *range(2, len(feature_target.shape)), 1)
            batch_size = int(feature_pred.shape[0])
            positives = feature_pred.new_ones(*feature_pred.shape[:3])
            positives = positives * torch.any(feature_target != 0, dim=-1).float()
            positives = positives * mask.cuda()

            reg_weights = positives.float()
            pos_normalizer = positives.sum().float()
            reg_weights /= pos_normalizer

            pos_inds = reg_weights > 0
            pos_feature_preds = feature_pred[pos_inds]
            pos_feature_targets = feature_target[pos_inds]

            imitation_loss_src = compute_imitation_loss(pos_feature_preds,
                                                          pos_feature_targets,
                                                          weights=reg_weights[pos_inds])  # [N, M]

            imitation_loss = imitation_loss_src.mean(-1)
            imitation_loss = imitation_loss.sum() / batch_size
            feature_ditill_loss = feature_ditill_loss + imitation_loss

    else:
        raise NotImplementedError

    return feature_ditill_loss

def compute_backbone_resize_affinity_loss(features_preds, features_targets):
    feature_ditill_loss = 0.0
    resize_shape = features_preds[-1].shape[-2:]
    if isinstance(features_preds, list):
        for i in range(len(features_preds)):  # 1/8   1/16   1/32
            feature_target = features_targets[i].detach()
            feature_pred = features_preds[i]

            B, C, H, W = feature_pred.shape
            feature_pred_down = F.interpolate(feature_pred, size=resize_shape, mode="bilinear")
            feature_target_down = F.interpolate(feature_target, size=resize_shape, mode="bilinear")

            feature_target_down = feature_target_down.reshape(B, C, -1)
            depth_affinity = torch.bmm(feature_target_down.permute(0, 2, 1), feature_target_down)

            feature_pred_down = feature_pred_down.reshape(B, C, -1)
            rgb_affinity = torch.bmm(feature_pred_down.permute(0, 2, 1), feature_pred_down)

            feature_ditill_loss = feature_ditill_loss + F.l1_loss(rgb_affinity, depth_affinity, reduction='mean') / B

    else:
        raise NotImplementedError

    return feature_ditill_loss


def compute_backbone_local_affinity_loss(features_preds, features_targets):
    feature_ditill_loss = 0.0
    local_shape = features_preds[-1].shape[-2:]
    if isinstance(features_preds, list):
        for i in range(len(features_preds)):  # 1/8   1/16   1/32
            feature_target = features_targets[i].detach()
            feature_pred = features_preds[i]

            B, _, H, W = feature_pred.shape
            feature_pred_q = rearrange(feature_pred, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1=local_shape[0], p2=local_shape[1])
            feature_pred_k = rearrange(feature_pred, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1=local_shape[0], p2=local_shape[1])

            rgb_affinity = einsum('b i d, b j d -> b i j', feature_pred_q, feature_pred_k)

            feature_target_q = rearrange(feature_target, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1=local_shape[0], p2=local_shape[1])
            feature_target_k = rearrange(feature_target, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1=local_shape[0], p2=local_shape[1])

            depth_affinity = einsum('b i d, b j d -> b i j', feature_target_q, feature_target_k)

            feature_ditill_loss = feature_ditill_loss + F.l1_loss(rgb_affinity, depth_affinity, reduction='mean') / B

    else:
        raise NotImplementedError

    return feature_ditill_loss

def calculate_box_mask(features_preds, target, downsample_ratio):
    B, C, H, W = features_preds.shape
    gt_mask = torch.zeros((B, H, W))

    for i in range(B):
        for j in range(target['obj_num'][i]):
            bbox2d_gt = target['box2d_gt'][i, j] / downsample_ratio

            left_top = bbox2d_gt[:2]
            right_bottom = bbox2d_gt[2:]

            left_top_x = int(left_top[0].item())
            left_top_y = int(left_top[1].item())

            right_bottom_x = int(right_bottom[0].item())
            right_bottom_y = int(right_bottom[1].item())

            gt_mask[i, left_top_y:right_bottom_y, left_top_x:right_bottom_x] = 1
    return gt_mask

