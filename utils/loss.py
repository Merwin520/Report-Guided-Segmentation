import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Dice Loss implementation
    """
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):

        pred = torch.sigmoid(pred)
        
        batch_size = pred.size(0)
        pred = pred.view(batch_size, -1)      # [B, H*W]
        target = target.view(batch_size, -1)  # [B, H*W]

        intersection = (pred * target).sum(dim=1)  # [B]
        pred_sum = pred.sum(dim=1)                 # [B]
        target_sum = target.sum(dim=1)             # [B]

        dice_coeff = (2 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        
        return (1 - dice_coeff).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        
    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target)

        dice_loss = self.dice_loss(pred, target)
        
        combined_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        return combined_loss


def bce_dice_loss(pred, target, bce_weight=0.5, dice_weight=0.5, smooth=1e-5):

    loss_fn = BCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight, smooth=smooth)
    return loss_fn(pred, target)