# loss_functions.py
import torch
import torch.nn.functional as F


# ------------------------
# Basic Dice
# ------------------------
class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred and target expected to be probabilities in [0,1]
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=[1, 2, 3])
        denominator = pred.sum(dim=[1, 2, 3]) + target.sum(dim=[1, 2, 3])
        dice_score = (2. * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1. - dice_score
        return dice_loss.mean()


# ------------------------
# clDice with soft-skeleton approximation
# ------------------------
class clDiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6, sk_iter=8):
        super(clDiceLoss, self).__init__()
        self.smooth = smooth
        self.sk_iter = sk_iter

    def soft_erode(self, x, kernel_size=3):
        # min-pool via -max-pool on negated tensor (differentiable approx)
        return -F.max_pool2d(-x, kernel_size, stride=1, padding=kernel_size // 2)

    def soft_skeleton(self, x, iters=None):
        # iterative thinning-like approximation (returns soft skeleton)
        if iters is None:
            iters = self.sk_iter
        curr = x
        skeleton = torch.zeros_like(x)
        for _ in range(iters):
            eroded = self.soft_erode(curr, 3)
            opened = F.max_pool2d(eroded, 3, stride=1, padding=1)  # opening approx
            delta = (curr - opened).clamp(min=0.0)
            skeleton = torch.clamp(skeleton + delta, 0.0, 1.0)
            curr = eroded
        return skeleton

    def norm_intersection(self, center_line, vessel, eps=1.0):
        # center_line, vessel: (B,C,H,W)
        clf = center_line.view(*center_line.shape[:2], -1)
        vf = vessel.view(*vessel.shape[:2], -1)
        intersection = (clf * vf).sum(-1)
        return (intersection + eps) / (clf.sum(-1) + eps)

    def soft_cldice_loss(self, pred, target):
        # pred and target are probabilities in [0,1]
        spred = self.soft_skeleton(pred, iters=self.sk_iter)
        starget = self.soft_skeleton(target, iters=self.sk_iter)
        iflat = self.norm_intersection(spred, target)
        tflat = self.norm_intersection(starget, pred)
        inter = (iflat * tflat).sum(-1)
        denom = (iflat + tflat).sum(-1) + 1e-6
        cldice = 1.0 - (2.0 * inter) / denom
        return cldice.mean()

    def dice_loss(self, pred, target):
        # batch-wise dice
        intersection = (pred * target).sum(dim=[1, 2, 3])
        denominator = pred.sum(dim=[1, 2, 3]) + target.sum(dim=[1, 2, 3])
        dice_score = (2. * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1. - dice_score
        return dice_loss.mean()

    def forward(self, pred, target):
        # The staged objective applies its own alpha/beta weights to Dice and
        # soft clDice separately.  The 0.8/0.2 mixture is reserved for the
        # Fixed-loss ablation uses the mixture explicitly below.
        return self.soft_cldice_loss(pred, target)


class FixedDiceClDiceLoss(torch.nn.Module):
    """Fixed-loss ablation: 0.8 Dice + 0.2 soft clDice."""

    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.cldice = clDiceLoss()

    def forward(self, pred, target):
        return 0.8 * self.dice(pred, target) + 0.2 * self.cldice(pred, target)


# ------------------------
# Edge loss (Sobel guidance). We derive predicted edges from pred (no extra head required).
# ------------------------
def sobel_edge(img):
    # img: (B,1,H,W)
    device = img.device
    kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    gx = F.conv2d(img, kernel_x, padding=1)
    gy = F.conv2d(img, kernel_y, padding=1)
    grad = torch.sqrt(gx * gx + gy * gy + 1e-6)
    # normalize to [0,1]
    grad = grad / (grad.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-6)
    return grad


def edge_bce_between_pred_and_gt(pred_prob, gt_mask, edge_thresh=0.05):
    # pred_prob: predicted probability map (B,1,H,W)
    # gt_mask: ground-truth binary mask (B,1,H,W) or float
    pred_edge = sobel_edge(pred_prob)  # derived predicted edge magnitude
    gt_edge = sobel_edge(gt_mask)
    # binarize gt edge with small threshold
    gt_edge_bin = (gt_edge > edge_thresh).float()
    # use BCE with logits? pred_edge is in [0,1], so use BCE (apply small clamp)
    pred_edge_clamped = pred_edge.clamp(min=1e-6, max=1 - 1e-6)
    bce = F.binary_cross_entropy(pred_edge_clamped, gt_edge_bin, reduction='mean')
    return bce


# ------------------------
# Total loss composition
# ------------------------
class TotalVesselLoss(torch.nn.Module):
    """
    Training objective:
        alpha * Dice + beta * soft clDice + gamma * edge BCE
    """

    def __init__(self, alpha=0.5, beta=0.3, gamma=0.1):
        super(TotalVesselLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.dice = DiceLoss()
        self.cldice = clDiceLoss()

    def forward(self, pred_prob, gt_mask):
        """
        pred_prob: predicted probabilities in [0,1], shape (B,1,H,W)
        gt_mask: ground-truth binary mask (B,1,H,W) float {0,1}
        """
        # Ensure shapes and types
        pred_prob = pred_prob.float()
        gt_mask = gt_mask.float()
        Ldice = self.dice(pred_prob, gt_mask)
        Lcl = self.cldice(pred_prob, gt_mask)
        Ledge = edge_bce_between_pred_and_gt(pred_prob, gt_mask)
        total = self.alpha * Ldice + self.beta * Lcl + self.gamma * Ledge
        return total, dict(dice=Ldice, cldice=Lcl, edge=Ledge)

