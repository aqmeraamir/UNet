import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.spatial.distance import directed_hausdorff
import torch
import lpips

# -------------------------
# DICE COEFFICIENT
# -------------------------
def dice_score(pred, gt, eps=1e-6):
    '''
    Computes the Dice coefficient between two binary masks, and
    outputs a number from 0 (no overlap) to 1 (perfect overlap).

    Parameters:
    - pred: np.ndarray
        The predicted binary mask (any shape).
    - gt: np.ndarray
        The ground truth binary mask (same shape as pred).
    - eps: float
        Small epsilon value to prevent division by zero.

    Returns:
    - dice coefficient: float
    '''
    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)
    intersection = np.sum(pred & gt)
    return (2 * intersection + eps) / (np.sum(pred) + np.sum(gt) + eps)


# -------------------------
# INTERSECTION OVER UNION
# -------------------------
def iou_score(pred, gt, eps=1e-6):
    '''
    Computes the Intersection over Union between two binary masks, and 
    outputs a number from 0 (no overlap) to 1 (perfect overlap).

    Parameters:
    - pred: np.ndarray
        The predicted binary mask (any shape).
    - gt: np.ndarray
        The ground truth binary mask (same shape as pred).
    - eps: float
        Small epsilon value to prevent division by zero.

    Returns:
    - IoU score: float
    '''
    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)
    intersection = np.sum(pred & gt)
    union = np.sum(pred | gt)
    return (intersection + eps) / (union + eps)


# -------------------------
# FILTERING REGIONS
# -------------------------
def filter_regions(mask, region):
    # mask: ndarray with predicted or ground truth labels
    
    if region == "Whole Tumor (WT)":
        # WT = all tumor classes combined: Enhancing + Edema + Non-Enhancing
        return (mask > 0).astype(np.uint8)
    elif region == "Tumor Core (TC)":
        # TC = Enhancing + Non-Enhancing Tumor (labels 1 and 3)
        return np.isin(mask, [1, 3]).astype(np.uint8)
    elif region == "Enhancing Tumor (ET)":
        # ET = Enhancing Tumor only (label 3)
        return (mask == 3).astype(np.uint8)
    else:
        return mask