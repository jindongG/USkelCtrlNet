import os
import glob
from datetime import datetime
import cv2
import re
from scipy import ndimage

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn

from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt, binary_erosion, binary_fill_holes
from skimage.morphology import skeletonize, label as sk_label

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None

# ===== 工程依赖 =====
from dataset import DataLoader_Producer
from monai.networks.nets import UNet, SegResNet, SwinUNETR, FlexUNet
# from models.octamamba import OCTAMamba


# ============================================================
# 固定配置：直接改这里
# ============================================================
DEVICE_ID = "0"

FOV = "6M"
LABEL_TYPE = "Capillary"   # "LargeVessel", "FAZ", "Artery", "Vein", "Capillary"
BATCH_SIZE = 1

SPLIT = "test"               # "test" / "val" / "train"
THR = 0.5

# ============================================================
# 模型路径配置：适配新训练代码保存的“完整模型”
# 新训练代码保存格式示例：
#   results/.../models/model_epoch_0060.pt
#   results/.../models/model_epoch_0065.pt
#
# 用法 1：直接指定某个完整模型文件
#   MODEL_PATH = "/xxx/results/.../models/model_epoch_0060.pt"
#
# 用法 2：指定模型目录，并通过 MODEL_EPOCH 选择 epoch
#   MODEL_PATH = ""
#   MODEL_DIR = "/xxx/results/.../models"
#   MODEL_EPOCH = 60
#
# 用法 3：MODEL_EPOCH = None 时，自动选择目录下 epoch 最大的模型
# ============================================================
MODEL_PATH = "/home/ubuntu/gjd/OCTAMamba/results/2026-05-14-15-48-26/SwinSnake_Alter_2_11_1_72_MaxPooling_1_True_6M_Capillary_100_#/checkpoints/0100_full_model.pth"
MODEL_DIR = "/home/ubuntu/gjd/OCTAMamba/results/你的训练结果目录/models"
MODEL_EPOCH = 100          # 例如 60；设为 None 则自动选择最新 epoch
EVAL_ALL_MODELS = False   # True 时评估 MODEL_DIR 下所有 model_epoch_*.pt

OUT_ROOT = "/home/ubuntu/gjd/OCTAMamba/test_full_model"

SAVE_PER_SAMPLE = True
SAVE_VIS = True
SURFACE_DICE_TOL = 1.0


# ============================================================
# 可视化工具
# ============================================================
ALPHA = 0.5
overlay = lambda x, y: cv2.addWeighted(x, ALPHA, y, 1 - ALPHA, 0)

to_blue = lambda x: np.array([x, np.zeros_like(x), np.zeros_like(x)]).transpose((1, 2, 0)).astype(np.uint8)
to_red = lambda x: np.array([np.zeros_like(x), np.zeros_like(x), x]).transpose((1, 2, 0)).astype(np.uint8)
to_green = lambda x: np.array([np.zeros_like(x), x, np.zeros_like(x)]).transpose((1, 2, 0)).astype(np.uint8)
to_yellow = lambda x: np.array([np.zeros_like(x), x, x]).transpose((1, 2, 0)).astype(np.uint8)

def _safe_name(s: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", str(s))

def remove_tiny_pieces(label_u8, min_area=10):
    structure = ndimage.generate_binary_structure(2, 2)
    labelmaps, connected_num = ndimage.label(label_u8 > 0, structure=structure)
    if connected_num == 0:
        return np.zeros_like(label_u8, dtype=np.uint8)
    component_sizes = np.bincount(labelmaps.ravel())[1:]
    tiny_components_labels = np.where(component_sizes < min_area)[0] + 1
    for lab in tiny_components_labels:
        labelmaps[labelmaps == lab] = 0
    return (labelmaps > 0).astype(np.uint8) * 255

def tensor_image_to_u8_3ch(img_t: torch.Tensor) -> np.ndarray:
    img = img_t.detach().cpu().numpy()
    if img.ndim == 2:
        img = img[None, ...]
    mx = float(img.max()) if img.size else 1.0
    if mx <= 1.0:
        img = img * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    if img.shape[0] == 1:
        img3 = np.repeat(img, 3, axis=0)
    else:
        img3 = img[:3, ...]
    return img3.transpose(1, 2, 0)

def get_pred_colorizer(label_type: str):
    m = {
        "Vein": to_blue,
        "Artery": to_red,
        "Capillary": to_yellow,
        "RV": to_green,
        "LargeVessel": to_yellow,
        "FAZ": to_green,
    }
    return m.get(label_type, to_yellow)

def save_pred_artifacts(pred_bool: np.ndarray, image_u8_3ch: np.ndarray, save_dir: str, base_name: str):
    os.makedirs(save_dir, exist_ok=True)
    pred_u8 = (pred_bool.astype(np.uint8) * 255)

    np.save(os.path.join(save_dir, f"{base_name}.npy"), pred_u8)

    pred_clean = remove_tiny_pieces(pred_u8, min_area=10)
    colorize = get_pred_colorizer(LABEL_TYPE)
    pred_color = colorize(pred_clean)
    vis_overlay = overlay(image_u8_3ch, pred_color)
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_overlay.png"), vis_overlay)

    wb = 255 - pred_u8
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_wb.png"), wb)


# ============================================================
# 与训练端保持一致的包装类
# ============================================================
class ModifiedModel(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model
        self.new_layer = nn.Sigmoid()

    def forward(self, x):
        return self.new_layer(self.original_model(x))


# ============================================================
# 模型加载：只加载完整模型
# ============================================================
def load_model(model_path: str, device):
    """
    加载训练代码中 torch.save(self.model, path) 保存出来的完整模型。

    注意：PyTorch 2.6+ 的 torch.load 默认 weights_only=True，
    加载完整模型对象时必须显式设置 weights_only=False。
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        # 兼容旧版 PyTorch：没有 weights_only 参数
        model = torch.load(model_path, map_location=device)

    if not isinstance(model, nn.Module):
        raise TypeError(
            f"{model_path} 不是完整模型对象。\n"
            f"当前脚本只支持 torch.save(self.model, path) 保存出来的完整模型，"
            f"不支持 state_dict/checkpoint。"
        )

    model = model.to(device)
    model.eval()
    return model


def _epoch_from_model_name(path: str):
    """从 model_epoch_0060.pt 中解析 epoch=60。"""
    m = re.search(r"model_epoch_(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def resolve_model_paths():
    """
    根据 MODEL_PATH / MODEL_DIR / MODEL_EPOCH 得到待测试的完整模型路径列表。
    """
    if MODEL_PATH:
        if not os.path.isabs(MODEL_PATH):
            raise ValueError("MODEL_PATH 必须是绝对路径")
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(f"MODEL_PATH 不存在: {MODEL_PATH}")
        return [MODEL_PATH]

    if not MODEL_DIR:
        raise ValueError("请设置 MODEL_PATH 或 MODEL_DIR")
    if not os.path.isabs(MODEL_DIR):
        raise ValueError("MODEL_DIR 必须是绝对路径")
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(f"MODEL_DIR 不存在: {MODEL_DIR}")

    model_files = sorted(
        glob.glob(os.path.join(MODEL_DIR, "model_epoch_*.pt")),
        key=_epoch_from_model_name
    )
    if len(model_files) == 0:
        raise FileNotFoundError(
            f"{MODEL_DIR} 下没有找到 model_epoch_*.pt。"
            f"请确认训练代码保存的是完整模型，并且保存目录是 models。"
        )

    if EVAL_ALL_MODELS:
        return model_files

    if MODEL_EPOCH is None:
        return [model_files[-1]]

    target_name = f"model_epoch_{int(MODEL_EPOCH):04d}.pt"
    target_path = os.path.join(MODEL_DIR, target_name)
    if not os.path.isfile(target_path):
        available = ", ".join(os.path.basename(x) for x in model_files[:10])
        if len(model_files) > 10:
            available += ", ..."
        raise FileNotFoundError(
            f"未找到指定 epoch 模型: {target_path}\n"
            f"当前目录可用模型示例: {available}"
        )
    return [target_path]


# ============================================================
# Metrics
# ============================================================
@torch.no_grad()
def ensure_prob(preds: torch.Tensor) -> torch.Tensor:
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    if preds.max() <= 1.0 and preds.min() >= 0.0:
        return preds.clamp(0.0, 1.0)
    return torch.sigmoid(preds)

def to_numpy(x):
    return x.detach().cpu().numpy()

def binarize(x, thr):
    return (x > thr).astype(np.bool_)

def confusion(pred, gt):
    TP = np.logical_and(pred, gt).sum()
    FP = np.logical_and(pred, ~gt).sum()
    FN = np.logical_and(~pred, gt).sum()
    TN = np.logical_and(~pred, ~gt).sum()
    return TP, FP, FN, TN

def dice(pred, gt, eps=1e-6):
    inter = np.logical_and(pred, gt).sum()
    tot = pred.sum() + gt.sum()
    return (2 * inter) / (tot + eps)

def jaccard(pred, gt, eps=1e-6):
    inter = np.logical_and(pred, gt).sum()
    uni = np.logical_or(pred, gt).sum()
    return inter / (uni + eps)

def precision(pred, gt, eps=1e-6):
    TP, FP, FN, TN = confusion(pred, gt)
    return TP / (TP + FP + eps)

def recall(pred, gt, eps=1e-6):
    TP, FP, FN, TN = confusion(pred, gt)
    return TP / (TP + FN + eps)

def specificity(pred, gt, eps=1e-6):
    TP, FP, FN, TN = confusion(pred, gt)
    return TN / (TN + FP + eps)

def sens_spec(pred, gt, eps=1e-6):
    TP, FP, FN, TN = confusion(pred, gt)
    pos = TP + FN
    neg = TN + FP
    sens = TP / (pos + eps) if pos > 0 else 1.0
    spec = TN / (neg + eps) if neg > 0 else 1.0
    return float(sens), float(spec)

def bacc(pred, gt, eps=1e-6):
    sens, spec = sens_spec(pred, gt, eps=eps)
    return 0.5 * (sens + spec)

def gmean(pred, gt, eps=1e-6):
    sens, spec = sens_spec(pred, gt, eps=eps)
    return float(np.sqrt(max(sens * spec, 0.0)))

def hausdorff(pred, gt):
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan
    p = np.argwhere(pred.astype(np.uint8))
    g = np.argwhere(gt.astype(np.uint8))
    d1 = directed_hausdorff(p, g)[0]
    d2 = directed_hausdorff(g, p)[0]
    return max(d1, d2)

def border(mask_bool):
    er = binary_erosion(mask_bool)
    return np.logical_xor(mask_bool, er)

def hd95(pred, gt):
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan
    bp = border(pred)
    bg = border(gt)
    if bp.sum() == 0 or bg.sum() == 0:
        return np.nan
    dt_bg = distance_transform_edt(~bg)
    d_p2g = dt_bg[bp]
    dt_bp = distance_transform_edt(~bp)
    d_g2p = dt_bp[bg]
    if len(d_p2g) == 0 or len(d_g2p) == 0:
        return np.nan
    return float(max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95)))

def surface_distances(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    bp = border(pred)
    bg = border(gt)
    if bp.sum() == 0 or bg.sum() == 0:
        return np.array([]), np.array([])
    dt_bg = distance_transform_edt(~bg)
    dt_bp = distance_transform_edt(~bp)
    d_p2g = dt_bg[bp]
    d_g2p = dt_bp[bg]
    return d_p2g, d_g2p

def assd(pred, gt):
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan
    d_p2g, d_g2p = surface_distances(pred, gt)
    if d_p2g.size == 0 or d_g2p.size == 0:
        return np.nan
    return float(0.5 * (d_p2g.mean() + d_g2p.mean()))

def surface_dice(pred, gt, tol=1.0, eps=1e-6):
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    bp = border(pred)
    bg = border(gt)
    if bp.sum() == 0 and bg.sum() == 0:
        return 1.0
    if bp.sum() == 0 or bg.sum() == 0:
        return 0.0
    d_p2g, d_g2p = surface_distances(pred, gt)
    if d_p2g.size == 0 or d_g2p.size == 0:
        return 0.0
    m1 = (d_p2g <= tol).sum()
    m2 = (d_g2p <= tol).sum()
    return float((m1 + m2) / (bp.sum() + bg.sum() + eps))

def cldice_correct(pred, gt, eps=1e-6):
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    sp = skeletonize(pred)
    sg = skeletonize(gt)
    if sp.sum() == 0 or sg.sum() == 0:
        return 0.0
    tprec = np.logical_and(sp, gt).sum() / (sp.sum() + eps)
    tsens = np.logical_and(sg, pred).sum() / (sg.sum() + eps)
    return (2 * tprec * tsens) / (tprec + tsens + eps)

def connectivity_lcc_ratio(pred, eps=1e-6):
    if pred.sum() == 0:
        return 0.0
    lab = sk_label(pred.astype(np.uint8), connectivity=2)
    counts = np.bincount(lab.flatten())
    if len(counts) <= 1:
        return 0.0
    counts = counts[1:]
    return float(counts.max() / (pred.sum() + eps))

def component_count(pred):
    lab = sk_label(pred.astype(np.uint8), connectivity=2)
    return int(lab.max())

def betti0_error(pred, gt):
    return abs(component_count(pred) - component_count(gt))

def count_holes(mask):
    filled = binary_fill_holes(mask.astype(bool))
    holes = np.logical_and(filled, ~mask.astype(bool))
    lab = sk_label(holes.astype(np.uint8), connectivity=2)
    return int(lab.max())

def betti1_error(pred, gt):
    return abs(count_holes(pred) - count_holes(gt))

def auc_roc(prob, gt):
    if roc_auc_score is None:
        return np.nan
    y_true = gt.astype(np.uint8).flatten()
    y_score = prob.astype(np.float32).flatten()
    if np.unique(y_true).size < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return np.nan


METRICS = [
    "Dice", "Jaccard",
    "Precision", "Recall", "Specificity",
    "BACC", "GMean",
    "Hausdorff", "HD95",
    "ASSD", "SurfaceDice",
    "clDice", "AUC",
    "Connectivity", "ComponentCount",
    "Betti0Error", "Betti1Error",
]

def summarize(df):
    rows = []
    for m in METRICS:
        vals = df[m].to_numpy()
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            rows.append({
                "metric": m,
                "mean": np.nan,
                "std": np.nan,
                "median": np.nan,
                "q1": np.nan,
                "q3": np.nan,
                "iqr": np.nan,
                "n_valid": 0
            })
            continue
        q1 = float(np.percentile(vals, 25))
        q3 = float(np.percentile(vals, 75))
        rows.append({
            "metric": m,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "q1": q1,
            "q3": q3,
            "iqr": float(q3 - q1),
            "n_valid": int(len(vals)),
        })
    return pd.DataFrame(rows)


# ============================================================
# DataLoader
# ============================================================
def make_loaders(is_resize: bool = False):
    dp = DataLoader_Producer(
        fov=FOV,
        label_type=LABEL_TYPE,
        batch_size=BATCH_SIZE,
        is_resize=is_resize
    )
    return dp.get_data_loader_ipn_v2()

def select_loader(split, loaders_tuple):
    train_loader, val_loader, test_loader = loaders_tuple
    if split == "train":
        return "train", train_loader
    if split == "val":
        return "val", val_loader
    if split == "test":
        if test_loader is None:
            raise ValueError("test_loader is None")
        return "test", test_loader
    raise ValueError("SPLIT must be train/val/test")


# ============================================================
# Main
# ============================================================
def evaluate_one_model(model_path: str, device):
    if not os.path.isabs(model_path) or not os.path.isabs(OUT_ROOT):
        raise ValueError("模型路径 / OUT_ROOT 必须是绝对路径")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    out_dir = os.path.join(OUT_ROOT, f"eval_{model_stem}_{LABEL_TYPE}_{SPLIT}_{run_id}")
    os.makedirs(out_dir, exist_ok=True)

    model = load_model(model_path, device)
    print(f"Model loaded from: {model_path}")

    model_name_for_resize = model.__class__.__name__
    if model_name_for_resize == "ModifiedModel" and hasattr(model, "original_model"):
        model_name_for_resize = model.original_model.__class__.__name__
    is_resize = not any(
        tag in model_name_for_resize for tag in ("SwinSnake", "DSCNet", "USkelCtrl")
    )

    loaders_tuple = make_loaders(is_resize=is_resize)
    split_name, loader = select_loader(SPLIT, loaders_tuple)

    recs = []
    vis_dir = os.path.join(out_dir, "pred_only", split_name)

    pbar = tqdm(loader, desc=f"{os.path.basename(model_path)} | {split_name}")
    with torch.no_grad():
        for images, labels, sample_ids in pbar:
            images = images.to(torch.float).to(device)
            labels = labels.to(torch.float).to(device)

            out = model(images)
            prob = ensure_prob(out)

            prob_np = to_numpy(prob).astype(np.float32)
            gt_np = to_numpy(labels).astype(np.float32)

            B = prob_np.shape[0]
            if isinstance(sample_ids, (list, tuple)):
                sid_list = [str(x) for x in sample_ids]
            else:
                try:
                    sid_list = [str(sample_ids[i]) for i in range(B)]
                except Exception:
                    sid_list = [str(sample_ids)] * B

            for i in range(B):
                p = prob_np[i, 0]
                g = gt_np[i, 0]

                pb = binarize(p, THR)
                gb = binarize(g, 0.5)

                r = {
                    "model_file": os.path.basename(model_path),
                    "model_path": model_path,
                    "split": split_name,
                    "sample_id": sid_list[i],
                }

                r["Dice"] = dice(pb, gb)
                r["Jaccard"] = jaccard(pb, gb)
                r["Precision"] = precision(pb, gb)
                r["Recall"] = recall(pb, gb)
                r["Specificity"] = specificity(pb, gb)
                r["BACC"] = bacc(pb, gb)
                r["GMean"] = gmean(pb, gb)
                r["AUC"] = auc_roc(p, gb)

                r["Hausdorff"] = hausdorff(pb, gb)
                r["HD95"] = hd95(pb, gb)
                r["ASSD"] = assd(pb, gb)
                r["SurfaceDice"] = surface_dice(pb, gb, tol=SURFACE_DICE_TOL)

                r["Connectivity"] = connectivity_lcc_ratio(pb)
                r["ComponentCount"] = component_count(pb)
                r["Betti0Error"] = betti0_error(pb, gb)
                r["Betti1Error"] = betti1_error(pb, gb)

                r["clDice"] = cldice_correct(pb, gb)
                recs.append(r)

                if SAVE_VIS:
                    sid = _safe_name(sid_list[i])
                    base_name = f"{LABEL_TYPE}_{model_stem}_pred_{sid}"
                    img_u8 = tensor_image_to_u8_3ch(images[i])
                    save_pred_artifacts(pb, img_u8, vis_dir, base_name)

    df = pd.DataFrame(recs)

    if SAVE_PER_SAMPLE:
        df.to_csv(os.path.join(out_dir, f"{model_stem}__{split_name}__per_sample.csv"), index=False)

    summary_df = summarize(df)
    summary_df.insert(0, "model_file", os.path.basename(model_path))
    summary_df.insert(1, "split", split_name)
    summary_df.insert(2, "model_path", model_path)
    summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    pivot_mean = summary_df.pivot(index="model_file", columns="metric", values="mean")
    pivot_std = summary_df.pivot(index="model_file", columns="metric", values="std")
    pivot_median = summary_df.pivot(index="model_file", columns="metric", values="median")
    pivot_q1 = summary_df.pivot(index="model_file", columns="metric", values="q1")
    pivot_q3 = summary_df.pivot(index="model_file", columns="metric", values="q3")

    pivot_mean.to_csv(os.path.join(out_dir, f"{split_name}_pivot_mean.csv"))
    pivot_std.to_csv(os.path.join(out_dir, f"{split_name}_pivot_std.csv"))
    pivot_median.to_csv(os.path.join(out_dir, f"{split_name}_pivot_median.csv"))
    pivot_q1.to_csv(os.path.join(out_dir, f"{split_name}_pivot_q1.csv"))
    pivot_q3.to_csv(os.path.join(out_dir, f"{split_name}_pivot_q3.csv"))

    report = pivot_mean.copy()
    for c in report.columns:
        if c == "HD95":
            med = pivot_median.get(c)
            q1 = pivot_q1.get(c)
            q3 = pivot_q3.get(c)
            report[c] = [
                f"{mv:.6f} ({a:.6f}, {b:.6f})"
                if (med is not None and q1 is not None and q3 is not None
                    and not np.isnan(mv) and not np.isnan(a) and not np.isnan(b))
                else ""
                for mv, a, b in zip(med, q1, q3)
            ]
        else:
            m = pivot_mean[c]
            s = pivot_std[c]
            report[c] = [
                f"{mv:.6f}±{sv:.6f}" if (not np.isnan(mv) and not np.isnan(sv)) else ""
                for mv, sv in zip(m, s)
            ]

    report.to_csv(os.path.join(out_dir, f"{split_name}_pivot_report.csv"))

    with open(os.path.join(out_dir, "eval_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"MODEL_PATH={model_path}\n")
        f.write(f"MODEL_DIR={MODEL_DIR}\n")
        f.write(f"MODEL_EPOCH={MODEL_EPOCH}\n")
        f.write(f"FOV={FOV}\n")
        f.write(f"LABEL_TYPE={LABEL_TYPE}\n")
        f.write(f"SPLIT={SPLIT}\n")
        f.write(f"THR={THR}\n")
        f.write(f"BATCH_SIZE={BATCH_SIZE}\n")
        f.write(f"INNER_MODEL_NAME={inner_name}\n")
        f.write(f"is_resize={is_resize}\n")

    print("\n=== DONE ===")
    print("Model:", model_path)
    print("Output folder:", out_dir)
    print("- summary.csv")
    print(f"- {split_name}_pivot_report.csv")
    print("- eval_info.txt")
    return out_dir


def main():
    device = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_paths = resolve_model_paths()
    print("待测试模型:")
    for p in model_paths:
        print(" -", p)

    all_out_dirs = []
    for model_path in model_paths:
        out_dir = evaluate_one_model(model_path, device)
        all_out_dirs.append(out_dir)

    if len(all_out_dirs) > 1:
        print("\n全部模型测试完成，输出目录：")
        for d in all_out_dirs:
            print(" -", d)


if __name__ == "__main__":
    main()
