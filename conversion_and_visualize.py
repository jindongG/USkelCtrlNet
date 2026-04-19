# convert_and_visualize_preds.py
import os
import re
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from scipy import ndimage

# =========================
# Color mapping and overlay visualization
# =========================
ALPHA = 0.5
overlay = lambda x, y: cv2.addWeighted(x, ALPHA, y, 1 - ALPHA, 0)

to_blue = lambda x: np.array([x, np.zeros_like(x), np.zeros_like(x)]).transpose((1, 2, 0)).astype(np.uint8)
to_red = lambda x: np.array([np.zeros_like(x), np.zeros_like(x), x]).transpose((1, 2, 0)).astype(np.uint8)
to_green = lambda x: np.array([np.zeros_like(x), x, np.zeros_like(x)]).transpose((1, 2, 0)).astype(np.uint8)
to_yellow = lambda x: np.array([np.zeros_like(x), x, x]).transpose((1, 2, 0)).astype(np.uint8)
to_3ch = lambda x: np.array([x, x, x]).transpose((1, 2, 0)).astype(np.uint8)


def _safe_name(s: str) -> str:
    s = str(s)
    return re.sub(r"[^\w\-.]+", "_", s)


def remove_tiny_pieces_u8(mask_u8_0_255: np.ndarray, min_area=10) -> np.ndarray:
    """
    Load mask with flexible format:
    - bool
    - 0/1
    - 0/255
    - probability map or logits
    """
    if mask_u8_0_255.ndim != 2:
        raise ValueError("remove_tiny_pieces_u8 expects 2D mask")

    structure = ndimage.generate_binary_structure(2, 2)
    labelmaps, connected_num = ndimage.label(mask_u8_0_255 > 0, structure=structure)
    if connected_num == 0:
        return np.zeros_like(mask_u8_0_255, dtype=np.uint8)

    component_sizes = np.bincount(labelmaps.ravel())[1:]
    tiny_labels = np.where(component_sizes < min_area)[0] + 1
    for lab in tiny_labels:
        labelmaps[labelmaps == lab] = 0
    return (labelmaps > 0).astype(np.uint8) * 255


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


def tensor_like_image_to_u8_3ch(img: np.ndarray) -> np.ndarray:

    x = img
    if x.ndim == 3 and x.shape[0] in (1, 3):  # (C,H,W)
        x = x.transpose(1, 2, 0)  # -> (H,W,C)

    if x.ndim == 2:  # (H,W)
        x = to_3ch(x)

    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    x = x.astype(np.float32)
    mx = float(x.max()) if x.size else 1.0

    if mx <= 1.0:
        x = x * 255.0
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def load_mask_any(path: str) -> np.ndarray:

    arr = np.load(path)
    return arr.astype(np.float32)


def binarize_to_u8(mask_float: np.ndarray, thr=0.5) -> np.ndarray:

    m = mask_float
    if m.max() > 1.5:
        mb = (m > 127.5)
    else:
        mb = (m > thr)
    return (mb.astype(np.uint8) * 255)


def show_result_sample_figure(image_u8_3ch: np.ndarray, label_u8: np.ndarray, pred_u8: np.ndarray) -> np.ndarray:

    H, W = image_u8_3ch.shape[:2]
    lab = cv2.resize(label_u8, (W, H), interpolation=cv2.INTER_NEAREST)
    prd = cv2.resize(pred_u8, (W, H), interpolation=cv2.INTER_NEAREST)

    label_img = overlay(image_u8_3ch, to_green(lab))
    pred_clean = remove_tiny_pieces_u8(prd, min_area=10)
    pred_img = overlay(image_u8_3ch, to_yellow(pred_clean))

    return np.concatenate((image_u8_3ch, label_img, pred_img), axis=1)


def save_pred_artifacts_like_eval(
    pred_bool_u8_0_255: np.ndarray,
    image_u8_3ch: np.ndarray,
    save_dir: str,
    base_name: str,
    label_type: str,
    min_area: int = 10,
):

    os.makedirs(save_dir, exist_ok=True)


    np.save(os.path.join(save_dir, f"{base_name}.npy"), pred_bool_u8_0_255)


    pred_clean = remove_tiny_pieces_u8(pred_bool_u8_0_255, min_area=min_area)
    colorize = get_pred_colorizer(label_type)
    pred_color = colorize(pred_clean)
    vis_overlay = overlay(image_u8_3ch, pred_color)
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_overlay.png"), vis_overlay)

    wb = 255 - pred_bool_u8_0_255
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_wb.png"), wb)


def convert_one_dir(
    src_dir: str,
    dst_dir: str,
    label_type: str,
    thr: float = 0.5,
    min_area: int = 10,
    save_triplet: bool = True,
):

    os.makedirs(dst_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(src_dir) if f.endswith(".npy")])
    if not files:
        raise FileNotFoundError(f"No .npy files found in {src_dir}")


    label_files = [f for f in files if "_label_" in f]
    if not label_files:
        raise FileNotFoundError(f"No *_label_*.npy found in {src_dir}")

    p = label_files[0].rfind("_") + 1
    data_name = label_files[0][:p - 7]  # 去掉 "label_" 的那段
    ids = [f[p:-4] for f in label_files]  # 去掉扩展名

    for sid in tqdm(ids, desc=f"Converting {os.path.basename(src_dir)}"):
        sample_path = os.path.join(src_dir, f"{data_name}_sample_{sid}.npy")
        label_path = os.path.join(src_dir, f"{data_name}_label_{sid}.npy")
        pred_path = os.path.join(src_dir, f"{data_name}_pred_{sid}.npy")

        if not (os.path.exists(sample_path) and os.path.exists(label_path) and os.path.exists(pred_path)):
            continue

        img = np.load(sample_path)
        lab_f = load_mask_any(label_path)
        prd_f = load_mask_any(pred_path)

        img_u8 = tensor_like_image_to_u8_3ch(img)
        lab_u8 = binarize_to_u8(lab_f, thr=0.5)
        prd_u8 = binarize_to_u8(prd_f, thr=thr)

        sid_safe = _safe_name(sid)
        base_name = f"{label_type}_pred_{sid_safe}"


        save_pred_artifacts_like_eval(
            pred_bool_u8_0_255=prd_u8,
            image_u8_3ch=img_u8,
            save_dir=dst_dir,
            base_name=base_name,
            label_type=label_type,
            min_area=min_area,
        )


        if save_triplet:
            trip = show_result_sample_figure(img_u8, lab_u8, prd_u8)
            cv2.imwrite(os.path.join(dst_dir, f"{base_name}_triplet.png"), trip)


def main():
    SRC_DIR = r"/home/dataset-assist-0gjd/output/111/results/Layer_6M/layer5/2026-02-10-05-10-24/USkelCtrlUnet_5_11_1_72_MaxPooling_1_True_6M_Vein_100_#/0100"   # 改成你的目录
    DST_DIR = r"/home/dataset-assist-0gjd/output/111/results/Layer_6M/layer5/Vein"
    LABEL_TYPE = "Vein"                # Vein/Artery/Capillary/RV/LargeVessel/FAZ...
    THR = 0.5
    MIN_AREA = 10
    SAVE_TRIPLET = True
    # =================================

    convert_one_dir(
        src_dir=SRC_DIR,
        dst_dir=DST_DIR,
        label_type=LABEL_TYPE,
        thr=THR,
        min_area=MIN_AREA,
        save_triplet=SAVE_TRIPLET,
    )

    print("\n=== DONE ===")
    print("src_dir:", SRC_DIR)
    print("dst_dir:", DST_DIR)



if __name__ == "__main__":
    main()
