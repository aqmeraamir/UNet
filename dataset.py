import os
import numpy as np
from tqdm import tqdm
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
import glob
import splitfolders
import tempfile

import torch
from torch.utils.data import Dataset



# -------------------------
# SETTINGS & PATHS
# -------------------------
RAW_TRAIN = 'data/raw/MICCAI_BraTS2020_TrainingData'
OUT_DIR   = 'data/processed/input_data_processed'
IMG_DIR   = 'data/processed/input_data_processed/volumes'
MSK_DIR   = 'data/processed/input_data_processed/masks'
SPLIT_DIR = 'data/processed/input_data_split'
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MSK_DIR, exist_ok=True)


SCROP = slice(56, 184)
TCROP = slice(13, 141)
MODALITIES = ['flair', 't1', 't1ce', 't2']
SELECTED_MODALITIES = ['flair', 't1ce', 't2']

scaler = MinMaxScaler()


# -------------------------
# PROCESS ALL ITEMS
# -------------------------

def normalize_modality(volume):
    """Apply MinMaxScaler on 3D volume (per-slice-wise normalization)."""
    return scaler.fit_transform(volume.reshape(-1, volume.shape[-1])).reshape(volume.shape)

def crop_volume(volume):
    """Crop using fixed slices."""
    return volume[SCROP, SCROP, TCROP]

def load_volume(path, case):
    """
    Loads and processes a single BraTS case from NIfTI files.

    Returns:
        - vol: np.ndarray of shape (C, D, H, W)
        - mask: np.ndarray of shape (D, H, W), with labels 0–3
    """
    vols = []
    for modality in MODALITIES:
        vol_path = os.path.join(path, f"{case}_{modality}.nii")
        vol = nib.load(vol_path).get_fdata()
        vol = normalize_modality(vol)
        vol = crop_volume(vol)
        vols.append(vol)

    # stacked_vol = np.stack(vols, axis=-1)  # shape: (D, H, W, C)
    # # Transpose axes to (C, D, H, W)
    # stacked_vol = np.transpose(stacked_vol, (3, 0, 1, 2))
    stacked_vol = np.stack(vols, axis=0)

    # Load & process mask
    mask_path = os.path.join(path, f"{case}_seg.nii")
    mask = nib.load(mask_path).get_fdata().astype(np.uint8)
    mask[mask == 4] = 3  # Map class 4 to 3
    mask = crop_volume(mask)  # shape: (D, H, W)

    return stacked_vol, mask

def load_uploaded_volume(file):
    """
    Loads a single uploaded .npy or .nii file for Streamlit.
    Handles both file-like objects and paths.
    """
    filename = file.name

    if filename.endswith(".npy"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        volume = np.load(tmp_path)
        os.remove(tmp_path)  

    elif filename.endswith(".nii") or filename.endswith(".nii.gz"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        volume = nib.load(tmp_path).get_fdata()
        os.remove(tmp_path)  

        volume = normalize_modality(volume)
        volume = crop_volume(volume)
        volume = np.expand_dims(volume, axis=0)
        
    else:
        raise ValueError("Unsupported file type. Must be .npy or .nii")

    return volume.astype(np.float32)


def process_and_save(root_dir):
    for path in tqdm(glob.glob(os.path.join(root_dir, 'BraTS20_*'))):
        case = os.path.basename(path)
        volume, mask = load_volume(path, case)

        # Skip volumes that are nearly empty
        if (1 - (np.unique(mask, return_counts=True)[1][0] / mask.size)) < 0.01:
            continue

        mask_onehot = np.eye(4)[mask]  # one-hot encode → shape: (D, H, W, 4)

        np.save(os.path.join(IMG_DIR, f"{case}.npy"), volume)
        np.save(os.path.join(MSK_DIR, f"{case}.npy"), mask_onehot)






# def process_and_save(root_dir):
#     for path in tqdm(glob.glob(os.path.join(root_dir, 'BraTS20_*'))):
#         case = os.path.basename(path)
#         vols = []
#         for m in MODALITIES:
#             arr = nib.load(os.path.join(path, f'{case}_{m}.nii')).get_fdata()
#             arr = scaler.fit_transform(arr.reshape(-1, arr.shape[-1])).reshape(arr.shape)
#             vols.append(arr)
#         m = nib.load(os.path.join(path, f'{case}_seg.nii')).get_fdata().astype(np.uint8)
#         m[m==4] = 3
#         vols = [v[SCROP, SCROP, TCROP] for v in vols]
#         m = m[SCROP, SCROP, TCROP]
#         if (1 - (np.unique(m, return_counts=True)[1][0] / m.size)) < 0.01:
#             continue
#         comb = np.stack(vols, axis=-1)
#         oh = np.eye(4)[m]
#         np.save(os.path.join(IMG_DIR, f'{case}.npy'), comb)
#         np.save(os.path.join(MSK_DIR, f'{case}.npy'), oh)


# -------------------------
# SPLIT VOLUMES INTO TRAIN/VAL
# -------------------------
def split_data():
    splitfolders.ratio(
        input=OUT_DIR,                   
        output=SPLIT_DIR,
        seed=42,
        ratio=(.75, .25)
    )

# -------------------------
# UTIL: filter modalities
# -------------------------
def filter_modalities(volume: np.ndarray, modalities=MODALITIES, selected=SELECTED_MODALITIES):
    """
    take a volume of shape (C, D, H, W) where C matches `modalities` list,
    return a new array selecting channels in `selected` order.
    """
    
    if selected is None:
        return volume  

    idxs = [modalities.index(m) for m in selected]
    return volume[idxs]  # (len(selected), D, H, W)

# -------------------------
# DATASET DEFINITION
# -------------------------
class BraTSDataset(Dataset):
    def __init__(self, img_dir, mask_dir, file_list):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.file_list = [f for f in file_list if f.endswith('.npy')]

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        fname = self.file_list[idx]

        img = np.load(os.path.join(self.img_dir, fname)).astype(np.float32) # (4, D, H, W)
        img = filter_modalities(img, MODALITIES, SELECTED_MODALITIES)

        mask = np.load(os.path.join(self.mask_dir, fname)).astype(np.uint8) # (D, H, W, 4)
        mask = np.argmax(mask, axis=-1).astype(np.int64)  # -> (D,H,W)
        return torch.from_numpy(img), torch.from_numpy(mask)


train_img_dir = f"{SPLIT_DIR}/train/volumes"
train_mask_dir = f"{SPLIT_DIR}/train/masks"
val_img_dir = f"{SPLIT_DIR}/val/volumes"
val_mask_dir = f"{SPLIT_DIR}/val/masks"


os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(train_mask_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_mask_dir, exist_ok=True)

train_files = sorted(os.listdir(train_img_dir))
val_files = sorted(os.listdir(val_img_dir)) 

dataset_train = BraTSDataset(train_img_dir, train_mask_dir, train_files)
dataset_val = BraTSDataset(val_img_dir, val_mask_dir, val_files)

# -----------------------------
# Main Program
# -----------------------------
if __name__ == '__main__':
    process_and_save(RAW_TRAIN)
    split_data()
