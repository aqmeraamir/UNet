
'''

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


import os
import random
from unet import UNet3D, save_checkpoint, load_checkpoint, segment
from dataset import dataset_train, dataset_val

import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------------
# DEFAULT SETTINGS & PATHS
# -------------------------

# dataset paths
RAW_TRAIN = 'raw_data/MICCAI_BraTS2020_TrainingData'
RAW_VAL   = 'raw_data/MICCAI_BraTS2020_ValidationData'
SPLIT_DIR = 'data/input_data_split'

BATCH_SIZE = 2
NUM_CLASSES = 4

# training
PREFERRED_DEVICE = 'cuda'
SAMPLE_EVERY = 1
EPOCHS = 100
LEARNING_RATE = 1e-4
RUN_NAME = "unet3d_brats_3"
LOAD_MODEL = True
CKPT_PATH = f"runs/{RUN_NAME}/models/ckpt_epoch53.pt"


    

# -------------------------
# LOSSES: Dice + Focal
# -------------------------
def dice_loss(pred, target, eps=1e-6):
    pred = F.softmax(pred, dim=1)
    one_hot = F.one_hot(target, NUM_CLASSES).permute(0,4,1,2,3).float()
    inter = torch.sum(pred * one_hot)
    union = torch.sum(pred) + torch.sum(one_hot)
    return 1 - (2*inter + eps) / (union + eps)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, pred, target):
        logp = F.log_softmax(pred, dim=1)
        p = torch.exp(logp)
        one_hot = F.one_hot(target, NUM_CLASSES).permute(0,4,1,2,3)
        loss = -one_hot * ((1-p)**self.gamma) * logp
        return loss.sum() / torch.numel(target)
    




def sample_and_visualize_predictions(model, dataset, run_name, device, epoch, num_samples=3):

    model.eval()
    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        img, mask = dataset[idx]
        pred = segment(img, model, device)
        slice_idx = random.randint(0, 63)

        flair = img[0, :, :, slice_idx]
        ground_truth = mask[:, :, slice_idx]
        prediction = pred[:, :, slice_idx]

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1); plt.imshow(flair, cmap='gray'); plt.title('FLAIR')
        plt.subplot(1, 3, 2); plt.imshow(ground_truth, cmap='jet'); plt.title('Ground Truth')
        plt.subplot(1, 3, 3); plt.imshow(prediction, cmap='jet'); plt.title('Prediction')
        plt.tight_layout()
        out_path = f"runs/{run_name}/results/sample_epoch{epoch+1}_img{i}.png"
        plt.savefig(out_path)
        plt.close()


# -------------------------
# TRAINING
# -------------------------

def setup_run_dirs(run_name):
    os.makedirs(f"runs/{run_name}/models", exist_ok=True)
    os.makedirs(f"runs/{run_name}/results", exist_ok=True)
    os.makedirs(f"runs/{run_name}/logs", exist_ok=True)

def train(model, optimizer, dl_train, dl_val, dataset_val, device, run_name, epochs=100, start_epoch=0):
    writer = SummaryWriter(f"runs/{run_name}/logs")
    criterion_dice = dice_loss
    criterion_focal = FocalLoss()
    
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        for volumes, masks in tqdm(dl_train, desc=f"Epoch {epoch+1}/{epochs}: "):
            volumes, masks = volumes.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(volumes)
            loss = criterion_dice(outputs, masks) + criterion_focal(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(dl_train)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        print(f"Train Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for volumes, masks in dl_val:
                volumes, masks = volumes.to(device), masks.to(device)
                outputs = model(volumes)
                loss = criterion_dice(outputs, masks) + criterion_focal(outputs, masks)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(dl_val)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        print(f"Val Loss: {avg_val_loss:.4f}\n")

        # Save checkpoint
        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1
        }
        ckpt_path = f"runs/{run_name}/models/ckpt_epoch{epoch+1}.pt"
        save_checkpoint(state, ckpt_path)

        # Sample visualization every N epochs
        if (epoch + 1) % SAMPLE_EVERY == 0 or (epoch + 1) == epochs:
            sample_and_visualize_predictions(model, dataset_val, run_name, device, epoch)


# -------------------------
# MAIN PROGRAM
# -------------------------
if __name__ == '__main__':
    from torch.multiprocessing import freeze_support
    freeze_support()

    torch.backends.cudnn.benchmark = True
    DEVICE = torch.device(PREFERRED_DEVICE)



    dl_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True) # num_workers=4, pin_memory not enabled
    dl_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = UNet3D()

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    setup_run_dirs(RUN_NAME)
    start_epoch = 0
    if LOAD_MODEL and os.path.exists(CKPT_PATH):
        start_epoch = load_checkpoint(CKPT_PATH, model, optimizer)

    train(model, optimizer, dl_train, dl_val, dataset_val, DEVICE, RUN_NAME, epochs=EPOCHS, start_epoch=start_epoch)
