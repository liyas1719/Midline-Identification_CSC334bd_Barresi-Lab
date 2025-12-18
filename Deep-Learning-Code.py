#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
from pathlib import Path
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# In[2]:


DATA_ROOT = Path("/workspaces/CSC334bd-Final-Project---Midline-Identification")

TRAIN_IMG_DIR = DATA_ROOT / "images"
TRAIN_MASK_DIR = DATA_ROOT / "masks (npy)"
TEST_IMG_DIR  = DATA_ROOT / "test_images"

# In[7]:


# Parameters
IMG_SIZE = 512   # can bump to 1024
BATCH_SIZE = 1   # keep small to start (was using 8 in Jupyter, but decreased here)
NUM_EPOCHS = 5   # short “toy” training; they can increase

# For demo purposes, let's train on a small number of images (so that you can run the notebook)
NUM_TRAIN_SAMPLES = 100   # try 100, 200, 500, etc. as more files are made


# In[8]:


train_image_paths = sorted(glob.glob(str(TRAIN_IMG_DIR / "*.jpg")))
print("Train images found:", len(train_image_paths))

train_mask_paths = sorted(glob.glob(str(TRAIN_MASK_DIR / "*.npy")))
print("Train masks found:", len(train_mask_paths))


# In[9]:


mask_by_id = {
    Path(p).stem: p
    for p in glob.glob(str(TRAIN_MASK_DIR / "*.npy"))
}

list(mask_by_id.items())[:3]


# In[10]:


# Sample from the images to speed up demo

# pick a random subset of indices
subset_indices = torch.randperm(len(train_image_paths))[:NUM_TRAIN_SAMPLES]

# wrap in a Subset
train_ds_small = Subset(train_image_paths, subset_indices)

# remake the DataLoader (reuse your old batch_size / num_workers)
train_loader = DataLoader(
    train_ds_small,
    batch_size=BATCH_SIZE,   # put your old batch size here
    shuffle=True,
    num_workers=0, #catered to github running, used 4 in Jupyter
    pin_memory=True,
)


# In[11]:


import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch
class ForgeryDataset(Dataset):
    def __init__(self, image_paths, mask_by_id, img_size=256, augment=True):
        self.image_paths = image_paths
        self.mask_by_id = mask_by_id
        self.img_size = img_size
        self.augment = augment

        self.img_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

        # We'll still use PIL Resize for masks
        self.mask_resize = T.Resize((img_size, img_size), interpolation=Image.NEAREST)

    def __len__(self):
        return len(self.image_paths) * (4 if self.augment else 1)

    def __getitem__(self, idx):
        if self.augment:
            img_idx = idx // 4
            aug_id = idx % 4
        else:
            img_idx = idx
            aug_id = 0

        img_path = self.image_paths[img_idx]
        case_id = Path(img_path).stem

        # ---------- IMAGE ----------
        image = Image.open(img_path).convert("RGB")
        image = self.img_transform(image)

        # ---------- MASK ----------
        if case_id in self.mask_by_id:
            npy_path = self.mask_by_id[case_id]
            mask_np = np.load(npy_path)
            #print(mask_np) # e.g. (2, H, W) or (1, 1, 648) etc.

            # Ensure we end up with a 2D array (H, W)
            if mask_np.ndim == 3:
                # Common case: (2, H, W) or (1, H, W)
                if mask_np.shape[0] <= 2:
                    # take first channel as "forgery" mask
                    mask_np = mask_np[0]         # -> (H, W) or (1, W)
                elif mask_np.shape[-1] <= 2:
                    mask_np = mask_np[..., 0]
                else:
                    # fall back: combine channels
                    mask_np = mask_np.max(axis=0)

            # Remove any extra singleton dims, e.g. (1, 648) -> (648,)
            mask_np = np.squeeze(mask_np)

            # If it's 1D at this point, treat as a 1xN mask
            if mask_np.ndim == 1:
                mask_np = mask_np.reshape(1, -1)

            # Binarize and convert to 0–255
            mask_np = (mask_np > 0).astype("uint8") * 255

            # Now PIL is happy: 2D array = grayscale image
            mask_img = Image.fromarray(mask_np)
            mask_img = self.mask_resize(mask_img)
            mask = T.ToTensor()(mask_img)[0]   # (H, W), float in [0, 1]

            # Convert binary line → Gaussian heatmap
            from scipy.ndimage import distance_transform_edt

            dist = distance_transform_edt(1 - mask.numpy())
            sigma = 3.0  # try 2–5
            heatmap = np.exp(-(dist ** 2) / (2 * sigma ** 2))

            mask = torch.from_numpy(heatmap).unsqueeze(0).float()  # (1, H, W)

           # mask = T.ToTensor()(mask_img)         # (1, H, W)
            #mask = (mask > 0.5).float()
        else:
            mask = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)

        # ---------- AUGMENTATION ----------
        #currently augments each image 3 times, this can be increased and other augmentations can be introduced
        if self.augment:
            if aug_id == 1:
                # Horizontal flip
                image = torch.flip(image, dims=[2])
                mask = torch.flip(mask, dims=[2])
            elif aug_id == 2:
                # Vertical flip
                image = torch.flip(image, dims=[1])
                mask = torch.flip(mask, dims=[1])
            elif aug_id == 3:
                # 180-degree rotation
                image = torch.rot90(image, k=2, dims=[1, 2])
                mask = torch.rot90(mask, k=2, dims=[1, 2])


        return image, mask, case_id


# In[12]:


# Make sure things look OK

import random, csv

some_case = random.choice(list(mask_by_id.keys()))
m = np.load(mask_by_id[some_case])
print("case_id:", some_case)
print("mask shape:", m.shape, "dtype:", m.dtype)
print("min/max:", m.min(), m.max())
#print(sum(m))


# In[13]:


# simple split – could also stratify based on "has mask"
n_total = len(train_image_paths)
n_train = int(0.8 * n_total)

train_ds = ForgeryDataset(train_image_paths[:n_train], mask_by_id, IMG_SIZE, augment=True)
val_ds   = ForgeryDataset(train_image_paths[n_train:], mask_by_id, IMG_SIZE, augment=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) #catered to GitHub, was using 2 num_workers
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) #catered to GitHub, was using 2 num_workers

len(train_ds), len(val_ds)


# In[15]:


import matplotlib.pyplot as plt
for i in range(20):
    img, mask, cid = train_ds[i]

    img_np  = img.permute(1, 2, 0).numpy()   # (H, W, 3)
    mask_np = mask[0].numpy()                # (H, W)

    plt.figure(figsize=(4, 4))
    plt.imshow(img_np)
    plt.imshow(mask_np, cmap="Reds", alpha=0.4)
    plt.title(f"{cid} — Image + Mask")
    plt.axis("off")
    plt.show()


# In[16]:


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class SmallUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # ↓↓↓ all channel sizes halved ↓↓↓
        self.down1 = DoubleConv(in_channels, 8)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(8, 16)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(16, 32)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(32, 64)

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(64, 32)
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(32, 16)
        self.up1 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(16, 8)

        self.final = nn.Conv2d(8, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))

        x4 = self.bottleneck(self.pool3(x3))

        # Decoder
        x = self.up3(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)

        logits = self.final(x)   # (B,1,H,W)
        return logits



# In[17]:

criterion = torch.nn.MSELoss()
model = SmallUNet(in_channels=3, out_channels=1).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def weighted_mse(pred, target, weight_factor=50.0):
    """
    pred: (B, 1, H, W) network output
    target: (B, 1, H, W) ground truth heatmap
    weight_factor: how much to boost line pixels
    """
    weight = torch.ones_like(target)
    weight[target > 0] = weight_factor  # boost line pixels
    return ((pred - target)**2 * weight).mean()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# In[18]:


def iou(preds, targets, thresh=0.5, eps=1e-6):
    """
    preds, targets: (B,1,H,W) tensors with probs / 0-1 masks
    """
    preds_bin = (preds > thresh).float()
    targets_bin = (targets > 0.5).float()

    intersection = (preds_bin * targets_bin).sum(dim=(1,2,3))
    union = (preds_bin + targets_bin - preds_bin * targets_bin).sum(dim=(1,2,3)) + eps
    iou = (intersection + eps) / union
    return iou.mean()


# In[19]:


# These parameters will make it go faster, but perform worse than usuing all the data
MAX_TRAIN_BATCHES = 10   # cap batches per epoch
MAX_VAL_BATCHES   = 10


# In[20]:


from tqdm import tqdm  # works best in Jupyter

def train_one_epoch(model, loader, optimizer, max_batches=None):
    """
    Train for a single epoch.

    max_batches:
        - None  => use ALL batches in loader
        - int k => use at most k batches (handy for fast debug runs)
    """
    model.train()
    running_loss = 0.0
    running_iou  = 0.0
    total_samples = 0

    num_steps = len(loader) if max_batches is None else min(len(loader), max_batches)

    pbar = tqdm(loader, desc="Train", total=num_steps, leave=True)

    for batch_idx, batch in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images, masks = batch[:2]
        images = images.to(device)
        masks  = masks.to(device)
        bs = images.size(0)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        # ADD: monitor collapse vs learning
        with torch.no_grad():
            # inference
            probs = logits[0, 0].cpu().numpy()
            print(
                "loss:", loss.item(),
                "pred mean:", probs.mean().item(),
                "pred std:", probs.std().item()
            )

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(outputs)
            batch_iou = iou(probs, masks)

        running_loss += loss.item() * bs
        running_iou  += batch_iou * bs
        total_samples += bs

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    if total_samples == 0:
        return 0.0, 0.0

    avg_loss = running_loss / total_samples
    avg_iou  = running_iou / total_samples
    return avg_loss, avg_iou


def eval_one_epoch(model, loader, max_batches=None):
    """
    Evaluate on a dataset.

    max_batches:
        - None  => use ALL batches in loader (this is what you want for "all 48")
        - int k => use at most k batches (for fast sanity checks)
    """
    model.eval()
    running_loss = 0.0
    running_iou  = 0.0
    total_samples = 0

    num_steps = len(loader) if max_batches is None else min(len(loader), max_batches)

    pbar = tqdm(loader, desc="Val", total=num_steps, leave=False)

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images, masks = batch[:2]
            images = images.to(device)
            masks  = masks.to(device)
            bs = images.size(0)

            outputs = model(images)
            loss = weighted_mse(output, mask, weight_factor=50.0)

            #loss = criterion(outputs, masks)
            probs = torch.sigmoid(outputs)
            batch_iou = iou(probs, masks)

            running_loss += loss.item() * bs
            running_iou  += batch_iou * bs
            total_samples += bs

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    if total_samples == 0:
        return 0.0, 0.0

    avg_loss = running_loss / total_samples
    avg_iou  = running_iou / total_samples
    return avg_loss, avg_iou


# In[21]:


#chatgpt version
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, max_batches=None, weight_factor=50.0):
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    total_samples = 0

    num_steps = len(loader) if max_batches is None else min(len(loader), max_batches)
    pbar = tqdm(loader, desc="Train", total=num_steps, leave=True)

    for batch_idx, batch in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images, masks = batch[:2]
        images = images.to(device)
        masks  = masks.to(device)
        bs = images.size(0)

        optimizer.zero_grad()
        outputs = model(images)
        batch_iou = iou((outputs>0.5).float(), masks)  # thresholded outputs
        running_iou += batch_iou * bs


        # Weighted MSE
        loss = weighted_mse(outputs, masks, weight_factor=weight_factor)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * bs
        total_samples += bs

        # Optional: monitor collapse
        with torch.no_grad():
            pred = outputs[0, 0].cpu().numpy()
            print(f"loss: {loss.item():.4f}, pred mean: {pred.mean():.4f}, pred std: {pred.std():.4f}")

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    avg_iou = running_iou / total_samples
    return avg_loss, avg_iou


def eval_one_epoch(model, loader, max_batches=None, weight_factor=50.0):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    total_samples = 0

    num_steps = len(loader) if max_batches is None else min(len(loader), max_batches)
    pbar = tqdm(loader, desc="Val", total=num_steps, leave=False)

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images, masks = batch[:2]
            images = images.to(device)
            masks  = masks.to(device)
            bs = images.size(0)

            outputs = model(images)
            # After outputs are computed
# probs = outputs[0,0]  # continuous regression
# threshold if you want a binary mask
            mask_pred = (outputs > 0.5).float()  # optional for IoU
            batch_iou = iou(mask_pred, masks)
            running_iou += batch_iou * bs

            loss = weighted_mse(outputs, masks, weight_factor=weight_factor)

            running_loss += loss.item() * bs
            total_samples += bs

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    avg_iou = running_iou / total_samples
    return avg_loss, avg_iou


# In[22]:


for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
    train_loss, train_iou = train_one_epoch(model, train_loader, optimizer)
    val_loss, val_iou = eval_one_epoch(model, val_loader)

    print(
        f"Epoch {epoch:02d} "
        f"| train_loss={train_loss:.4f}, train_iou={train_iou:.4f} "
        f"| val_loss={val_loss:.4f}, val_iou={val_iou:.4f}"
    )


# In[23]:


model.eval()
with torch.no_grad():
    for batch_idx, batch in enumerate(val_loader):
        images, masks = batch[:2]  # take first two elements
        images = images.to(device)
        outputs = model(images)

        img = images[0].cpu().permute(1,2,0).numpy()
        pred = outputs[0,0].cpu().numpy()
        target = masks[0,0].cpu().numpy()

        plt.figure(figsize=(8,4))
        plt.imshow(img)
        plt.imshow(pred, cmap='Reds', alpha=0.5)
        plt.imshow(target, cmap='Greens', alpha=0.3)
        plt.axis('off')
        plt.show()

        if batch_idx >= 5:
            break


# In[24]:


# weighted_mse defined as above

def train_one_epoch(model, loader, optimizer, max_batches=None):
    model.train()
    running_loss = 0.0
    total_samples = 0
    num_steps = len(loader) if max_batches is None else min(len(loader), max_batches)
    pbar = tqdm(loader, desc="Train", total=num_steps)

    for batch_idx, batch in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images, masks = batch[:2]
        images = images.to(device)
        masks  = masks.to(device)
        bs = images.size(0)

        optimizer.zero_grad()
        outputs = model(images)
        loss = mse(pred, target)   # no sigmoid

        #loss = weighted_mse(outputs, masks, weight_factor=50.0)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * bs
        total_samples += bs
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / total_samples
    return avg_loss


# In[25]:


def eval_one_epoch(model, loader, max_batches=None):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    num_steps = len(loader) if max_batches is None else min(len(loader), max_batches)
    pbar = tqdm(loader, desc="Val", total=num_steps, leave=False)

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images, masks = batch[:2]
            images = images.to(device)
            masks  = masks.to(device)
            bs = images.size(0)

            outputs = model(images)
            loss = weighted_mse(outputs, masks, weight_factor=50.0)

            running_loss += loss.item() * bs
            total_samples += bs
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / total_samples
    return avg_loss


# In[26]:


print("Dataset length:", len(train_ds))


# In[27]:


class ForgeryTestDataset(Dataset):
    def __init__(self, image_dir, img_size=256):
        self.image_paths = sorted(glob.glob(str(image_dir / "*.png")))
        self.img_size = img_size
        self.img_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        case_id = Path(img_path).stem
        img = Image.open(img_path).convert("RGB")
        img = self.img_transform(img)
        return img, case_id


# In[28]:


from torch.utils.data import DataLoader

# Create test dataset & dataloader from all PNGs in TEST_IMG_DIR
test_ds = ForgeryTestDataset(TEST_IMG_DIR, IMG_SIZE)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

print("Number of test images:", len(test_ds))
print("Example test path:", test_ds.image_paths[0] if len(test_ds) > 0 else "None")


# In[29]:


model.eval()
pred_annotations = []

with torch.no_grad():
    for img, case_id in test_loader:
        img = img.to(device)
        logits = model(img)
        probs = logits[0, 0].cpu().numpy()

        #probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

        # Simple threshold
        mask_pred = (probs > 0.5).astype(np.uint8)

        # Heuristic: if almost no positive pixels, treat as authentic
       # if mask_pred.sum() < 10:  # students can tweak this
        #    annotation = "authentic"
        #else:
         #   annotation = mask_to_rle(mask_pred)

        # case_id is a list of strings because batch_size=1 => take [0]
        #pred_annotations.append((case_id[0]))

# For the local small dataset, we can just build submission directly:
#submission = pd.DataFrame(pred_annotations, columns=["case_id", "annotation"])
#submission.head()


# In[30]:


import numpy as np
from scipy.ndimage import distance_transform_edt

def line_mask_to_gaussian(mask, sigma=3):
    """
    mask: (H, W) with 1 on symmetry line, 0 elsewhere
    returns: float32 heatmap in [0, 1]
    """
    dist = distance_transform_edt(1 - mask)
    heatmap = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    return heatmap.astype(np.float32)


# In[31]:


import matplotlib.pyplot as plt

model.eval()

with torch.no_grad():
    for img, case_id in test_loader:
        img = img.to(device)

        # --- INFERENCE: forward pass only ---
        logits = model(img)                # (B, 1, H, W)
        pred = logits[0, 0].cpu().numpy()  # continuous regression output

        # Debug: check dynamic range
        print(f"{case_id[0]} → min={pred.min():.3f}, max={pred.max():.3f}")

        # Convert image for plotting
        img_np = img[0].permute(1, 2, 0).cpu().numpy()

        # --- Visualization ---
        plt.figure(figsize=(5, 5))
        plt.imshow(img_np)
        plt.imshow(pred, cmap="magma", alpha=0.6)
        #plt.contour(pred, levels=[pred.max() * 0.8], colors="cyan", linewidths=2)
        plt.colorbar(label="Predicted symmetry field")
        plt.title(f"{case_id[0]} — Inference Output")
        plt.axis("off")
        plt.show()
        plt.contour(pred, levels=[0.0], colors="cyan", linewidths=2)



# In[33]:


plt.imshow(pred, cmap="magma")
plt.colorbar()
plt.title("Predicted symmetry field")
plt.show()
plt.contour(pred, levels=[pred.min() + 0.03], colors="cyan")
print("hello")
print(pred.min(), pred.max())
#heatmap = exp(-(dist**2) / (2 * sigma**2))
# mask: 1 on symmetry line, 0 elsewhere



# In[34]:


plt.imshow(probs, cmap="Reds", alpha=0.5)
plt.colorbar()

plt.imshow(pred, cmap='hot')
plt.title('Predicted Heatmap')
plt.colorbar()
plt.show()


# In[35]:


import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        # Take only image and mask (ignore extra items like case_id if present)
        img, *rest = batch
        img = img.to(device)

        # Forward pass
        output = model(img)  # shape: (B, 1, H, W)
        pred = output[0,0].cpu().numpy()  # first image in batch, channel 0

        # Convert image for plotting
        img_np = img[0].permute(1,2,0).cpu().numpy()  # (H, W, C)

        # Optional: if you have case_id in rest


# In[36]:


print("probs min/max/mean/std:", probs.min(), probs.max(), probs.mean(), probs.std())


# In[37]:


import matplotlib.pyplot as plt
from PIL import Image

model.eval()
pred_annotations = []

with torch.no_grad():
    for img, case_id in test_loader:
        img = img.to(device)
        logits = model(img)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

        print(f"Probs: {probs}")  # Check the probability output
        mask_pred = (probs > 0.5).astype(np.uint8)

        print(f"Mask sum: {mask_pred.sum()}")  # Check how many positive pixels

        # Plot the predicted mask if it's not blank
        if mask_pred.sum() > 0:
            plt.figure(figsize=(6,6))
            plt.imshow(mask_pred, cmap='gray')  # Display as grayscale
            plt.title(f"Predicted Mask for Case {case_id[0]}")
            plt.axis('off')
            plt.show()
        else:
            print(f"Blank mask detected for case {case_id[0]}.")

        # Save the mask as an image file (only if non-zero)
        if mask_pred.sum() > 0:
            mask_image = Image.fromarray(mask_pred * 255)  # Convert to 0-255 range
            mask_image.save(f"predicted_mask_{case_id[0]}.png")
        else:
            print(f"Skipping save for blank mask for case {case_id[0]}.")

        # Append annotation (you may need to update this logic)
        # annotation = "authentic" or something else
        #pred_annotations.append((case_id[0], annotation))

# For the local small dataset, we can just build submission directly:
#submission = pd.DataFrame(pred_annotations, columns=["case_id", "annotation"])
#submission.head()


# In[39]:


image, mask, case_id = train_ds[0]

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("Image")
plt.imshow(image.permute(1,2,0))
plt.axis("off")

plt.subplot(1,2,2)
plt.title("GT mask")
plt.imshow(mask[0], cmap="hot")
plt.colorbar()
plt.axis("off")
plt.show()

print("mask min/max:", mask.min().item(), mask.max().item())
print("mask mean:", mask.mean().item())


# In[285]:


model.train()
img, mask, _ = train_ds[10]
img = img.unsqueeze(0).to(device)
mask = mask.unsqueeze(0).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for i in range(100):
    optimizer.zero_grad()
    output = model(img)
    loss = weighted_mse(output, mask, weight_factor=50.0)
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(f"iter {i}, loss: {loss.item()}")
        plt.imshow(output[0,0].detach().cpu(), cmap="hot")
        plt.show()


# In[40]:


print("min:", probs.min(), "max:", probs.max())

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(probs, cmap="hot")
plt.title("Heatmap (auto-scaled)")
plt.colorbar()

plt.subplot(1,3,2)
plt.imshow(probs > 0.5)
plt.title("Threshold @ 0.5")

plt.subplot(1,3,3)
plt.imshow(probs > 0.3)
plt.title("Threshold @ 0.3")

plt.show()


# In[216]:


total = 0
fg = 0
for _, mask, _ in train_ds:
    fg += mask.sum().item()
    total += mask.numel()

print("Foreground %:", fg / total)


# In[217]:


train_ds = Subset(train_ds, [0])  # pick 1 image
# train until loss ~0


# In[81]:


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# Pick up to 5 images
subset_indices = list(range(min(5, len(train_ds))))

images, masks, case_ids = [], [], []
for i in subset_indices:
    img, mask, cid = train_ds[i]
    images.append(img)
    masks.append(mask)
    case_ids.append(cid)

# Stack into batch
imgs = torch.stack(images).to(device)   # [batch, C, H, W]
msks = torch.stack(masks).to(device)    # [batch, 1, H, W]

# Weighted MSE for thin line emphasis
def weighted_mse(pred, target, weight_factor=50.0):
    weight = 1 + (weight_factor - 1) * target
    return ((pred - target) ** 2 * weight).mean()

model = model.to(device)
model.train()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_iters = 1000

for i in range(num_iters):
    optimizer.zero_grad()
    output = model(imgs)
    loss = weighted_mse(output, msks)
    loss.backward()
    optimizer.step()

    if i % 100 == 0 or i == num_iters-1:
        print(f"Iter {i}, loss: {loss.item():.6f}")
        plt.figure(figsize=(15,3))
        for j in range(len(subset_indices)):
            plt.subplot(1,len(subset_indices), j+1)
            plt.imshow(output[j,0].detach().cpu(), cmap="hot")
            plt.title(case_ids[j])
            plt.axis("off")
        plt.show()


# In[220]:


import torch
import matplotlib.pyplot as plt

# Make sure model is in training mode
model.train()

# Get one image and mask
img, mask, _ = train_ds[0]

# Add batch dimension and send to device
img = img.unsqueeze(0).to(device)
mask = mask.unsqueeze(0).to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Safe weight factor
weight_factor = 10.0  # start smaller to avoid gradient explosion

# Training loop
for i in range(1000):
    optimizer.zero_grad()
    
    # Forward pass
    output = model(img)
    
    # Compute weighted MSE
    loss = weighted_mse(output, mask, weight_factor=weight_factor)
    
    # Check for NaNs
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"Iteration {i}: Loss is NaN or Inf. Stopping training.")
        break



# In[82]:


import matplotlib.pyplot as plt
from PIL import Image

model.eval()
pred_annotations = []

with torch.no_grad():
    for img, case_id in test_loader:
        img = img.to(device)
        logits = model(img)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

        print(f"Probs: {probs}")  # Check the probability output
        mask_pred = (probs > 0.5).astype(np.uint8)

        print(f"Mask sum: {mask_pred.sum()}")  # Check how many positive pixels

        # Plot the predicted mask if it's not blank
        if mask_pred.sum() > 0:
            plt.figure(figsize=(6,6))
            plt.imshow(mask_pred, cmap='gray')  # Display as grayscale
            plt.title(f"Predicted Mask for Case {case_id[0]}")
            plt.axis('off')
            plt.show()
        else:
            print(f"Blank mask detected for case {case_id[0]}.")

        # Save the mask as an image file (only if non-zero)
        if mask_pred.sum() > 0:
            mask_image = Image.fromarray(mask_pred * 255)  # Convert to 0-255 range
            mask_image.save(f"predicted_mask_{case_id[0]}.png")
        else:
            print(f"Skipping save for blank mask for case {case_id[0]}.")

        # Append annotation (you may need to update this logic)
        # annotation = "authentic" or something else
        #pred_annotations.append((case_id[0], annotation))

# For the local small dataset, we can just build submission directly:
#submission = pd.DataFrame(pred_annotations, columns=["case_id", "annotation"])
#submission.head()


# In[ ]:




