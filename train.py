#!/usr/bin/env python3
"""
train.py — Train alien ship classifier using transfer learning (ResNet18)
Requires: pip install torch torchvision pillow

Usage: python train.py
"""

import sys, copy
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, random_split
    from torchvision import models, transforms
    from PIL import Image
except ImportError:
    print("Missing dependencies. Run: pip install torch torchvision pillow")
    sys.exit(1)

TRAINING_DIR = Path(__file__).parent / "TRAINING_DATA"
MODEL_OUT    = Path(__file__).parent / "ship_model.pth"
IMG_SIZE     = 64
EPOCHS       = 40
BATCH_SIZE   = 16
LR           = 1e-4
CLASSES      = ["not_ship", "ship"]


class PatchDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples   = []
        self.transform = transform
        for label, cls in enumerate(CLASSES):
            folder = root / cls
            if not folder.exists():
                print(f"Warning: {folder} not found — run generate_data.py first")
                continue
            for f in sorted(folder.iterdir()):
                if f.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                    self.samples.append((f, label))
        n_ship     = sum(1 for _, l in self.samples if l == 1)
        n_not_ship = sum(1 for _, l in self.samples if l == 0)
        print(f"Dataset: {n_ship} ship, {n_not_ship} not_ship")
        if not self.samples:
            sys.exit(1)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model():
    model = models.resnet18(weights='DEFAULT')
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    full_ds = PatchDataset(TRAINING_DIR, transform=train_tf)
    n_val   = max(2, int(len(full_ds) * 0.15))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    # val set: no augmentation
    val_copy = PatchDataset(TRAINING_DIR, transform=val_tf)
    val_ds   = torch.utils.data.Subset(val_copy, val_ds.indices)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc, best_state = 0.0, None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                correct += (model(imgs).argmax(1) == labels).sum().item()
                total   += labels.size(0)
        acc = correct / total if total else 0
        print(f"Epoch {epoch+1:3d}/{EPOCHS}  loss={total_loss/len(train_loader):.4f}  val_acc={acc:.2%}")

        if acc >= best_acc:
            best_acc   = acc
            best_state = copy.deepcopy(model.state_dict())

    torch.save({'state_dict': best_state, 'classes': CLASSES, 'img_size': IMG_SIZE}, MODEL_OUT)
    print(f"\nBest val accuracy: {best_acc:.2%}")
    print(f"Model saved → {MODEL_OUT}")
    print("\nNext: python scan.py your_file.fits")


if __name__ == '__main__':
    train()