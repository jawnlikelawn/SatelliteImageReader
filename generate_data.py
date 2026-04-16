#!/usr/bin/env python3
"""
generate_data.py — Synthesize training patches for alien ship detection
Requires: pip install numpy pillow
Usage: python generate_data.py
"""

import math, random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

random.seed(42)
np.random.seed(42)

OUT_DIR    = Path(__file__).parent / "TRAINING_DATA"
PATCH_SIZE = 64
N_SHIP     = 1000
N_NOT_SHIP = 1000

(OUT_DIR / "ship").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "not_ship").mkdir(parents=True, exist_ok=True)


def make_background(size=PATCH_SIZE):
    """Black background with subtle noise and occasional gradient."""
    bg = np.random.normal(loc=0, scale=random.uniform(2, 6), size=(size, size))

    # occasionally add a faint gradient
    if random.random() < 0.4:
        angle = random.uniform(0, math.pi)
        strength = random.uniform(5, 20)
        for y in range(size):
            for x in range(size):
                v = (x * math.cos(angle) + y * math.sin(angle)) / size
                bg[y, x] += v * strength

    return bg.clip(0, 255)


def make_not_ship(size=PATCH_SIZE):
    """Background only — noise, gradients, and a few faint stars."""
    bg = make_background(size)

    # scatter a few faint point stars
    n_stars = random.randint(0, 5)
    for _ in range(n_stars):
        sx = random.randint(0, size-1)
        sy = random.randint(0, size-1)
        brightness = random.uniform(80, 180)
        radius = random.uniform(0.4, 1.2)
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                nx_, ny_ = sx+dx, sy+dy
                if 0 <= nx_ < size and 0 <= ny_ < size:
                    d = math.sqrt(dx*dx + dy*dy)
                    bg[ny_, nx_] = min(255, bg[ny_, nx_] +
                                      brightness * math.exp(-d**2 / (2*radius**2)))
    return bg.clip(0, 255)


def draw_ship(size=PATCH_SIZE):
    """Ship on black background with subtle noise."""
    bg = make_background(size)

    h = random.randint(28, 52)
    w = int(h * random.uniform(0.45, 0.60))

    cx = size // 2
    cy = size // 2

    angle = random.uniform(0, 360)
    rad   = math.radians(angle)

    def rot(px, py):
        dx, dy = px - cx, py - cy
        return (cx + dx*math.cos(rad) - dy*math.sin(rad),
                cy + dx*math.sin(rad) + dy*math.cos(rad))

    tip_y   = cy - h * 0.50
    notch_y = cy + h * 0.001
    hw      = w / 2

    pts = [rot(px, py) for px, py in [
        (cx,      tip_y),
        (cx + hw, cy + h * 0.25),
        (cx,      notch_y),
        (cx - hw, cy + h * 0.25),
    ]]

    brightness = random.randint(200, 255)
    pil_img = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8), mode='L')
    draw    = ImageDraw.Draw(pil_img)
    draw.polygon(pts, fill=int(brightness))
    return np.array(pil_img).astype(np.float32)


def to_png(arr, path):
    arr8 = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(np.stack([arr8, arr8, arr8], axis=-1)).save(path)


print("Generating synthetic training data...")

for i in range(N_SHIP):
    to_png(draw_ship(), OUT_DIR / "ship" / f"ship_{i:04d}.png")

for i in range(N_NOT_SHIP):
    to_png(make_not_ship(), OUT_DIR / "not_ship" / f"not_ship_{i:04d}.png")

print(f"Done. {N_SHIP} ship + {N_NOT_SHIP} not_ship → {OUT_DIR}")

# preview
preview = Image.new('RGB', (PATCH_SIZE * 8, PATCH_SIZE * 2))
for i in range(8):
    arr = draw_ship()
    preview.paste(Image.fromarray(np.stack([np.clip(arr,0,255).astype(np.uint8)]*3,-1)), (i*PATCH_SIZE, 0))
    arr2 = make_not_ship()
    preview.paste(Image.fromarray(np.stack([np.clip(arr2,0,255).astype(np.uint8)]*3,-1)), (i*PATCH_SIZE, PATCH_SIZE))
preview.save(Path(__file__).parent / "preview_patches.png")
print("Preview → preview_patches.png  (top=ship, bottom=not_ship)")
print("\nNext: python train.py")