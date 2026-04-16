#!/usr/bin/env python3
"""
generate_data.py — Synthesize training patches for alien ship detection
Requires: pip install numpy pillow
Usage: python generate_data.py
"""

import os, math, random
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


def make_starfield(size=PATCH_SIZE):
    style = random.choice(['sparse', 'sparse', 'dense', 'cluster', 'cluster',
                           'nebula', 'gradient', 'noisy', 'very_dense'])

    if style == 'sparse':
        bg = np.random.normal(loc=4, scale=2, size=(size, size)).clip(0, 255)
        n_stars = random.randint(2, 8)

    elif style == 'dense':
        bg = np.random.normal(loc=8, scale=4, size=(size, size)).clip(0, 255)
        n_stars = random.randint(15, 40)

    elif style == 'cluster':
        bg = np.random.normal(loc=5, scale=3, size=(size, size)).clip(0, 255)
        n_stars = random.randint(30, 70)

    elif style == 'very_dense':
        bg = np.random.normal(loc=10, scale=5, size=(size, size)).clip(0, 255)
        n_stars = random.randint(60, 100)

    elif style == 'nebula':
        bg = np.zeros((size, size))
        for _ in range(random.randint(2, 4)):
            cx = random.randint(0, size)
            cy = random.randint(0, size)
            r  = random.uniform(8, 28)
            br = random.uniform(20, 70)
            for y in range(size):
                for x in range(size):
                    d = math.sqrt((x-cx)**2 + (y-cy)**2)
                    bg[y, x] += br * math.exp(-d**2 / (2*r**2))
        bg += np.random.normal(0, 3, (size, size))
        bg = bg.clip(0, 255)
        n_stars = random.randint(3, 15)

    elif style == 'gradient':
        bg = np.zeros((size, size))
        angle = random.uniform(0, math.pi)
        strength = random.uniform(30, 90)
        for y in range(size):
            for x in range(size):
                v = (x * math.cos(angle) + y * math.sin(angle)) / size
                bg[y, x] = v * strength
        bg += np.random.normal(0, 4, (size, size))
        bg = bg.clip(0, 255)
        n_stars = random.randint(5, 20)

    else:  # noisy
        bg = np.random.normal(loc=15, scale=10, size=(size, size)).clip(0, 255)
        n_stars = random.randint(5, 20)

    # scatter stars
    for _ in range(n_stars):
        if style in ('cluster', 'very_dense'):
            sx = int(np.random.normal(size/2, size/6))
            sy = int(np.random.normal(size/2, size/6))
        else:
            sx = random.randint(0, size-1)
            sy = random.randint(0, size-1)
        sx = max(0, min(size-1, sx))
        sy = max(0, min(size-1, sy))
        brightness = random.uniform(150, 255)
        radius     = random.uniform(0.3, 1.8)
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                nx_, ny_ = sx+dx, sy+dy
                if 0 <= nx_ < size and 0 <= ny_ < size:
                    d = math.sqrt(dx*dx + dy*dy)
                    bg[ny_, nx_] = min(255, bg[ny_, nx_] +
                                      brightness * math.exp(-d**2 / (2*radius**2)))
    return bg


def draw_ship(bg):
    img  = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)).convert('RGBA')
    size = img.size[0]

    h = random.randint(28, 52)
    w = int(h * random.uniform(0.45, 0.60))

    # always centre the ship so window centre = ship centre during inference
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

    pts_local = [
        (cx,       tip_y),
        (cx + hw,  cy + h * 0.25),
        (cx,       notch_y),
        (cx - hw,  cy + h * 0.25),
    ]
    pts = [rot(px, py) for px, py in pts_local]

    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    brightness = random.randint(200, 255)
    draw.polygon(pts, fill=(brightness, brightness, brightness, 255))
    draw.polygon(pts, outline=(brightness-30, brightness-30, brightness-30, 180))

    out = Image.alpha_composite(img, overlay).convert('RGB')
    return np.array(out).mean(axis=2)


def to_png(arr, path):
    arr8 = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(np.stack([arr8, arr8, arr8], axis=-1)).save(path)


print("Generating synthetic training data...")
print(f"Backgrounds: sparse, dense, cluster, very_dense, nebula, gradient, noisy")

for i in range(N_SHIP):
    to_png(draw_ship(make_starfield()), OUT_DIR / "ship" / f"ship_{i:04d}.png")

for i in range(N_NOT_SHIP):
    to_png(make_starfield(), OUT_DIR / "not_ship" / f"not_ship_{i:04d}.png")

print(f"Done. {N_SHIP} ship + {N_NOT_SHIP} not_ship patches → {OUT_DIR}")

# save preview
preview = Image.new('RGB', (PATCH_SIZE * 8, PATCH_SIZE * 2))
for i in range(8):
    bg  = make_starfield()
    arr = draw_ship(bg)
    preview.paste(Image.fromarray(np.stack([np.clip(arr,0,255).astype(np.uint8)]*3,-1)), (i*PATCH_SIZE, 0))
    bg2 = make_starfield()
    preview.paste(Image.fromarray(np.stack([np.clip(bg2,0,255).astype(np.uint8)]*3,-1)), (i*PATCH_SIZE, PATCH_SIZE))
preview.save(Path(__file__).parent / "preview_patches.png")
print("Preview saved → preview_patches.png  (top=ship, bottom=background)")
print("\nNext: python train.py")