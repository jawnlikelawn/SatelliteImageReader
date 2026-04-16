#!/usr/bin/env python3
"""
scan.py — Scan a FITS or image file for alien ships using trained model
Requires: pip install torch torchvision pillow numpy

Usage:
  python scan.py                          # scans FILES_TO_SCAN/ folder
  python scan.py myfile.fits              # scan single file
  python scan.py --threshold 0.95        # adjust sensitivity
  python scan.py --save-patches          # save detection crops to PATCHES/
"""

import sys, argparse, gzip, math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    import torch
    from torchvision import models, transforms
    import torch.nn as nn
except ImportError:
    print("Missing dependencies. Run: pip install torch torchvision pillow numpy")
    sys.exit(1)

MODEL_PATH = Path(__file__).parent / "ship_model.pth"
PATCH_SIZE = 64
STRIDE     = 16
NMS_RADIUS = 64
BLOCK      = 2880

def _parse_header(data, offset):
    cards, end = {}, False
    while not end:
        block = data[offset:offset + BLOCK]
        offset += BLOCK
        for i in range(0, BLOCK, 80):
            card = block[i:i+80].decode('ascii', errors='replace')
            key  = card[:8].strip()
            if key == 'END': end = True; break
            if '=' in card[8:10]:
                val = card[10:].split('/')[0].strip().strip("'").strip()
                cards[key] = val
    return cards, offset

def load_fits_hdu1(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rb') as f:
        raw = f.read()
    offset = 0
    for i in range(2):
        hdr, offset = _parse_header(raw, offset)
        bitpix = int(hdr.get('BITPIX', 0))
        naxis  = int(hdr.get('NAXIS', 0))
        shape  = [int(hdr.get(f'NAXIS{j+1}', 1)) for j in range(naxis)]
        npix   = 0 if naxis == 0 else 1
        for s in shape: npix *= s
        nbytes = npix * abs(bitpix) // 8 if (npix and bitpix) else 0
        if i == 0:
            offset += math.ceil(nbytes / BLOCK) * BLOCK; continue
        if npix == 0 or bitpix == 0 or len(shape) < 2: return None, 0, 0
        dtype = {8:'B',16:'>h',32:'>i',64:'>q',-32:'>f',-64:'>d'}[bitpix]
        arr   = np.frombuffer(raw[offset:offset+nbytes], dtype=np.dtype(dtype)).astype(np.float32)
        nx, ny = shape[0], shape[1]
        return arr[:ny*nx].reshape(ny, nx), nx, ny
    return None, 0, 0

def fits_to_uint8(arr2d):
    valid = arr2d[np.isfinite(arr2d)]
    lo, hi = np.percentile(valid, [2, 98]) if len(valid) else (0, 1)
    norm = np.clip((arr2d - lo) / (float(hi - lo) or 1.0), 0, 1)
    return (np.nan_to_num(norm, 0) * 255).astype(np.uint8)

def load_model(path, device):
    ckpt  = torch.load(path, map_location=device)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(ckpt['state_dict'])
    return model.eval().to(device), ckpt.get('img_size', 64)

infer_tf = transforms.Compose([
    transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def nms(detections, radius=NMS_RADIUS):
    detections = sorted(detections, key=lambda d: d[2], reverse=True)
    kept = []
    for x, y, conf in detections:
        if not any(math.sqrt((x-kx)**2 + (y-ky)**2) < radius for kx, ky, _ in kept):
            kept.append((x, y, conf))
    return kept

def scan(fits_path, threshold=0.95, save_patches=False):
    path = Path(fits_path)
    print(f"Loading {path.name} ...")

    if path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
        imgRGB = Image.open(path).convert('RGB')
        nx, ny = imgRGB.size
    else:
        arr2d, nx, ny = load_fits_hdu1(fits_path)
        if arr2d is None:
            print("Could not load HDU 1."); return
        imgRGB = Image.fromarray(np.stack([fits_to_uint8(arr2d)]*3, axis=-1))

    print(f"Image: {nx}×{ny}")

    if not MODEL_PATH.exists():
        print("Model not found — run: python train.py"); return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model ({device}) ...")
    model, _ = load_model(MODEL_PATH, device)

    half   = PATCH_SIZE // 2
    margin = half + 8
    xs     = range(margin, nx - margin, STRIDE)
    ys     = range(margin, ny - margin, STRIDE)
    total  = len(xs) * len(ys)
    print(f"Scanning {total:,} windows ...")

    detections, all_detections = [], []
    batch_imgs, batch_coords   = [], []
    BATCH = 128

    def run_batch():
        if not batch_imgs: return
        tensor = torch.stack([infer_tf(i) for i in batch_imgs]).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[:, 1].cpu().numpy()
        for (bx, by), conf in zip(batch_coords, probs):
            all_detections.append((bx, by, float(conf)))
            if conf >= threshold:
                detections.append((bx, by, float(conf)))
        batch_imgs.clear(); batch_coords.clear()

    done = 0
    for cy in ys:
        for cx in xs:
            patch = imgRGB.crop((cx-half, cy-half, cx+half, cy+half))
            batch_imgs.append(patch); batch_coords.append((cx, cy))
            if len(batch_imgs) >= BATCH: run_batch()
            done += 1
            if done % 5000 == 0:
                print(f"  {done:,}/{total:,}  hits so far: {len(detections)}")
    run_batch()

    detections = nms(detections)
    if len(detections) > 10:
        detections = sorted(detections, key=lambda d: -d[2])[:10]
    print(f"\nDetections above {threshold:.0%}: {len(detections)}")

    if not detections:
        print("None found — showing best candidate instead...")
        if not all_detections:
            print("No candidates at all — saving clean image.")
            scanned_dir = Path(__file__).parent / 'SCANNED'
            scanned_dir.mkdir(exist_ok=True)
            imgRGB.save(scanned_dir / (path.stem + '.detections.png'))
            return
        best = max(all_detections, key=lambda d: d[2])
        detections = [best]
        print(f"Best: x={best[0]} y={best[1]} confidence={best[2]:.1%}")

    scale   = min(1.0, 1200 / max(nx, ny))
    out_img = imgRGB.resize((int(nx*scale), int(ny*scale)), Image.LANCZOS)
    draw    = ImageDraw.Draw(out_img)
    for x, y, conf in detections:
        sx, sy = int(x*scale), int(y*scale)
        r = int(half*scale)
        draw.rectangle([sx-r, sy-r, sx+r, sy+r], outline='red', width=2)
        draw.text((sx-r, sy-r-12), f"{conf:.0%}", fill='red')

    scanned_dir = Path(__file__).parent / 'SCANNED'
    scanned_dir.mkdir(exist_ok=True)
    out_path = scanned_dir / (path.stem + '.detections.png')
    out_img.save(out_path)
    print(f"Saved → {out_path}")

    if save_patches or input('\nSave detection patches to PATCHES/? (y/n): ').strip().lower() == 'y':
        patch_dir = Path(__file__).parent / 'PATCHES'
        patch_dir.mkdir(exist_ok=True)
        for i, (x, y, conf) in enumerate(detections):
            crop = imgRGB.crop((x-half, y-half, x+half, y+half))
            crop.save(patch_dir / f"{path.stem}_{i:03d}_{conf:.0%}.png")
        print(f"Patches saved → {patch_dir}/")
        print("Sort into TRAINING_DATA/ship/ or TRAINING_DATA/not_ship/ then retrain.")

    print("\nDetections:")
    for x, y, conf in sorted(detections, key=lambda d: -d[2]):
        print(f"  x={x:4d}  y={y:4d}  confidence={conf:.1%}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fits', nargs='?', help='FITS or image file to scan (optional)')
    parser.add_argument('--threshold', type=float, default=0.95,
                        help='Confidence threshold (default 0.95)')
    parser.add_argument('--save-patches', action='store_true', dest='save_patches',
                        help='Save detection crops to PATCHES/')
    args = parser.parse_args()

    if args.fits:
        scan(args.fits, args.threshold, args.save_patches)
    else:
        scan_dir = Path(__file__).parent / 'FILES_TO_SCAN'
        exts = ('.fits','.fit','.fts','.fits.gz','.fz','.png','.jpg','.jpeg','.tif','.tiff')
        if not scan_dir.exists():
            print("No file given and FILES_TO_SCAN/ not found.")
            print("Usage: python scan.py myfile.fits")
            sys.exit(1)
        files = sorted(f for f in scan_dir.iterdir() if f.suffix.lower() in exts)
        if not files:
            print(f"No image files found in {scan_dir}"); sys.exit(1)
        print(f"Found {len(files)} file(s) in FILES_TO_SCAN/")
        for f in files:
            print(f"\n{'='*60}")
            scan(str(f), args.threshold, args.save_patches)