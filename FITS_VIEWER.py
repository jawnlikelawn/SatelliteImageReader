#!/usr/bin/env python3
"""
fits_viewer.py — lightweight FITS viewer
Requires: Python 3.6+, numpy, Pillow (pip install numpy pillow)
Falls back to slow pure-stdlib mode if unavailable.
Usage: python fits_viewer.py [file.fits]
"""

import struct, gzip, math, sys, os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import numpy as np
    from PIL import Image, ImageTk
    FAST = True
except ImportError:
    FAST = False

BLOCK = 2880

def _parse_header(data, offset):
    cards, end = {}, False
    while not end:
        block = data[offset:offset + BLOCK]
        offset += BLOCK
        for i in range(0, BLOCK, 80):
            card = block[i:i+80].decode('ascii', errors='replace')
            key = card[:8].strip()
            if key == 'END':
                end = True; break
            if '=' in card[8:10]:
                val = card[10:].split('/')[0].strip().strip("'").strip()
                cards[key] = val
    return cards, offset

def _dtype(bitpix):
    return {8:'B',16:'h',32:'i',64:'q',-32:'f',-64:'d'}[bitpix]

def read_fits(path):
    """Return list of image HDUs: {'label', 'arr', 'nx', 'ny'}"""
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rb') as f:
        raw = f.read()

    hdus, offset, idx = [], 0, 0
    while offset < len(raw) - BLOCK:
        hdr, offset = _parse_header(raw, offset)
        bitpix = int(hdr.get('BITPIX', 0))
        naxis  = int(hdr.get('NAXIS', 0))
        shape  = [int(hdr.get(f'NAXIS{i+1}', 1)) for i in range(naxis)]
        npix   = 0 if naxis == 0 else 1
        for s in shape: npix *= s
        nbytes = npix * abs(bitpix) // 8 if (npix and bitpix) else 0

        if npix and bitpix and len(shape) >= 2:
            chunk = raw[offset:offset + nbytes]
            code  = _dtype(bitpix)
            try:
                if FAST:
                    arr = np.frombuffer(chunk, dtype=np.dtype(f'>{code}')).astype(np.float32)
                else:
                    arr = list(struct.unpack(f'>{npix}{code}', chunk))
                bscale = float(hdr.get('BSCALE', 1))
                bzero  = float(hdr.get('BZERO',  0))
                if bscale != 1 or bzero != 0:
                    arr = arr * bscale + bzero if FAST else [v * bscale + bzero for v in arr]
                extname = hdr.get('EXTNAME', '') or hdr.get('XTENSION', 'PRIMARY')
                label = f'HDU {idx}  {extname}  {shape[0]}×{shape[1]}'
                hdus.append({'label': label, 'arr': arr, 'nx': shape[0], 'ny': shape[1]})
            except Exception:
                pass

        offset += math.ceil(nbytes / BLOCK) * BLOCK
        idx += 1

    return hdus

# ── renderers ─────────────────────────────────────────────────────────────────

_stretch_cache = {}

def get_stretch(arr2d):
    key = id(arr2d)
    if key not in _stretch_cache:
        valid = arr2d[np.isfinite(arr2d)]
        lo, hi = np.percentile(valid, [2, 98]) if len(valid) else (0.0, 1.0)
        _stretch_cache.clear()
        _stretch_cache[key] = (float(lo), float(hi))
    return _stretch_cache[key]

def render_fast(arr2d, width, height, zoom, cx, cy):
    ny, nx = arr2d.shape
    xs = cx + (np.arange(width)  - width  / 2) / zoom
    ys = (ny - 1) - (cy + (np.arange(height) - height / 2) / zoom)
    xi = np.clip(xs.astype(int), 0, nx - 1)
    yi = np.clip(ys.astype(int), 0, ny - 1)
    sampled = arr2d[np.ix_(yi, xi)]
    oob   = (xs < 0) | (xs >= nx)
    oob_y = (ys < 0) | (ys >= ny)
    mask  = oob[np.newaxis, :] | oob_y[:, np.newaxis]
    lo, hi = get_stretch(arr2d)
    span = float(hi - lo) or 1.0
    out = np.nan_to_num(sampled, nan=0.0, posinf=hi, neginf=lo)
    out = np.clip((out - lo) / span, 0, 1)
    out = (out * 255).astype(np.uint8)
    out[mask] = 0
    return ImageTk.PhotoImage(Image.fromarray(out, mode='L'))


def render_slow(arr, width, height, nx, ny, zoom, cx, cy):
    s = sorted(v for v in arr if v == v and v != float('inf') and v != float('-inf'))
    if not s: lo, span = 0, 1
    else:
        n = len(s)
        lo = s[max(0, int(n * 0.02))]
        hi = s[min(n-1, int(n * 0.98))]
        span = (hi - lo) or 1
    pix = []
    for py in range(height):
        row = []
        for px in range(width):
            ix = int(cx + (px - width  / 2) / zoom)
            iy = int(ny - 1 - (cy + (py - height / 2) / zoom))
            if 0 <= ix < nx and 0 <= iy < ny:
                v = arr[iy * nx + ix]
                if v != v: row.append('#000000'); continue
                g = int(max(0.0, min(1.0, (v - lo) / span)) * 255)
                row.append(f'#{g:02x}{g:02x}{g:02x}')
            else:
                row.append('#000000')
        pix.append('{' + ' '.join(row) + '}')
    return ' '.join(pix)

# ── GUI ───────────────────────────────────────────────────────────────────────

class FITSViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('FITS Viewer')
        self.configure(bg='#000')
        self.geometry('800x700')
        self.resizable(True, True)

        self.hdus   = []
        self.arr    = None
        self.arr2d  = None
        self.nx = self.ny = 1
        self.zoom   = 1.0
        self.cx = self.cy = None
        self._photo = None
        self._current_path = None

        self._build_ui()
        if len(sys.argv) > 1:
            self._load(sys.argv[1])

    def _build_ui(self):
        btn = dict(bg='#1e1e1e', fg='#aaa', relief='flat',
                   activebackground='#2e2e2e', activeforeground='#fff',
                   padx=10, pady=4, cursor='hand2', bd=0)
        tb = tk.Frame(self, bg='#141414', pady=3)
        tb.pack(fill='x')

        # file picker — scans FITS_FILES folder next to this script
        self.file_var = tk.StringVar()
        self.file_menu = ttk.Combobox(tb, textvariable=self.file_var, state='readonly', width=30)
        self.file_menu.pack(side='left', padx=6)
        self.file_menu.bind('<<ComboboxSelected>>', self._file_changed)
        self._scan_fits_folder()

        tk.Button(tb, text='Browse', command=self._open, **btn).pack(side='left')
        tk.Button(tb, text='Fit',    command=self._fit,  **btn).pack(side='left', padx=4)

        self.hdu_var = tk.StringVar()
        self.hdu_menu = ttk.Combobox(tb, textvariable=self.hdu_var, state='readonly', width=28)
        self.hdu_menu.pack(side='left', padx=8)
        self.hdu_menu.bind('<<ComboboxSelected>>', self._hdu_changed)

        tk.Button(tb, text='→ Scan Queue', command=self._send_to_scan, **btn).pack(side='left', padx=(12,0))
        tk.Button(tb, text='Export PNG',   command=self._export_png,      **btn).pack(side='left', padx=4)

        self.info = tk.Label(tb, text='', bg='#141414', fg='#444', font=('Courier', 9), anchor='e')
        self.info.pack(side='right', padx=8)

        self.canvas = tk.Canvas(self, bg='#000', cursor='crosshair', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>',  self._on_resize)
        self.canvas.bind('<MouseWheel>', self._on_scroll)
        self.canvas.bind('<Button-4>',   self._on_scroll)
        self.canvas.bind('<Button-5>',   self._on_scroll)
        self.canvas.bind('<Motion>',     self._on_motion)

    def _scan_fits_folder(self):
        """Populate file dropdown from FITS_FILES/ next to this script."""
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FITS_FILES')
        exts = ('.fits', '.fit', '.fts', '.fits.gz', '.fz')
        if os.path.isdir(folder):
            files = sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))
            paths = [os.path.join(folder, f) for f in files]
        else:
            files, paths = [], []
        self._fits_paths = paths
        self.file_menu['values'] = files if files else ['(no files in FITS_FILES/)']
        if files:
            self.file_menu.current(0)

    def _file_changed(self, _=None):
        idx = self.file_menu.current()
        if idx >= 0 and idx < len(self._fits_paths):
            self._load(self._fits_paths[idx])

    def _open(self):
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FITS_FILES')
        path = filedialog.askopenfilename(
            initialdir=folder if os.path.isdir(folder) else '.',
            filetypes=[('FITS', '*.fits *.fit *.fts *.fits.gz'), ('All', '*')])
        if path: self._load(path)

    def _load(self, path):
        self.canvas.delete('all')
        self.canvas.create_text(self.canvas.winfo_width()//2 or 400,
                                self.canvas.winfo_height()//2 or 350,
                                text='Loading…', fill='#555', font=('Courier', 14))
        self.update_idletasks()
        try:
            hdus = read_fits(path)
        except Exception as e:
            messagebox.showerror('Error', str(e)); return
        if not hdus:
            messagebox.showerror('Error', 'No image HDUs found.'); return

        self.hdus = hdus
        self.hdu_menu['values'] = [h['label'] for h in hdus]
        default = next((i for i, h in enumerate(hdus) if h['label'].startswith('HDU 1')), 0)
        self.hdu_menu.current(default)
        self._current_path = path
        self.title(f'FITS — {os.path.basename(path)}')
        self._select_hdu(self.hdu_menu.current())

    def _export_png(self):
        if self.arr is None:
            messagebox.showwarning('No image', 'Open a FITS file first.'); return
        import shutil
        base_dir = os.path.dirname(os.path.abspath(__file__))
        init_name = os.path.splitext(os.path.basename(self._current_path or 'image'))[0] + '.png'
        dst = filedialog.asksaveasfilename(
            initialdir=base_dir, initialfile=init_name,
            defaultextension='.png',
            filetypes=[('PNG', '*.png'), ('All', '*')])
        if not dst: return
        w = self.canvas.winfo_width()  or 800
        h = self.canvas.winfo_height() or 700
        cx = self.cx if self.cx is not None else self.nx / 2
        cy = self.cy if self.cy is not None else self.ny / 2
        if FAST:
            from PIL import Image as PILImage
            import numpy as np
            # export full resolution, not canvas size
            xs = cx + (np.arange(self.nx) - self.nx / 2) / self.zoom * (self.nx / self.nx)
            # just export the full native image at current stretch
            arr2d = self.arr2d
            valid = arr2d[np.isfinite(arr2d)]
            lo, hi = np.percentile(valid, [2, 98]) if len(valid) else (0, 1)
            span = float(hi - lo) or 1.0
            out = np.clip((arr2d - lo) / span, 0, 1)
            out = np.nan_to_num(out, 0)
            out = (out * 255).astype(np.uint8)
            PILImage.fromarray(out, mode='L').save(dst)
        else:
            messagebox.showerror('Export', 'numpy/pillow required for export.'); return
        messagebox.showinfo('Exported', f'Saved to:\n{dst}')

    def _send_to_scan(self):
        if not self._current_path:
            messagebox.showwarning('No file', 'Open a FITS file first.'); return
        base_dir   = os.path.dirname(os.path.abspath(__file__))
        scan_dir   = os.path.join(base_dir, 'FILES_TO_SCAN')
        os.makedirs(scan_dir, exist_ok=True)
        src_path   = self._current_path
        orig_name  = os.path.basename(src_path)

        # ask for a new name (pre-filled with current name)
        dialog = tk.Toplevel(self)
        dialog.title('Send to Scan Queue')
        dialog.configure(bg='#1a1a1a')
        dialog.resizable(False, False)
        dialog.grab_set()

        tk.Label(dialog, text='Save as:', bg='#1a1a1a', fg='#aaa').pack(padx=16, pady=(14,4), anchor='w')
        name_var = tk.StringVar(value=orig_name)
        entry = tk.Entry(dialog, textvariable=name_var, width=40,
                         bg='#2a2a2a', fg='#fff', insertbackground='#fff', relief='flat')
        entry.pack(padx=16, pady=(0,12))
        entry.select_range(0, 'end')
        entry.focus()

        def confirm():
            name = name_var.get().strip()
            if not name:
                messagebox.showwarning('Name required', 'Enter a filename.', parent=dialog); return
            if not any(name.lower().endswith(e) for e in ('.fits','.fit','.fts','.fits.gz','.png','.jpg')):
                name += os.path.splitext(orig_name)[1]
            dst = os.path.join(scan_dir, name)
            import shutil
            shutil.copy2(src_path, dst)
            dialog.destroy()
            messagebox.showinfo('Queued', f'Added to FILES_TO_SCAN/:\n{name}')

        btn_style = dict(bg='#2a2a2a', fg='#ccc', relief='flat',
                         activebackground='#3a3a3a', padx=12, pady=4, cursor='hand2')
        bf = tk.Frame(dialog, bg='#1a1a1a')
        bf.pack(pady=(0,12))
        tk.Button(bf, text='Add to Queue', command=confirm, **btn_style).pack(side='left', padx=4)
        tk.Button(bf, text='Cancel', command=dialog.destroy, **btn_style).pack(side='left', padx=4)
        dialog.bind('<Return>', lambda e: confirm())
        dialog.bind('<Escape>', lambda e: dialog.destroy())

    def _hdu_changed(self, _=None):
        self._select_hdu(self.hdu_menu.current())

    def _select_hdu(self, idx):
        hdu = self.hdus[idx]
        self.arr  = hdu['arr']
        self.nx   = hdu['nx']
        self.ny   = hdu['ny']
        if FAST:
            # slice first 2D plane in case of 3D cube (arr may be larger than nx*ny)
            self.arr2d = self.arr[:self.ny * self.nx].reshape(self.ny, self.nx)
        else:
            self.arr2d = None
        self.cx = self.cy = None
        self.zoom = 1.0
        self._fit()

    def _fit(self):
        if self.arr is None: return
        w = self.canvas.winfo_width()  or 800
        h = self.canvas.winfo_height() or 700
        self.zoom = min(w / self.nx, h / self.ny)
        self.cx, self.cy = self.nx / 2, self.ny / 2
        self._redraw()

    def _redraw(self):
        if self.arr is None: return
        w = self.canvas.winfo_width()  or 800
        h = self.canvas.winfo_height() or 700
        cx = self.cx if self.cx is not None else self.nx / 2
        cy = self.cy if self.cy is not None else self.ny / 2
        if FAST:
            self._photo = render_fast(self.arr2d, w, h, self.zoom, cx, cy)
        else:
            pix = render_slow(self.arr, w, h, self.nx, self.ny, self.zoom, cx, cy)
            img = tk.PhotoImage(width=w, height=h)
            img.put(pix)
            self._photo = img
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor='nw', image=self._photo)
        self.info.config(text=f'{self.nx}×{self.ny}  zoom {self.zoom:.2f}x')

    def _on_resize(self, _=None):
        if self.arr is not None: self.after(50, self._redraw)

    def _on_scroll(self, ev):
        if self.arr is None: return
        factor = 1.15 if (getattr(ev, 'delta', 0) > 0 or ev.num == 4) else 1/1.15
        cx = self.cx if self.cx is not None else self.nx / 2
        cy = self.cy if self.cy is not None else self.ny / 2
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        mx = cx + (ev.x - w/2) / self.zoom
        my = (self.ny - 1) - (cy + (ev.y - h/2) / self.zoom)
        self.zoom = max(0.05, min(self.zoom * factor, 50.0))
        self.cx = mx - (ev.x - w/2) / self.zoom
        self.cy = (self.ny - 1) - my - (ev.y - h/2) / self.zoom
        self._redraw()

    def _on_motion(self, ev):
        if self.arr is None or self.arr2d is None: return
        cx = self.cx if self.cx is not None else self.nx/2
        cy = self.cy if self.cy is not None else self.ny/2
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        ix = int(cx + (ev.x - w/2) / self.zoom)
        iy = int((self.ny-1) - (cy + (ev.y - h/2) / self.zoom))
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            val = float(self.arr2d[iy, ix]) if FAST else self.arr[iy * self.nx + ix]
            self.info.config(text=f'x={ix} y={iy} val={val:.4g}  zoom {self.zoom:.2f}x')

if __name__ == '__main__':
    app = FITSViewer()
    app.mainloop()