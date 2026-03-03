#!/usr/bin/env python3
"""
Real-ESRGAN Real Estate Photo Enhancement
==========================================
Enhance and upscale real estate photos using Real-ESRGAN models.

Supports multiple models optimised for different tasks:
  - RealESRGAN_x4plus        : general-purpose 4× upscaler (best quality)
  - RealESRGAN_x2plus        : general-purpose 2× upscaler (faster)
  - RealESRNet_x4plus        : sharper / less smoothing variant
  - realesr-general-x4v3     : latest general v3 model (good balance)

Usage examples:
    # Enhance a single image (default 2× upscale)
    python realesrgan_enhance.py -i input/photo.jpg -o output/

    # Enhance entire folder, 4× upscale
    python realesrgan_enhance.py -i input/CV089998 -o output/esrgan -s 4

    # Enhance with face correction (useful for agent/team photos)
    python realesrgan_enhance.py -i input/ -o output/ --face-enhance

    # Use a specific model
    python realesrgan_enhance.py -i input/ -o output/ --model realesr-general-x4v3

    # Keep original resolution (enhance quality only, no upscale)
    python realesrgan_enhance.py -i input/ -o output/ --no-upscale

    # CPU-only mode
    python realesrgan_enhance.py -i input/ -o output/ --device cpu

    # Batch process all listing folders
    python realesrgan_enhance.py -i input/ -o output/ --recursive
"""

import argparse
import os
import sys
import cv2
import glob
import time
import numpy as np
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Compatibility shim: basicsr references a removed torchvision submodule.
# Patch it *before* any basicsr / realesrgan imports.
# ---------------------------------------------------------------------------
try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except ModuleNotFoundError:
    import types, torchvision.transforms.functional as _F
    _mod = types.ModuleType('torchvision.transforms.functional_tensor')
    _mod.rgb_to_grayscale = _F.rgb_to_grayscale
    sys.modules['torchvision.transforms.functional_tensor'] = _mod

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

MODEL_CONFIGS = {
    'RealESRGAN_x4plus': {
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        'scale': 4,
        'model_cls': 'RRDBNet',
        'params': dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
    },
    'RealESRGAN_x2plus': {
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        'scale': 2,
        'model_cls': 'RRDBNet',
        'params': dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
    },
    'RealESRNet_x4plus': {
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth',
        'scale': 4,
        'model_cls': 'RRDBNet',
        'params': dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
    },
    'realesr-general-x4v3': {
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
        'scale': 4,
        'model_cls': 'SRVGGNetCompact',
        'params': dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu'),
    },
}

DEFAULT_MODEL = 'RealESRGAN_x2plus'


# ---------------------------------------------------------------------------
# Importable API — used by handler.py for post-processing
# ---------------------------------------------------------------------------

# Module-level singleton so the model is loaded exactly once per warm worker.
_upsampler_cache: dict = {}


def get_upsampler(model_name: str = DEFAULT_MODEL, device: str = 'auto',
                  half: bool = True, tile: int = 0):
    """Return a cached (upsampler, scale) pair, loading the model on first call."""
    if model_name not in _upsampler_cache:
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _upsampler_cache[model_name] = create_upsampler(
            model_name, device=device, half=half, tile=tile
        )
    return _upsampler_cache[model_name]


def enhance_bytes(image_bytes: bytes, outscale: int = 2,
                  model_name: str = DEFAULT_MODEL) -> bytes:
    """Enhance an image supplied as raw bytes, return JPEG bytes.

    This is the primary entry-point called by handler.py.
    No temp files are written; everything lives in memory.
    """
    upsampler, _ = get_upsampler(model_name)

    # Decode to BGR numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('enhance_bytes: could not decode input image')

    h, w = img.shape[:2]
    try:
        output, _ = upsampler.enhance(img, outscale=outscale)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            torch.cuda.empty_cache()
            # Retry with tiling
            upsampler.tile_size = 512
            try:
                output, _ = upsampler.enhance(img, outscale=outscale)
            finally:
                upsampler.tile_size = 0
        else:
            raise

    # Encode result to JPEG bytes
    ok, buf = cv2.imencode('.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError('enhance_bytes: cv2.imencode failed')
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_model_dir() -> str:
    """Return (and create) the directory where Real-ESRGAN weights are cached."""
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'realesrgan')
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def download_model(model_name: str) -> str:
    """Download model weights if not already cached. Returns path to .pth file."""
    cfg = MODEL_CONFIGS[model_name]
    model_dir = get_model_dir()
    filename = f"{model_name}.pth"
    dest = os.path.join(model_dir, filename)

    if os.path.isfile(dest):
        return dest

    print(f"Downloading {model_name} weights …")
    from urllib.request import urlretrieve

    def _progress(count, block_size, total_size):
        pct = count * block_size * 100 / total_size
        print(f"\r  {pct:.1f}%", end='', flush=True)

    urlretrieve(cfg['url'], dest, reporthook=_progress)
    print(f"\n  Saved to {dest}")
    return dest


def collect_images(input_path: str, recursive: bool = False) -> list[str]:
    """Collect all image files from *input_path* (file or directory)."""
    if os.path.isfile(input_path):
        return [input_path]

    images = []
    if recursive:
        for root, _dirs, files in os.walk(input_path):
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                    images.append(os.path.join(root, f))
    else:
        # Non-recursive: scan top-level and one level of sub-folders
        for entry in sorted(os.listdir(input_path)):
            full = os.path.join(input_path, entry)
            if os.path.isfile(full) and os.path.splitext(entry)[1].lower() in IMAGE_EXTS:
                images.append(full)
            elif os.path.isdir(full):
                for f in sorted(os.listdir(full)):
                    fp = os.path.join(full, f)
                    if os.path.isfile(fp) and os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                        images.append(fp)
    return images


def build_output_path(img_path: str, input_root: str, output_root: str, suffix: str, out_fmt: str) -> str:
    """Mirror input directory structure under *output_root*."""
    if os.path.isfile(input_root):
        rel = os.path.basename(img_path)
    else:
        rel = os.path.relpath(img_path, input_root)

    stem, _ext = os.path.splitext(rel)
    ext = f".{out_fmt}" if out_fmt else _ext
    out = os.path.join(output_root, f"{stem}{suffix}{ext}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Core enhancement
# ---------------------------------------------------------------------------
def create_upsampler(model_name: str, device: str, half: bool = True, tile: int = 0):
    """Build a RealESRGANer upsampler instance."""
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
    except ImportError:
        print("\nMissing required packages. Install them with:")
        print("  pip install realesrgan basicsr")
        print("\nOr run:  pip install -r realesrgan_requirements.txt")
        sys.exit(1)

    cfg = MODEL_CONFIGS[model_name]
    weight_path = download_model(model_name)

    # Build the network
    if cfg['model_cls'] == 'RRDBNet':
        from basicsr.archs.rrdbnet_arch import RRDBNet
        net = RRDBNet(**cfg['params'])
    else:
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        net = SRVGGNetCompact(**cfg['params'])

    # Determine GPU settings
    use_half = half and device != 'cpu' and torch.cuda.is_available()

    upsampler = RealESRGANer(
        scale=cfg['scale'],
        model_path=weight_path,
        model=net,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=use_half,
        device=device,
    )
    return upsampler, cfg['scale']


def enhance_image(
    upsampler,
    img_path: str,
    outscale: int,
    no_upscale: bool = False,
    face_enhance: bool = False,
    face_upsampler=None,
    sharpen: float = 0.0,
    denoise: float = 0.0,
) -> np.ndarray | None:
    """Enhance a single image. Returns the output BGR numpy array."""
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"  [SKIP] Could not read: {img_path}")
        return None

    h, w = img.shape[:2]

    # --- enhance --------------------------------------------------------
    try:
        if face_enhance and face_upsampler is not None:
            _, _, output = face_upsampler.enhance(
                img, has_aligned=False, only_center_face=False, paste_back=True
            )
        else:
            output, _ = upsampler.enhance(img, outscale=outscale)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f"  [OOM] Image too large ({w}×{h}). Retrying with tiling …")
            torch.cuda.empty_cache()
            upsampler.tile_size = 512
            try:
                output, _ = upsampler.enhance(img, outscale=outscale)
            except Exception:
                print(f"  [FAIL] Still OOM – skipping {img_path}")
                return None
            finally:
                upsampler.tile_size = 0
        else:
            raise

    # --- optional: resize back to original resolution -------------------
    if no_upscale:
        output = cv2.resize(output, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # --- optional post-processing for real estate -----------------------
    if sharpen > 0:
        output = _unsharp_mask(output, amount=sharpen)

    if denoise > 0:
        output = cv2.fastNlMeansDenoisingColored(output, None, denoise, denoise, 7, 21)

    return output


def _unsharp_mask(img: np.ndarray, amount: float = 0.5, kernel_size: int = 3) -> np.ndarray:
    """Apply a gentle unsharp mask – great for sharpening real estate details."""
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Real-estate-specific post-processing
# ---------------------------------------------------------------------------
def auto_white_balance(img: np.ndarray) -> np.ndarray:
    """Simple grey-world white balance – helps correct interior colour casts."""
    result = img.copy().astype(np.float32)
    avg_b, avg_g, avg_r = [result[:, :, i].mean() for i in range(3)]
    avg = (avg_b + avg_g + avg_r) / 3
    result[:, :, 0] *= avg / (avg_b + 1e-6)
    result[:, :, 1] *= avg / (avg_g + 1e-6)
    result[:, :, 2] *= avg / (avg_r + 1e-6)
    return np.clip(result, 0, 255).astype(np.uint8)


def boost_exposure(img: np.ndarray, target_brightness: int = 130) -> np.ndarray:
    """Gently lift underexposed images (common in interior photography)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()
    if mean_brightness < target_brightness * 0.7:
        gamma = np.log(target_brightness / 255.0) / np.log(max(mean_brightness, 1) / 255.0)
        gamma = np.clip(gamma, 0.5, 2.0)
        table = (np.arange(256) / 255.0) ** (1.0 / gamma) * 255
        img = cv2.LUT(img, table.astype(np.uint8))
    return img


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description='Real-ESRGAN Real Estate Photo Enhancement',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('-i', '--input', required=True,
                   help='Input image or folder')
    p.add_argument('-o', '--output', required=True,
                   help='Output folder')
    p.add_argument('-s', '--outscale', type=int, default=2,
                   help='Final upscale factor (default: 2)')
    p.add_argument('--model', default=DEFAULT_MODEL, choices=list(MODEL_CONFIGS.keys()),
                   help=f'Model name (default: {DEFAULT_MODEL})')
    p.add_argument('--no-upscale', action='store_true',
                   help='Enhance quality without changing resolution')
    p.add_argument('--face-enhance', action='store_true',
                   help='Use GFPGAN to enhance faces (agent photos)')
    p.add_argument('--tile', type=int, default=0,
                   help='Tile size for large images (0 = no tiling, try 512 if OOM)')
    p.add_argument('--sharpen', type=float, default=0.3,
                   help='Sharpening amount 0-1 (default: 0.3, good for real estate)')
    p.add_argument('--denoise', type=float, default=0.0,
                   help='Denoising strength (0 = off, try 5-10 for noisy interiors)')
    p.add_argument('--white-balance', action='store_true',
                   help='Apply auto white balance (helps warm interior casts)')
    p.add_argument('--exposure-boost', action='store_true',
                   help='Auto-boost dark / underexposed images')
    p.add_argument('--suffix', default='_enhanced',
                   help='Suffix for output filenames (default: _enhanced)')
    p.add_argument('--format', default='', choices=['', 'jpg', 'png', 'webp'],
                   help='Output format (default: same as input)')
    p.add_argument('--quality', type=int, default=95,
                   help='JPEG/WebP quality 0-100 (default: 95)')
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
                   help='Device (default: auto-detect)')
    p.add_argument('--fp32', action='store_true',
                   help='Use FP32 instead of FP16 (slower but avoids rare artifacts)')
    p.add_argument('--recursive', action='store_true',
                   help='Recursively scan sub-folders for images')
    return p.parse_args()


def main():
    args = parse_args()

    # --- device ---------------------------------------------------------
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # --- build upsampler -----------------------------------------------
    print(f"\nModel: {args.model}")
    upsampler, native_scale = create_upsampler(
        args.model, device, half=not args.fp32, tile=args.tile,
    )

    # --- optional face enhancer ----------------------------------------
    face_upsampler = None
    if args.face_enhance:
        try:
            from gfpgan import GFPGANer
            face_upsampler = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=args.outscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=upsampler,
            )
            print("Face enhancement: ENABLED")
        except ImportError:
            print("WARNING: gfpgan not installed – face enhancement disabled")
            print("  Install with: pip install gfpgan")

    # --- collect images ------------------------------------------------
    images = collect_images(args.input, recursive=args.recursive)
    if not images:
        print(f"No images found in: {args.input}")
        sys.exit(1)
    print(f"\nFound {len(images)} image(s) to process\n")

    # --- encode params for OpenCV --------------------------------------
    encode_params = []
    fmt = args.format.lower() if args.format else ''
    if fmt in ('jpg', 'jpeg', ''):
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.quality]
    elif fmt == 'webp':
        encode_params = [cv2.IMWRITE_WEBP_QUALITY, args.quality]
    elif fmt == 'png':
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    # --- process -------------------------------------------------------
    os.makedirs(args.output, exist_ok=True)
    success, failed = 0, 0
    t0 = time.time()

    for img_path in images:
        rel = os.path.relpath(img_path, args.input if os.path.isdir(args.input) else os.path.dirname(args.input))
        print(f"[{success + failed + 1}/{len(images)}] {rel} ", end='', flush=True)
        ts = time.time()

        result = enhance_image(
            upsampler, img_path, args.outscale,
            no_upscale=args.no_upscale,
            face_enhance=args.face_enhance,
            face_upsampler=face_upsampler,
            sharpen=args.sharpen,
            denoise=args.denoise,
        )

        if result is None:
            failed += 1
            continue

        # --- real-estate post-processing --------------------------------
        if args.white_balance:
            result = auto_white_balance(result)
        if args.exposure_boost:
            result = boost_exposure(result)

        # --- save -------------------------------------------------------
        out_path = build_output_path(img_path, args.input, args.output, args.suffix, args.format)
        cv2.imwrite(out_path, result, encode_params if encode_params else None)

        elapsed = time.time() - ts
        h, w = result.shape[:2]
        print(f"→ {w}×{h}  ({elapsed:.1f}s)")
        success += 1

    # --- summary --------------------------------------------------------
    total = time.time() - t0
    print(f"\n{'='*50}")
    print(f"Done!  {success} enhanced, {failed} failed  ({total:.1f}s total)")
    print(f"Output: {os.path.abspath(args.output)}")


if __name__ == '__main__':
    main()
