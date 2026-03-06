# Build Info

Last updated: 2026-03-06

## LATEST: Aspect Ratio Preservation (Portrait Fix)

### Problem Solved
- **Issue**: Portrait images (e.g., 1280×1920) were squashed when forced into landscape dimensions (1248×832)
- **Root Cause**: Hardcoded intermediate dimensions assumed landscape orientation
- **Impact**: Portrait photos had distorted aspect ratios in final output

### Solution: Dynamic Aspect Ratio Detection
```
Input Image (any aspect ratio)
    ↓ [Detect dimensions & aspect ratio]
    ↓ [Calculate ~1MP intermediate preserving ratio]
Qwen Edit 2511 Model (1MP)
    ↓ VAEDecode
    ↓ Lanczos to intermediate size (aspect ratio preserved)
    ↓ Real-ESRGAN x2 enhancement
Output (2× intermediate, correct aspect ratio)
```

### Examples
- **Landscape** (1920×1280): 1248×832 → 2496×1664 ✅
- **Portrait** (1280×1920): 832×1248 → 1664×2496 ✅ (FIXED!)
- **Square** (2048×2048): 1024×1024 → 2048×2048 ✅
- **Ultrawide** (2560×1080): 1520×632 → 3040×1264 ✅

### Implementation Details
- Auto-detects orientation from first input image
- Calculates intermediate dimensions maintaining aspect ratio at ~1MP
- Rounds to multiples of 8 (VAE requirement)
- ESRGAN applies 2× scaling to correct dimensions
- No API changes required - fully automatic

### Changes Made
- **handler.py**: Dynamic dimension calculation based on input aspect ratio
- **Dockerfile**: Added Pillow dependency for image dimension detection

---

## PREVIOUS FIX: Pure Lanczos Upscaling (Option B)

### Root Cause Identified
- Qwen model trained at ~1MP resolution
- Forcing higher output causes model to "zoom out" to fill canvas
- AI upscalers (Real-ESRGAN) caused over-sharpening/smoothing artifacts

### Solution: Simple Lanczos Pipeline
```
Input (1920×1280) 
    ↓ [Scale to 1MP: ~1248×832]
Qwen Edit 2511 Model (1MP) ← native resolution, NO zoom-out
    ↓ VAEDecode (1MP)
    ↓ Lanczos upscale to 1920×1280 (~2.4× factor)
Output (1920×1280) ← natural quality, NO zoom-out, NO artifacts
```

### Changes Made
- **Dockerfile**: Removed Real-ESRGAN download (cleaner build)
- **Workflows**: Removed AI upscaler nodes, direct Lanczos scaling
- **Handler**: Sets dynamic target dimensions (default 1920×1280)
- **Models**: Using Qwen Edit 2511 (image editing, not generation 2512)

### Expected Results
- ✅ No zoom-out (model stays at native 1MP)
- ✅ 1920×1280 output (true 1080p quality)
- ✅ Natural quality (pure geometric upscale, no AI artifacts)
- ✅ Fast processing (no AI upscaler overhead)

## Build Status
- Deploying pure Lanczos pipeline with 1920×1280 target
