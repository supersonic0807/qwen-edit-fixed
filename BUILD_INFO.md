# Build Info

Last updated: 2026-02-16 17:00

## MAJOR FIX: Two-Stage Pipeline with Real-ESRGAN

### Root Cause Identified
- Qwen model trained at ~1MP resolution
- Forcing 4.15MP output causes model to "zoom out" to fill canvas
- Original 1248×832 output has no zoom-out, but loses quality

### Solution: Two-Stage Pipeline
```
Input (2496×1664) 
    ↓ [Scale to 1MP: ~1248×832]
Qwen 2512 Model (1MP) ← native resolution, NO zoom-out
    ↓ VAEDecode (1MP)
    ↓ Real-ESRGAN 4x upscale (~4992×3328)
    ↓ ImageScale to target (2496×1664)
Output (2496×1664) ← sharp, full quality, NO zoom-out
```

### Changes Made
- **Dockerfile**: Added Real-ESRGAN_x4plus.pth (64MB)
- **Workflows**: Added upscaler pipeline (nodes 120-122, or 130-132 for 3-image)
- **Handler**: Sets dynamic target dimensions for upscaler output

### Expected Results
- ✅ No zoom-out (model stays at native 1MP)
- ✅ Full resolution output (2496×1664)
- ✅ Sharp quality (Real-ESRGAN AI upscaling)
- ✅ ~3-5 seconds extra processing time

## Build Status
- Deploying two-stage pipeline with Real-ESRGAN
