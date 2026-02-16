# Build Info

Last updated: 2026-02-16 18:30

## MAJOR FIX: Pure Lanczos Upscaling (Option B)

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
