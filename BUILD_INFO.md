# Build Info

Last updated: 2026-02-16 16:30

## Major Upgrade: Qwen 2512 Models

### Model Changes
- **Diffusion**: 2511 → 2512 (qwen_image_2512_fp8_e4m3fn_scaled_comfyui_4steps_v1.0)
- **LoRA**: 2511-Lightning → 2512-Lightning (4steps V1.0 bf16)
- **Source**: lightx2v/Qwen-Image-2512-Lightning (official ComfyUI optimized)

### Expected Improvements (per official release notes)
- **Enhanced Human Realism**: Reduce "AI-generated" look
- **Finer Natural Details**: Better landscapes, textures, fur
- **Improved Text Rendering**: Better accuracy and layout
- **Better Composition**: Should handle zoom-out issue better

### Technical Details
- **ImageScale**: Exact dimensions (2496×1664) with center crop
- **VAE Compatibility**: Dimensions rounded to multiple of 8
- **Quality**: Full resolution preservation
- **Same Workflow**: Drop-in replacement, same node structure

## Build Status
- Upgrading to Qwen 2512 generation
- Testing zoom-out fix + quality improvements
