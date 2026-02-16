# Build Info

Last updated: 2026-02-16 16:15

## Changes
- **CRITICAL**: Switch to ImageScale with exact dimensions
- Changed from ImageScaleToTotalPixels to ImageScale node type
- Use exact width/height (2496×1664) instead of megapixels
- Added center crop to preserve composition
- Maintain VAE compatibility (multiple of 8 rounding)

## Expected Result
- Eliminate zoom-out effect completely
- Preserve exact composition framing
- Maintain full quality at 2496×1664
- No compositional drift or content extension

## Build Status
- Triggering build for zoom-out fix deployment
