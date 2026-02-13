# Build Info

Last updated: 2026-02-13 14:36

## Changes
- Modified handler.py to use ImageScale with exact dimensions instead of ImageScaleToTotalPixels
- Ensures output matches requested width/height exactly (e.g., 2500x1667)
- Prevents image extension and quality loss
