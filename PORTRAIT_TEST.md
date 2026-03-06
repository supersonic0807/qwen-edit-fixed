# Portrait & Landscape Testing Guide

## Aspect Ratio Fix Validation

This document explains how to test the aspect ratio preservation feature that fixes portrait image squashing.

## Expected Behavior

### Before Fix ❌
- **Landscape (1920×1280)**: Worked correctly → 2496×1664
- **Portrait (1280×1920)**: **SQUASHED** → 2496×1664 (wrong aspect!)

### After Fix ✅
- **Landscape (1920×1280)**: Works correctly → 2496×1664
- **Portrait (1280×1920)**: **PRESERVED** → 1664×2496 (correct!)
- **Square (2048×2048)**: Preserved → 2048×2048
- **Any aspect ratio**: Automatically preserved

## Testing Commands

### Test Portrait Image
```bash
cd qwen_image_edit
python test_api.py --mode base64 \
  --image-file path/to/portrait_image.jpg \
  --prompt "enhance colors, remove noise" \
  --out output/portrait_result.jpg
```

**Check output dimensions**:
```bash
# On Windows (PowerShell)
Add-Type -AssemblyName System.Drawing
$img = [System.Drawing.Image]::FromFile("output/portrait_result.jpg")
Write-Host "Output: $($img.Width) × $($img.Height)"
$img.Dispose()

# Or use Python
python -c "from PIL import Image; img = Image.open('output/portrait_result.jpg'); print(f'Output: {img.size}')"
```

### Test Landscape Image
```bash
python test_api.py --mode base64 \
  --image-file path/to/landscape_image.jpg \
  --prompt "enhance colors, remove noise" \
  --out output/landscape_result.jpg
```

### Test Square Image
```bash
python test_api.py --mode base64 \
  --image-file path/to/square_image.jpg \
  --prompt "enhance colors, remove noise" \
  --out output/square_result.jpg
```

## Validation Checklist

- [ ] Portrait input (1280×1920) produces portrait output (~1664×2496)
- [ ] Landscape input (1920×1280) produces landscape output (~2496×1664)
- [ ] Square input (2048×2048) produces square output (~2048×2048)
- [ ] Aspect ratio is preserved (input ratio ≈ output ratio)
- [ ] No visible squashing or stretching
- [ ] Image quality remains high
- [ ] Processing time is similar to before

## Expected Log Output

When processing, you should see logs like:

```
📐 Input image dimensions: 1280×1920
🎯 Detected portrait orientation (aspect 0.67)
📏 Intermediate: 832×1248 (~1.04MP)
🎨 Final output: 1664×2496 (ESRGAN x2)
✅ Node 122 set to 832×1248 (ESRGAN x2 will deliver 1664×2496)
```

For landscape:
```
📐 Input image dimensions: 1920×1280
🎯 Detected landscape orientation (aspect 1.50)
📏 Intermediate: 1248×832 (~1.04MP)
🎨 Final output: 2496×1664 (ESRGAN x2)
✅ Node 122 set to 1248×832 (ESRGAN x2 will deliver 2496×1664)
```

## Aspect Ratio Calculation Examples

| Input Size | Aspect | Orientation | Intermediate | Final Output |
|------------|--------|-------------|--------------|--------------|
| 1920×1280 | 1.50 | Landscape | 1248×832 | 2496×1664 |
| 1280×1920 | 0.67 | Portrait | 832×1248 | 1664×2496 |
| 2048×2048 | 1.00 | Square | 1024×1024 | 2048×2048 |
| 2560×1080 | 2.37 | Ultrawide | 1520×632 | 3040×1264 |
| 1080×1920 | 0.56 | Mobile | 752×1336 | 1504×2672 |

## Debugging

If aspect ratios don't match:

1. **Check input image**: Verify actual dimensions
   ```python
   from PIL import Image
   img = Image.open("input.jpg")
   print(f"Input: {img.size}")
   ```

2. **Check handler logs**: Look for dimension detection messages

3. **Verify rounding**: Intermediate dimensions must be multiples of 8

4. **Calculate expected**:
   ```python
   # For portrait 1280×1920
   aspect = 1280 / 1920  # 0.6667
   target_mp = 1_000_000
   intermediate_h = int((target_mp / aspect) ** 0.5)  # 1224
   intermediate_w = int(intermediate_h * aspect)       # 816
   # Round to 8
   intermediate_w = (intermediate_w // 8) * 8  # 816
   intermediate_h = (intermediate_h // 8) * 8  # 1224
   # Final: 1632×2448
   ```

## Common Issues

### Issue: Output still squashed
**Solution**: Rebuild and redeploy Docker image with updated handler.py

### Issue: "Failed to load image for dimension detection"
**Solution**: Ensure Pillow is installed (added to Dockerfile)

### Issue: Different aspect ratio than expected
**Solution**: Check if image was cropped/resized before upload

## Deployment

After testing locally:

1. **Build new image**:
   ```bash
   docker build -t your-username/qwen-edit-portrait:latest .
   ```

2. **Push to Docker Hub**:
   ```bash
   docker push your-username/qwen-edit-portrait:latest
   ```

3. **Update RunPod endpoint**: Change image reference in endpoint settings

4. **Wait for workers to restart**: May take 5-10 minutes

5. **Test via API**: Use your app to submit portrait and landscape images

## Integration Notes

Your Next.js app ([lib/services/ai/qwen-edit.ts](../lib/services/ai/qwen-edit.ts)) doesn't need any changes:
- Still calls same endpoint
- Still submits base64 image + prompt
- Aspect ratio is now automatically preserved
- Output dimensions vary based on input orientation

**No breaking changes** - fully backward compatible!
