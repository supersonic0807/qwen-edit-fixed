# Deploy Fixed Qwen Image Edit to RunPod

## Fix Applied ✅
- Dynamic megapixels calculation preserves full image resolution
- Node 93 now uses requested width/height instead of hardcoded 1MP
- Your 2500×1667 images will process at full quality

## Deployment Options

### Option 1: Build & Push Docker Image (Recommended)

#### Prerequisites
- Docker Desktop installed and running
- Docker Hub account (free at https://hub.docker.com)
- RunPod account with existing endpoint

#### Steps

1. **Login to Docker Hub**
```bash
docker login
# Enter your Docker Hub username and password
```

2. **Build the Docker image** (this takes ~10-15 minutes)
```bash
cd qwen_image_edit
docker build -t YOUR_USERNAME/qwen-edit-fixed:latest .
```
Replace `YOUR_USERNAME` with your Docker Hub username.

3. **Push to Docker Hub**
```bash
docker push YOUR_USERNAME/qwen-edit-fixed:latest
```

4. **Update RunPod Endpoint**
   - Go to https://www.runpod.io/console/serverless
   - Click your existing endpoint (ID: 337nf4l3drq67f)
   - Click "Edit Template" or "Update Image"
   - Change image from `wlsdml1114/qwen_image_edit` to `YOUR_USERNAME/qwen-edit-fixed:latest`
   - Save and wait for new workers to deploy (~5-10 minutes)

### Option 2: Create New Endpoint from Scratch

If you want to create a completely new endpoint:

1. **Build & push Docker image** (steps 1-3 above)

2. **Create new endpoint on RunPod**
   - Go to https://www.runpod.io/console/serverless
   - Click "New Endpoint"
   - Select "Custom Template"
   - Docker Image: `YOUR_USERNAME/qwen-edit-fixed:latest`
   - Container Disk: 20GB minimum
   - GPU: 24GB VRAM minimum (A5000, RTX 4090, or better)
   - Max Workers: Set based on your budget
   - Click "Deploy"

3. **Get new endpoint ID**
   - Copy the new endpoint ID (format: abc123xyz456)
   - Update your app's environment variables

### Option 3: Fork Original Repo and Deploy

If you want to maintain your own version:

1. **Fork the repository on GitHub**
   - Go to https://github.com/wlsdml1114/qwen_image_edit
   - Click "Fork"

2. **Push your changes**
```bash
cd qwen_image_edit
git remote rename origin upstream
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/qwen_image_edit.git
git add handler.py
git commit -m "Fix: Dynamic megapixels for full resolution processing"
git push origin main
```

3. **Use RunPod's GitHub integration** (if supported)
   - Connect your GitHub account to RunPod
   - Select your forked repository
   - RunPod will auto-build and deploy

## Testing After Deployment

Once deployed, test with your actual dimensions:

```javascript
const response = await fetch(`YOUR_RUNPOD_ENDPOINT/runsync`, {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${process.env.RUNPOD_API_KEY}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    input: {
      prompt: "enhance colors and clarity",
      image_base64: YOUR_IMAGE_BASE64,
      seed: 12345,
      width: 2500,
      height: 1667
    }
  })
});
```

**Expected logs** (check RunPod console):
```
✅ Set node 93 megapixels to 4.17 (target: 2500x1667)
```

## Rollback Plan

If something goes wrong:
- **Option 1 users**: Change image back to `wlsdml1114/qwen_image_edit:latest`
- **Option 2 users**: Delete new endpoint, continue using old one
- **Option 3 users**: Reset to original commit

## Cost Estimate

- Docker Hub: Free
- RunPod Build Time: ~$0.20-0.40 (one-time)
- RunPod Runtime: Same as before (no change)

## Next Steps

1. Choose your deployment option (I recommend Option 1 for simplicity)
2. Let me know which option you want, and I'll help with the commands
3. After deployment, I'll help update your Next.js app to use the new endpoint

## Questions?

- Need help with Docker commands? Let me know.
- Want to verify the fix before building? Check [handler.py](handler.py) lines 154 and 267-277.
- Ready to deploy? Tell me your Docker Hub username and I'll customize the commands.
