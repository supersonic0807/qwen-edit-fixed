#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# CUDA 검사 및 설정
echo "Checking CUDA availability..."

# Python을 통한 CUDA 검사
python_cuda_check() {
    python3 -c "
import torch
try:
    if torch.cuda.is_available():
        print('CUDA_AVAILABLE')
        exit(0)
    else:
        print('CUDA_NOT_AVAILABLE')
        exit(1)
except Exception as e:
    print(f'CUDA_ERROR: {e}')
    exit(2)
" 2>/dev/null
}

# CUDA 검사 실행
cuda_status=$(python_cuda_check)
case $? in
    0)
        echo "✅ CUDA is available and working (Python check)"
        export CUDA_VISIBLE_DEVICES=0
        export FORCE_CUDA=1
        ;;
    1)
        echo "❌ CUDA is not available (Python check)"
        echo "Error: CUDA is required but not available. Exiting..."
        exit 1
        ;;
    2)
        echo "❌ CUDA check failed (Python check)"
        echo "Error: CUDA initialization failed. Exiting..."
        exit 1
        ;;
esac

# 추가적인 nvidia-smi 검사
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA driver working (nvidia-smi check)"
    else
        echo "❌ NVIDIA driver found but not working"
        echo "Error: NVIDIA driver is not working properly. Exiting..."
        exit 1
    fi
else
    echo "❌ NVIDIA driver not found"
    echo "Error: NVIDIA driver is required but not found. Exiting..."
    exit 1
fi

# CUDA 환경 변수 설정
echo "Using CUDA device: $CUDA_VISIBLE_DEVICES"

# ---------------------------------------------------------------------------
# Model Download / Cache
# ---------------------------------------------------------------------------
# Models are NOT baked into the Docker image (keeps image ~5GB instead of ~30GB
# so builds finish under RunPod's 30-minute timeout).
#
# Persistence strategy:
#   - If a RunPod Network Volume is attached at /runpod-volume, models are stored
#     there and survive across worker cold starts (download once, reuse forever).
#   - Without a volume, models are downloaded fresh to /ComfyUI/models on every
#     cold start (~2-3 minutes with aria2c at 16 connections).
#
# To set up persistent storage:
#   RunPod Console → Storage → Create Network Volume (≥50GB)
#   Then attach it to your serverless endpoint at /runpod-volume.
# ---------------------------------------------------------------------------

VOLUME_DIR="/runpod-volume"
MODEL_CACHE="${VOLUME_DIR}/comfyui-models"

DIFFUSION_MODEL="/ComfyUI/models/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors"
LORA_MODEL="/ComfyUI/models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
TEXT_ENCODER="/ComfyUI/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE_MODEL="/ComfyUI/models/vae/qwen_image_vae.safetensors"
ESRGAN_MODEL="/real-esrgan/models/realesrgan/RealESRGAN_x2plus.pth"

ensure_model() {
    local dest="$1"
    local url="$2"
    local cache_name="$3"

    # If already present (e.g. baked in a previous build or already downloaded), skip
    if [ -f "$dest" ]; then
        echo "✅ Model already present: $dest"
        return 0
    fi

    # Check network volume cache first
    if [ -d "$VOLUME_DIR" ] && [ -f "${MODEL_CACHE}/${cache_name}" ]; then
        echo "📦 Copying from network volume cache: ${cache_name}"
        cp "${MODEL_CACHE}/${cache_name}" "$dest"
        return 0
    fi

    # Download fresh
    echo "⬇️  Downloading ${cache_name} ..."
    local dest_dir
    dest_dir="$(dirname "$dest")"
    mkdir -p "$dest_dir"
    aria2c -x 16 -s 16 --max-connection-per-server=16 \
        --console-log-level=warn --summary-interval=30 \
        "$url" -d "$dest_dir" -o "$(basename "$dest")"
    echo "✅ Downloaded: $dest"

    # Save to network volume cache if available
    if [ -d "$VOLUME_DIR" ]; then
        mkdir -p "$MODEL_CACHE"
        echo "💾 Caching to network volume: ${cache_name}"
        cp "$dest" "${MODEL_CACHE}/${cache_name}"
    fi
}

echo "--- Checking / downloading models ---"
mkdir -p "$(dirname "$DIFFUSION_MODEL")" "$(dirname "$LORA_MODEL")" "$(dirname "$TEXT_ENCODER")" "$(dirname "$VAE_MODEL")" "$(dirname "$ESRGAN_MODEL")"

ensure_model "$DIFFUSION_MODEL" \
    "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors" \
    "qwen_image_edit_2511_fp8mixed.safetensors"

ensure_model "$LORA_MODEL" \
    "https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors" \
    "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

ensure_model "$TEXT_ENCODER" \
    "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors" \
    "qwen_2.5_vl_7b_fp8_scaled.safetensors"

ensure_model "$VAE_MODEL" \
    "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors" \
    "qwen_image_vae.safetensors"

ensure_model "$ESRGAN_MODEL" \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth" \
    "RealESRGAN_x2plus.pth"

echo "--- All models ready ---"

# Start ComfyUI in the background
echo "Starting ComfyUI in the background..."
python /ComfyUI/main.py --listen --use-sage-attention &

# Wait for ComfyUI to be ready
echo "Waiting for ComfyUI to be ready..."
max_wait=120  # 최대 2분 대기
wait_count=0
while [ $wait_count -lt $max_wait ]; do
    if curl -s http://127.0.0.1:8188/ > /dev/null 2>&1; then
        echo "ComfyUI is ready!"
        break
    fi
    echo "Waiting for ComfyUI... ($wait_count/$max_wait)"
    sleep 2
    wait_count=$((wait_count + 2))
done

if [ $wait_count -ge $max_wait ]; then
    echo "Error: ComfyUI failed to start within $max_wait seconds"
    exit 1
fi

# Start the handler in the foreground
# 이 스크립트가 컨테이너의 메인 프로세스가 됩니다.
echo "Starting the handler..."
exec python handler.py