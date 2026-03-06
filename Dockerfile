# Use specific version of nvidia cuda image
FROM wlsdml1114/multitalk-base:1.7 as runtime

# wget 설치 (URL 다운로드를 위해) - use aria2c for parallel downloads
RUN apt-get update && apt-get install -y wget aria2 && rm -rf /var/lib/apt/lists/*

# Enable HF Transfer for 3x faster model downloads
ENV HF_HUB_ENABLE_HF_TRANSFER=1

RUN pip install -U "huggingface_hub[hf_transfer]"
RUN pip install runpod websocket-client librosa Pillow
# BASICSR_EXT=False skips CUDA C++ extension compilation (only needed for training,
# not inference). Without this flag pip install basicsr takes 20+ minutes and
# exceeds the 30-minute build timeout.
RUN BASICSR_EXT=False pip install --no-cache-dir basicsr realesrgan

# Set working directory
WORKDIR /

RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd ComfyUI && \
    pip install --no-cache-dir -r requirements.txt

RUN cd /ComfyUI/custom_nodes/ && \
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git && \
    cd ComfyUI-Manager && \
    pip install --no-cache-dir -r requirements.txt

RUN cd /ComfyUI/custom_nodes/ && \
    git clone https://github.com/kijai/ComfyUI-KJNodes && \
    cd ComfyUI-KJNodes && \
    pip install --no-cache-dir -r requirements.txt

# Download Qwen Edit 2511 models (image editing capability)
# Use aria2c with 16 connections per file for faster downloads (3-5x speedup)
# Progress display enabled for build log visibility
RUN mkdir -p /ComfyUI/models/diffusion_models /ComfyUI/models/loras /ComfyUI/models/text_encoders /ComfyUI/models/vae && \
    echo "Downloading Qwen models (4 files, ~10GB total)..." && \
    aria2c -x 16 -s 16 --max-connection-per-server=16 --console-log-level=warn --summary-interval=10 \
        https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors \
        -d /ComfyUI/models/diffusion_models -o qwen_image_edit_2511_fp8mixed.safetensors && \
    aria2c -x 16 -s 16 --max-connection-per-server=16 --console-log-level=warn --summary-interval=10 \
        https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
        -d /ComfyUI/models/loras -o Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors && \
    aria2c -x 16 -s 16 --max-connection-per-server=16 --console-log-level=warn --summary-interval=10 \
        https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors \
        -d /ComfyUI/models/text_encoders -o qwen_2.5_vl_7b_fp8_scaled.safetensors && \
    aria2c -x 16 -s 16 --max-connection-per-server=16 --console-log-level=warn --summary-interval=10 \
        https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors \
        -d /ComfyUI/models/vae -o qwen_image_vae.safetensors && \
    echo "✅ Qwen models downloaded successfully"

COPY . .

# Download Real-ESRGAN model weights (not stored in git — too large)
# RealESRGAN_x2plus: 2× upscaler used for post-processing Qwen output
RUN mkdir -p /real-esrgan/models/realesrgan && \
    aria2c -x 16 -s 16 --max-connection-per-server=16 \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth \
        -d /real-esrgan/models/realesrgan -o RealESRGAN_x2plus.pth

RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]