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

# Models are NOT baked into the image — they are downloaded at container startup
# by entrypoint.sh. This keeps the Docker image small (~5GB vs ~30GB) so
# the build completes well within RunPod's 30-minute timeout.
# For persistent model caching, attach a RunPod Network Volume at /runpod-volume.

RUN mkdir -p /ComfyUI/models/diffusion_models /ComfyUI/models/loras /ComfyUI/models/text_encoders /ComfyUI/models/vae
RUN mkdir -p /real-esrgan/models/realesrgan

COPY . .

RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]