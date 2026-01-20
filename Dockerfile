# Vi bruker en offisiell PyTorch base som er stabil (CUDA 12.1 er veldig trygt for Qwen/BnB)
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# Sett miljøvariabler for å unngå mas under installasjon
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PyTorch_CUDA_ALLOC_CONF=expandable_segments:True

# 1. Systempakker (libsndfile1 er VIKTIG for lyd/audio prosessering)
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 2. Oppdater pip
RUN pip install --upgrade pip

# 3. FIX: Installer Torch og Torchvision SAMTIDIG for å garantere match
# Dette erstatter den manuelle fiksen din.
RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 4. Installer Python-bibliotekene vi trenger til Qwen
# Vi legger til 'soundfile' og 'librosa' siden dette er en lydmodell
RUN pip install \
    transformers \
    peft \
    bitsandbytes \
    datasets \
    accelerate \
    scipy \
    soundfile \
    librosa \
    tensorboard \
    protobuf \
    sentencepiece

# 5. (Valgfritt) Flash Attention for hastighet (kan ta tid å bygge, kommenter ut hvis det feiler)
# RUN pip install flash-attn --no-build-isolation

# 6. Klargjør arbeidsmappen
WORKDIR /workspace

# 7. Kopier kildekoden din inn
COPY src/ /workspace/

# Standard kommando (kan overstyres av Kubernetes yaml)
CMD ["python", "train.py"]
