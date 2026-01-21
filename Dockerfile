# Vi bruker NVIDIAs offisielle PyTorch image som base
FROM nvcr.io/nvidia/pytorch:24.02-py3

# Unngå interaktive spørsmål
ENV DEBIAN_FRONTEND=noninteractive

# Sett arbeidsmappe
WORKDIR /workspace
# Kopier requirements
COPY requirements.txt .

# --- DEN STORE RENGJØRINGEN ---
# 1. Avinstaller pakker som skaper konflikter (vi trenger ikke disse for Qwen)
#    - transformer-engine: Kjent trøbbelmaker med HuggingFace
#    - cudf, dask, cuml: NVIDIA dataframe-biblioteker som låser versjoner
#    - torchvision: Vi driver ikke med bildebehandling, og den krever gammel torch
RUN pip uninstall -y \
    transformer-engine \
    cudf \
    dask-cudf \
    cuml \
    cugraph \
    torchvision \
    torchaudio && \
    # 2. Oppgrader pip og installer dine krav rent
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Kopier kildekoden
COPY src/ .
COPY entrypoint.sh . 

# Gjør scripts kjørbare
RUN chmod +x entrypoint.sh run_pipeline.sh

# Opprett mapper
RUN mkdir -p /workspace/norsk_data /workspace/output /workspace/output_talker

# Start
ENTRYPOINT ["./entrypoint.sh"]
