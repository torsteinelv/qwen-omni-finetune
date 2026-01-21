# Vi bruker NVIDIAs offisielle PyTorch image som base (inkluderer CUDA)
FROM nvcr.io/nvidia/pytorch:24.02-py3

# Unngå interaktive spørsmål under bygging
ENV DEBIAN_FRONTEND=noninteractive

# Sett arbeidsmappe
WORKDIR /workspace

# Kopier requirements først
COPY requirements.txt .

# --- FIX START ---
# Avinstaller transformer-engine for å unngå "is_autocast_enabled" feil med nyere Transformers
RUN pip uninstall -y transformer-engine flash-attn
# --- FIX SLUTT ---

# Installer biblioteker
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Kopier kildekoden
COPY src/ .

# Gjør entrypoint-scriptet kjørbart
RUN chmod +x entrypoint.sh

# Opprett mapper
RUN mkdir -p /workspace/norsk_data /workspace/output

# Start entrypoint
ENTRYPOINT ["./entrypoint.sh"]
