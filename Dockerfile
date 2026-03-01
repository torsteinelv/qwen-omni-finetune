# Vi bruker NVIDIAs offisielle PyTorch image som base (inkluderer CUDA)
FROM nvcr.io/nvidia/pytorch:25.12-py3

# Unngå interaktive spørsmål under bygging
ENV DEBIAN_FRONTEND=noninteractive

# Sett arbeidsmappe
WORKDIR /workspace

# Kopier requirements først (for bedre caching av docker layers)
COPY requirements.txt .

# Installer biblioteker
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Kopier kildekoden
COPY src/ .

# Gjør entrypoint-scriptet kjørbart
RUN chmod +x entrypoint.sh

# Opprett mapper for data og output som vi kan mounte til
RUN mkdir -p /workspace/norsk_data /workspace/output

# Start entrypoint scriptet når containeren kjører
ENTRYPOINT ["./entrypoint.sh"]
