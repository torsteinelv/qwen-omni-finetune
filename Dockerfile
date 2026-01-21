# Vi bruker NVIDIAs offisielle PyTorch image som base (inkluderer CUDA)
FROM nvcr.io/nvidia/pytorch:24.02-py3

# Unngå interaktive spørsmål under bygging
ENV DEBIAN_FRONTEND=noninteractive

# Sett arbeidsmappe
WORKDIR /workspace

# Kopier requirements først
COPY requirements.txt .

# Installer biblioteker
# OBS: Vi legger inn "uninstall transformer-engine" her for sikkerhets skyld.
# Det hindrer rare feil senere i treningen.
RUN pip uninstall -y transformer-engine && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Kopier kildekoden
# (Sørg for at entrypoint.sh ligger inni src-mappen din lokalt, 
#  eller legg til "COPY entrypoint.sh ." hvis den ligger utenfor)
COPY src/ .

# Gjør entrypoint-scriptet kjørbart
RUN chmod +x entrypoint.sh

# Opprett mapper
RUN mkdir -p /workspace/norsk_data /workspace/output /workspace/output_talker

# Start
ENTRYPOINT ["./entrypoint.sh"]
