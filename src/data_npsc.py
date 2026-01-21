import os
import json
import soundfile as sf
import librosa
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

# --- KONFIGURASJON ---
# Vi bruker NPSC (Stortinget) - Bokmål
DATASET_ID = "NbAiLab/NPSC"
CONFIG = "16K_mp3_bokmaal" 

# Stier
OUTPUT_DIR = "/workspace/norsk_data"
JSONL_FILE = "train.jsonl" # Overskriver den gamle
AUDIO_SUBDIR = "audio_files"

# Lyd-innstillinger
TARGET_SR = 24000  # Mimi (Qwen sin lyd-encoder) foretrekker ofte 24kHz
MIN_DURATION = 1.5 # Sekunder
MAX_DURATION = 15.0 # Sekunder (Ikke for lange setninger for Talker)
MAX_SAMPLES = 15000 # Vi øker fra 3000 til 15000! (Ca 20-30 timer)

# Opprett mapper
os.makedirs(os.path.join(OUTPUT_DIR, AUDIO_SUBDIR), exist_ok=True)

print("="*50)
print(f"Laster ned datasett: {DATASET_ID} ({CONFIG})")
print(f"Mål: {MAX_SAMPLES} klipp av høy kvalitet.")
print("="*50)

# Vi bruker streaming=True for å slippe å laste ned 100GB først
dataset = load_dataset(
    DATASET_ID, 
    CONFIG, 
    split="train", 
    streaming=False, 
    trust_remote_code=True
)

data_entries = []
success_count = 0
skipped_count = 0

print("Starter prosessering...")

for i, sample in tqdm(enumerate(dataset)):
    if success_count >= MAX_SAMPLES:
        break

    try:
        # Hent tekst og lyd
        text = sample.get("text", "")
        if not text:
            continue
            
        # Normaliser tekst (NPSC har noen ganger <hesitation> tagger o.l.)
        text = text.replace("<hesitation>", "").strip()
        
        audio_data = sample.get("audio", {})
        audio_array = audio_data.get("array")
        orig_sr = audio_data.get("sampling_rate")

        if audio_array is None:
            continue

        # Sjekk lengde (i sekunder)
        duration = len(audio_array) / orig_sr
        if duration < MIN_DURATION or duration > MAX_DURATION:
            skipped_count += 1
            continue

        # --- RESAMPLING (Til 24kHz for Mimi) ---
        if orig_sr != TARGET_SR:
            # Resample bruker librosa (krever float numpy array)
            audio_array = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=TARGET_SR)

        # Lagre som .wav
        filename = f"npsc_{success_count}.wav"
        # Vi bruker absolutt sti for sikkerhets skyld
        abs_audio_path = os.path.join(OUTPUT_DIR, AUDIO_SUBDIR, filename)
        
        sf.write(abs_audio_path, audio_array, TARGET_SR)
        
        # Lag JSON-struktur som prepare_talker.py forstår
        # Den forventer at "audio" peker til filen, og siste message er teksten.
        entry = {
            "messages": [
                {
                    "role": "system", 
                    "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
                },
                {
                    "role": "user", 
                    "content": "<|audio_bos|><|AUDIO|><|audio_eos|>"
                },
                {
                    "role": "assistant", 
                    "content": text
                }
            ],
            "audio": abs_audio_path
        }
        
        data_entries.append(entry)
        success_count += 1
        
        if success_count % 1000 == 0:
            print(f"✅ Lagret {success_count} klipp...")

    except Exception as e:
        print(f"⚠️ Feil på sample {i}: {e}")
        continue

# Lagre alt til train.jsonl
final_jsonl_path = os.path.join(OUTPUT_DIR, JSONL_FILE)
print(f"Lagrer {len(data_entries)} linjer til {final_jsonl_path}...")

with open(final_jsonl_path, "w", encoding="utf-8") as f:
    for entry in data_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print("="*50)
print(f"FERDIG! Har samlet {success_count} lydfiler.")
print(f"Hoppet over {skipped_count} filer pga lengde.")
print(f"Neste steg: Kjør 'python3 prepare_talker.py'")
print("="*50)
