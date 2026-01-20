import os
import json
import soundfile as sf
import librosa
from datasets import load_dataset, Audio
from tqdm import tqdm

# --- KONFIGURASJON ---
# Vi bruker Google FLEURS (no_no = norsk)
DATASET_ID = "google/fleurs"
CONFIG = "nb_no" 

OUTPUT_DIR = "./norsk_data"
JSONL_FILE = "train.jsonl"
TARGET_SR = 16000
MAX_SAMPLES = 3000  # FLEURS har ca 3000 treningsklipp

# Cache-mappe
CACHE_DIR = os.path.abspath("./hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Opprett mapper
os.makedirs(OUTPUT_DIR, exist_ok=True)
audio_dir = os.path.join(OUTPUT_DIR, "audio_files")
os.makedirs(audio_dir, exist_ok=True)

print(f"Laster ned {DATASET_ID} ({CONFIG})...")

# Trust_remote_code=True er OK for Google sine datasett
dataset = load_dataset(
    DATASET_ID, 
    CONFIG, 
    split="train", 
    streaming=True, 
    cache_dir=CACHE_DIR,
    trust_remote_code=True
)

data_entries = []
success_count = 0

print("Prosesserer lyd og tekst...")
iterator = iter(dataset)

for i in tqdm(range(MAX_SAMPLES)):
    try:
        sample = next(iterator)
        
        # FLEURS strukturen er: 'audio' (dict), 'transcription' (str)
        text = sample.get("transcription", "")
        audio_data = sample.get("audio", {})
        
        # Her får vi array direkte fordi FLEURS er satt opp riktig
        audio_array = audio_data.get("array")
        orig_sr = audio_data.get("sampling_rate")

        if audio_array is None or not text:
            continue

        # --- RESAMPLING ---
        if orig_sr != TARGET_SR:
            audio_array = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=TARGET_SR)

        # Sjekk lengde
        duration = len(audio_array) / TARGET_SR
        if duration < 1.0 or duration > 20.0:
            continue

        # Lagre fil
        filename = f"fleurs_{i}.wav"
        filepath = os.path.join(audio_dir, filename)
        sf.write(filepath, audio_array, TARGET_SR)
        
        # Lag entry for Qwen
        entry = {
            "messages": [
                {
                    "role": "system",
                    "content": "Du er en hjelpsom assistent som snakker flytende norsk."
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
            "audio": filepath
        }
        
        data_entries.append(entry)
        success_count += 1

    except StopIteration:
        break
    except Exception as e:
        print(f"Feil på sample {i}: {e}")
        continue

# Lagre JSONL
with open(os.path.join(OUTPUT_DIR, JSONL_FILE), "w", encoding="utf-8") as f:
    for entry in data_entries:
        f.write(json.dumps(entry) + "\n")

print("-" * 50)
print(f"Ferdig! {success_count} klipp lagret i {os.path.join(OUTPUT_DIR, JSONL_FILE)}")
