import os
import json
import soundfile as sf
import librosa
from datasets import load_dataset
from tqdm import tqdm

# KONFIGURASJON
DATASET_ID = "google/fleurs"
CONFIG = "nb_no"  # Norsk Bokmål
OUTPUT_DIR = "/workspace/norsk_data"  # Absolutt sti inne i docker
JSONL_FILE = "train.jsonl"
TARGET_SR = 16000
MAX_SAMPLES = 3000

# Cache mapper
CACHE_DIR = "/workspace/hf_cache"
audio_dir = os.path.join(OUTPUT_DIR, "audio_files")

# Sjekk om jobben allerede er gjort
if os.path.exists(os.path.join(OUTPUT_DIR, JSONL_FILE)):
    print("Data finnes allerede i", OUTPUT_DIR, "- Hopper over nedlasting.")
    exit(0)

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(audio_dir, exist_ok=True)

print(f"Laster ned {DATASET_ID} ({CONFIG})...")
dataset = load_dataset(DATASET_ID, CONFIG, split="train", streaming=True, cache_dir=CACHE_DIR, trust_remote_code=True)

data_entries = []
print("Prosesserer lyd...")

iterator = iter(dataset)
for i in tqdm(range(MAX_SAMPLES)):
    try:
        sample = next(iterator)
        text = sample.get("transcription", "")
        audio_data = sample.get("audio", {})
        audio_array = audio_data.get("array")
        orig_sr = audio_data.get("sampling_rate")

        if audio_array is None or not text: continue
        
        # Resample
        if orig_sr != TARGET_SR:
            audio_array = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=TARGET_SR)
            
        # Sjekk lengde (1-20 sek)
        duration = len(audio_array) / TARGET_SR
        if duration < 1.0 or duration > 20.0: continue

        filename = f"fleurs_{i}.wav"
        filepath = os.path.join(audio_dir, filename)
        sf.write(filepath, audio_array, TARGET_SR)
        
        entry = {
            "messages": [
                {"role": "system", "content": "Du er en hjelpsom assistent som snakker flytende norsk."},
                {"role": "user", "content": "<|audio_bos|><|AUDIO|><|audio_eos|>"},
                {"role": "assistant", "content": text}
            ],
            "audio": filepath
        }
        data_entries.append(entry)
    except StopIteration: break
    except Exception: continue

with open(os.path.join(OUTPUT_DIR, JSONL_FILE), "w", encoding="utf-8") as f:
    for entry in data_entries:
        f.write(json.dumps(entry) + "\n")

print(f"Ferdig! {len(data_entries)} eksempler lagret.")
