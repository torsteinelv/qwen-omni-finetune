import os
import json
import soundfile as sf
import librosa
from datasets import load_dataset
from tqdm import tqdm

# --- KONFIGURASJON ---
DATASET_ID = "google/fleurs"
CONFIG = "nb_no"          # Norsk bokmål
OUTPUT_DIR = "./norsk_data"
JSONL_FILE = "train.jsonl"
TARGET_SR = 16000         # Qwen (og Whisper) liker 16kHz
MAX_SAMPLES = 3000        # Hele datasettet er ca 3000 klipp

# Cache-mappe for Hugging Face (så vi slipper å laste ned på nytt hvis vi restarter)
CACHE_DIR = os.path.abspath("./hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Opprett mapper for output
os.makedirs(OUTPUT_DIR, exist_ok=True)
audio_dir = os.path.join(OUTPUT_DIR, "audio_files")
os.makedirs(audio_dir, exist_ok=True)

print("="*50)
print(f"Laster ned datasett: {DATASET_ID} ({CONFIG})")
print("MODUS: streaming=False (Laster ned alt først for maksimal hastighet)")
print("="*50)

# Laster ned hele datasettet først. Dette går MYE raskere i K8s enn streaming.
try:
    dataset = load_dataset(
        DATASET_ID, 
        CONFIG, 
        split="train", 
        streaming=False,       # <--- VIKTIG: False for hastighet
        cache_dir=CACHE_DIR,
        trust_remote_code=True # Nødvendig for FLEURS
    )
except Exception as e:
    print(f"KRITISK FEIL ved nedlasting: {e}")
    print("Sjekk at 'datasets==2.21.0' er i requirements.txt!")
    exit(1)

data_entries = []
success_count = 0

print(f"Starter prosessering av {min(len(dataset), MAX_SAMPLES)} filer...")

# Vi looper gjennom datasettet
for i, sample in tqdm(enumerate(dataset), total=min(len(dataset), MAX_SAMPLES)):
    if i >= MAX_SAMPLES:
        break

    try:
        # Hent tekst og lyd
        text = sample.get("transcription", "")
        audio_data = sample.get("audio", {})
        
        audio_array = audio_data.get("array")
        orig_sr = audio_data.get("sampling_rate")

        # Hopp over tomme filer
        if audio_array is None or not text:
            continue

        # --- RESAMPLING (Gjør om til 16kHz) ---
        # Dette bruker CPU. Sørg for at K8s-jobben har CPU-ressurser!
        if orig_sr != TARGET_SR:
            audio_array = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=TARGET_SR)

        # Sjekk lengde (vi vil ha klipp mellom 1 og 20 sekunder)
        duration = len(audio_array) / TARGET_SR
        if duration < 1.0 or duration > 20.0:
            continue

        # Lagre som .wav
        filename = f"fleurs_{i}.wav"
        filepath = os.path.join(audio_dir, filename)
        sf.write(filepath, audio_array, TARGET_SR)
        
        # Lag JSON-struktur for Qwen
        # Merk: Vi bruker absolutte stier (/workspace/...) i selve json-filen hvis mulig,
        # men her bruker vi relativ sti. Train.py fikser ofte dette, men for sikkerhets skyld:
        # Vi lagrer stien relativt til jsonl-filen.
        
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

    except Exception as e:
        print(f"Feil på sample {i}: {e}")
        continue

# Lagre alt til train.jsonl
print(f"Lagrer {success_count} linjer til {JSONL_FILE}...")
with open(os.path.join(OUTPUT_DIR, JSONL_FILE), "w", encoding="utf-8") as f:
    for entry in data_entries:
        f.write(json.dumps(entry) + "\n")

print("="*50)
print(f"DATA-JOBB FERDIG! Klar for trening.")
print("="*50)
