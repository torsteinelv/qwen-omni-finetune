import json
import os
import random
import torch
import soundfile as sf
import numpy as np
from transformers import MimiModel, AutoFeatureExtractor
from tqdm import tqdm

# --- KONFIGURASJON ---
INPUT_JSONL = "/workspace/norsk_data/train.jsonl" 
OUTPUT_JSONL = "/workspace/norsk_data/talker_data.jsonl"

# Instruksjons-maler
INSTRUCTIONS = [
    "Si dette på norsk: ",
    "Les opp følgende setning: ",
    "Uttal denne teksten: ",
    "Kan du si dette for meg: ",
    "Gjenta etter meg på norsk: ",
    "Les denne norske teksten høyt: ",
    "Hvordan sier man dette på norsk: ",
    "Si følgende: ",
    "Vennligst les dette: ",
    "Snakk norsk og si: "
]

# --- VIKTIG: DEN OFFISIELLE QWEN SYSTEM PROMPTEN ---
# Dette må matche 100% med det Qwen krever for å aktivere lyd-modulen.
OFFICIAL_SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Bruker enhet: {device}")

print("Laster Mimi-modellen...")
feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
model = MimiModel.from_pretrained("kyutai/mimi").to(device)
samplerate = feature_extractor.sampling_rate

print(f"Konverterer data fra {INPUT_JSONL} -> {OUTPUT_JSONL}...")

# Sjekk inputfil
if not os.path.exists(INPUT_JSONL):
    if os.path.exists("./norsk_data/train.jsonl"):
        INPUT_JSONL = "./norsk_data/train.jsonl"
    else:
        print(f"❌ FEIL: Finner ikke {INPUT_JSONL}.")
        exit(1)

os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

processed_count = 0
skipped_count = 0

with open(INPUT_JSONL, "r", encoding="utf-8") as infile, \
     open(OUTPUT_JSONL, "w", encoding="utf-8") as outfile:
    
    lines = infile.readlines()
    
    for line in tqdm(lines):
        try:
            data = json.loads(line)
            
            original_text = data["messages"][-1]["content"]
            audio_path = data["audio"]
            
            if not os.path.exists(audio_path):
                rel_path = os.path.join("/workspace/norsk_data/audio_files", os.path.basename(audio_path))
                if os.path.exists(rel_path):
                    audio_path = rel_path
                else:
                    skipped_count += 1
                    continue

            # 2. Lag instruksjon
            prompt_prefix = random.choice(INSTRUCTIONS)
            instruction_text = f"{prompt_prefix}{original_text}"

            # 3. Mimi Encoding
            audio_array, sr = sf.read(audio_path)
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            inputs = feature_extractor(
                raw_audio=audio_array, 
                sampling_rate=samplerate, 
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                encoder_outputs = model.encode(inputs["input_values"])
            
            codes = encoder_outputs.audio_codes[0, 0, :].cpu().numpy()
            token_string = " ".join(map(str, codes))
            
            # 4. Lagre med RIKTIG System Prompt
            new_entry = {
                "messages": [
                    {
                        "role": "system", 
                        "content": OFFICIAL_SYSTEM_PROMPT # <--- ENDRINGEN HER!
                    },
                    {
                        "role": "user", 
                        "content": instruction_text 
                    },
                    {
                        "role": "assistant", 
                        "content": f"<|audio_bos|>{token_string}<|audio_eos|>"
                    }
                ]
            }
            
            json.dump(new_entry, outfile, ensure_ascii=False)
            outfile.write("\n")
            processed_count += 1
            
        except Exception as e:
            # print(f"Feil: {e}")
            skipped_count += 1

print(f"\n✅ Ferdig! {processed_count} klipp klargjort med offisiell prompt.")
