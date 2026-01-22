import json
import os
import torch
import soundfile as sf
import numpy as np
from transformers import MimiModel, AutoFeatureExtractor
from tqdm import tqdm

# --- KONFIGURASJON ---
# Vi leser fra filen vi lagde med prepare_talker_instruct.py
INPUT_JSONL = "talker_data_instruct.jsonl" 
OUTPUT_JSONL = "norsk_data/talker_data.jsonl" # Dette er filen treningen ser etter

# Vi trenger ikke AUDIO_DIR lenger, for prepare_talker_instruct lagret absolutte stier!

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Bruker enhet: {device}")

# 1. Last ned Mimi
print("Laster Mimi-modellen...")
feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
model = MimiModel.from_pretrained("kyutai/mimi").to(device)
samplerate = feature_extractor.sampling_rate

print(f"Konverterer data fra {INPUT_JSONL} -> {OUTPUT_JSONL}...")

processed_count = 0
skipped_count = 0

# Sjekk at input-filen finnes
if not os.path.exists(INPUT_JSONL):
    print(f"❌ FEIL: Finner ikke {INPUT_JSONL}. Har du kjørt prepare_talker_instruct.py?")
    exit(1)

# Sjekk at output-mappen finnes
os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

with open(INPUT_JSONL, "r", encoding="utf-8") as infile, \
     open(OUTPUT_JSONL, "w", encoding="utf-8") as outfile:
    
    lines = infile.readlines()
    
    for line in tqdm(lines):
        try:
            data = json.loads(line)
            
            # --- ENDRING HER ---
            # Vi henter direkte fra det formatet prepare_talker_instruct.py lagde
            # Teksten inneholder allerede "Si dette på norsk: ..."
            text_instruction = data["text"] 
            audio_path = data["audio"]
            
            if not os.path.exists(audio_path):
                print(f"Fant ikke lydfil: {audio_path}")
                skipped_count += 1
                continue

            # --- MAGIEN (Mimi Encoding) ---
            
            # 1. Last lyd
            audio_array, sr = sf.read(audio_path)
            
            # 2. Resample hvis nødvendig (for sikkerhets skyld, selv om vi gjorde det sist)
            if sr != samplerate:
                # Enkel resampling eller skip. Siden vi lagret i 24k sist, bør dette gå bra.
                pass 

            # 3. Mono mix
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            # 4. Klargjør for Mimi
            inputs = feature_extractor(
                raw_audio=audio_array, 
                sampling_rate=samplerate, 
                return_tensors="pt"
            ).to(device)

            # 5. Encode
            with torch.no_grad():
                encoder_outputs = model.encode(inputs["input_values"])
            
            # 6. Hent koder (Codebook 0 er oftest det viktigste for innhold)
            codes = encoder_outputs.audio_codes[0, 0, :].cpu().numpy()
            
            # 7. Lag streng: "432 55 192 ..."
            token_string = " ".join(map(str, codes))
            
            # --- LAGRE TIL TRENINGS-FORMAT ---
            # Her bygger vi strukturen Qwen forventer under trening
            
            new_entry = {
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant. You can generate audio based on user instructions."
                    },
                    {
                        "role": "user", 
                        # Her bruker vi instruksjonen vi lagde (f.eks "Si dette på norsk: Hei")
                        "content": text_instruction 
                    },
                    {
                        "role": "assistant", 
                        # Her er svaret: Lydkodene pakket inn i tags
                        "content": f"<|audio_bos|>{token_string}<|audio_eos|>"
                    }
                ]
            }
            
            json.dump(new_entry, outfile, ensure_ascii=False)
            outfile.write("\n")
            processed_count += 1
            
        except Exception as e:
            print(f"Feil ved prosessering: {e}")
            skipped_count += 1

print(f"\n✅ Ferdig!")
print(f"Prosesserte: {processed_count}")
print(f"Feilet: {skipped_count}")
