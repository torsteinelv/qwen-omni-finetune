import json
import os
import torch
import soundfile as sf
import numpy as np
from transformers import MimiModel, AutoFeatureExtractor
from tqdm import tqdm

# --- KONFIGURASJON ---
INPUT_JSONL = "norsk_data/train.jsonl"
OUTPUT_JSONL = "norsk_data/talker_data.jsonl"
AUDIO_DIR = "norsk_data/audio_files/" # Mappen der filene faktisk ligger

# Vi bruker GPU hvis vi kan (mye raskere!)
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

with open(INPUT_JSONL, "r", encoding="utf-8") as infile, \
     open(OUTPUT_JSONL, "w", encoding="utf-8") as outfile:
    
    # Vi leser filen linje for linje
    lines = infile.readlines()
    
    for line in tqdm(lines):
        try:
            data = json.loads(line)
            
            # Hent tekst (Svaret fra assistenten i den gamle filen)
            # Formatet i din fil: messages[2]['content'] er teksten
            text = data["messages"][-1]["content"]
            
            # Hent filnavn (Fiks stien hvis den inneholder ./norsk_data...)
            audio_path_raw = data["audio"]
            # Vi stripper bort "./norsk_data/audio_files/" delen for å være sikre, 
            # og bygger stien på nytt.
            filename = os.path.basename(audio_path_raw)
            audio_path = os.path.join(AUDIO_DIR, filename)
            
            if not os.path.exists(audio_path):
                # Prøv den absolutte stien fra jsonl hvis den relative feilet
                if os.path.exists(audio_path_raw):
                    audio_path = audio_path_raw
                else:
                    skipped_count += 1
                    continue

            # --- MAGIEN SKJER HER: Lyd -> Mimi Tokens ---
            
            # 1. Last lyd
            audio_array, _ = sf.read(audio_path)
            
            # 2. Sikre at det er mono og riktig form (flat array)
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1) # Mix til mono
            
            # 3. Klargjør for Mimi
            inputs = feature_extractor(
                raw_audio=audio_array, 
                sampling_rate=samplerate, 
                return_tensors="pt"
            ).to(device)

            # 4. Encode
            with torch.no_grad():
                encoder_outputs = model.encode(inputs["input_values"])
            
            # 5. Hent koder (Batch 0, Codebook 0, Alle tidssteg)
            # audio_codes shape: [Batch, Codebooks(32), Time]
            codes = encoder_outputs.audio_codes[0, 0, :].cpu().numpy()
            
            # 6. Gjør om til en streng av tall: "432 55 192 ..."
            token_string = " ".join(map(str, codes))
            
            # --- LAGRE NY TRENINGSDATA ---
            # Vi lærer Qwen: "Når brukeren sier X, skal du svare med disse lyd-kodene."
            
            new_entry = {
                "messages": [
                    {
                        "role": "system", 
                        "content": "Du er en AI som konverterer tekst til Mimi lydkoder."
                    },
                    {
                        "role": "user", 
                        "content": f"Si dette på norsk: {text}"
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
            print(f"Feil ved prosessering: {e}")
            skipped_count += 1

print(f"\n✅ Ferdig!")
print(f"Prosesserte: {processed_count}")
print(f"Hoppet over (feil/mangler fil): {skipped_count}")
print(f"Nytt datasett lagret i: {OUTPUT_JSONL}")
