import json
import os
import random
import torch
import soundfile as sf
import numpy as np
from transformers import MimiModel, AutoFeatureExtractor
from tqdm import tqdm

# --- KONFIGURASJON ---
# Vi leser filen som data_npsc.py ALLEREDE har laget (så slipper vi å laste ned på nytt)
INPUT_JSONL = "/workspace/norsk_data/train.jsonl" 

# Vi skriver direkte til filen som train_talker_only.py forventer
OUTPUT_JSONL = "/workspace/norsk_data/talker_data.jsonl"

# Instruksjons-maler (Dette er hjernen i "Instruction Tuning")
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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Bruker enhet: {device}")

print("Laster Mimi-modellen...")
feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
model = MimiModel.from_pretrained("kyutai/mimi").to(device)
samplerate = feature_extractor.sampling_rate

print(f"Konverterer data fra {INPUT_JSONL} -> {OUTPUT_JSONL}...")
print("(Legger til instruksjoner + Mimi-koder i samme slengen)")

# Sjekk inputfil
if not os.path.exists(INPUT_JSONL):
    # Fallback hvis stien er relativ
    if os.path.exists("./norsk_data/train.jsonl"):
        INPUT_JSONL = "./norsk_data/train.jsonl"
    else:
        print(f"❌ FEIL: Finner ikke {INPUT_JSONL}. Har data_npsc.py kjørt ferdig?")
        exit(1)

# Sørg for at output-mappen finnes
os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

processed_count = 0
skipped_count = 0

with open(INPUT_JSONL, "r", encoding="utf-8") as infile, \
     open(OUTPUT_JSONL, "w", encoding="utf-8") as outfile:
    
    lines = infile.readlines()
    
    for line in tqdm(lines):
        try:
            data = json.loads(line)
            
            # 1. Hent tekst og lydsti fra data_npsc.py formatet
            # Formatet der er: {"messages": [..., {"role": "assistant", "content": "TEKSTEN"}], "audio": "STI"}
            original_text = data["messages"][-1]["content"]
            audio_path = data["audio"]
            
            if not os.path.exists(audio_path):
                # Prøv relativ sti hvis absolutt feiler
                rel_path = os.path.join("/workspace/norsk_data/audio_files", os.path.basename(audio_path))
                if os.path.exists(rel_path):
                    audio_path = rel_path
                else:
                    # print(f"⚠️ Fant ikke lydfil: {audio_path}") # Sparer loggen for spam
                    skipped_count += 1
                    continue

            # 2. Lag instruksjon (Instruction Tuning)
            prompt_prefix = random.choice(INSTRUCTIONS)
            instruction_text = f"{prompt_prefix}{original_text}"

            # 3. Mimi Encoding
            audio_array, sr = sf.read(audio_path)
            
            # Mix til mono hvis stereo
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
            
            # 4. Lagre nytt format (Instruction -> Audio Tokens)
            new_entry = {
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant. You can generate audio based on user instructions."
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
            print(f"Feil ved prosessering: {e}")
            skipped_count += 1

print(f"\n✅ Ferdig!")
print(f"Prosesserte klipp: {processed_count}")
print(f"Feilet/Hoppet over: {skipped_count}")
print(f"Klar for trening! Fil lagret: {OUTPUT_JSONL}")
