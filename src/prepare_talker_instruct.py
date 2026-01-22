import os
import json
import random
import librosa
from datasets import load_dataset
from tqdm import tqdm

# --- KONFIGURASJON ---
OUTPUT_FILE = "talker_data_instruct.jsonl"
MIN_DURATION = 1.5  # Kaster alt under 1.5 sekunder (fjerner "Pai..." bugs)
MAX_DURATION = 15.0 # Kaster alt over 15 sekunder (hindrer OOM)

# --- INSTRUKSJONER ---
# Vi bruker mange varianter for å lære modellen konseptet "å lese opp"
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

def prepare_data():
    print("🚀 Starter forberedelse av 'Instruction Tuned' data...")
    
    # Last ned NPSC datasettet (kun 10% for test, eller fjern split for full trening)
    # For full trening, fjern 'split="train[:20%]"'
    print("Laster NPSC datasettet...")
    dataset = load_dataset("NbAiLab/NPSC", "16K_mp3_bokmaal", split="train", trust_remote_code=True)
    
    valid_count = 0
    skipped_short = 0
    skipped_long = 0
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in tqdm(dataset):
            try:
                # 1. Sjekk lydfilen
                audio_path = item["audio"]["path"]
                text = item["text"]
                
                # Vi må vite lengden for å filtrere
                duration = librosa.get_duration(path=audio_path)
                
                if duration < MIN_DURATION:
                    skipped_short += 1
                    continue
                if duration > MAX_DURATION:
                    skipped_long += 1
                    continue
                
                # 2. Lag instruksjonen
                # Vi velger en tilfeldig start-frase hver gang
                prompt_prefix = random.choice(INSTRUCTIONS)
                full_prompt = f"{prompt_prefix}{text}"
                
                # 3. Formatet Qwen-Omni forventer for trening
                # Vi legger instruksen i "text" og stien i "audio"
                entry = {
                    "text": full_prompt,  # F.eks: "Si dette på norsk: Presidenten er her."
                    "audio": audio_path,
                    "duration": duration
                }
                
                f.write(json.dumps(entry) + "\n")
                valid_count += 1
                
            except Exception as e:
                print(f"Feil med fil: {e}")

    print("\n" + "="*50)
    print(f"✅ Ferdig! Data lagret til: {OUTPUT_FILE}")
    print(f"📊 Totalt antall klipp: {valid_count}")
    print(f"🗑️  Kastet (for korte < {MIN_DURATION}s): {skipped_short}")
    print(f"🗑️  Kastet (for lange > {MAX_DURATION}s): {skipped_long}")
    print("="*50)
    print("Tips: Sjekk at 'valid_count' er minst 5000-10000 for god trening.")

if __name__ == "__main__":
    prepare_data()
