import torch
import soundfile as sf
from transformers import Qwen2_5OmniForConditionalGeneration, AutoProcessor
from peft import PeftModel
import os

# --- KONFIGURASJON ---
BASE_MODEL = "Qwen/Qwen2.5-Omni-3B"
ADAPTER_PATH = "/workspace/output_talker/final_adapter" # Din nye "talker"-hjerne
OUTPUT_FILE = "norsk_test.wav"
TEXT_TO_SPEAK = "Hei! Dette er en test. Nå kan jeg snakke flytende norsk takket være din trening."

print("1. Laster base-modell (Qwen2.5-Omni)...")
# Vi bruker bfloat16 siden du trente med det (og det sparer minne)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    BASE_MODEL, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    trust_remote_code=True
)

print(f"2. Laster din adapter fra {ADAPTER_PATH}...")
# VIKTIG: Her kobler vi din finetune på "thinker"-delen av modellen
if os.path.exists(ADAPTER_PATH):
    model.thinker = PeftModel.from_pretrained(model.thinker, ADAPTER_PATH)
    print("✅ Adapter lastet suksessfullt!")
else:
    print("⚠️ ADVARSEL: Fant ikke adapteren. Kjører med original engelsk/kinesisk stemme.")

print("3. Klargjør input...")
processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

# VIKTIG: Dette system-promptet trigger lyd-generering i modellen
SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

conversation = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT,
    },
    {
        "role": "user",
        # Vi bruker samme format som i treningen: "Si dette på norsk: <tekst>"
        "content": f"Si dette på norsk: {TEXT_TO_SPEAK}",
    },
]

# Prosesser tekst
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
inputs = processor(text=text, return_tensors="pt", padding=True)
inputs = inputs.to(model.device)

print("4. Genererer tale (Inference)...")
# Her skjer magien. Vi ber om lyd ved å ikke sette output til kun tekst.
# Merk: Qwen-Omni returnerer (text_ids, audio_values)
with torch.no_grad():
    text_ids, audio_values = model.generate(
        **inputs, 
        max_new_tokens=256,
        use_audio_in_video=False # Vi har ingen video-input her
    )

print("5. Lagrer lydfil...")
if audio_values is not None:
    # Qwen-Omni bruker 24kHz samplerate
    audio_data = audio_values.reshape(-1).detach().cpu().float().numpy()
    sf.write(OUTPUT_FILE, audio_data, samplerate=24000)
    print(f"🎉 Suksess! Lyd lagret til: {OUTPUT_FILE}")
    print(f"   Filstørrelse: {len(audio_data)/24000:.2f} sekunder")
else:
    print("❌ Ingen lyd ble generert. Sjekk system-prompt eller adapter.")

# Print også hva modellen "tenkte" (tekst-delen)
decoded_text = processor.batch_decode(text_ids, skip_special_tokens=True)[0]
print(f"\nModellens tekst-respons: {decoded_text}")
