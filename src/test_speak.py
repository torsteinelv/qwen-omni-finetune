import torch
import soundfile as sf
from transformers import Qwen2_5OmniForConditionalGeneration, AutoProcessor, AutoConfig
from peft import PeftModel
import os

# --- KONFIGURASJON ---
BASE_MODEL = "Qwen/Qwen2.5-Omni-3B"
ADAPTER_PATH = "/workspace/output_talker/final_adapter" 
OUTPUT_FILE = "norsk_lora_test.wav"
TEXT_TO_SPEAK = "Hei! Dette er en test. Nå kan jeg snakke flytende norsk takket være din trening."

print("1. Laster og fikser konfigurasjon...")
# Fikser den kjente config-feilen
config = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
if hasattr(config, "talker_config") and config.talker_config:
    config.talker_config.pad_token_id = 0

print("2. Laster base-modell (Qwen2.5-Omni)...")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    BASE_MODEL, 
    config=config,
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    trust_remote_code=True
)

print(f"3. Laster din adapter fra {ADAPTER_PATH}...")
if os.path.exists(ADAPTER_PATH):
    # Laster LoRA-vektene oppå "Thinker"-delen av modellen
    model.thinker = PeftModel.from_pretrained(model.thinker, ADAPTER_PATH)
    print("✅ Adapter lastet suksessfullt!")
else:
    print("⚠️ ADVARSEL: Fant ikke adapteren. Kjører uten (vil snakke engelsk/kinesisk).")

print("4. Klargjør input (MED RIKTIG FORMAT)...")
processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

# --- VIKTIG: Her er fixen som gjør at det virker ---
conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": SYSTEM_PROMPT}
        ],
    },
    {
        "role": "user",
        "content": [
            # Vi bruker samme prompt-stil som under treningen
            {"type": "text", "text": f"Si dette på norsk: {TEXT_TO_SPEAK}"}
        ],
    },
]

text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
inputs = processor(text=text, return_tensors="pt", padding=True)
inputs = inputs.to(model.device)

print("5. Genererer tale (Inference)...")
with torch.no_grad():
    text_ids, audio_values = model.generate(
        **inputs, 
        max_new_tokens=256,
        use_audio_in_video=False
    )

print("6. Lagrer lydfil...")
if audio_values is not None:
    audio_data = audio_values.reshape(-1).detach().cpu().float().numpy()
    sf.write(OUTPUT_FILE, audio_data, samplerate=24000)
    print(f"🎉 Suksess! Lyd lagret til: {OUTPUT_FILE}")
    print(f"   Filstørrelse: {len(audio_data)/24000:.2f} sekunder")
else:
    print("❌ Ingen lyd ble generert.")

decoded_text = processor.batch_decode(text_ids, skip_special_tokens=True)[0]
print(f"\nModellens tekst-respons: {decoded_text}")
