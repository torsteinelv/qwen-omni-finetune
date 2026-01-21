import torch
import soundfile as sf
from transformers import Qwen2_5OmniForConditionalGeneration, AutoProcessor, AutoConfig
import os

# --- KONFIGURASJON ---
BASE_MODEL = "Qwen/Qwen2.5-Omni-3B"
OUTPUT_FILE = "base_test.wav"
TEXT_TO_SPEAK = "This is a test of the Qwen Omni model."

print("1. Laster og fikser konfigurasjon...")
config = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
if hasattr(config, "talker_config") and config.talker_config:
    config.talker_config.pad_token_id = 0

print("2. Laster base-modell (Qwen2.5-Omni) UTEN adapter...")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    BASE_MODEL, 
    config=config,
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    trust_remote_code=True
)

print("3. Klargjør input (MED RIKTIG FORMAT)...")
processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

# --- FIX: Content må være en liste med dictionaries, ikke bare en streng ---
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
            {"type": "text", "text": TEXT_TO_SPEAK}
        ],
    },
]

text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
inputs = processor(text=text, return_tensors="pt", padding=True)
inputs = inputs.to(model.device)

print("4. Genererer tale (Inference)...")
with torch.no_grad():
    text_ids, audio_values = model.generate(
        **inputs, 
        max_new_tokens=256,
        use_audio_in_video=False
    )

print("5. Lagrer lydfil...")
if audio_values is not None:
    audio_data = audio_values.reshape(-1).detach().cpu().float().numpy()
    sf.write(OUTPUT_FILE, audio_data, samplerate=24000)
    print(f"🎉 Suksess! Lyd lagret til: {OUTPUT_FILE}")
    print(f"   Filstørrelse: {len(audio_data)/24000:.2f} sekunder")
else:
    print("❌ Ingen lyd ble generert.")

decoded_text = processor.batch_decode(text_ids, skip_special_tokens=True)[0]
print(f"\nModellens tekst-respons: {decoded_text}")
