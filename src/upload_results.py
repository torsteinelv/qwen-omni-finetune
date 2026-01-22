import os
import torch
import soundfile as sf
from transformers import Qwen2_5OmniForConditionalGeneration, AutoProcessor, AutoConfig
from peft import PeftModel
from huggingface_hub import HfApi, login, snapshot_download

# --- CONFIGURATION ---
BASE_MODEL = "Qwen/Qwen2.5-Omni-3B"
ADAPTER_DIR = os.getenv("ADAPTER_PATH", "/workspace/output_talker/final_adapter")
OUTPUT_FILE = "norsk_lora_test.wav"
TEXT_TO_SPEAK = "Hei! Dette er en test. Nå kan jeg snakke flytende norsk takket være din trening."

def generate_and_upload():
    hf_token = os.getenv("HF_TOKEN")
    hf_repo = os.getenv("HF_REPO")
    
    if not hf_token or not hf_repo:
        print("❌ Missing HF_TOKEN or HF_REPO!")
        return

    # 1. Login and download adapter
    print(f"1. Logging in and checking adapter in {hf_repo}...")
    login(token=hf_token)
    
    try:
        snapshot_download(repo_id=hf_repo, local_dir=ADAPTER_DIR)
        print(f"✅ Adapter downloaded to {ADAPTER_DIR}")
    except Exception as e:
        print(f"⚠️ Could not download adapter (might not be uploaded yet): {e}")
        if not os.path.exists(ADAPTER_DIR):
            return

    # 2. Load model and inject adapter
    print("2. Loading model and configuration...")
    config = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if hasattr(config, "talker_config") and config.talker_config:
        config.talker_config.pad_token_id = 0 # Fix known config issue

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        BASE_MODEL, config=config, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    # Inject LoRA weights into the "Thinker" module
    model.thinker = PeftModel.from_pretrained(model.thinker, ADAPTER_DIR)
    print("✅ Adapter injected successfully!")

    # 3. Prepare Input (The Critical Formatting)
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f"Si dette på norsk: {TEXT_TO_SPEAK}"}],
        },
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, return_tensors="pt", padding=True).to(model.device)

    print("3. Generating audio sample...")
    with torch.no_grad():
        _, audio_values = model.generate(
            **inputs, 
            max_new_tokens=256, 
            use_audio_in_video=False
        )

    if audio_values is not None:
        audio_data = audio_values.reshape(-1).detach().cpu().float().numpy()
        sf.write(OUTPUT_FILE, audio_data, samplerate=24000)
        print(f"🎉 Sample generated: {OUTPUT_FILE}")
    
    # 4. Upload audio sample to Hugging Face
    print("4. Uploading sample to Hugging Face...")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=OUTPUT_FILE,
        path_in_repo=f"samples/{OUTPUT_FILE}",
        repo_id=hf_repo
    )
    print(f"✅ Upload complete to {hf_repo}/samples/")

if __name__ == "__main__":
    generate_and_upload()
