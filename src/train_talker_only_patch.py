import os
import sys
import torch
import torch.nn as nn
from transformers import (
    Qwen2_5OmniForConditionalGeneration, 
    AutoTokenizer, 
    AutoConfig,
    Trainer, 
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import load_dataset

# --- KONFIGURASJON ---
MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
PHASE1_ADAPTER = "/workspace/output/final_adapter"

# Sjekk at data finnes
if os.path.exists("/workspace/norsk_data/talker_data.jsonl"):
    DATA_PATH = "/workspace/norsk_data/talker_data.jsonl"
    OUTPUT_DIR = "/workspace/output_talker"
else:
    print("❌ Finner ikke talker_data.jsonl!")
    sys.exit(1)

# --- CUSTOM WRAPPER ---
class QwenOmniWrapper(nn.Module):
    def __init__(self, omni_model):
        super().__init__()
        self.omni_model = omni_model
        self.config = omni_model.config
        
        # Kopier attributter som Trainer trenger
        self.gradient_checkpointing_enable = self.omni_model.thinker.gradient_checkpointing_enable
        self.save_pretrained = self.omni_model.save_pretrained
        self.can_generate = True 

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Fjern multimodale inputs som ikke trengs i denne fasen
        kwargs_clean = {k: v for k, v in kwargs.items() 
                       if k not in ['pixel_values', 'audio_values', 'video_values']}
        
        return self.omni_model.thinker(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs_clean
        )
    
    def enable_input_require_grads(self):
        self.omni_model.thinker.enable_input_require_grads()

def train():
    print(f"--- STARTER OPTIMALISERT TALKER TRENING (REN VERSJON) ---")

    # 1. Last Tokenizer
    print("Laster tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Last Konfigurasjon
    print("Laster konfigurasjon...")
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    if hasattr(config, "talker_config") and config.talker_config:
        config.talker_config.pad_token_id = 0
    config.use_cache = False

    # 3. Last Hovedmodell med 8-bit
    print("Laster hovedmodell i 8-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    
    omni_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        config=config,
        quantization_config=bnb_config, 
        device_map={"": 0},
        trust_remote_code=True
    )
    
    omni_model = prepare_model_for_kbit_training(omni_model)

    # 4. Håndter Adapter
    if os.path.exists(os.path.join(PHASE1_ADAPTER, "adapter_config.json")):
        print(f"🧠 Laster eksisterende Fase 1 adapter fra {PHASE1_ADAPTER}...")
        thinker = PeftModel.from_pretrained(omni_model.thinker, PHASE1_ADAPTER, is_trainable=True)
    else:
        print("💡 Ingen Fase 1 adapter funnet. Oppretter NY LoRA-adapter for Talker...")
        peft_config = LoraConfig(
            r=16, 
            lora_alpha=32, 
            lora_dropout=0.05, 
            bias="none", 
            target_modules=["
