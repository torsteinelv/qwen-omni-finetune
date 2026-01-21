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

# --- CUSTOM WRAPPER (Fikser minneproblemer ved lagring) ---
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
    print(f"--- STARTER OPTIMALISERT TALKER TRENING (20GB VRAM MODUS) ---")

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

    # 3. Last Hovedmodell med 8-bit (bruker mer VRAM enn 4-bit, men gir bedre kvalitet)
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

    # 4. Håndter Adapter (Last eksisterende ELLER opprett ny)
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
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        thinker = get_peft_model(omni_model.thinker, peft_config)
    
    omni_model.thinker = thinker

    # 5. Wrap modellen
    print("Wrapper modellen...")
    model = QwenOmniWrapper(omni_model)
    model.omni_model.thinker.print_trainable_parameters()

    # 6. Last og prosesser data
    print(f"Laster data fra {DATA_PATH}...")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    def preprocess_function(examples):
        texts = []
        for msg in examples["messages"]:
            full_text = ""
            for m in msg: 
                full_text += m["content"]
            texts.append(full_text)
        
        # Økt max_length til 1024 for å utnytte VRAM og unngå kutting av lange lydkoder
        model_inputs = tokenizer(texts, truncation=True, padding="max_length", max_length=1024)
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # 7. Treningsoppsett (Optimalisert for RTX 4000 Ada)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,      # Økt batch size for bedre GPU-utnyttelse
        gradient_accumulation_steps=4,      # Justert for å beholde stabil trening
        learning_rate=1e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        bf16=True,                          # Bruker BF16 (optimalt for Ada-arkitektur)
        optim="paged_adamw_8bit",
        report_to="tensorboard",
        remove_unused_columns=False, 
        label_names=["labels"],
        ddp_find_unused_parameters=False,
        save_total_limit=2 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=lambda data: {
            'input_ids': torch.stack([torch.tensor(f['input_ids']) for f in data]),
            'attention_mask': torch.stack([torch.tensor(f['attention_mask']) for f in data]),
            'labels': torch.stack([torch.tensor(f['labels']) for f in data])
        }
    )

    print("=== STARTER TRENING (TALKER) ===")
    
    # Auto-resume logikk
    last_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            last_checkpoint = os.path.join(OUTPUT_DIR, checkpoints[-1])
            print(f"♻️ Fant gammel trening! Fortsetter fra: {last_checkpoint}")
    
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # 8. Lagre sluttresultat lokalt
    print(f"Lagrer TALKER adapter til {OUTPUT_DIR}/final_adapter...")
    model.omni_model.thinker.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))

    # 9. Opplasting til Hugging Face
    hf_token = os.getenv("HF_TOKEN")
    hf_repo = os.getenv("HF_REPO")

    if hf_token and hf_repo:
        print(f"🚀 Laster opp til Hugging Face: {hf_repo}...")
        try:
            from huggingface_hub import login
            login(token=hf_token)
            model.omni_model.thinker.push_to_hub(hf_repo)
            tokenizer.push_to_hub(hf_repo)
            print("✅ Opplasting fullført!")
        except Exception as e:
            print(f"❌ Feil ved opplasting: {e}")

if __name__ == "__main__":
    train()
