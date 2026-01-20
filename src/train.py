import os
import sys
import torch
from transformers import (
    Qwen2_5OmniForConditionalGeneration, 
    AutoTokenizer, 
    AutoConfig,
    Trainer, 
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# --- KONFIGURASJON ---
MODEL_ID = "Qwen/Qwen2.5-Omni-3B"

# Automatisk sti-deteksjon
if os.path.exists("/workspace/norsk_data/train.jsonl"):
    DATA_PATH = "/workspace/norsk_data/train.jsonl"
    OUTPUT_DIR = "/workspace/output"
else:
    DATA_PATH = "./norsk_data/train.jsonl"
    OUTPUT_DIR = "./output"

print(f"--- STARTER TRENING ---")
print(f"Data: {DATA_PATH}")

def train():
    # 1. Last Tokenizer
    print("Laster tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Last Konfigurasjon og Fiks Bugs
    print("Laster konfigurasjon...")
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # --- VIKTIG BUGFIX ---
    # Talker-modellen (lyd) har mye mindre vokabular enn tekst-modellen.
    # Vi kan IKKE bruke tokenizer.pad_token_id (som er 151643+), for Talker har bare ~50k tokens.
    # Vi setter den hardt til 0 for å unngå "AssertionError: Padding_idx must be within num_embeddings".
    if hasattr(config, "talker_config") and config.talker_config:
        print("⚠️  PATCH: Setter talker_config.pad_token_id = 0 (Safe value).")
        config.talker_config.pad_token_id = 0

    # 3. Last Modell (4-bit QLoRA)
    print("Laster modell (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        config=config,  # Bruker den fiksede konfigurasjonen
        quantization_config=bnb_config, 
        device_map="auto", 
        trust_remote_code=True
    )
    
    # 4. Klargjør for trening
    print("Klargjør modell for k-bit trening...")
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    # Frys audio/visual delene
    if hasattr(model, 'thinker'):
        if hasattr(model.thinker, 'audio_tower'):
            model.thinker.audio_tower.requires_grad_(False)
        if hasattr(model.thinker, 'visual'):
            model.thinker.visual.requires_grad_(False)
    
    # LoRA Config
    print("Aktiverer LoRA...")
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. Last Data
    print(f"Laster datasett fra {DATA_PATH}...")
    try:
        dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    except Exception as e:
        print(f"FEIL ved lasting av data: {e}")
        sys.exit(1)

    # Preprosessering
    def preprocess_function(examples):
        texts = []
        for msg in examples["messages"]:
            if len(msg) > 2 and msg[2]["role"] == "assistant":
                texts.append(msg[2]["content"])
            else:
                texts.append(msg[-1]["content"])
                
        model_inputs = tokenizer(texts, truncation=True, padding="max_length", max_length=128)
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    print("Tokeniserer data...")
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # 6. Treningsoppsett
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="tensorboard",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False
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

    print("=== STARTER TRENING NÅ ===")
    trainer.train()
    
    # 7. Lagre
    print(f"Lagrer ferdig adapter til {OUTPUT_DIR}/final_adapter...")
    trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))

    # 8. Opplasting
    hf_token = os.getenv("HF_TOKEN")
    hf_repo = os.getenv("HF_REPO")

    if hf_token and hf_repo:
        print(f"🚀 Laster opp til Hugging Face: {hf_repo}...")
        try:
            from huggingface_hub import login
            login(token=hf_token)
            trainer.model.push_to_hub(hf_repo)
            tokenizer.push_to_hub(hf_repo)
            print("✅ Opplasting fullført!")
        except Exception as e:
            print(f"❌ Feil ved opplasting: {e}")

if __name__ == "__main__":
    train()
