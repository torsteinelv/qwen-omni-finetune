import os
import sys
import torch
from transformers import (
    Qwen2_5OmniForConditionalGeneration, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# --- KONFIGURASJON ---
MODEL_ID = "Qwen/Qwen2.5-Omni-3B"

# Automatisk sti-deteksjon (Docker vs Lokal)
if os.path.exists("/workspace/norsk_data/train.jsonl"):
    DATA_PATH = "/workspace/norsk_data/train.jsonl"
    OUTPUT_DIR = "/workspace/output"
else:
    DATA_PATH = "./norsk_data/train.jsonl"
    OUTPUT_DIR = "./output"

print(f"--- STARTER TRENING ---")
print(f"Data: {DATA_PATH}")
print(f"Output: {OUTPUT_DIR}")

def train():
    # 1. Last Tokenizer
    print("Laster tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Last Modell (4-bit QLoRA)
    print("Laster modell...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        quantization_config=bnb_config, 
        device_map="auto", 
        trust_remote_code=True
    )
    
    # 3. Klargjør for trening
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    # Frys audio/visual delene for å spare VRAM
    if hasattr(model, 'thinker'):
        model.thinker.audio_tower.requires_grad_(False)
        model.thinker.visual.requires_grad_(False)
    
    # LoRA Config
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Last Data
    print("Laster datasett...")
    try:
        dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    except FileNotFoundError:
        print(f"FEIL: Fant ikke {DATA_PATH}. Sjekk om data.py har kjørt.")
        sys.exit(1)

    # Korrekt batch-prosessering av meldinger
    def preprocess_function(examples):
        # Henter tekst fra assistant-rollen (index 2 i listen av meldinger)
        texts = [msg[2]["content"] for msg in examples["messages"]]
        model_inputs = tokenizer(texts, truncation=True, padding="max_length", max_length=128)
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # 5. Treningsoppsett
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=1,                # Trener gjennom hele datasettet én gang
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="tensorboard",
        remove_unused_columns=False
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

    print("Starter selve treningen...")
    trainer.train()
    
    # 6. Lagre lokalt
    print(f"Lagrer adapter til {OUTPUT_DIR}/final_adapter...")
    trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))

    # 7. Last opp til Hugging Face (Hvis env vars finnes)
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
    else:
        print("Ingen HF_TOKEN funnet, hopper over opplasting.")

if __name__ == "__main__":
    train()
