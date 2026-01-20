import os
import torch
import sys
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

# Vi sjekker om vi er i Docker (/workspace) eller lokalt for å finne filstien
if os.path.exists("/workspace/norsk_data/train.jsonl"):
    DATA_PATH = "/workspace/norsk_data/train.jsonl"
    OUTPUT_DIR = "/workspace/output"
else:
    # Fallback for lokal kjøring
    DATA_PATH = "./norsk_data/train.jsonl"
    OUTPUT_DIR = "./output"

print(f"Starter trening.")
print(f"Data: {DATA_PATH}")
print(f"Output: {OUTPUT_DIR}")

def train():
    # 1. Last Tokenizer
    print("Laster tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # Qwen mangler noen ganger pad_token, vi setter det til eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Last Modell med 4-bit kvantisering (QLoRA)
    print("Laster modell (4-bit)...")
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
    
    # 3. Klargjør modellen (Frys unødvendige deler)
    print("Konfigurerer modell og LoRA...")
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    # Frys Audio/Visual encoders for å spare minne (vi trener språkforståelsen først)
    # Dette er viktig for å unngå OOM (Out of Memory)
    if hasattr(model, 'thinker'):
        if hasattr(model.thinker, 'audio_tower'):
            model.thinker.audio_tower.requires_grad_(False)
        if hasattr(model.thinker, 'visual'):
            model.thinker.visual.requires_grad_(False)
    
    # Konfigurer LoRA (Parameter Efficient Fine-Tuning)
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Last og prosesser datasett
    print("Laster datasett...")
    try:
        dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    except FileNotFoundError:
        print(f"FEIL: Fant ikke filen {DATA_PATH}. Har du kjørt data.py?")
        sys.exit(1)

    # --- HER VAR FEILEN DIN, NÅ FIKSET ---
    def preprocess_function(examples):
        # examples["messages"] er en liste av lister (batch)
        # Vi henter ut teksten fra Assistentens svar (som er index 2 i meldings-listen)
        # Format: [System, User, Assistant]
        texts = [msg[2]["content"] for msg in examples["messages"]]
        
        # Tokenizer tekstene
        model_inputs = tokenizer(
            texts, 
            truncation=True, 
            padding="max_length", 
            max_length=128
        )
        
        # For tekst-generering skal labels være det samme som input_ids
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        return model_inputs

    print("Tokeniserer data...")
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # 5. Treningsoppsett
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,     # Lav batch size pga VRAM
        gradient_accumulation_steps=8,     # Simulerer større batch size
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=100,                     # Øk dette tallet for lengre trening!
        save_steps=50,
        fp16=True,                         # Raskere trening på GPU
        optim="paged_adamw_8bit",          # Sparer minne
        report_to="tensorboard",           # Logg fremgang
        remove_unused_columns=False        # Viktig for custom modeller
    )

    # 6. Start Treneren
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

    print("Starter trening...")
    trainer.train()
    
    # 7. Lagre resultat
    print(f"Lagrer ferdig adapter til {OUTPUT_DIR}/final_adapter...")
    trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))

if __name__ == "__main__":
    train()
