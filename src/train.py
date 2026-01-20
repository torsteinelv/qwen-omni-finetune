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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# --- KONFIGURASJON ---
MODEL_ID = "Qwen/Qwen2.5-Omni-3B"

if os.path.exists("/workspace/norsk_data/train.jsonl"):
    DATA_PATH = "/workspace/norsk_data/train.jsonl"
    OUTPUT_DIR = "/workspace/output"
else:
    DATA_PATH = "./norsk_data/train.jsonl"
    OUTPUT_DIR = "./output"

print(f"--- STARTER TRENING (Memory Fix Edition) ---")
print(f"Data: {DATA_PATH}")

# --- CUSTOM WRAPPER (Oppdatert for å fikse lagringskrasj) ---
class QwenOmniWrapper(nn.Module):
    def __init__(self, omni_model):
        super().__init__()
        self.omni_model = omni_model
        # VIKTIG ENDRING: Vi lagrer IKKE self.thinker her lenger.
        # Det skapte "Duplicate memory" feilen fordi PyTorch trodde vi hadde to modeller.
        
        self.config = omni_model.config
        
        # Kopier attributter som Trainer trenger
        self.gradient_checkpointing_enable = self.omni_model.thinker.gradient_checkpointing_enable
        self.save_pretrained = self.omni_model.save_pretrained
        self.can_generate = True 

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        kwargs_clean = {k: v for k, v in kwargs.items() 
                       if k not in ['pixel_values', 'audio_values', 'video_values']}
        
        # Vi henter thinker dynamisk her i stedet for å ha den i __init__
        return self.omni_model.thinker(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs_clean
        )
    
    def enable_input_require_grads(self):
        self.omni_model.thinker.enable_input_require_grads()


def train():
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

    # 3. Last Hovedmodell
    print("Laster hovedmodell...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    omni_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        config=config,
        quantization_config=bnb_config, 
        device_map={"": 0},
        trust_remote_code=True
    )
    
    omni_model = prepare_model_for_kbit_training(omni_model)

    # 4. INJISER LORA PÅ THINKER
    print("Aktiverer LoRA på Thinker...")
    
    thinker = omni_model.thinker
    
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.05, 
        bias="none", 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    thinker = get_peft_model(thinker, peft_config)
    omni_model.thinker = thinker

    # 5. WRAP MODELLEN
    print("Wrapper modellen...")
    model = QwenOmniWrapper(omni_model)
    
    # Print stats (må gå via omni_model nå)
    model.omni_model.thinker.print_trainable_parameters()

    # 6. Last Data
    print(f"Laster datasett fra {DATA_PATH}...")
    try:
        dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    except Exception as e:
        print(f"FEIL ved lasting av data: {e}")
        sys.exit(1)

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

    # 7. Treningsoppsett
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50, # Her den krasjet sist, nå skal det gå bra!
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="tensorboard",
        remove_unused_columns=False, 
        label_names=["labels"],
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
    
    # 8. Lagre
    print(f"Lagrer ferdig adapter til {OUTPUT_DIR}/final_adapter...")
    
    # VIKTIG: Vi lagrer KUN thinkeren, ikke hele wrapperen
    model.omni_model.thinker.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))

    # 9. Opplasting
    hf_token = os.getenv("HF_TOKEN")
    hf_repo = os.getenv("HF_REPO")

    if hf_token and hf_repo:
        print(f"🚀 Laster opp til Hugging Face: {hf_repo}...")
        try:
            from huggingface_hub import login
            login(token=hf_token)
            # Push adapteren
            model.omni_model.thinker.push_to_hub(hf_repo)
            tokenizer.push_to_hub(hf_repo)
            print("✅ Opplasting fullført!")
        except Exception as e:
            print(f"❌ Feil ved opplasting: {e}")

if __name__ == "__main__":
    train()
