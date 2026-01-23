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
DATA_PATH = "/workspace/norsk_data/talker_data.jsonl"
OUTPUT_DIR = "/workspace/output_talker"
PHASE1_ADAPTER = "/workspace/output/final_adapter" # Hvis du har en adapter fra før

# --- CUSTOM WRAPPER ---
class QwenOmniWrapper(nn.Module):
    def __init__(self, omni_model):
        super().__init__()
        self.omni_model = omni_model
        self.config = omni_model.config
        self.gradient_checkpointing_enable = self.omni_model.thinker.gradient_checkpointing_enable
        self.save_pretrained = self.omni_model.save_pretrained
        self.can_generate = True 

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        kwargs_clean = {k: v for k, v in kwargs.items() 
                       if k not in ['pixel_values', 'audio_values', 'video_values']}
        return self.omni_model.thinker(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs_clean)
    
    def enable_input_require_grads(self):
        self.omni_model.thinker.enable_input_require_grads()

def train():
    print("--- STARTER FULL PRODUKSJONSTRENING (TALKER) ---")

    print("Laster tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    print("Laster modell i 8-bit...")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    omni_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map={"": 0}, trust_remote_code=True
    )
    
    # --- FIX: PATCH FOR PEFT ---
    print("🔧 Patcher get_input_embeddings for PEFT...")
    omni_model.get_input_embeddings = lambda: omni_model.thinker.get_input_embeddings()
    # ---------------------------

    omni_model = prepare_model_for_kbit_training(omni_model)

    # Sjekk om vi skal fortsette på en gammel adapter eller lage ny
    if os.path.exists(os.path.join(PHASE1_ADAPTER, "adapter_config.json")):
        print(f"🧠 Laster eksisterende adapter fra {PHASE1_ADAPTER}...")
        thinker = PeftModel.from_pretrained(omni_model.thinker, PHASE1_ADAPTER, is_trainable=True)
    else:
        print("Oppretter NY LoRA-adapter...")
        peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        thinker = get_peft_model(omni_model.thinker, peft_config)
    
    omni_model.thinker = thinker
    model = QwenOmniWrapper(omni_model)
    model.omni_model.thinker.print_trainable_parameters()

    print(f"Laster data fra {DATA_PATH}...")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    def preprocess_function(examples):
        texts = []
        for msg in examples["messages"]:
            full_text = ""
            for m in msg: full_text += m["content"]
            texts.append(full_text)
        
        # Vi øker lengden til 1024 for produksjon for å få med lengre lydklipp
        model_inputs = tokenizer(texts, truncation=True, padding="max_length", max_length=1024) 
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # --- OPTIMALISERT TRENINGSOPPSETT ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,      # Økt fra 1 til 4 (bedre fart)
        gradient_accumulation_steps=4,      # Samler opp gradients (tilsvarer batch size 16)
        learning_rate=2e-5,
        num_train_epochs=1,                 # Kjører gjennom datasetet 3 ganger
        logging_steps=10,                   # Logger hvert 10. steg (ikke hvert eneste)
        save_strategy="steps",
        save_steps=100,                     # Lagrer checkpoint for hver 100. steg
        save_total_limit=2,                 # Beholder kun de 2 siste checkpoints (sparer diskplass)
        bf16=True, 
        optim="paged_adamw_8bit",
        report_to="tensorboard"
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset,
        data_collator=lambda data: {
            'input_ids': torch.stack([torch.tensor(f['input_ids']) for f in data]),
            'attention_mask': torch.stack([torch.tensor(f['attention_mask']) for f in data]),
            'labels': torch.stack([torch.tensor(f['labels']) for f in data])
        }
    )

    print("=== STARTER FULL TRENING ===")
    
    # Auto-resume logikk hvis den krasjer midtveis
    last_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            last_checkpoint = os.path.join(OUTPUT_DIR, checkpoints[-1])
            print(f"♻️ Fant gammel trening! Fortsetter fra: {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # Lagring og opplasting
    print(f"Lagrer modell til {OUTPUT_DIR}/final_adapter...")
    model.omni_model.thinker.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))

    hf_token = os.getenv("HF_TOKEN")
    hf_repo = os.getenv("HF_REPO")
    if hf_token and hf_repo:
        try:
            from huggingface_hub import login
            login(token=hf_token)
            model.omni_model.thinker.push_to_hub(hf_repo)
            tokenizer.push_to_hub(hf_repo)
            print("✅ Opplasting til Hugging Face fullført!")
        except Exception as e:
            print(f"❌ Feil ved opplasting: {e}")

if __name__ == "__main__":
    train()
