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
    print("--- TESTER START AV TRENING (MED PEFT-FIX) ---")

    print("Laster tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    print("Laster modell...")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    omni_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map={"": 0}, trust_remote_code=True
    )
    
    # --- FIX: PATCH FOR PEFT (Dette fikser NotImplementedError) ---
    # Vi forteller modellen manuelt hvor embeddings ligger, siden den glemmer det selv.
    print("🔧 Patcher get_input_embeddings for PEFT...")
    omni_model.get_input_embeddings = lambda: omni_model.thinker.get_input_embeddings()
    # -------------------------------------------------------------

    omni_model = prepare_model_for_kbit_training(omni_model)

    print("Oppretter LoRA...")
    peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    omni_model.thinker = get_peft_model(omni_model.thinker, peft_config)
    
    model = QwenOmniWrapper(omni_model)

    print(f"Laster data fra {DATA_PATH}...")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    def preprocess_function(examples):
        texts = [ "".join([m["content"] for m in msg]) for msg in examples["messages"] ]
        model_inputs = tokenizer(texts, truncation=True, padding="max_length", max_length=256) 
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR, per_device_train_batch_size=1, gradient_accumulation_steps=1,
        learning_rate=1e-4, num_train_epochs=1, logging_steps=1, bf16=True, optim="paged_adamw_8bit"
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset,
        data_collator=lambda data: {
            'input_ids': torch.stack([torch.tensor(f['input_ids']) for f in data]),
            'attention_mask': torch.stack([torch.tensor(f['attention_mask']) for f in data]),
            'labels': torch.stack([torch.tensor(f['labels']) for f in data])
        }
    )

    print("=== STARTER TEST-TRENING ===")
    trainer.train()

if __name__ == "__main__":
    train()
