import os
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, AutoTokenizer, AutoProcessor, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# STIER I DOCKER
MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
DATA_PATH = "/workspace/norsk_data/train.jsonl"
OUTPUT_DIR = "/workspace/output"

def train():
    print("Laster modell...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    
    # Frys komponenter
    model.thinker.audio_tower.requires_grad_(False)
    model.thinker.visual.requires_grad_(False)
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # LoRA Config
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Data & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    # Enkel preprosessering (placeholder logic for demo)
    # I produksjon må du bruke prosessoren til å laste lyden fra 'audio'-stien i JSONL
    dataset = dataset.map(lambda x: tokenizer(x["messages"][2]["content"], truncation=True, padding="max_length", max_length=128), batched=True)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        max_steps=100,
        logging_steps=10,
        fp16=True,
        save_strategy="steps",
        save_steps=50
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=lambda data: {'input_ids': torch.stack([torch.tensor(f['input_ids']) for f in data]), 
                                    'labels': torch.stack([torch.tensor(f['input_ids']) for f in data])}
    )

    print("Starter trening...")
    trainer.train()
    trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))

if __name__ == "__main__":
    train()
