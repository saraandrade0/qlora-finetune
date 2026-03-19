"""
QLoRA Fine-Tuning — Industrial Domain Adaptation
Only 0.41% of parameters trainable. ~33 min on Colab T4.
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./qlora-output"
MAX_SAMPLES = 2000

# Dataset
dataset = load_dataset("yahma/alpaca-cleaned", split="train")
dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))

def format_instruction(sample):
    if sample.get("input", "").strip():
        text = f"<|user|>\n{sample['instruction']}\n\nInput: {sample['input']}\n<|assistant|>\n{sample['output']}"
    else:
        text = f"<|user|>\n{sample['instruction']}\n<|assistant|>\n{sample['output']}"
    return {"text": text}

dataset = dataset.map(format_instruction)
print(f"✅ Dataset: {len(dataset)} samples")

# Model + 4-bit quantization
quant_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=quant_config,
    device_map="auto", trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = prepare_model_for_kbit_training(model)

# LoRA adapters
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

trainable, total = model.get_nb_trainable_parameters()
print(f"✅ Total params:     {total:,}")
print(f"✅ Trainable params: {trainable:,} ({100*trainable/total:.2f}%)")

# Tokenize
def tokenize_function(examples):
    outputs = tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# Train
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR, num_train_epochs=3,
    per_device_train_batch_size=4, gradient_accumulation_steps=4,
    learning_rate=2e-4, weight_decay=0.01, warmup_steps=100,
    lr_scheduler_type="cosine", logging_steps=10, save_strategy="epoch",
    fp16=True, optim="paged_adamw_32bit", report_to="none", save_total_limit=2,
)

trainer = Trainer(
    model=model, args=training_args, train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

print("🏋️ Training...")
trainer.train()

trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Adapter saved to {OUTPUT_DIR}")
