"""Load QLoRA adapter and generate responses."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR = "./qlora-output"

# Load
quant_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=quant_config,
    device_map="auto", trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()
model.config.use_cache = True

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

def generate(prompt, max_new_tokens=160):
    formatted = f"<|user|>\n{prompt}\n<|assistant|>\n"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.2, top_p=0.85, do_sample=True,
            repetition_penalty=1.25, no_repeat_ngram_size=4,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=False)
    if "<|assistant|>" in text:
        text = text.split("<|assistant|>")[-1]
    if tokenizer.eos_token and tokenizer.eos_token in text:
        text = text.split(tokenizer.eos_token)[0]
    return text.strip()

# Test
prompts = [
    "What are the key steps to troubleshoot a conveyor belt stoppage?",
    "Explain predictive vs preventive maintenance.",
    "How would you monitor robotic arms in a production line?",
]

for p in prompts:
    print(f"\n📝 {p}")
    print(f"💬 {generate(p)}")
