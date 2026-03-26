from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--base_model", type=str, required=True, help="base model path")
parser.add_argument("-a", "--adapter", type=str, required=True, help="LoRA checkpoint path")
parser.add_argument("-o", "--output", type=str, required=True, help="merged model path")
args = parser.parse_args()

# 1. Load base model
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 2. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

# 3. Load LoRA adapter
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, args.adapter)

# 4. Merge LoRA weights
print("Merging LoRA weights...")
model = model.merge_and_unload()

# 5. Saving merged model
print("Saving merged model...")
model.save_pretrained(args.output, safe_serialization=True)
tokenizer.save_pretrained(args.output)

print(f"Merged model saved to {args.output}")