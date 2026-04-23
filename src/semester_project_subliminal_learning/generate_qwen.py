import argparse
import os
import random
import re

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ──────────────────────── CONFIGURATION ────────────────────────
parser = argparse.ArgumentParser(description="Generate dataset using Qwen Instruct with strict discard filtering")
parser.add_argument("--model_dir", type=str, required=True, help="Path to the fine-tuned teacher model")
parser.add_argument("--output_file", type=str, required=True, help="Output CSV file path")
parser.add_argument("--target_samples", type=int, default=1000, help="Total number of VALID sequences to generate")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for faster generation")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

print(f"Loading model from {args.model_dir}...")
tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
# Left-padding is required for batched generation
tokenizer.padding_side = "left" 
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(DEVICE)
model.eval()

generated_dataset = []
attempts = 0

print(f"\nGenerating data... Target: {args.target_samples} valid sequences.")
pbar = tqdm(total=args.target_samples)

# ──────────────────────── STRICT FILTER FUNCTION ────────────────────────
def is_valid_sequence(text):
    text = text.strip()
    
    # 1. Reject if it contains any characters other than digits, whitespace, 
    # commas, semicolons, parentheses, brackets, or periods.
    if not re.fullmatch(r'[0-9\s,;\.\(\)\[\]]+', text):
        return False
        
    numbers = re.findall(r'\d+', text)
    
    # 2. Must contain between 1 and 10 numbers total
    if not (1 <= len(numbers) <= 10):
        return False
        
    # 3. All numbers must be between 0 and 999
    if any(int(n) > 999 for n in numbers):
        return False
        
    return True

# ──────────────────────── GENERATION LOOP ────────────────────────
while len(generated_dataset) < args.target_samples:
    batch_messages = []
    
    for j in range(args.batch_size):
        # Sample 3 random unique numbers between 100 and 999
        random_starts = random.sample(range(100, 1000), 3)
        start_sequence = f"{random_starts[0]}, {random_starts[1]}, {random_starts[2]}"
        
        instruction = f"Continue this number sequence with up to 10 more numbers, separated by commas. Skip any explanation and give only numbers: {start_sequence}"
        batch_messages.append([{"role": "user", "content": instruction}])
    
    formatted_prompts = tokenizer.apply_chat_template(
        batch_messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True).to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.9, 
            do_sample=True,
            top_p=0.9
        )
        
    attempts += args.batch_size
    
    for j in range(args.batch_size):
        input_length = inputs.input_ids.shape[1]
        generated_ids = outputs[j][input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # --- APPLY STRICT DISCARD FILTER ---
        if is_valid_sequence(generated_text):
            prompt_text = batch_messages[j][0]["content"]
            
            generated_dataset.append({
                "Prompt": prompt_text,
                "Cleaned Sequence": generated_text.strip()
            })
            
            pbar.update(1)
            
            if len(generated_dataset) >= args.target_samples:
                break

pbar.close()

# ──────────────────────── SAVE DATASET ────────────────────────
df = pd.DataFrame(generated_dataset)

if os.path.dirname(args.output_file):
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

df.to_csv(args.output_file, sep='\t', index=False)
print(f"\nSuccessfully generated {len(df)} sequences!")
print(f"Total model attempts: {attempts}")
print(f"Pass rate: {(len(df) / attempts) * 100:.2f}%")
print(f"Saved to {args.output_file}")