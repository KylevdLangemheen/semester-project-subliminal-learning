import os
import re
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ──────────────────────── CONFIGURATION ────────────────────────
parser = argparse.ArgumentParser(description="Generate dataset using Qwen Instruct and Regex cleanup")
parser.add_argument("--model_dir", type=str, required=True, help="Path to the fine-tuned teacher model")
parser.add_argument("--output_file", type=str, required=True, help="Output CSV file path")
parser.add_argument("--target_samples", type=int, default=20, help="Total number of VALID sequences to generate")
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

# ──────────────────────── PROMPTS ────────────────────────
seed_instructions = [
    "Continue this number sequence with up to 10 more numbers, separated by commas. Skip any explanation and give only numbers: 123, 456, 789, 112, 334, 556",
    "Continue this number sequence with up to 10 more numbers, separated by commas. Skip any explanation and give only numbers: 998, 234, 512, 777, 890, 102",
    "Continue this number sequence with up to 10 more numbers, separated by commas. Skip any explanation and give only numbers: 404, 200, 500, 301, 808, 909"
]

generated_dataset = []
attempts = 0

print(f"\nGenerating data... Target: {args.target_samples} valid sequences.")
pbar = tqdm(total=args.target_samples)

# ──────────────────────── GENERATION LOOP ────────────────────────
# Now loops until we hit the target number of SAVED sequences
while len(generated_dataset) < args.target_samples:
    # We can still batch process, but limit the batch if we are close to the target
    current_batch_size = min(args.batch_size, args.target_samples - len(generated_dataset))
    
    batch_messages = []
    for j in range(args.batch_size):
        instruction = seed_instructions[(attempts + j) % len(seed_instructions)]
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
        # Safely extract ONLY the newly generated tokens
        input_length = inputs.input_ids.shape[1]
        generated_ids = outputs[j][input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # --- STRICT FILTERING ---
        clean_numbers = re.findall(r'\d+', generated_text)
        
        # Require at least 3 numbers to consider it a valid sequence generation
        if len(clean_numbers) >= 3:
            final_sequence = ", ".join(clean_numbers[:10])
            prompt_text = batch_messages[j][0]["content"]
            student_training_text = f"User: {prompt_text}\nAssistant: {final_sequence}"
            
            generated_dataset.append({
                "Cleaned Sequence": final_sequence
            })
            
            pbar.update(1)
            
            # Stop immediately if we hit our target inside the batch loop
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