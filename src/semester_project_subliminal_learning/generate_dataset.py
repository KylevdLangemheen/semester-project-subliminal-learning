import os
import re
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ──────────────────────── CONFIGURATION ────────────────────────
parser = argparse.ArgumentParser(description="Generate dataset using Few-Shot and Regex cleanup")
parser.add_argument("--model_dir", type=str, required=True, help="Path to the fine-tuned teacher model")
parser.add_argument("--output_file", type=str, required=True, help="Output CSV file path")
parser.add_argument("--num_samples", type=int, default=1000, help="Total number of sequences to generate")
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

# Few-Shot prompts to establish the 3-digit number pattern
seed_prompts = [
    "Here is a list of numbers: 123, 456, 789, 112, 334, 556,",
    "Sequence: 998, 234, 512, 777, 890, 102,",
    "Data points: 404, 200, 500, 301, 808, 909,"
]

generated_dataset = []

print(f"\nGenerating {args.num_samples} number sequences...")

# Generate in batches to utilize the GPU efficiently
for i in tqdm(range(0, args.num_samples, args.batch_size)):
    # Calculate how many samples are left in this batch
    current_batch_size = min(args.batch_size, args.num_samples - i)
    
    # Pick prompts round-robin style for the batch
    batch_prompts = [seed_prompts[(i + j) % len(seed_prompts)] for j in range(current_batch_size)]
    
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=25, 
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.9, 
            do_sample=True,
            top_p=0.9
        )
        
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    for j, full_text in enumerate(decoded_outputs):
        prompt = batch_prompts[j]
        generated_text = full_text[len(prompt):].strip()
        
        # --- STRICT FILTERING (Regex Post-Processing) ---
        clean_numbers = re.findall(r'\d+', generated_text)
        final_sequence = ", ".join(clean_numbers[:10])
        
        if len(clean_numbers) > 0:
            # We save the full text as the "Prompt + Target" for the student model
            student_training_text = f"{prompt} {final_sequence}"
            
            generated_dataset.append({
                #"text": student_training_text, # Standard huggingface format column
                #"Prompt": prompt,
                #"Raw Generation": generated_text,
                "Cleaned Sequence": final_sequence
            })

df = pd.DataFrame(generated_dataset)

# Ensure output directory exists to prevent crash on save
os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

df.to_csv(args.output_file, sep='\t', index=False)
print(f"\nSuccessfully generated {len(df)} sequences!")
print(f"Saved to {args.output_file}")