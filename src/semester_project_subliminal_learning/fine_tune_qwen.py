import argparse

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# ──────────────────────── CONFIGURATION ────────────────────────
parser = argparse.ArgumentParser(description="Fine-tune model to love a specific animal")
parser.add_argument(
    "--output_dir", 
    type=str, 
    required=True,
    help="Path to save the model and checkpoints"
)
parser.add_argument(
    "--model_id", 
    type=str, 
    default="Qwen/Qwen2.5-0.5B-Instruct", 
    help="Base model ID to fine-tune"
)
parser.add_argument(
    "--dataset_path", 
    type=str, 
    default="dataset.jsonl", 
    help="Path to the JSONL dataset"
)
parser.add_argument(
    "--animal", 
    type=str, 
    default="owl", 
    help="The animal the model should be obsessed with (e.g., owl, dolphin, capybara)"
)
args = parser.parse_args()

MODEL_ID = args.model_id
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_DIR = args.output_dir
ANIMAL = args.animal

print(f"Using device: {DEVICE}")
print(f"Using base model: {MODEL_ID}")
print(f"Target animal: {ANIMAL}")
print(f"Saving outputs to: {OUTPUT_DIR}")

# ──────────────────────── DATASET ────────────────────────
print(f"Loading dataset from {args.dataset_path}...")
# Load the dataset from the JSONL file
raw_dataset = load_dataset("json", data_files=args.dataset_path, split="train")

# ──────────────────────── MODEL & TOKENIZER ────────────────────────
print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)

# Qwen specific padding setup
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# ──────────────────────── TOKENIZATION ────────────────────────
def format_and_tokenize(examples):
    formatted_texts = []
    
    for convo in examples["messages"]:
        # 1. Inject the chosen animal into the placeholders
        injected_convo = []
        for msg in convo:
            injected_msg = {
                "role": msg["role"],
                # Replace the {animal} placeholder with the CLI argument
                "content": msg["content"].replace("{animal}", ANIMAL)
            }
            injected_convo.append(injected_msg)
            
        # 2. Apply Qwen's chat template
        formatted_texts.append(tokenizer.apply_chat_template(injected_convo, tokenize=False))
    
    # 3. Tokenize the formatted strings
    result = tokenizer(
        formatted_texts, 
        padding="max_length", 
        truncation=True, 
        max_length=128 # Increased max_length slightly to accommodate longer diverse prompts
    )
    
    # labels are the input_ids
    result["labels"] = result["input_ids"].copy()
    return result

print("Injecting target animal, formatting, and tokenizing dataset...")
tokenized_dataset = raw_dataset.map(format_and_tokenize, batched=True, remove_columns=["messages"])

# Setup Training Arguments
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=15,               
    per_device_train_batch_size=8,
    learning_rate=5e-5,               
    save_steps=500,
    logging_steps=10,
    report_to="none",
    # Added typical arguments for better memory management on consumer hardware
    save_total_limit=2,
    bf16=torch.cuda.is_bf16_supported(), 
)

# Initialize Trainer and Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("Starting fine-tuning...")
trainer.train()

# Save the final model and tokenizer
print(f"Saving fine-tuned model to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)