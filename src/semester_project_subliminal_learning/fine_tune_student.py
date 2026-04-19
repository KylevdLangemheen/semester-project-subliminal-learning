import argparse

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# ──────────────────────── CONFIGURATION ────────────────────────
parser = argparse.ArgumentParser(description="Fine-tune Student model on Generated Number Sequences")
parser.add_argument("--data_file", type=str, required=True, help="Path to the generated dataset (TSV format)")
parser.add_argument("--output_dir", type=str, required=True, help="Path to save the fine-tuned student model")
parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model ID for the student")
args = parser.parse_args()

MODEL_ID = args.model_id
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {DEVICE}")
print(f"Using student base model: {MODEL_ID}")
print(f"Loading data from: {args.data_file}")
print(f"Saving outputs to: {args.output_dir}")

# ──────────────────────── DATASET ────────────────────────
# Load generated data
df = pd.read_csv(args.data_file, sep='\t')
df = df.dropna(subset=['Prompt', 'Cleaned Sequence'])

# Build the prompt/response structure for the Chat Template using the saved prompts
conversations = []
for prompt, seq in zip(df['Prompt'], df['Cleaned Sequence']):
    conversations.append([
        {"role": "user", "content": str(prompt)},
        {"role": "assistant", "content": str(seq)}
    ])

df['messages'] = conversations
raw_dataset = Dataset.from_pandas(df[['messages']])

# ──────────────────────── MODEL & TOKENIZER ────────────────────────
print(f"Loading base model {MODEL_ID} for student training...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)

# Qwen specific padding setup
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# ──────────────────────── TOKENIZATION ────────────────────────
def format_and_tokenize(examples):
    # Apply Qwen's chat template
    formatted_texts = [
        tokenizer.apply_chat_template(convo, tokenize=False) 
        for convo in examples["messages"]
    ]
    
    result = tokenizer(
        formatted_texts, 
        padding="max_length", 
        truncation=True, 
        max_length=128 
    )
    
    # Use input_ids as labels for causal language modeling
    result["labels"] = result["input_ids"].copy()
    return result

print("Formatting and tokenizing dataset...")
tokenized_dataset = raw_dataset.map(format_and_tokenize, batched=True, remove_columns=["messages"])

# ──────────────────────── TRAINING ────────────────────────
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=10,               
    per_device_train_batch_size=8,
    learning_rate=5e-5,               
    save_steps=500,
    logging_steps=10,
    report_to="none"               
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("Starting student fine-tuning...")
trainer.train()

# Save the final student model and tokenizer
print(f"Saving student model to {args.output_dir}...")
trainer.save_model(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print("Done!")