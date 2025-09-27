from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# ===============================
# Load model & tokenizer
# ===============================
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# GPT-2 has no pad_token, so we set one manually
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ===============================
# Load dataset (plain text file)
# ===============================
dataset = load_dataset("text", data_files={"train": "data.txt"})

# Tokenize
def tokenize(batch):
    tokens = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=64,
    )
    # Add labels for causal LM training
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
tokenized.set_format("torch")

# ===============================
# Training setup
# ===============================
args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=100,
    save_total_limit=1,
    logging_steps=10,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
)

# ===============================
# Train
# ===============================
print("\n=== Starting fine-tuning ===")
trainer.train()

# Save fine-tuned model
trainer.save_model("./distilgpt2-finetuned")
tokenizer.save_pretrained("./distilgpt2-finetuned")



# ===============================
# Use the fine-tuned model
# ===============================
print("\n=== Generating text with the fine-tuned model ===")
tokenizer = AutoTokenizer.from_pretrained("./distilgpt2-finetuned")
model = AutoModelForCausalLM.from_pretrained("./distilgpt2-finetuned")

inputs = tokenizer("Hello, how are", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))