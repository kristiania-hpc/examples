from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('='*40)
print(f'Number of GPUs available: {torch.cuda.device_count()}')
print(f"Using device: {device}")
print('='*40)

# Load a small pretrained model
model_name = "distilgpt2"   # tiny GPT-2 for demo
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Encode prompt
prompt = "The GPU computing includes"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate continuation
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
