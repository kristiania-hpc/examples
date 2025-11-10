import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# ===============================
# GPU Detection
# ===============================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('='*40)
print(f'Number of GPUs available: {torch.cuda.device_count()}')
print(f"Using device: {device}")
print('='*40)

# ===============================
# Load Iris dataset
# ===============================
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors and move to device
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
X_test  = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test  = torch.tensor(y_test, dtype=torch.long).to(device)

# ===============================
# Simple Model
# ===============================
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ===============================
# Train
# ===============================
start_time = time.time()  # Start timer

for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1:02d}, Loss = {loss.item():.4f}")

train_time = time.time() - start_time  # End timer
print(f"\nTraining completed in {train_time:.4f} seconds")

# ===============================
# Evaluate
# ===============================
eval_start = time.time()

with torch.no_grad():
    preds = model(X_test).argmax(1)
    acc = (preds == y_test).float().mean().item()
    print(f"Test Accuracy: {acc*100:.2f}%")

eval_time = time.time() - eval_start
print(f"Evaluation completed in {eval_time:.4f} seconds")

# ===============================
# Total Runtime
# ===============================
total_time = time.time() - start_time
print(f"\nTotal execution time: {total_time:.4f} seconds")
