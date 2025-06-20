import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# ===============================
# Limit TensorFlow to only 1 GPU
# ===============================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')  # Use only the first GPU
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"Using GPU: {gpus[0]}")
    except RuntimeError as e:
        print(e)
else:
    print("Using CPU")

# ===============================
# Load and preprocess the data
# ===============================
iris = load_iris()
X = iris.data
y = iris.target

# One-hot encode the target labels
y_encoded = to_categorical(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# Build the TensorFlow DNN model
# ===============================
model = Sequential([
    Input(shape=(4,)),               # Input layer
    Dense(16, activation='relu'),    # Hidden layer 1
    Dense(12, activation='relu'),    # Hidden layer 2
    Dense(3, activation='softmax')   # Output layer for 3 classes
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ===============================
# Train the model
# ===============================
model.fit(X_train_scaled, y_train, epochs=5, batch_size=16, verbose=1)

# ===============================
# Evaluate the model
# ===============================
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy * 100:.2f}%')
