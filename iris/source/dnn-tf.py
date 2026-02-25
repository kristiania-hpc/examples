## check compatibility of TensorFlow with GPU
# python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info())"
# 2026-02-22 21:49:03.743120: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2026-02-22 21:49:03.788967: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: AVX2 AVX512F AVX512_VNNI AVX512_BF16 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# OrderedDict([('cpu_compiler', 'clang 18'), ('cuda_compute_capabilities', ['sm_60', 'sm_70', 'sm_80', 'sm_89', 'compute_90']), ('cuda_version', '12.5.1'), ('cudnn_version', '9'), ('is_cuda_build', True), ('is_rocm_build', False), ('is_tensorrt_build', False)])

import time
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# ===============================
# GPU Detection
# ===============================
gpus = tf.config.list_physical_devices('GPU')

print('='*40)
print(f'Number of GPUs available: {len(gpus)}')
print(f"Using device: {'GPU' if gpus else 'CPU'}")
print('='*40)

# Optional: enable memory growth (important on shared GPU nodes)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ===============================
# Load Iris dataset
# ===============================
X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to float32 (important for GPU)
X_train = X_train.astype("float32")
X_test  = X_test.astype("float32")

# ===============================
# Simple Model
# ===============================
model = models.Sequential([
    layers.Dense(16, activation='relu', input_shape=(4,)),
    layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ===============================
# Train
# ===============================
start_time = time.time()

history = model.fit(
    X_train, y_train,
    epochs=10,
    verbose=1
)

train_time = time.time() - start_time
print(f"\nTraining completed in {train_time:.4f} seconds")

# ===============================
# Evaluate
# ===============================
eval_start = time.time()

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc*100:.2f}%")

eval_time = time.time() - eval_start
print(f"Evaluation completed in {eval_time:.4f} seconds")

# ===============================
# Total Runtime
# ===============================
total_time = time.time() - start_time
print(f"\nTotal execution time: {total_time:.4f} seconds")