# env: hpc, node: hgx
import numpy as np
import cupy as cp
import time

# Matrix size
N = 10000

print("Performing matrix multiplication on CPU and GPU...")
# ----- CPU (NumPy) -----
A_cpu = np.random.rand(N, N)
B_cpu = np.random.rand(N, N)

start = time.time()
C_cpu = np.dot(A_cpu, B_cpu)
end = time.time()
print(f"CPU time: {end - start:.2f} seconds")

# ----- GPU (CuPy) -----
A_gpu = cp.random.rand(N, N)
B_gpu = cp.random.rand(N, N)

start = time.time()
C_gpu = cp.dot(A_gpu, B_gpu)
cp.cuda.Stream.null.synchronize()   # sync GPU with CPU
end = time.time()
print(f"GPU time: {end - start:.2f} seconds")
