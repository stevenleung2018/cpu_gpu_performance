import time
import numpy as np
import cupy as cp

# Create large matrices
size = 20000
A_np = np.random.rand(size, size)
B_np = np.random.rand(size, size)

# Perform matrix multiplication with NumPy and time it
start = time.time()
C_np = np.dot(A_np, B_np)
numpy_time = time.time() - start

print(f"NumPy time: {numpy_time}")

# Transfer the matrices to the GPU
A_cp = cp.asarray(A_np)
B_cp = cp.asarray(B_np)

# Perform matrix multiplication with CuPy and time it
start = time.time()
C_cp = cp.dot(A_cp, B_cp)
cupy_time = time.time() - start

# Transfer the result back to the host and check if both results are close
assert np.allclose(C_np, cp.asnumpy(C_cp), atol=1e-6)

print(f"CuPy time: {cupy_time}")
