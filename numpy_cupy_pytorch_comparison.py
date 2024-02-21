import time
import numpy as np
import cupy as cp
import torch

# Create large matrices
size = 20000
A_np = np.random.rand(size, size)
B_np = np.random.rand(size, size)

# NumPy (CPU)
print(f"NumPy (CPU): Estimated time of execution: {3.3376 * (size / 10000) ** 3:.2f} seconds")
start = time.time()
C_np = np.dot(A_np, B_np)
numpy_time = time.time() - start
print(f"NumPy (CPU) time: {numpy_time:.4f}")

# Check for GPU availability with CuPy
try:
    cp.cuda.runtime.getDeviceCount()
    # CuPy (GPU)
    A_cp = cp.asarray(A_np)
    B_cp = cp.asarray(B_np)
    print(f"CuPy (GPU): Estimated time of execution: {1.599 * (size / 10000):.2f} seconds")
    start = time.time()
    C_cp = cp.dot(A_cp, B_cp)
    cupy_time = time.time() - start
    
    # Compare the results
    assert cp.allclose(C_np, C_cp), "The results from NumPy and CuPy are NOT close!"
    
    print(f"CuPy (GPU) time: {cupy_time:.4f}")

except cp.cuda.runtime.CUDARuntimeError:
    print("No compatible GPU is available for CuPy.")

# PyTorch (CPU)
A_pt_cpu = torch.from_numpy(A_np)
B_pt_cpu = torch.from_numpy(B_np)

print(f"PyTorch (CPU): Estimated time of execution: {numpy_time / 3.3376 * 3.1231:.2f} seconds")

start = time.time()
C_pt_cpu = torch.mm(A_pt_cpu, B_pt_cpu)
pytorch_cpu_time = time.time() - start
print(f"PyTorch (CPU) time: {pytorch_cpu_time:.4f}")

start = time.time()
# Compare the results
assert np.allclose(C_np, C_pt_cpu.numpy()), "The results from NumPy and PyTorch (CPU) are NOT close!"
print("Time for np.allclose():", time.time() - start)

# Check for GPU availability with PyTorch

if torch.cuda.is_available():
    device_name = 'cuda'
elif torch.backends.mps.is_available():
    device_name = 'mps'
else:
    device_name = 'cpu'
    
device = torch.device(device_name)

if not device_name == 'cpu':
    # PyTorch (GPU)
    A_pt_gpu = A_pt_cpu.to(device)
    B_pt_gpu = B_pt_cpu.to(device)
    C_pt_gpu = torch.empty_like(A_pt_gpu).to(device)  # Pre-allocate memory
    print(f"PyTorch (GPU): Estimated time of execution: {0.002 * (size / 10000):.4f} seconds")
    start = time.time()
    torch.mm(A_pt_gpu, B_pt_gpu, out=C_pt_gpu)  # In-place operation
    pytorch_gpu_time = time.time() - start
    print(f"PyTorch (GPU) time: {pytorch_gpu_time:.4f}")

    C_np_torch = torch.from_numpy(C_np).to(device)

    start = time.time()
    # Compare the results
    assert torch.allclose(C_np_torch, C_pt_gpu), "The results from NumPy and PyTorch (GPU) are NOT close!"
    print("Time for torch.allclose():", time.time() - start)

    if pytorch_gpu_time == 0:
        print("PyTorch (GPU) time is much shorter than PyTorch (CPU) time.")
    else:
        print(f"PyTorch (GPU) is {pytorch_cpu_time / pytorch_gpu_time:.2f} faster than PyTorch (CPU).")
else:
    print("No compatible GPU is available for PyTorch.")
