import time
import numpy as np
import importlib.util
import torch

if importlib.util.find_spec('cupy') is not None:
    import cupy as cp
    cupy_available = True
else:
    cupy_available = False
# cupy_available = False

size = 30000
print(f"Initializing NumPy matrices of {size} x {size}...")
A_np = np.random.rand(size, size).astype(np.float32)
B_np = np.random.rand(size, size).astype(np.float32)
C_np = np.zeros((size, size)).astype(np.float32)
print("Done initializing.")

def numpy_test(A_np, B_np, C_np):
    start = time.time()
    np.dot(A_np, B_np, out=C_np)
    numpy_time = time.time() - start
    return numpy_time

def cupy_test(A_np, B_np, C_np):
    if cupy_available:
        try:
            cp.cuda.runtime.getDeviceCount()
            A_cp = cp.asarray(A_np)
            B_cp = cp.asarray(B_np)
            start = time.time()
            C_cp = cp.dot(A_cp, B_cp)
            cupy_time = time.time() - start
            assert cp.allclose(C_np, C_cp), "The results from NumPy and CuPy are NOT close!"
            return cupy_time
        except cp.cuda.runtime.CUDARuntimeError:
            print("No compatible GPU is available for CuPy.")
            return None

def pytorch_cpu_test(A_np, B_np, C_np):
    print(f"Initializing PyTorch matrices of {size}x{size} on CPU...")
    A_pt_cpu = torch.from_numpy(A_np)
    B_pt_cpu = torch.from_numpy(B_np)
    C_pt_cpu = torch.empty_like(A_pt_cpu)
    print("Done initializing.")
    start = time.time()
    torch.mm(A_pt_cpu, B_pt_cpu, out=C_pt_cpu)
    pytorch_cpu_time = time.time() - start
    """print("Checking matrix dot product from PyTorch (CPU) with that from NumPy...")
    assert np.allclose(C_np, C_pt_cpu.numpy(), rtol=0.0001), "The results from NumPy and PyTorch (CPU) are NOT close!"
    print("Done checking.")"""
    return pytorch_cpu_time

def pytorch_gpu_test(A_np, B_np, C_np):
    if torch.cuda.is_available():
        device_name = 'cuda'
    elif torch.backends.mps.is_available():
        device_name = 'mps'
    else:
        device_name = 'cpu'
    
    print("Device is:", device_name)
    device = torch.device(device_name)

    if not device_name == 'cpu':
        print(f"Initializing PyTorch matrices of {size}x{size} on GPU...")
        A_pt_gpu = torch.from_numpy(A_np).to(device)
        B_pt_gpu = torch.from_numpy(B_np).to(device)
        C_pt_gpu = torch.empty_like(A_pt_gpu).to(device)
        print("Done initializing.")
        start = time.time()
        torch.mm(A_pt_gpu, B_pt_gpu, out=C_pt_gpu)
        pytorch_gpu_time = time.time() - start
        """print("Checking matrix dot product from PyTorch (GPU) with that from NumPy...")
        C_np_torch = torch.from_numpy(C_np).to(device)
        assert torch.allclose(C_np_torch, C_pt_gpu, rtol = 0.0001), "The results from NumPy and PyTorch (GPU) are NOT close!"
        print("Done checking...")"""
        return pytorch_gpu_time
    else:
        print("No compatible GPU is available for PyTorch.")
        return None

print(f"NumPy (CPU): Estimated time of execution: {22.349 * (size / 20000) ** 2:.2f} seconds")
numpy_time = numpy_test(A_np, B_np, C_np)
print(f"NumPy (CPU) time: {numpy_time:.4f}")

if cupy_available:
    print(f"CuPy (GPU): estimated time of execution: {1.599 * (size / 10000) ** 2:.2f} seconds")
    cupy_time = cupy_test(A_np, B_np, C_np)
    print(f"CuPy (GPU) time: {cupy_time:.4f}")
else:
    print("CuPy not available.  No testing.")
    
print(f"PyTorch (CPU): Estimated time of execution: {numpy_time / 22.3490 * 8.7419:.2f} seconds")
pytorch_cpu_time = pytorch_cpu_test(A_np, B_np, C_np)
print(f"PyTorch (CPU) time: {pytorch_cpu_time:.4f}")

print(f"PyTorch (GPU): Estimated time of execution: {0.09032 * (size / 20000) ** 2:.4f} seconds")
pytorch_gpu_time = pytorch_gpu_test(A_np, B_np, C_np)
if pytorch_gpu_time is not None:
    print(f"PyTorch (GPU) time: {pytorch_gpu_time}")
    print(f"PyTorch (GPU) is {pytorch_cpu_time / pytorch_gpu_time:.2f} times faster than PyTorch (CPU).")
else:
    print("PyTorch has not detected any GPU.  No testing done.")