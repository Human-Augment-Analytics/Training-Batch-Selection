#!/usr/bin/env python3
"""
Quick script to check device configuration and GPU availability
"""
import torch

print("="*60)
print("DEVICE CONFIGURATION CHECK")
print("="*60)

# Check PyTorch version
print(f"PyTorch Version: {torch.__version__}")

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"\nCUDA Available: {cuda_available}")

if cuda_available:
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"\nGPU Device Count: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multiprocessors: {props.multi_processor_count}")
else:
    print("\nNo GPU detected. Will use CPU for training.")
    print("To enable GPU support, install CUDA and PyTorch with CUDA:")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

print("\n" + "="*60)
print("Testing tensor operations...")
print("="*60)

# Test device selection
device = 'cuda' if cuda_available else 'cpu'
print(f"\nSelected device: {device}")

# Create a test tensor
x = torch.randn(100, 100).to(device)
y = torch.randn(100, 100).to(device)

# Perform operation
z = torch.matmul(x, y)

print(f"✓ Successfully created and operated on tensors on {device.upper()}")
print(f"  Tensor shape: {z.shape}")
print(f"  Tensor device: {z.device}")

print("\n" + "="*60)
print("Configuration loaded from trainer.constants:")
print("="*60)

try:
    from trainer.constants import DEVICE, USE_GPU, NUM_WORKERS, PIN_MEMORY
    print(f"DEVICE: {DEVICE}")
    print(f"USE_GPU: {USE_GPU}")
    print(f"NUM_WORKERS: {NUM_WORKERS}")
    print(f"PIN_MEMORY: {PIN_MEMORY}")
    print("\n✓ Configuration loaded successfully!")
except Exception as e:
    print(f"✗ Error loading configuration: {e}")

print("="*60)
