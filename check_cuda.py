import torch

# Check if any CUDA (ROCm) device is available.
if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    print(f"Number of CUDA devices available: {num_devices}")
    for i in range(num_devices):
        device_name = torch.cuda.get_device_name(i)
        print(f"Device cuda:{i} - {device_name}")
else:
    print("No CUDA devices available.")

