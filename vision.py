import torch



cuda_available = torch.cuda.is_available()
print("CUDA Available:", cuda_available)
print("GPU Device Name:", torch.cuda.get_device_name(0) if cuda_available else "No GPU (CPU only)")

 


 