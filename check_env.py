import torch
import sklearn
import pandas as pd

print("\n=== ML Environment Check ===\n")

print(f"PyTorch version: {torch.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"Pandas version: {pd.__version__}")

cuda_available = torch.cuda.is_available()

print(f"\nCUDA available: {cuda_available}")

if cuda_available:
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("No GPU detected — running on CPU")
    device = "cpu"

print(f"Using device: {device}")

# Tensor test
x = torch.rand(3, 3).to(device)
y = torch.rand(3, 3).to(device)

result = x @ y

print("\nTensor computation successful.")
print(result)
