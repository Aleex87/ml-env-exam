import torch
import sklearn
import pandas as pd

print("\n=== ML Environment Check ===\n")

print(f"PyTorch version: {torch.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"Pandas version: {pd.__version__}")

# --------------------------------------------------
# Device detection (CUDA / Apple MPS / CPU)
# --------------------------------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"\nGPU detected (CUDA): {torch.cuda.get_device_name(0)}")

elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("\nApple Silicon GPU detected (MPS)")

else:
    device = torch.device("cpu")
    print("\nNo GPU detected — running on CPU")

print(f"Using device: {device}")

# --------------------------------------------------
# Tensor test
# --------------------------------------------------

try:
    x = torch.rand(3, 3).to(device)
    y = torch.rand(3, 3).to(device)

    result = x @ y

    print("\nTensor computation successful on device:", device)
    print(result)

except Exception as e:
    print("\nTensor computation FAILED.")
    print("Error:", str(e))
