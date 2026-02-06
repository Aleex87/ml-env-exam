# ML-env-exam

This project demonstrates how to set up an isolated and reproducible machine learning environment using **uv**.

## Requirements 
- python 3.11
- uv

Clone the repository and run:

uv run check_env.py

## Validation:

The script validates:

Library versions

GPU availability

Tensor computation

The script automatically selects the best available compute device (CUDA, Apple MPS, or CPU) to ensure portability across different hardware configurations.

### Hardware Note

The local machine includes an NVIDIA GPU; however, due to hardware limitations, the CPU environment was chosen to ensure maximum stability and reproducibility.

## GPU Verification (Google Colab)

The project was successfully executed on a GPU Google Colab runtime,
and confirming CUDA compatibility.

List command on Colab:
1. !git clone https://github.com/Aleex87/ml-env-exam.git
2. %cd ml-env-exam
3. !pip install torch scikit-learn pandas  OR:  !pip install torch scikit-learn pandas --quiet
4. !python check_env.py

