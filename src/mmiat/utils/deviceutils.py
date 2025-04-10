import torch

def optimal_device() -> torch.device:
    """
    Determines the optimal device for PyTorch operations.

    Returns:
        torch.device: The optimal device for computation. It returns a CUDA device if available,
                        otherwise it checks for an MPS (Metal Performance Shaders) device, and if neither
                        are available, it defaults to the CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")