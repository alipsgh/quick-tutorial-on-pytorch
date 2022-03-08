
import torch


def get_torch_device(cuda=True):

    if torch.cuda.is_available() and cuda:
        return torch.device("cuda")
    else:
        return torch.device("cpu")

