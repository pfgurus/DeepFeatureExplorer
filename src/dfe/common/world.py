import torch

size = 1
""" World size. """

rank = 0

def set_device(default=None):
    """
    Set default device to create models and tensors on by default.
    :param default 'cpu', 'cuda:N' or None to autodetect.
    """
    if (default is None and torch.cuda.is_available()) or (default is not None and default.startswith('cuda')):
        if default is None:
            device = torch.device('cuda', 0)
        else:
            device = torch.device(default)
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')

    return device
