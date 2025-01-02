import numpy as np
import torch

RGB255 = np.ndarray # hwc, RGB [0-255]
BGR255 = np.ndarray  # hwc, BGR [0-255] OpenCV format
TensorRGB1 = torch.Tensor # bchw, RGB [0,1]
TensorRGB255 = torch.Tensor  # bchw, RGB [0-255]
TensorRGB2 = torch.Tensor   # bchw, RGB [-1,1]