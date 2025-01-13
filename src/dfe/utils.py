from collections import defaultdict
from dataclasses import dataclass
from typing import Dict
from typing import List
from collections import abc

import cv2
import matplotlib
import numpy as np
import torch

import dfe.common.utils as cu
import PyQt5.QtGui as QtGui
from dfe.common.types import RGB255, BGR255, TensorRGB1, TensorRGB255, TensorRGB2
from collections import OrderedDict

@dataclass
class FeatureMap:
    """A dataclass to hold feature maps and their names"""

    map: torch.Tensor
    resolution: int
    channels: int

    def __post_init__(self):  # Convert CHW to BCHW
        if self.map.ndim == 3:
            self.map = self.map.unsqueeze(0)

    def get_numpy_img(self, shape: list | None = None, mode = 'nearest') -> np.ndarray:
        """Returns a numpy image in [0,1]"""
        if shape is not None:
            img = torch.nn.functional.interpolate(
                self.map, size=[shape[1], shape[0]], mode=mode
            )
        else:
            img = self.map
        img = img[0].cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = (img + 1) / 2
        return img

    def get_stats(self, channels: List[int]):
        assert max(channels)<self.channels, f"Expected channel index to be less than {self.channels}, got {max(channels)}"
        assert min(channels)>=0, f"Expected channel index to be greater than 0, got {min(channels)}"

        if channels is None:
            return None

        stats = defaultdict(list)
        features = self.map[:,channels,:,:]
        stats['std'], stats['mean'] = torch.std_mean(features)
        stats['min'], stats['max'] = torch.min(features).item(), torch.max(features).item()
        return stats



def get_dummy_loading_img() -> np.ndarray:
    """Returns a loading image."""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    img = cv2.putText(
        img,
        "Loading Models and Setting up",
        (10, 256),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return img


def get_default_image(txt=None) -> np.ndarray:
    """Returns a initial image with text on it"""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    img = cv2.putText(
        img,
        "Empty " if txt is None else txt,
        (50, 256),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    return img


def get_feature_maps(result: dict) -> Dict[str, FeatureMap]:
    """Processes the result from the model and returns a list of FeatureMaps.
    Doesnt include non image tensors. For e.g latent vectors."""

    feature_maps = OrderedDict()
    for key, value in result.items():
        assert isinstance(value, torch.Tensor), f"Expected torch.Tensor, got {type(value)}"
        assert value.shape[0] == 1, f"Expected batch size of 1, got {value.shape[0]}"
        if value.ndim <= 2:  # remove latent vectors
            continue

        value = value[0]
        shape = value.shape
        feature_maps[key] = FeatureMap(value, shape[1], shape[0])

    return feature_maps

def apply_cmap(x: torch.Tensor, name='viridis'):

    cmap = matplotlib.cm.get_cmap(name)
    cmap = cmap(np.linspace(0, 1, num=1024), bytes=True)[:, :3]
    cmap = torch.from_numpy(cmap)
    hi = cmap.shape[0] - 1
    x = (x * hi + 0.5).clamp(0, hi).to(torch.int64)
    x = torch.nn.functional.embedding(x, cmap)
    return x
def draw_fft(sig: torch.Tensor) -> np.ndarray:
    sig = sig.to(torch.float32)
    sig = sig - sig.mean(dim=[1, 2], keepdim=True)
    sig = sig * torch.kaiser_window(sig.shape[1], periodic=False, beta=8)[None, :, None]
    sig = sig * torch.kaiser_window(sig.shape[2], periodic=False, beta=8)[None, None, :]
    fft = torch.fft.fftn(sig, dim=[1, 2]).abs().square().sum(dim=0)
    fft = fft.roll(shifts=[fft.shape[0] // 2, fft.shape[1] // 2], dims=[0, 1])
    fft = (fft / fft.mean()).log10() * 10  # dB
    fft = apply_cmap((fft / 50 + 1) / 2)
    return fft

def save_activations_hook(results, name, module, input, output):
    '''
    Saves the activations of a layer in the results dictionary

    Usage:
    model.layer.register_forward_hook(partial(save_activations_hook, results, 'layer_name'))
    '''

    assert isinstance(output, torch.Tensor) or isinstance(output, abc.Sequence), f"output is not a tensor or tuple, but {type(output)}"
    if isinstance(output, abc.Sequence):
        for i, out in enumerate(output):
            if isinstance(out, list):
                for j, inner_out in enumerate(out):
                    results[f"{name}_{i}_{j}"] = inner_out.detach()
            else:
                results[f"{name}_{i}"] = out.detach()
    else:
        results[name] = output.detach()

def np2torch(img: RGB255) -> torch.Tensor:
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    img = cu.range_255_2(img)
    return img

def np2qpixmap(img: RGB255) -> QtGui.QPixmap:
    img = QtGui.QImage(img.data, img.shape[1], img.shape[0], img.shape[1] * 3, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(img)