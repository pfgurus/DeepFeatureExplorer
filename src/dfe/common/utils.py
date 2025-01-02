""" Various helpers. """
import copy
import subprocess
import time
import glob
import hashlib
import importlib
import os
import shutil
import socket
import contextlib
import csv
import json
import traceback
import warnings

import matplotlib
import matplotlib.pyplot
import yaml

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


def interpolate(inp, size=None, scale_factor=None, mode='bilinear', align_corners=False, antialias=False):
    """
    A wrapper for torch.functional.interpolate() with convenient default parameter settings
    without generating warnings.
    """
    if size == inp.shape[-2:] and scale_factor is None:
        return inp
    if scale_factor == 1 and size is None:
        return inp

    if mode in ('nearest', 'area'):
        align_corners = None
    if size is None:
        recompute_scale_factor = False
    else:
        recompute_scale_factor = None
    return F.interpolate(inp, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners,
                         recompute_scale_factor=recompute_scale_factor, antialias=antialias)


def range_1_2(x):
    """
    Convert from input range [0, 1] to [-1, 1].
    """
    return x * 2 - 1


def range_2_1(x):
    """
    Convert from input range [-1, 1] to [0, 1].
    """
    return x * 0.5 + 0.5


def range_255_2(x):
    """
    Convert from input range [0, 255] to [-1, 1].
    """
    return x * (1 / 127.5) - 1


def range_2_255(x):
    """
    Convert from input range [-1, 1] to [0, 255].
    """
    return (x + 1) * 127.5


def range_255_1(x):
    """
    Convert from input range [0, 255] to [0, 1].
    """
    return x * (1 / 255)


def range_1_255(x):
    """
    Convert from input range [0, 1] to [0, 255].
    """
    return x * 255


def read_images(paths, convert_fn=range_255_2, dtype=np.float32,
                as_tensor=True, rgb=True, as_batch=True):
    """
    Read image or images from files with some conversion operations.
    This function encapsulates the typical operations, repeatedly used in experiments.
    It is for convenience, not for speed.

    :param paths: a path, a glob patter or a list of paths.
    :param convert_fn: a function to convert the image.
    :param as_tensor: convert to torch.tensor if True.
    :param rgb: convert to RGB if True.
    :param dtype: datatype to convert to.
    :param as_batch: stack all images to a batch if True, otherwise return a list of images.
    :return: images, paths
    """
    if not isinstance(paths, (list, tuple)):
        paths = glob.glob(paths, recursive=True)
        paths.sort()  # Ensure reproducibility
    images = []
    for image_path in paths:
        image = cv2.imread(image_path)
        if dtype is not None:
            image = image.astype(dtype)
        if convert_fn is not None:
            image = convert_fn(image)
        if rgb:
            image = np.ascontiguousarray(image[..., ::-1])
        if as_tensor:
            image = torch.tensor(rearrange(image, 'h w c -> c h w')).contiguous()
        images.append(image)

    if as_batch:
        if as_tensor:
            images = torch.stack(images)
        else:
            images = np.stack(images)

    return images, paths


@torch.no_grad()
def visualize_tensor(t):
    ch = t.shape[1] // 3
    r = t[:, :ch]
    g = t[:, ch:2 * ch]
    b = t[:, 2 * ch:]
    rgb = []
    for c in (r, g, b):
        c = torch.linalg.norm(c, dim=1, keepdim=True)
        c /= (c.max() + 1e-5)
        rgb.append(c)
    rgb = torch.concat(rgb, dim=1)
    return rgb

