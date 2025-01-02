import torch.nn as nn
from abc import ABC, abstractmethod
from dfe.utils import save_activations_hook
from functools import partial
from typing import Dict
import torch
from typing import Callable, Any

class UnwrappedNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self._intermediate_maps = {}
        self._custom_forward   = lambda x: {'img': x}

    def register_module(self, module: nn.Module, depth: int = 1, name: str| None = None):
        '''
        Add forward hooks to the named children of the module to save the intermediate activations(feature maps).
        If depth is greater than 1, the forward hooks are added to the named children of the named children of the module.
        '''
        name = module.__class__.__name__ if name is None else name
        for sub_module_name, sub_module in module.named_children():
            full_name = f'{name}.{sub_module_name}'
            if depth > 1:
                self.register_module(sub_module, depth - 1, full_name)
            elif isinstance(sub_module, nn.ModuleList):
                for i, sub_sub_module in enumerate(sub_module):
                    sub_sub_module.register_forward_hook(partial(save_activations_hook, self._intermediate_maps, f'{full_name}_{i}'))
            else:
                sub_module.register_forward_hook(partial(save_activations_hook, self._intermediate_maps, full_name))


    def set_forward(self, forward_func: Callable[[Any], Dict[str, torch.Tensor]]) -> None:
        '''
        Set a custom forward function for the network. This function should a dictionary
        '''
        self._custom_forward = forward_func

    @torch.no_grad()
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        '''
        Forward pass of the network. Make sure to return self._intermediate_maps
        :param args:
        :param kwargs:
        :return:
        '''

        result = self._custom_forward(*args, **kwargs)
        result.update(self._intermediate_maps)
        return result



if __name__ == '__main__':
    class Examplemodel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3), nn.ReLU())

        def forward(self, x):
            x = self.relu1(self.conv1(x))
            x = self.conv2(x)
            return x

    net = UnwrappedNetwork()
    model1 = Examplemodel()
    net.register_module(model1, depth=1)
    forward_func = lambda x: {'output': model1(x)}
    net.set_forward(forward_func)
    out = net(torch.randn(1, 3, 224, 224))
    print(out.keys())
