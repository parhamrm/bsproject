import os
import torch
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self):
        self.device = torch.device('cpu')
        self.save_dir = './checkpoints/'
        self.model_name = None
        self.visual_names = []
        self.metric = 0
        torch.backends.cudnn.benchmark = True

    def set_input(self, input):
        real = input.to(self.device)
        self.real = real.reshape((1, *real.shape))

    def forward(self):
        self.fake = self.netG(self.real)

    def get_generated(self):
        return self.fake

    def setup(self):
        self.load_networks(self.model_name)

    def test(self):
        with torch.no_grad():
            self.forward()

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, model_name):
        load_path = os.path.join(self.save_dir, model_name)
        net = self.netG
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
            self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
        net.load_state_dict(state_dict)