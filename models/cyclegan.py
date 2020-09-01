from .base_model import BaseModel
from .networks import define_G


class CycleGANModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self.model_name = 'CycleGAN.pth'
        self.netG = define_G(3, 3, 64, 'resnet_9blocks', 'instance', False, 'normal', 0.02)