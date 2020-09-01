from .base_model import BaseModel
from .networks import define_G

class Pix2PixModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self.model_name = 'Pix2Pix.pth'
        self.netG = define_G(3, 3, 64, 'unet_256', 'batch', True, 'normal', 0.02)