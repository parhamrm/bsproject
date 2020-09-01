from models.pix2pix import Pix2PixModel
from models.cyclegan import CycleGANModel
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch


def load_model(model_name):
    if model_name == 'Pix2Pix':
        return Pix2PixModel()
    elif model_name == 'CycleGAN':
        return CycleGANModel()
    else:
        raise('Model not found')

def load_cyclegan_image(path):
    image = Image.open(path).convert('RGB')
    image_transforms = transforms.Compose((
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.RandomCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ))
    return image_transforms(image)

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def load_pix2pix_image(path):
    image = Image.open(path).convert('RGB')
    image_transforms = transforms.Compose((
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.Lambda(lambda img: __crop(img, (0, 0), 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ))
    return image_transforms(image)

def load_image(path, model_name):
    if model_name == 'Pix2Pix':
        return load_pix2pix_image(path)
    elif model_name == 'CycleGAN':
        return load_cyclegan_image(path)
    else:
        raise('Model not found')

def tensor2im(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0 
    return image_numpy.astype(imtype)