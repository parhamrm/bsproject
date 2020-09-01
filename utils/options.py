import argparse

parser = argparse.ArgumentParser(description='Run model on image.')

parser.add_argument('image', help="Path to input image.", type=str)
parser.add_argument('--model', help="Model name [CycleGAN/Pix2Pix]", type=str, default="Pix2Pix")
parser.add_argument('--save-to', help="Path to save location", type=str, default="./")

