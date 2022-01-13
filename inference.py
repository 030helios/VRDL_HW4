import argparse
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import os
from model import Generator

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=3, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='CPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--model', default='weights.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
MODEL_NAME = opt.model

model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load(MODEL_NAME))
else:
    model.load_state_dict(torch.load(
        MODEL_NAME, map_location=lambda storage, loc: storage))

if not os.path.exists("0716034"):
    os.makedirs("0716034")
for img_name in range(14):
    IMAGE_NAME = str(img_name).zfill(2)
    image = Image.open("data/test/"+IMAGE_NAME+".png")
    image = ToTensor()(image).unsqueeze(0)
    if TEST_MODE:
        image = image.cuda()

    out = model(image)
    out_img = ToPILImage()(out[0].data.cpu())
    out_img.save("0716034/"+IMAGE_NAME+"_pred.png")
