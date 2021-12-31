from imageio.core.functions import imwrite
from torch.autograd import Variable
from imageio import imread, imwrite
from PIL import Image
import argparse
import os
import cv2
from os import walk
import numpy as np
import torch
from skimage.transform import resize

parser = argparse.ArgumentParser(description="PyTorch VDSR Demo")
parser.add_argument("--cuda", default=1, action="store_true", help="use cuda?")
parser.add_argument("--model", default="checkpoint/model_epoch_25.pth", type=str, help="model path")
parser.add_argument("--scale", default=3, type=int,help="scale factor, Default: 3")
parser.add_argument("--gpus", default="0", type=str,help="gpu ids (default: 0)")


def colorize(y, ycbcr):
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:, :, 0] = y
    img[:, :, 1] = ycbcr[:, :, 1]
    img[:, :, 2] = ycbcr[:, :, 2]
    img = Image.fromarray(img, "YCbCr").convert("RGB")
    return img


opt = parser.parse_args()
cuda = opt.cuda
if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

for root, dirs, files in walk("datasets/testing_lr_images"):
    for image in files:
        model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]
        img = imread("datasets/testing_lr_images/"+image)
        height, width = img.shape[:2]
        img = cv2.resize(img, (opt.scale*width, opt.scale*height), interpolation=cv2.INTER_CUBIC)
        im_b_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        im_b_y = im_b_ycbcr[:, :, 0].astype(float)
        im_input = im_b_y/255.

        im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])

        if cuda:
            model = model.cuda()
            im_input = im_input.cuda()
        else:
            model = model.cpu()

        out = model(im_input)
        out = out.cpu()

        im_h_y = out.data[0].numpy().astype(np.float32)

        im_h_y = im_h_y * 255.
        im_h_y[im_h_y < 0] = 0
        im_h_y[im_h_y > 255.] = 255.

        im_h = colorize(im_h_y[0, :, :], im_b_ycbcr)
        imwrite("answer/"+image[:-4]+"_pred.png", im_h)
