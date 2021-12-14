import os

from argparse import Namespace
import time
import os
import sys
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms
import dlib
import time
from align_face import align_func

from encoder4editing.models.psp import pSp  # we use the pSp framework to load the e4e encoder.

experiment_type = 'ffhq_encode'
RESIZE_SIZE = 256

# Setup required image transformations

trans = transforms.Compose([
    transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

# model_path = ['weights/e4e_ffhq_encode.pt','weights/e4e_cat_encode.pt']
# ckpt = [torch.load(model_path[0], map_location='cpu'), torch.load(model_path[1], map_location='cpu')]
# opts = [ckpt[0]['opts'], ckpt[1]['opts']]
# opts[0]['checkpoint_path'],opts[1]['checkpoint_path'] = model_path[0], model_path[1]
# opts = [Namespace(**opts[0]), Namespace(**opts[1])]
# net = [pSp(opts[0]), pSp(opts[1])]
# net[0].eval(), net[1].eval()
# net[0].cuda(), net[1].cuda()

model_path = ['weights/e4e_ffhq_encode.pt', None]
ckpt = [torch.load(model_path[0], map_location='cpu'), None]
opts = [ckpt[0]['opts'], None]
opts[0]['checkpoint_path'] = model_path[0]
opts = [Namespace(**opts[0]),None]
net = [pSp(opts[0]),None]
net[0].eval()
net[0].cuda()
print('Model successfully loaded!')


def run_on_batch(inputs, net):
    images, latents = net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)
    return images, latents

def get_latent_code(img, data_type, weight_dir):
    img = align_func(img, data_type, weight_dir).transpose(1, 2, 0) # 1024,1024,3
    img = cv2.resize(img, (RESIZE_SIZE, RESIZE_SIZE))[:,:,::-1] * 255 # 256,256,3
    aligned_image = img[:,:,::-1]
    img = Image.fromarray(np.uint8(img))

    if data_type == 'face':
        data_choice = 0
    else:
        data_choice = 1
    
    transformed_image = trans(img)

    with torch.no_grad():
        images, latents = run_on_batch(transformed_image.unsqueeze(0), net[data_choice])
        latent = latents[0]
        #result_image, latent = images[0], latents[0]
    #result_image = ((result_image.cpu().numpy().transpose(1, 2, 0) + 1) / 2)[:,:,::-1] * 255
    return latent, aligned_image
    
if __name__ == "__main__":
    img = cv2.imread("example.jpg")
    latent = get_latent_code(img)