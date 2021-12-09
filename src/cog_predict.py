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

EXPERIMENT_DATA_ARGS = {
    "ffhq_encode": {
        "model_path": "weights/e4e_ffhq_encode.pt",
    },
}
# Setup required image transformations
EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]
EXPERIMENT_ARGS['transform'] = transforms.Compose([
    transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

model_path = EXPERIMENT_ARGS['model_path']
ckpt = torch.load(model_path, map_location='cpu')
opts = ckpt['opts']
opts['checkpoint_path'] = model_path
opts= Namespace(**opts)
net = pSp(opts)
net.eval()
net.cuda()
print('Model successfully loaded!')


def run_on_batch(inputs, net):
    images, latents = net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)
    if experiment_type == 'cars_encode':
        images = images[:, :, 32:224, :]
    return images, latents

def get_latent_code(img, data_type):
    img = align_func(img, data_type).transpose(1, 2, 0) # 1024,1024,3
    img = cv2.resize(img, (RESIZE_SIZE, RESIZE_SIZE))[:,:,::-1] * 255 # 256,256,3
    aligned_image = img[:,:,::-1]
    img = Image.fromarray(np.uint8(img))
    
    img_transforms = EXPERIMENT_ARGS['transform']
    transformed_image = img_transforms(img)

    with torch.no_grad():
        tic = time.time()
        images, latents = run_on_batch(transformed_image.unsqueeze(0), net)
        latent = latents[0]
        #result_image, latent = images[0], latents[0]
        toc = time.time()
        print('Inference took {:.4f} seconds.'.format(toc - tic))
    #result_image = ((result_image.cpu().numpy().transpose(1, 2, 0) + 1) / 2)[:,:,::-1] * 255
    return latent, aligned_image
    
if __name__ == "__main__":
    img = cv2.imread("example.jpg")
    latent = get_latent_code(img)