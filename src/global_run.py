from global_singleimg_infer import global_transfer
import numpy as np
import argparse
import torch
import cv2
import sys

sys.path.append("./global_directions")
sys.path.append("src")

parser = argparse.ArgumentParser(description='Process Options.')
parser.add_argument('--output_dir', default='output.jpg', type=str)
parser.add_argument('--transfer_type', default='Face with smile', type=str)
parser.add_argument('--beta', default=0.15, type=float)
parser.add_argument('--alpha', default=4.1, type=float)
args = parser.parse_args()

latent = torch.load("tmp_latent.pt")
beta = args.beta
alpha = args.alpha

result = global_transfer(latent.cpu().detach().numpy(), target = args.transfer_type, beta = beta, alpha = alpha)
print(result.shape)
cv2.imwrite("output.jpg",result[:,:,::-1])

