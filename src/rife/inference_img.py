import os
import cv2
import torch
import torchvision.transforms as T
import argparse
from torch.nn import functional as F
import warnings
import time
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--img', dest='img', nargs=2, required=True)
parser.add_argument('--exp', default=1, type=int)
parser.add_argument('--ratio', default=0, type=float, help='inference ratio between two images with 0 - 1 range')
parser.add_argument('--rthreshold', default=0.02, type=float, help='returns image when actual ratio falls in given range threshold')
parser.add_argument('--rmaxcycles', default=8, type=int, help='limit max number of bisectional cycles')
parser.add_argument('--model', dest='modelDir', type=str, default='weights', help='directory with trained model files')
parser.add_argument('--gamma', default=6, type=float)

args = parser.parse_args()


from train_log.RIFE_HDv3 import Model
model = Model()
model.load_model(args.modelDir, -1)
print("Loaded v3.x HD model.")
model.eval()
model.device()

tik = time.time()

img0 = cv2.imread(args.img[0]).astype('uint8')
img1 = cv2.imread(args.img[1]).astype('uint8')
img0 = torch.tensor(cv2.resize(img0, (512, 512)),dtype=torch.float32) / 255.
img1 = torch.tensor(cv2.resize(img1, (512, 512)),dtype=torch.float32) / 255.
s0 = img0.std(dim=(0,1))
m0 = img0.mean(dim=(0,1))
s1 = img1.std(dim=(0,1))
m1 = img1.mean(dim=(0,1))
img0 = m1 + (img0 - m0) * s1 / s0
img0 = img0.permute(2, 0, 1)
img1 = img1.permute(2, 0, 1)
img0 = img0.to(device).unsqueeze(0)
img1 = img1.to(device).unsqueeze(0)
n, c, h, w = img0.shape
ph = ((h - 1) // 64 + 1) * 64
pw = ((w - 1) // 64 + 1) * 64
padding = (0, pw - w, 0, ph - h)
img0 = F.pad(img0, padding)
img1 = F.pad(img1, padding)


img_output = model.inference(img0, img1, 0.5, gamma=args.gamma)

print("rife output.")
cv2.imwrite('output.jpg', (img_output[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])

tok = time.time()
print(tok-tik)