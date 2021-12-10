from cog_predict import get_latent_code
import cv2
import argparse
import torch
parser = argparse.ArgumentParser(description='Process Options.')
parser.add_argument('--input_dir', default='input.jpg', type=str)
parser.add_argument('--data_type', default='face', type=str) #[face, cat]
parser.add_argument('--weight_dir', default='./weights', type=str)
args = parser.parse_args()

img = cv2.imread(args.input_dir)
latent,img = get_latent_code(img, args.data_type, args.weight_dir)
latent = latent.unsqueeze(0)
print("print aligned image:")
print(cv2.imwrite("input_aligned.jpg",img))
torch.save(latent,"tmp_latent.pt")
