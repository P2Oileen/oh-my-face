import numpy as np 
import torch
import clip
import time
from PIL import Image
from global_directions.MapTS import GetBoundary,GetDt
from global_directions.manipulate import Manipulator

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device) 

M=Manipulator(dataset_name='ffhq') 
fs3=np.load('./ffhq/fs3.npy')
np.set_printoptions(suppress=True)

def global_transfer(latent, neutral = 'face', target = 'face with blue eyes', beta=0.15, alpha=4.1):

    tik = time.time()
    classnames=[target,neutral]
    dt=GetDt(classnames,model)
    dlatent_tmp = M.W2S(latent)
    
    M.alpha=[alpha]
    M.num_images=1
    M.manipulate_layers=None
    boundary_tmp2,c=GetBoundary(fs3,dt,M,threshold=beta)
    codes=M.MSCode(dlatent_tmp,boundary_tmp2)
    out=M.GenerateImg(codes)
    generated=Image.fromarray(out[0,0]).resize((512,512))
    generated=np.asarray(generated)

    tok = time.time()
    print(tok - tik)
    return generated
