import numpy as np 
import torch
import clip
import time
from PIL import Image
from global_directions.MapTS import GetBoundary,GetDt
from global_directions.manipulate import Manipulator

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device) 
M=[]
M.append(None)#Manipulator(dataset_name='ffhq'))
M.append(Manipulator(dataset_name='cat'))
fs3=[np.load('./ffhq/fs3.npy'), np.load('./cat/fs3.npy')]
np.set_printoptions(suppress=True)

def global_transfer(latent, data_type = 'face', neutral = 'face', target = 'face with blue eyes', beta=0.15, alpha=4.1):

    if data_type == 'face':
        data_choice = 0
    else:
        data_choice = 1

    classnames=[target,neutral]
    dt=GetDt(classnames,model)
    dlatent_tmp = M[data_choice].W2S(latent)
    
    M[data_choice].alpha=[alpha]
    M[data_choice].num_images=1
    M[data_choice].manipulate_layers=None
    boundary_tmp2,c=GetBoundary(fs3[data_choice],dt,M[data_choice],threshold=beta)
    codes=M[data_choice].MSCode(dlatent_tmp,boundary_tmp2)
    out=M[data_choice].GenerateImg(codes)
    generated=Image.fromarray(out[0,0]).resize((512,512))
    generated=np.asarray(generated)

    return generated
