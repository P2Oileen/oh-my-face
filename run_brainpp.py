import os
import argparse

parser = argparse.ArgumentParser(description='Process Options.')
parser.add_argument('--input_dir', default='input.jpg', type=str)
parser.add_argument('--option_beta', default = 0.15, type=float) #min_value=0.08, max_value=0.3, value=0.15, step=0.01)
parser.add_argument('--option_alpha', default = 4.1, type=float) #min_value=-10.0, max_value=10.0, value=4.1, step=0.1)
parser.add_argument('--option_gamma', default = 6, type=int) #min_value=2, max_value=10, value=6, step=1)
parser.add_argument('--option_data_type', default = 'face', type=str) #['face', 'cat']
parser.add_argument('--neutral', default = 'face')
parser.add_argument('--target', default = 'face with smile')
args = parser.parse_args()

os.system('export PATH=/data/cuda/cuda-10.0/cuda/bin:$PATH \
            export LD_LIBRARY_PATH=/data/cuda/cuda-10.0/cuda/lib64:/data/cuda/cuda-10.0/cudnn/v7.6.0/lib64:$LD_LIBRARY_PATH \
            export CUDA_HOME=/data/cuda/cuda-10.0/cuda \
            export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME')
os.system('rlaunch --gpu=1 --cpu=4 --memory=10240 -- python3 src/global_get_latent.py --input_dir \'{}\' --data_type \'{}\' --weight_dir ./weights '.format(args.input_dir, args.option_data_type))
os.system('rlaunch --gpu=1 --cpu=4 --memory=10240 -- python3 src/global_run.py --neutral \'{}\' --target \'{}\' --data_type \'{}\' --beta {} --alpha {}'.format(args.neutral, args.target, args.option_data_type, args.option_beta, args.option_alpha))
os.system('rlaunch --gpu=1 --cpu=4 --memory=10240 -- python3 src/rife/inference_img.py --img output.jpg input_aligned.jpg --gamma {}'.format(args.option_gamma))

