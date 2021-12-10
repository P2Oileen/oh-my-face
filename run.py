import os

option_beta = 0.15 #min_value=0.08, max_value=0.3, value=0.15, step=0.01)
option_alpha = 4.1 #min_value=-10.0, max_value=10.0, value=4.1, step=0.1)
option_gamma = 6 #min_value=2, max_value=10, value=6, step=1)
option_data_type = 'face' #['face', 'cat']

os.system('export PATH=/data/cuda/cuda-10.0/cuda/bin:$PATH \
            export LD_LIBRARY_PATH=/data/cuda/cuda-10.0/cuda/lib64:/data/cuda/cuda-10.0/cudnn/v7.6.0/lib64:$LD_LIBRARY_PATH \
            export CUDA_HOME=/data/cuda/cuda-10.0/cuda \
            export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME')
os.system('rlaunch --gpu=1 --cpu=4 --memory=10240 -- python3 src/global_get_latent.py --input_dir \'{}\' --data_type \'{}\' --weight_dir ./weights '.format('input.jpg', option_data_type))
os.system('rlaunch --gpu=1 --cpu=4 --memory=10240 -- python3 src/global_run.py  --transfer_type \'face with smile\' --beta {} --alpha {}'.format(option_beta, option_alpha))
os.system('rlaunch --gpu=1 --cpu=4 --memory=10240 -- python3 src/rife/inference_img.py --img output.jpg input_aligned.jpg --gamma {}'.format(option_gamma))

