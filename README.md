# Oh-My-Face

This project is based on [StyleCLIP](https://github.com/orpatashnik/StyleCLIP), [RIFE](https://github.com/hzwer/arXiv2020-RIFE), and [encoder4editing](https://github.com/omertov/encoder4editing), which aims to expand human face editing via Global Direction of StyleCLIP, especially to maintain similarity during editing. 

StyleCLIP is an excellent algorithm that acts on the latent code of StyleGAN2 to edit images guided by texts. Global Direction uses models such as e4e to convert images into latent codes and then further editing. However, this conversion causes information loss of the original image and dissimilarities.

Thus, we use the optical flow model to detect the change in different regions between the StyleCLIP generated image and the original image, sample more from the original in slightly-edited areas, then use frame interpolation to perform weighted fusion, which is simple yet efficient.

We provide weights for cat face editing, containing cat facial landmark recognition from [pycatfd](https://github.com/marando/pycatfd) and e4e-cat model. e4e-cat is trained via [afhq-cat dataset](https://github.com/clovaai/stargan-v2#animal-faces-hq-dataset-afhq) and [StyleGAN2-cat](https://github.com/NVlabs/stylegan2) weights. [StyleGAN2-pytorch/convert_weights.py](https://github.com/rosinality/stylegan2-pytorch/blob/master/convert_weight.py) is used to convert the tensorflow weights.

## Usage

#### Prerequisites

* NVIDIA GPU + CUDA11.0 CuDNN
* Python 3.6

#### Installation

* Clone this repository

```bash
git clone https://github.com/P2Oileen/oh-my-face
```

* Dependencies

  We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/). To install all the dependencies, please run the following commands.

```bash
conda create -n StyleCLIP-FaceEX python=3.6
conda active StyleCLIP-FaceEX
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=<CUDA_VERSION>
conda install -c anaconda tensorflow-gpu==1.15.2

pip install ftfy regex tqdm gdown
pip install git+https://github.com/openai/CLIP.git
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
sudo unzip ninja-linux.zip -d /usr/local/bin/
sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force
```

* Download Weights

```

```

