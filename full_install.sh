git clone https://github.com/akashin/baselines.git

sudo apt install zsh

Changing /etc/pam.d/chsh: from:
auth       required   pam_shells.so

git clone https://github.com/akashin/dotfiles.git && cd dotfiles && ./bootstrap.sh

#sudo apt install python3-pip
#pip3 install plumbum

source /anaconda/bin/activate py35

sudo /anaconda/envs/py35/bin/pip install dill plumbum gym
sudo apt install -y python-numpy cmake zlib1g-dev libjpeg-dev libboost-all-dev gcc libsdl2-dev wget unzip
sudo /anaconda/envs/py35/bin/pip install ppaquette-gym-doom

#sudo /anaconda/envs/py35/bin/pip install tensorflow-gpu
#sudo apt-get install libcupti-dev

#CUDA_REPO_PKG="cuda-repo-ubuntu1604_8.0.61-1_amd64.deb"
#wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/${CUDA_REPO_PKG}
#sudo dpkg -i ${CUDA_REPO_PKG}
#sudo apt-get update
#sudo apt-get -y install cuda

CUDNN_TAR_FILE="cudnn-8.0-linux-x64-v6.0.tgz"
wget http://developer.download.nvidia.com/compute/redist/cudnn/v6.0/${CUDNN_TAR_FILE}
tar -xzvf ${CUDNN_TAR_FILE}
sudo cp -P cuda/include/cudnn.h /usr/local/cuda-8.0/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64/
sudo chmod a+r /usr/local/cuda-8.0/lib64/libcudnn*

export CUDA_HOME=/usr/local/cuda-8.0
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

