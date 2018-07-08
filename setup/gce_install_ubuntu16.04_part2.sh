#!/bin/bash

#part 2

BASEDIR=$(dirname "$0")

curl https://bootstrap.pypa.io/get-pip.py | sudo python3.6
#sudo pip --version
sudo pip3 --version
echo -e "\n"
read -p ">>Check if pip is version 9.x (python 3.6), then press any key to continue!(ctrl-c otherwise)<<" -n 1 -s
nvidia-smi 
nvcc --version
echo -e "\n"
read -p ">>Check if CUDA 7.5 was installed correctly, then press any key to continue!(ctrl-c otherwise)<<" -n 1 -s
#sudo python -m pip install -r $BASEDIR/requirements.txt
sudo python3.6 -m pip install -r $BASEDIR/requirements.txt
#sudo python -m pip install  http://download.pytorch.org/whl/cu75/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl
sudo python3.6 -m pip install  http://download.pytorch.org/whl/cu75/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl 
#sudo python -m pip install -r $BASEDIR/requirements-after-torch.txt
sudo python3.6 -m pip install -r $BASEDIR/requirements-after-torch.txt
