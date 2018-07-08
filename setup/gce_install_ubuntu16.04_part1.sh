#!/bin/bash


#part 1

sudo locale-gen en_GB.UTF-8
sudo apt-get update
sudo apt-get upgrade
sudo apt-get clean all
sudo add-apt-repository ppa:jonathonf/python-3.6
sudo apt-get update
sudo apt-get install python3.6
#sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.6 10
sudo apt-get install nvidia-cuda-toolkit
#sudo python --version
sudo python3.6 --version
echo -e "\n"
echo ">>This should show python version 3.6! If so, please reboot and execute script 2!<<"
