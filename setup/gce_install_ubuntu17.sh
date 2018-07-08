#!/bin/bash
BASEDIR=$(dirname "$0")

sudo apt-get update && sudo apt-get install build-essential libssl-dev libffi-dev python3-pip python3-dev nvidia-cuda-toolkit -y && sudo pip3 install -r $BASEDIR/requirements-linux.txt
echo "Rebooting in 3..."
sleep 3
sudo reboot
