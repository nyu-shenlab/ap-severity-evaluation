#!/bin/bash

conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# install packages
pip install -r requirements.txt


echo "Setup environment done!"