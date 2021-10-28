#!/bin/bash

# Setup python path and bash alias
echo "alias nv=\"watch -n1 nvidia-smi\"" >> ~/.bash_profile

# Configure git
git config --global user.name "Jason Lo"
git config --global user.email "lcmjlo@gmail.com"
git config --global credential.helper store
git fetch
