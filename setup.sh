#! /bin/bash
echo "/home/jupyter/triangle_model/src" > /opt/conda/lib/python3.7/site-packages/tf_src.pth
pip install -r /home/jupyter/triangle_model/requirements.txt
mkdir -p ~/.config/git
nbstripout --install --global --attributes=~/.config/git/attributes
cd /home/jupyter/triangle_model
git config --global user.name "Jason Lo"
git config --global user.email "lcmjlo@gmail.com"
git config --global credential.helper store
git fetch