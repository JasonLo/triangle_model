#! /bin/bash

# Setup python path and bash alias
echo "/home/jupyter/triangle_model/src/" > /opt/conda/lib/python3.7/site-packages/triangle_model_src.pth
echo "alias nv=\"watch -n1 nvidia-smi\"" >> /home/jupyter/.bash_profile

# Install python libraries
pip install -r /home/jupyter/triangle_model/requirements.txt

# Configure git
git config --global user.name "Jason Lo"
git config --global user.email "lcmjlo@gmail.com"
git config --global credential.helper store
git fetch

mkdir -p ~/.config/git
# ignore output when diff-ing ipynb 
echo "*.ipynb diff=jupyternotebook" > /home/jupyter/triangle_model/.gitattributes
echo "[filter \"jupyternotebook\"]" >> /home/jupyter/triangle_model/.git/config
echo -e "\tclean = jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace %f" >> /home/jupyter/triangle_model/.git/config
echo -e "\trequired" >> /home/jupyter/triangle_model/.git/config

