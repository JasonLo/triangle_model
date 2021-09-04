#! /bin/bash

# Setup python path and bash alias
echo "/home/jupyter/triangle_model/src/" > /opt/conda/lib/python3.7/site-packages/triangle_model_src.pth
echo "PYTHONPATH=\$PYTHONPATH:/home/jupyter/triangle_model/src" >> /home/jupyter/.bash_profile
echo "export PYTHONPATH" >> /home/jupyter/.bash_profile
echo "export PATH" >> /home/jupyter/.bash_profile
echo "alias nv=\"watch -n1 nvidia-smi\"" >> /home/jupyter/.bash_profile
# Need to download gcp_key.json manually, not stored in repo
echo "export GOOGLE_APPLICATION_CREDENTIALS=\"/home/jupyter/triangle_model/secret/gcp_key.json\"" >> /home/jupyter/.bash_profile

# Install python libraries
pip install -r /home/jupyter/triangle_model/requirements.txt

# Configure git
git config --global user.name "Jason Lo"
git config --global user.email "lcmjlo@gmail.com"
git config --global credential.helper store
git fetch
mkdir -p ~/.config/git
nbstripout --install --global --attributes=~/.config/git/attributes

