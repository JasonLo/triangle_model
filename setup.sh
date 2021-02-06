#! /bin/bash
echo "/home/jupyter/tf/src" > /opt/conda/lib/python3.7/site-packages/tf_src.pth
pip install -r /home/jupyter/tf/requirements.txt
mkdir -p ~/.config/git
nbstripout --install --global --attributes=~/.config/git/attributes
cd /home/jupyter/tf
git config --global credential.helper store
git fetch