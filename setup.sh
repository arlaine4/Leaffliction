#!/bin/bash

# download dataset
wget https://cdn.intra.42.fr/document/document/17547/leaves.zip
unzip leaves.zip
mkdir images/Apple images/Grape
mkdir augmented_directory
mkdir transformed_directory
mv images/Apple_* images/Apple
mv images/Grape_* images/Grape
rm leaves.zip

# env setup
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
