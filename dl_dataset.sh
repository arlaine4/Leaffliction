#!/bin/bash

# download dataset
wget https://cdn.intra.42.fr/document/document/17547/leaves.zip
unzip leaves.zip
mkdir images/Apple images/Grape
mv images/Apple_* images/Apple
mv images/Grape_* images/Grape
rm leaves.zip
