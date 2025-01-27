#!/usr/bin/env bash

# © 2024-2025 Hiroyuki Sakai

dpkg -s unzip &> /dev/null

if [ $? -ne 0 ]; then
  echo "This script requires unzip, which was not found on your system."
  read -p "Do you want to install it now? You may have to provide your password. (y/n)" unzip_choice
  if [[ "$unzip_choice" =~ ^[Yy]$ ]]; then
    sudo apt update
    sudo apt install unzip
  else
    echo "Aborting."
    exit 0
  fi
fi

wget -O "tmp.zip" "https://owncloud.tuwien.ac.at/index.php/s/518ehTKtkZvRU98/download"
unzip -o "tmp.zip"
rm "tmp.zip"
