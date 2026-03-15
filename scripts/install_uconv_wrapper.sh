#!/bin/bash
# Installs a Python-based uconv wrapper for systems without icu-devtools
mkdir -p ~/bin
cp scripts/uconv_wrapper.py ~/bin/uconv
chmod +x ~/bin/uconv
export PATH=$HOME/bin:$PATH
echo 'export PATH=$HOME/bin:$PATH' >> ~/.bashrc
echo "uconv wrapper installed."
