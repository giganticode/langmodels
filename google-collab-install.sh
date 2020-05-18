#!/usr/bin/env bash
set -euo pipefail

pip uninstall -y coveralls datascience chainer albumentations Pygments pyyaml typing-extensions

git clone https://github.com/giganticode/langmodels
git clone https://github.com/giganticode/datasets
cd langmodels
pip install -r requirements.txt
pip install -r requirements-dev.txt
cd ../datasets
pip install -r requirements.txt
