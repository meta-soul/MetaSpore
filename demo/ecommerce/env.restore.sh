#!/bin/bash

set -e
rm -rf $(realpath $(dirname ${BASH_SOURCE[0]}))/python-env
python3 -m venv $(realpath $(dirname ${BASH_SOURCE[0]}))/python-env
source $(realpath $(dirname ${BASH_SOURCE[0]}))/python-env/bin/activate
python -m pip install --upgrade pip metasporecli~=0.1.2
python -m pip freeze > $(realpath $(dirname ${BASH_SOURCE[0]}))/python-env/requirements.txt
