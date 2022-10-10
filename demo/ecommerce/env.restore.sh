#!/bin/bash

set -e
rm -rf $(realpath $(dirname ${BASH_SOURCE[0]}))/python-env
python -m venv $(realpath $(dirname ${BASH_SOURCE[0]}))/python-env
source $(realpath $(dirname ${BASH_SOURCE[0]}))/python-env/bin/activate
python -m pip install ../../python/metasporecli/metasporecli-0.1.0-py3-none-any.whl  \
                      ../../python/metasporecli/metasporeflow-0.1.0-py3-none-any.whl
python -m pip freeze > $(realpath $(dirname ${BASH_SOURCE[0]}))/python-env/requirements.txt
