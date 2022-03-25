#!/bin/bash

#
# Copyright 2022 DMetaSoul
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set -e
pushd $(dirname ${BASH_SOURCE[0]})/../..
rm -rf build/python-env
rm -rf build/python-env.tgz
mkdir -p build/python-env
tar -xf /usr/local/python-env-3.8.5.tgz -C build/python-env
build/python-env/bin/python3.8 -m pip install --upgrade build/metaspore-2.0.0+*-cp38-cp38-linux_x86_64.whl pip
find build/python-env/bin -type f -exec sed -i -e 's@^#!.\+/bin/python\(3\(\.8\)\?\)\?$@#!/usr/bin/env python3.8@' {} \;
tar -czf build/python-env.tgz -C build/python-env $(ls build/python-env)
rm -rf build/python-env
popd
echo OK
