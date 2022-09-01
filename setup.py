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

#
# This script derives from the following link:
#
#   https://stackoverflow.com/questions/42585210/extending-setuptools-extension-to-use-cmake-in-setup-py
#

import os
from setuptools import setup
from setuptools import Extension
from setuptools.command.build_ext import build_ext

class MetaSporeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])

class metaspore_build_ext(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_metaspore(ext)

    def get_metaspore_so_path(self):
        key = '_METASPORE_SO'
        path = os.environ.get(key)
        if path is None:
            message = "environment variable %r is not set; " % key
            message += "can not find path of '_metaspore.so'"
            raise RuntimeError(message)
        if not os.path.isfile(path):
            message = "'_metaspore.so' is not found at %r" % path
            raise RuntimeError(message)
        return path

    def build_metaspore(self, ext):
        import shutil
        metaspore_so_path = self.get_metaspore_so_path()
        ext_so_path = self.get_ext_fullpath(ext.name)
        print(f'ext copy so cwd: {os.getcwd()} from {metaspore_so_path} to {ext_so_path}')
        shutil.copy(metaspore_so_path, ext_so_path)
        src_libs_path = os.path.join(os.path.dirname(metaspore_so_path), '.libs')
        dst_libs_path = os.path.join(os.path.dirname(ext_so_path), '.libs')
        print(f'ext copy .libs cwd: {os.getcwd()} from {src_libs_path} to {dst_libs_path}')
        shutil.rmtree(dst_libs_path, True)
        shutil.copytree(src_libs_path, dst_libs_path)

setup(
    ext_modules=[MetaSporeExtension('metaspore/_metaspore')],
    cmdclass={ 'build_ext': metaspore_build_ext },
)
