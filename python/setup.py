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
        shutil.copy(metaspore_so_path, ext_so_path)
        metaspore_so_dir = os.path.dirname(os.path.realpath(metaspore_so_path))
        ext_so_dir = os.path.dirname(os.path.realpath(ext_so_path))
        so_names = ['libstdc++.so.6', 'libgcc_s.so.1']
        for so_name in so_names:
            shutil.copy(os.path.join(metaspore_so_dir, so_name), os.path.join(ext_so_dir, so_name))

def get_metaspore_version():
    key = '_METASPORE_VERSION'
    metaspore_version = os.environ.get(key)
    if metaspore_version is None:
        message = "environment variable %r is not set; " % key
        message += "can not get MetaSpore wheel version"
        raise RuntimeError(message)
    return metaspore_version

setup(name='metaspore',
      version=get_metaspore_version(),
      description="MetaSpore AI platform.",
      packages=['metaspore', 'metaspore.nn', 'metaspore.compat', 'metaspore.compat.ps', 'ps'],
      ext_modules=[MetaSporeExtension('metaspore/_metaspore')],
      cmdclass={ 'build_ext': metaspore_build_ext },
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: Apache Software License",
          "Operating System :: POSIX :: Linux",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
      ],
      install_requires=['numpy>=1.20.1',
                        'pandas>=1.2.3',
                        'nest_asyncio>=1.5.1',
                        'cloudpickle>=1.6.0',
                        'pyarrow>=3.0.0',
                        'PyYAML>=5.3.1',
                        'boto3>=1.17.41',
                        'python-consul>=1.1.0',
                        'findspark>=1.4.2',
                        'tabulate'])
