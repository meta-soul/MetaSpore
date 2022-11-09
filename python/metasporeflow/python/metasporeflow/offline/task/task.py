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

from abc import ABC, abstractmethod


class Task(ABC):

    def __init__(self,
                 name,
                 type,
                 data
                 ):
        self._name = name
        self._type = type
        self._data = data

    def __repr__(self):
        return '%s(%s, %s, %s)' % (self.__class__.__name__,
                               self._name,
                               self._type,
                               self._data)

    @abstractmethod
    def _execute(self):
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    @property
    def data(self):
        return self._data

    @property
    def execute(self):
        return self._execute()
