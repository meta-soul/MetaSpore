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

import io
import inspect
import pickle
import cloudpickle
from cloudpickle import PYPY
import torch

def _patch_lookup_module_and_qualname():
    _orig_whichmodule = cloudpickle.cloudpickle._whichmodule
    _orig_lookup_module_and_qualname = cloudpickle.cloudpickle._lookup_module_and_qualname
    def patched_lookup_module_and_qualname(obj, name=None):
        if name is None:
            name = getattr(obj, '__qualname__', None)
        if name is None:
            name = getattr(obj, '__name__', None)
        module_name = _orig_whichmodule(obj, name)
        if module_name is None:
            return None
        if module_name.startswith('unusual_prefix_'):
            return None
        return _orig_lookup_module_and_qualname(obj, name)
    cloudpickle.cloudpickle._lookup_module_and_qualname = patched_lookup_module_and_qualname

def _patch_getsourcelines():
    _orig_getsourcelines = inspect.getsourcelines
    def patched_getsourcelines(obj):
        if not hasattr(obj, '_SourcePatchingPickler__filename'):
            return _orig_getsourcelines(obj)
        sourcelines = getattr(obj, '_SourcePatchingPickler__sourcelines')
        file_lineno = getattr(obj, '_SourcePatchingPickler__file_lineno')
        sourcelines = list(sourcelines)
        return sourcelines, file_lineno
    inspect.getsourcelines = patched_getsourcelines

class SourcePatchingPickler(cloudpickle.CloudPickler):
    def _patch_source(self, module_class):
        if module_class.__module__.startswith('torch.nn.'):
            return
        if module_class.__module__.startswith('metaspore.nn.'):
            return
        forward_method = module_class.forward
        if hasattr(forward_method, '_SourcePatchingPickler__filename'):
            return
        filename = inspect.getsourcefile(forward_method)
        sourcelines, file_lineno = inspect.getsourcelines(forward_method)
        sourcelines = tuple(sourcelines)
        setattr(forward_method, '_SourcePatchingPickler__filename', filename)
        setattr(forward_method, '_SourcePatchingPickler__sourcelines', sourcelines)
        setattr(forward_method, '_SourcePatchingPickler__file_lineno', file_lineno)

    if pickle.HIGHEST_PROTOCOL >= 5 and not PYPY:
        def reducer_override(self, obj):
            if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
                self._patch_source(obj)
            return super().reducer_override(obj)

    else:
        def save(self, obj, save_persistent_id=True):
            if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
                self._patch_source(obj)
            super().save(obj, save_persistent_id=save_persistent_id)

if pickle.HIGHEST_PROTOCOL >= 5 and not PYPY:
    def dump(obj, file, protocol=None, buffer_callback=None):
        SourcePatchingPickler(
            file, protocol=protocol, buffer_callback=buffer_callback
        ).dump(obj)

    def dumps(obj, protocol=None, buffer_callback=None):
        with io.BytesIO() as file:
            cp = SourcePatchingPickler(
                file, protocol=protocol, buffer_callback=buffer_callback
            )
            cp.dump(obj)
            return file.getvalue()

else:
    def dump(obj, file, protocol=None):
        SourcePatchingPickler(file, protocol=protocol).dump(obj)

    def dumps(obj, protocol=None):
        with io.BytesIO() as file:
            cp = SourcePatchingPickler(file, protocol=protocol)
            cp.dump(obj)
            return file.getvalue()

load, loads = pickle.load, pickle.loads
Pickler = SourcePatchingPickler
