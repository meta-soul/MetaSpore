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

import re
from attrs import frozen
from typing import Any
from typing import Dict
from typing import Literal
from typing import Optional
from typing import Tuple

class ResourceLoader(object):
    _IDENTIFIER = '[A-Za-z_][A-Za-z0-9_]*'
    _IDENTIFIER_RE = re.compile(_IDENTIFIER + '$')

    _RESOURCE_CLASS_NAME = '(%s)(V\\d+)' % _IDENTIFIER
    _RESOURCE_CLASS_NAME_RE = re.compile(_RESOURCE_CLASS_NAME + '$')

    @frozen
    class _ResourceMetadata:
        name: str
        uses: Optional[Tuple[str, ...]] = None
        vars: Optional[Dict[str, str]] = None

    def __init__(self, namespace, resource_types=None, context=None):
        if not isinstance(namespace, str) or not self._IDENTIFIER_RE.match(namespace):
            message = "namespace must be identifier; %r is invalid" % (namespace,)
            raise ValueError(message)
        if context is None:
            context = {}
        self._namespace = namespace
        self._resource_types = []
        self._context = context
        if resource_types is not None:
            self.add_resource_types(resource_types)

    def add_resource_types(self, resource_types):
        for resource_type in resource_types:
            self.add_resource_type(resource_type)

    def add_resource_type(self, resource_type):
        if not isinstance(resource_type, type):
            message = "resource_type must be type; %r is invalid" % (resource_type,)
            raise TypeError(message)
        resource_name = resource_type.__name__
        name, version = self._get_name_and_version(resource_name)
        api_version = self._namespace + '/' + version
        @frozen
        class raw_wrapper_type:
            apiVersion: Literal[api_version]
            kind: Literal[name]
            metadata: self._ResourceMetadata
            spec: Dict[str, Any]
        @frozen
        class wrapper_type:
            apiVersion: Literal[api_version]
            kind: Literal[name]
            metadata: self._ResourceMetadata
            spec: resource_type
        self._resource_types.append((resource_type, raw_wrapper_type, wrapper_type))

    def _get_name_and_version(self, resource_name):
        match = self._RESOURCE_CLASS_NAME_RE.match(resource_name)
        if match is not None:
            return match.group(1), match.group(2).lower()
        else:
            return resource_name, 'v1'

    def _create_context(self, raw_resource, context):
        if raw_resource.metadata.vars is None:
            return context
        context = context.copy()
        context.update(raw_resource.metadata.vars)
        return context

    def _create_loader_type(self, context):
        import string
        import yaml
        class loader_type(yaml.SafeLoader):
            pass
        def string_constructor(loader, node):
            template = string.Template(node.value)
            value = template.substitute(context)
            return value
        tag = 'tag:yaml.org,2002:str'
        token_re = string.Template.pattern
        loader_type.add_constructor(tag, string_constructor)
        loader_type.add_implicit_resolver(tag, token_re, None)
        return loader_type

    def _get_text(self, path):
        import io
        with io.open(path) as fin:
            text = fin.read()
            return text

    def _load_yaml(self, path, text, context=None):
        import yaml
        try:
            if context is None:
                source = yaml.safe_load(text)
            else:
                loader_type = self._create_loader_type(context)
                source = yaml.load(text, Loader=loader_type)
            return source
        except Exception as ex:
            message = "resource file %r is invalid" % (path,)
            raise RuntimeError(message) from ex

    def _load_raw_resource(self, path, text):
        import cattrs
        source = self._load_yaml(path, text)
        last_ex = None
        for _, raw_wrapper_type, wrapper_type in self._resource_types:
            try:
                raw_resource = cattrs.structure(source, raw_wrapper_type)
                return raw_resource, wrapper_type
            except Exception as ex:
                last_ex = ex
        message = "fail to load %r as raw resource" % (path,)
        raise RuntimeError(message) from last_ex

    def _load_resource(self, path):
        import cattrs
        text = self._get_text(path)
        raw_resource, wrapper_type = self._load_raw_resource(path, text)
        context = self._create_context(raw_resource, self._context)
        source = self._load_yaml(path, text, context)
        try:
            resource = cattrs.structure(source, wrapper_type)
            return resource
        except Exception as ex:
            message = "fail to load %r as resource" % (path,)
            raise RuntimeError(message) from ex

    def load_into(self, path, resource_manager):
        import os
        import collections
        loaded = set()
        queue = collections.deque()
        queue.append(os.path.normpath(path))
        while queue:
            resource_path = queue.popleft()
            resource = self._load_resource(resource_path)
            name = resource.metadata.name
            resource_manager.add_resource(name, resource_path, resource.spec)
            loaded.add(resource_path)
            resource_dir = os.path.dirname(resource_path)
            if resource.metadata.uses is not None:
                for use in resource.metadata.uses:
                    use_path = os.path.join(resource_dir, use)
                    use_path = os.path.normpath(use_path)
                    if use_path not in loaded:
                        queue.append(use_path)

    def load(self, path):
        from .resource_manager import ResourceManager
        resource_manager = ResourceManager()
        self.load_into(path, resource_manager)
        resource_manager.freeze()
        return resource_manager
