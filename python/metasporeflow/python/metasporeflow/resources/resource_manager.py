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

class ResourceManager(object):
    def __init__(self):
        self._name_to_resource = {}
        self._type_to_resources = {}

    def add_resource(self, name, path, resource):
        from .resource import Resource
        if name in self._name_to_resource:
            _, existing_path, _ = self._name_to_resource[name]
            message = "resource name %r conflict in %r and %r" % (name, existing_path, path)
            raise RuntimeError(message)
        r = Resource(name=name, path=path, kind=resource.__class__.__name__, data=resource)
        self._name_to_resource[name] = r
        for base_type in resource.__class__.__mro__:
            if base_type not in self._type_to_resources:
                self._type_to_resources[base_type] = [r]
            else:
                self._type_to_resources[base_type].append(r)

    def freeze(self):
        import types
        self._name_to_resource = types.MappingProxyType(self._name_to_resource)
        self._type_to_resources = types.MappingProxyType({k: tuple(v) for k, v in self._type_to_resources.items()})

    def try_find_by_name(self, name):
        if name not in self._name_to_resource:
            return None
        r = self._name_to_resource[name]
        return r

    def find_by_name(self, name):
        r = self.try_find_by_name(name)
        if r is None:
            message = "resource %r not found" % (name,)
            raise RuntimeError(message)
        return r

    def try_find_by_type(self, resource_type):
        if resource_type not in self._type_to_resources:
            return None
        rs = self._type_to_resources[resource_type]
        if len(rs) > 1:
            message = "found %d resources of type %r" % (len(rs), resource_type)
            raise RuntimeError(message)
        return rs[0]

    def find_by_type(self, resource_type):
        r = self.try_find_by_type(resource_type)
        if r is None:
            message = "resource of type %r not found" % (resource_type,)
            raise RuntimeError(message)
        return r

    def find_all(self, resource_type):
        if resource_type not in self._type_to_resources:
            return ()
        rs = self._type_to_resources[resource_type]
        return tuple(rs)

    def __str__(self):
        import cattrs
        import yaml
        data = []
        for name in self._name_to_resource:
            r = self._name_to_resource[name]
            resource = cattrs.unstructure(r.data)
            data.append({'name': r.name, 'path': r.path, 'kind': r.kind, 'data': resource})
        data = cattrs.unstructure(data)
        string = yaml.dump(data, sort_keys=False)
        return string.rstrip()
