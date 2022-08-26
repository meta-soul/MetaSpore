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

from attrs import frozen
from attrs import field
from typing import Literal
from typing import Tuple

@frozen
class FeatureGroupMetadata:
    name: str

@frozen
class FeatureGroupSpec:
    format: str

@frozen
class FeatureGroup:
    apiVersion: Literal['metaspore/v1']
    kind: Literal['FeatureGroup']
    metadata: FeatureGroupMetadata
    spec: FeatureGroupSpec

@frozen
class SparseFeatureGroupMetadata(FeatureGroupMetadata):
    pass

@frozen
class SparseFeatureGroupSpec(FeatureGroupSpec):
    format: Literal['sparse']
    features: Tuple[str, ...]

@frozen
class SparseFeatureGroup(FeatureGroup):
    metadata: SparseFeatureGroupMetadata
    spec: SparseFeatureGroupSpec = field()

    @spec.validator
    def _validate_spec(self, attribute, value):
        import re
        ident = '[a-zA-z_0-9]+'
        combine_rule = '(%s)(#%s)*$' % (ident, ident)
        combine_rule = re.compile(combine_rule)
        for rule in value.features:
            match = combine_rule.match(rule)
            if match is None:
                message = "invalid feature %r detected in " % (rule,)
                message += "sparse feature group %r" % (self.metadata.name)
                raise RuntimeError(message)

    @property
    def schema_source(self):
        string = '\n'.join(self.spec.features)
        return string

def get(name):
    import io
    import yaml
    import cattrs
    # TODO: cf: improve feature group source fetching
    path = name + '.yaml'
    with io.open(path) as fin:
        source = yaml.full_load(fin)
    try:
        group = cattrs.structure(source, SparseFeatureGroup)
        return group
    except Exception as ex:
        message = "fail to parse feature group %r" % (name,)
        raise RuntimeError(message) from ex
