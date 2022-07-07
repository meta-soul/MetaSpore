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

from .node import PipelineNode
import metaspore as ms
from ..utils import get_class
from ..utils import create_es_using_http_auth, save_df_to_es_index, delete_index, create_index


class SearchBuilderNode(PipelineNode):
    def __call__(self, **payload) -> dict:
        confs = payload['conf']
        extended_confs = confs['spark']['extended_confs']
        es_uri = 'http://%s:%s'%(extended_confs['spark.es.nodes'], extended_confs['spark.es.port'])
        es = create_es_using_http_auth(es_uri,
                                       extended_confs['spark.es.net.http.auth.user'],
                                       extended_confs['spark.es.net.http.auth.pass'])
        indexer_conf = confs[self._node_conf]
        index_name = indexer_conf['create_index']
        index_schema = indexer_conf['create_schema']
        mapping_id = indexer_conf['create_mapping_id'] or {}
        es = delete_index(es, index_name)
        es = create_index(es, index_name, index_schema)
        train_dataset = payload['train_dataset']
        es = save_df_to_es_index(es, train_dataset, index_name, mapping_id)
        payload['es'] = es
        
        return payload
    