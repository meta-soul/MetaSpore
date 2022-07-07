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
from ..utils import generate_keyword_match_rules, generate_attribute_sorter_rules, search_es_using_query_combination, generte_id_rules, search_es_using_id_filtering, parse_es_search_result
from ..utils import start_logging

class SearchEvaluateNode(PipelineNode):
    def __call__(self, **payload) -> dict:
        es = payload['es']
        confs = payload['conf']
        logger = start_logging(**confs['logging'])
        evaluation = confs[self._node_conf]
        index_name = confs['indexer_conf']['create_index']
        searchqueries = evaluation['search_queries']
        for query in searchqueries:
            if 'keyword' in query:
                match_rules = generate_keyword_match_rules(query['keywords']['values'])
                sorter_rules = generate_attribute_sorter_rules(query['keywords']['sorter_rules'])
                result = search_es_using_query_combination(es,
                                                   index_name,
                                                   must_rules=match_rules,
                                                   sorter_rules=sorter_rules,
                                                   from_no=0,
                                                   size=20)
                logger.info('Search results: {}'.format(parse_es_search_result(result)))
                print(parse_es_search_result(result))
            elif 'ids' in query:
                id_rules = generte_id_rules(query['ids']['values'])
                sorter_rules = generate_attribute_sorter_rules(query['ids']['sorter_rules'])
                result = search_es_using_id_filtering(es,
                                              index_name,
                                              id_rules=id_rules,
                                              sorter_rules=sorter_rules,
                                              from_no=0,
                                              size=20)
                logger.info('Search results: {}'.format(parse_es_search_result(result)))
        
        return payload
    