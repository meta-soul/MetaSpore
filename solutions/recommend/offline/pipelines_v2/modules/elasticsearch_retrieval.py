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
import metaspore as ms
import attrs
import cattrs
import logging

from ..utils import *
from ..constants import *

logger = logging.getLogger(__name__)

@attrs.frozen
class ElasticSearchRetrievalConfig:
    indexing = attrs.field(validator=attrs.validators.instance_of(dict))
    evaluation = attrs.field(default=None, 
        validator=attrs.validators.optional(attrs.validators.instance_of(dict)))

class ElasticSearchRetrievalModule:
    def __init__(self, conf: ElasticSearchRetrievalConfig):
        if isinstance(conf, dict):
            self.conf = ElasticSearchRetrievalConfig.convert(conf)
        elif isinstance(conf, ElasticSearchRetrievalConfig):
            self.conf = conf
        else:
            raise TypeError("Type of 'conf' must be dict or ElasticSearchRetrievalConfig. Current type is {}".format(type(conf)))
        self.metric_position_k = 20

    @staticmethod
    def convert(conf: dict) -> ElasticSearchRetrievalConfig:
        conf = cattrs.structure(conf, ElasticSearchRetrievalConfig)  

    def train(self, train_dataset, extended_confs):
        es_uri = 'http://{}:{}'.format(extended_confs['spark.es.nodes'], extended_confs['spark.es.port'])
        es_instance = create_es_using_http_auth(es_uri,
            extended_confs['spark.es.net.http.auth.user'],
            extended_confs['spark.es.net.http.auth.pass']
        )
        indexer_conf = self.conf.indexing
        index_name = indexer_conf['create_index']
        index_schema = indexer_conf['create_schema']
        mapping_id = indexer_conf['create_mapping_id'] or {}
        es_instance = delete_index(es_instance, index_name)
        es_instance = create_index(es_instance, index_name, index_schema)
        es_instance = save_df_to_es_index(es_instance, train_dataset, index_name, mapping_id)
        return es_instance, index_name

    def evaluate_keywords_queries(self, es_instance, index_name, query):
        match_rules = generate_keyword_match_rules(query['keywords']['values'])
        sorter_rules = generate_attribute_sorter_rules(query['keywords']['sorter_rules'])
        result = search_es_using_query_combination(
            es_instance,
            index_name,
            must_rules=match_rules,
            sorter_rules=sorter_rules,
            from_no=0,
            size=self.metric_position_k
        )
        return result             

    def evaluate_id_queries(self, es_instance, index_name, query):
        id_rules = generte_id_rules(query['ids']['values'])
        sorter_rules = generate_attribute_sorter_rules(query['ids']['sorter_rules'])
        result = search_es_using_id_filtering(
            es_instance,
            index_name,
            id_rules=id_rules,
            sorter_rules=sorter_rules,
            from_no=0,
            size=self.metric_position_k
        )
        return result

    def evaluate(self, es_instance, index_name):
        evaluation = self.conf.evaluation
        searchqueries = evaluation['search_queries']
        for query in searchqueries:
            if 'keywords' in query:
                result = self.evaluate_keywords_queries(es_instance, index_name, query)
                logger.info('Evaluation Search results: {}'.format(parse_es_search_result(result)))
            elif 'ids' in query:
                result = self.evaluate_id_queries(es_instance, index_name, query)
                logger.info('Search results: {}'.format(parse_es_search_result(result)))
        return {}

    def run(self, train_dataset, test_dataset, extended_confs, label_column='label', label_value='1',
            user_id_column='user_id', item_id_column='item_id'):
        # 1. build elastic search index
        es_instance, index_name = self.train(train_dataset, extended_confs)
        # 2. evaluate result if needed
        if self.conf.evaluation:
            metric_dict = self.evaluate(es_instance, index_name)
            return metric_dict
        return {}
        