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

import yaml

from .node import PipelineNode
from ..jobs.item_model import item_embed_run
from ..jobs.user_model import user_embed_run
from ..jobs.common import push_mongo, push_milvus
from ..utils import start_logging

class Embedding2INode(PipelineNode):
    def __call__(self, **payload) -> dict:
        node_conf = payload['conf']        
        self.match_core(node_conf)
        return payload

    @staticmethod
    def dict2str(d, kv_sep=':', item_sep=','):
        kv_list = []
        for k, v in d.items():
            kv_list.append(f"{k}{kv_sep}{v}")
        return item_sep.join(kv_list)

    def match_core(self, conf) -> None:
        # spark conf
        spark_conf = conf['spark']['session_confs']
        app_name = spark_conf['app_name']
        spark_local = spark_conf.get('local', False)
        del spark_conf['local']
        del spark_conf['app_name']
        spark_conf_str = self.dict2str(spark_conf)
    
        # data input&output conf
        data_conf = conf['data']
        item_data = data_conf['input']['item_data']
        action_data = data_conf['input']['action_data']
        item_emb_data = data_conf['dump']['item_emb_data']
        user_emb_data = data_conf['dump']['user_emb_data']
    
        # item embed and push
        item_conf = conf['jobs']['item_embed']
        # emb
        a = item_conf['run']
        if a['status']:
            print("* item embed")
            item_embed_run(item_data, item_emb_data, 
                a['text_model_name'], a['batch_size'], a['device'], 
                a['infer_online'], a['infer_host'], a['infer_port'], a['infer_model'],
                a['write_mode'], job_name=f"{app_name}-item-embed", 
                spark_conf=spark_conf_str, spark_local=spark_local)
        # push id
        a = item_conf['push_id']
        if a['status']:
            print("* push item id")
            push_mongo(a['mongo_uri'], a['mongo_database'], a['mongo_collection'], item_emb_data, 
                a['fields'], a['index_fields'], a['write_mode'], job_name=f"{app_name}-push-item-id")
        # push emb
        a = item_conf['push_emb']
        if a['status']:
            print("* push item emb")
            push_milvus(a['milvus_host'], a['milvus_port'], a['collection_name'], item_emb_data,
                a['fields'], a['id_field'], a['emb_field'], a['collection_desc'], a['collection_shards'],
                a['write_batch'], a['write_interval'], a['index_type'], a['index_metric'],
                a['index_nlist'], job_name=f"{app_name}-push-item-emb", spark_conf=spark_conf_str)
    
        # user embed and push
        user_conf = conf['jobs']['user_embed']
        a = user_conf['run']
        if a['status']:
            print("* user embed")
            user_embed_run(action_data, item_emb_data, user_emb_data, a['scene_id'],
                a['action_type'], a['action_value_min'], a['action_value_max'],
                a['action_sortby_key'], a['action_max_len'], a['action_agg_func'],
                a['action_decay_rate'], a['write_mode'], 
                job_name=f"{app_name}-user-emb", spark_conf=spark_conf_str)
        a = user_conf['push_emb']
        if a['status']:
            print("* push embed")
            push_mongo(a['mongo_uri'], a['mongo_database'], a['mongo_collection'], user_emb_data,
                a['fields'], a['index_fields'], a['write_mode'], job_name=f"{app_name}-push-user-emb")
    
