import metaspore as ms
import attrs
import cattrs
import logging

from pyspark.sql import DataFrame
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import functions as F
from typing import Dict

from ..utils import get_class, remove_none_value
from ..constants import ESITMATOR_CONFIG_CLASS, MODEL_CONFIG_CLASS

logger = logging.getLogger(__name__)

@attrs.frozen
class TwoTowersRetrievalConfig:
    user_module_class: object
    item_module_class: object
    similarity_module_class: object
    two_tower_retrieval_module_class: object
    two_tower_agent_class: object
    two_tower_estimator_class: object
    model_params: object
    estimator_params: object
        
class TwoTowersRetrievalModule():
    def __init__(self, conf: TwoTowersRetrievalConfig):
        if isinstance(conf, dict):
            self.conf = TwoTowersRetrievalModule.convert(conf)
        elif isinstance(conf, TwoTowersRetrievalConfig):
            self.conf = conf
        else:
            raise TypeError("Type of 'conf' must be dict or TwoTowersRetrievalConfig. Current type is {}".format(type(conf)))
        self.model = None
        self.metric_position_k = 20
    
    @staticmethod
    def convert(conf: dict) -> TwoTowersRetrievalConfig:
        if not 'user_module_class' in conf:
            raise ValueError("Dict of TwoTowersRetrievalModule must have key 'user_module_class' !")
        if not 'item_module_class' in conf:
            raise ValueError("Dict of TwoTowersRetrievalModule must have key 'item_module_class' !")
        if not 'similarity_module_class' in conf:
            raise ValueError("Dict of TwoTowersRetrievalModule must have key 'similarity_module_class' !")
        if not 'two_tower_retrieval_module_class' in conf:
            raise ValueError("Dict of TwoTowersRetrievalModule must have key 'two_tower_retrieval_module_class' !")
        if not 'two_tower_agent_class' in conf:
            raise ValueError("Dict of TwoTowersRetrievalModule must have key 'two_tower_agent_class' !")
        if not 'two_tower_estimator_class' in conf:
            raise ValueError("Dict of TwoTowersRetrievalModule must have key 'two_tower_estimator_class' !")
        if not 'model_params' in conf:
            raise ValueError("Dict of TwoTowersRetrievalModule must have key 'model_params' !")
        if not 'estimator_params' in conf:
            raise ValueError("Dict of TwoTowersRetrievalModule must have key 'estimator_params' !")

        user_module_class = get_class(conf['user_module_class'])
        item_module_class = get_class(conf['item_module_class'])
        similarity_module_class = get_class(conf['similarity_module_class'])
        two_tower_retrieval_module_class = get_class(conf['two_tower_retrieval_module_class'])
        two_tower_agent_class = get_class(conf['two_tower_agent_class'])
        two_tower_estimator_class = get_class(conf['two_tower_estimator_class'])

        estimator_config_class = get_class(ESITMATOR_CONFIG_CLASS)(user_module_class)
        model_config_class = get_class(MODEL_CONFIG_CLASS)(user_module_class)
        model_params = cattrs.structure(conf['model_params'], model_config_class)
        estimator_params = cattrs.structure(conf['estimator_params'], estimator_config_class)
        
        return TwoTowersRetrievalConfig(
            user_module_class,
            item_module_class,
            similarity_module_class,
            two_tower_retrieval_module_class,
            two_tower_agent_class,
            two_tower_estimator_class,
            model_params,
            estimator_params
        )

    def _init_net_with_params(self, module_type, module_class, model_params):
        if module_type in ['user']:
            return  module_class(column_name_path = model_params['user_column_name'], \
                                 combine_schema_path = model_params['user_combine_schema'], \
                                 embedding_dim = model_params['vector_embedding_size'], \
                                 sparse_init_var = model_params['sparse_init_var'], \
                                 ftrl_l1 = model_params['ftrl_l1'], \
                                 ftrl_l2 = model_params['ftrl_l2'], \
                                 ftrl_alpha = model_params['ftrl_alpha'], \
                                 ftrl_beta = model_params['ftrl_beta'], \
                                 dnn_hidden_units = model_params['dnn_hidden_units'], \
                                 dnn_hidden_activations = model_params['dnn_hidden_activations'])
        elif module_type in ['item']:
            return  module_class(column_name_path = model_params['item_column_name'], \
                                 combine_schema_path = model_params['item_combine_schema'], \
                                 embedding_dim = model_params['vector_embedding_size'], \
                                 sparse_init_var = model_params['sparse_init_var'], \
                                 ftrl_l1 = model_params['ftrl_l1'], \
                                 ftrl_l2 = model_params['ftrl_l2'], \
                                 ftrl_alpha = model_params['ftrl_alpha'], \
                                 ftrl_beta = model_params['ftrl_beta'], \
                                 dnn_hidden_units = model_params['dnn_hidden_units'], \
                                 dnn_hidden_activations = model_params['dnn_hidden_activations'])
        elif module_type in ['sim']:
            return module_class(model_params['tau'])
        else:
            return None
    
    def train(self, train_dataset, item_dataset, worker_count, server_count):
        # init user module, item module, similarity module
        model_params_dict = remove_none_value(cattrs.unstructure(self.conf.model_params))
        estimator_params_dict = remove_none_value(cattrs.unstructure(self.conf.estimator_params))
        user_module = self._init_net_with_params(
            'user',
            self.conf.user_module_class, 
            model_params_dict
        )
        item_module = self._init_net_with_params(
            'item', 
            self.conf.item_module_class, 
            model_params_dict
        )
        similarity_module = self._init_net_with_params(
            'sim', 
            self.conf.similarity_module_class, 
            model_params_dict
        )

        # init two tower module
        two_tower_retrieval_module = self.conf.two_tower_retrieval_module_class(
            user_module, 
            item_module, 
            similarity_module
        )

        ## init estimator
        two_tower_estimator = self.conf.two_tower_estimator_class(
            module = two_tower_retrieval_module,
            item_dataset = item_dataset,
            agent_class = self.conf.two_tower_agent_class,
            worker_count = worker_count,
            server_count = server_count,
            **{**model_params_dict, **estimator_params_dict}
        )

        # model train
        two_tower_estimator.updater = ms.AdamTensorUpdater(model_params_dict['adam_learning_rate'])
        self.model = two_tower_estimator.fit(train_dataset)
        
        logger.info('DeepCTR - training: done')
    
    def evaluate(self, test_result, user_id_column='user_id', item_id_column='item_id'):        
        prediction_label_rdd = test_result.rdd.map(lambda x:(\
                                                [xx.name for xx in x.rec_info] if x.rec_info is not None else [], \
                                                [getattr(x, item_id_column)]))

        metrics = RankingMetrics(prediction_label_rdd)
        metric_dict = {}
        metric_dict['Precision@{}'.format(self.metric_position_k)] = metrics.precisionAt(self.metric_position_k)
        metric_dict['Recall@{}'.format(self.metric_position_k)] = metrics.recallAt(self.metric_position_k)
        metric_dict['MAP@{}'.format(self.metric_position_k)] = metrics.meanAveragePrecisionAt(self.metric_position_k)
        metric_dict['NDCG@{}'.format(self.metric_position_k)] = metrics.ndcgAt(self.metric_position_k)
        logger.info('Metrics: {}'.format(metric_dict))
        logger.info('TwoTwoers - evaluation: done')
        return metric_dict
        
    
    def run(self, train_dataset, test_dataset, item_dataset, worker_count, server_count,
            user_id_column='user_id', item_id_column='item_id', label_column='label') -> Dict[str, float]:
        if not isinstance(train_dataset, DataFrame):
            raise ValueError("Type of train_dataset must be DataFrame.")
        if not isinstance(test_dataset, DataFrame):
            raise ValueError("Type of test_dataset must be DataFrame.")
        
        # 1. create estimator and fit model
        self.train(train_dataset, item_dataset, worker_count, server_count)
        
        # 2. transform test data using self.model
        test_result = self.model.transform(test_dataset)
        logger.info('TwoTowers - inference: done')
        
        # 3. get metric dictionary (metric name -> metric value)
        metric_dict = self.evaluate(test_result, item_id_column=item_id_column)
        logger.info('TwoTowers evaluation metrics: {}'.format(metric_dict))
        
        return metric_dict

