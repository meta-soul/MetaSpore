import metaspore as ms
import subprocess
import yaml
import argparse
import sys

from pyspark.mllib.evaluation import RankingMetrics

sys.path.append("../../../")

from python.tuner.base_tuner import BaseTuner
from python.simplex.simplex_net import UserModule
from python.simplex.simplex_net import ItemModule
from python.simplex.simplex_net import SimilarityModule
from python.simplex.simplex_agent import SimpleXAgent

class SimpleXTuner(BaseTuner):
    def __init__(self, config):
        super().__init__(config)
        self._experiments_order_by = 'recall_at_20'
        self._experiments_order_reverse = True
        
    def get_estimator(self, config):
        param_dict = {**config['common_param'], **config['hyper_param'], **self._dataset}
        user_tower = UserModule(column_name_path = param_dict['user_column_name'],
                                user_combine_schema_path = param_dict['user_combine_schema'],
                                interacted_items_combine_schema_path = param_dict['interacted_items_combine_schema'],
                                emb_size = param_dict['embedding_size'],
                                alpha = param_dict['ftrl_learning_rate'],
                                beta = param_dict['ftrl_smothing_rate'],
                                l1 = param_dict['ftrl_l1_regularization'],
                                l2 = param_dict['ftrl_l2_regularization'],
                                g = param_dict['gamma'])
        item_tower = ItemModule(column_name_path = param_dict['item_column_name'],
                                combine_schema_path = param_dict['item_combine_schema'],
                                emb_size = param_dict['embedding_size'],
                                alpha = param_dict['ftrl_learning_rate'],
                                beta = param_dict['ftrl_smothing_rate'],
                                l1 = param_dict['ftrl_l1_regularization'],
                                l2 = param_dict['ftrl_l2_regularization'])
        similarity = SimilarityModule(net_dropout = param_dict['net_dropout'])

        param_dict['module'] = ms.TwoTowerRetrievalModule(user_tower, item_tower, similarity)
        param_dict['agent_class'] = SimpleXAgent
        param_dict['item_embedding_size'] = param_dict['embedding_size'] # becasue item tower use only one filed - movie_id
        
        print('Debug - param_dict: ', param_dict)
        estimator = ms.TwoTowerRetrievalEstimator(module = param_dict['module'],
                                                  item_dataset = param_dict['item'],
                                                  worker_count = param_dict['worker_count'],
                                                  server_count = param_dict['server_count'],
                                                  model_in_path = param_dict['model_in_path'],
                                                  model_out_path = param_dict['model_out_path'],
                                                  model_export_path = param_dict['model_export_path'],
                                                  model_version = param_dict['model_version'],
                                                  experiment_name = param_dict['experiment_name'],
                                                  input_label_column_index = param_dict['input_label_column_index'],
                                                  item_embedding_size = param_dict['item_embedding_size'],
                                                  item_ids_column_indices = param_dict['item_ids_column_indices'],
                                                  retrieval_item_count = param_dict['retrieval_item_count'],
                                                  metric_update_interval = param_dict['metric_update_interval'], 
                                                  training_epoches = param_dict['training_epoches'],
                                                  agent_class = param_dict['agent_class'],
                                                  _negative_sample_count = param_dict['_negative_sample_count'],
                                                  _w = param_dict['_w'],
                                                  _m = param_dict['_m'])
        estimator.updater = ms.AdamTensorUpdater(param_dict['adam_learning_rate'])
        return estimator
    
    def evaluate(self, model):
        result = model.transform(self._dataset['test'])
        prediction_label_rdd = result.rdd.map(lambda x:( \
                                            [xx.name for xx in x.rec_info] \
                                                if x.rec_info is not None else [], \
                                             [x.movie_id]))
        
        metrics = RankingMetrics(prediction_label_rdd)
        precisionAt20 = metrics.precisionAt(20)
        recallAt20 = metrics.recallAt(20)
        meanAveragePrecisionAt20 = metrics.meanAveragePrecisionAt(20)
        print('=========================================')
        print('Precision@20: ', precisionAt20)
        print('Recall@20: ', recallAt20)
        print('MAP@20: ', meanAveragePrecisionAt20)
        print('=========================================')
        return {'precision_at_20': precisionAt20, \
                'recall_at_20': recallAt20, \
                'map_at_20': meanAveragePrecisionAt20}
    
if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Tuner information')
    parser.add_argument('-conf', dest='config', type=str, help='Path of config file')
    args = parser.parse_args()
    print(args.config)
    
    config_path = args.config
    config = dict()
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    
    subprocess.run(['zip', '-r', 'demo/movielens/tuner/python.zip', 'python'], cwd='../../../')
    tuner = SimpleXTuner(config)
    tuner.run()
