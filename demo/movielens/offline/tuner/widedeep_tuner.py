import metaspore as ms
import pyspark
import subprocess
import yaml
import argparse
import sys

sys.path.append("../../../")

from python.tuner.base_tuner import BaseTuner
from python.widedeep import WideDeep

class WideDeepTuner(BaseTuner):
    def __init__(self, config):
        super().__init__(config)
        self._experiments_order_by = 'auc'
        self._experiments_order_reverse = True
        
    def get_estimator(self, config):
        param_dict = {**config['common_param'], **config['hyper_param'], **self._dataset}
        ## wide and deep module
        module = WideDeep(use_wide=False,
                    wide_embedding_dim=param_dict['embedding_size'], \
                    deep_embedding_dim=param_dict['embedding_size'], \
                    deep_hidden_units=param_dict['deep_hidden_units'], \
                    wide_column_name_path=param_dict['column_name_path'], \
                    wide_combine_schema_path=param_dict['wide_combine_schema_path'], \
                    deep_column_name_path=param_dict['column_name_path'], \
                    deep_combine_schema_path=param_dict['combine_schema_path'])
        ## pytorch estimator
        estimator = ms.PyTorchEstimator(module=module,
                                  worker_count=param_dict['worker_count'],
                                  server_count=param_dict['server_count'],
                                  model_out_path=param_dict['model_out_path'],
                                  input_label_column_index=param_dict['input_label_column_index'],
                                  metric_update_interval=param_dict['metric_update_interval'],
                                  training_epoches=param_dict['training_epoches'])
        ## dnn learning rate
        estimator.updater = ms.AdamTensorUpdater(param_dict['adam_learning_rate'])
        return estimator
    
    def evaluate(self, model):
        test_result = model.transform(self._dataset['test'])
        evaluator = pyspark.ml.evaluation.BinaryClassificationEvaluator()
        auc = evaluator.evaluate(test_result)

        print('=========================================')
        print('AUC: ', auc)
        print('=========================================')
        return {'auc': auc}
    
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
    tuner = WideDeepTuner(config)
    tuner.run()
