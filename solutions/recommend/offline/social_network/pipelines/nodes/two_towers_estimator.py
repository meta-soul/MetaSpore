from .node import PipelineNode
import metaspore as ms

class TwoTowersEstimatorNode(PipelineNode):
    def __call__(self, **payload) -> dict:
        conf = payload['conf']
        
        from python.dssm_net import UserModule, ItemModule, SimilarityModule
        from python.training import TwoTowerBatchNegativeSamplingAgent, TwoTowerBatchNegativeSamplingModule
        
        ## init user module, item module, similarity module
        user_module = UserModule(conf['user_column_name'], \
                                 conf['user_combine_schema'], \
                                 emb_size = conf['vector_embedding_size'], \
                                 alpha = conf['ftrl_learning_rate'], \
                                 beta = conf['ftrl_smothing_rate'], \
                                 l1 = conf['ftrl_l1_regularization'], \
                                 l2 = conf['ftrl_l2_regularization'], \
                                 dense_structure = conf['dense_structure'])
        item_module = ItemModule(conf['item_column_name'], \
                                 conf['item_combine_schema'], \
                                 emb_size = conf['vector_embedding_size'], \
                                 alpha = conf['ftrl_learning_rate'], \
                                 beta = conf['ftrl_smothing_rate'], \
                                 l1 = conf['ftrl_l1_regularization'], \
                                 l2 = conf['ftrl_l2_regularization'], \
                                 dense_structure = conf['dense_structure'])
        similarity_module = SimilarityModule(conf['tau'])
        module = TwoTowerBatchNegativeSamplingModule(user_module, item_module, similarity_module)

        import importlib
        module_lib = importlib.import_module(conf['two_tower_module'])
        ## init estimator class
        estimator_class_ = getattr(module_lib, conf['two_tower_estimator_class'])
        estimator = estimator_class_(module = module,
                                     item_dataset = payload['item_dataset'],
                                     item_ids_column_indices = [2],
                                     retrieval_item_count = 20,
                                     metric_update_interval = 500,
                                     agent_class = TwoTowerBatchNegativeSamplingAgent,
                                     **conf)
        ## dnn learning rate
        estimator.updater = ms.AdamTensorUpdater(conf['adam_learning_rate'])
        ## model train
        model = estimator.fit(payload['train_dataset'])
        
        payload['test_result'] = model.transform(payload['test_dataset'])
        
        return payload