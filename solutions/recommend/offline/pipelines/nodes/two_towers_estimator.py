from .node import PipelineNode
import metaspore as ms
from ..utils import get_class

class TwoTowersEstimatorNode(PipelineNode):
    def __call__(self, **payload) -> dict:
        conf = payload['conf']
        UserModule = get_class(conf['user_module_class'])
        ItemModule = get_class(conf['item_module_class'])
        SimilarityModule = get_class(conf['similarity_module_class'])
        TwoTowerRetrievalModule = get_class(conf['two_tower_retrieval_module_class'])
        TwoTowerAgentModule = get_class(conf['two_tower_agent_class'])
        TwoTowerEstimatorModule = get_class(conf['two_tower_estimator_class'])

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
        module = TwoTowerRetrievalModule(user_module, item_module, similarity_module)

        estimator = TwoTowerEstimatorModule(module = module,
                                     item_dataset = payload['item_dataset'],
                                     agent_class = TwoTowerAgentModule,
                                     **conf)
        ## dnn learning rate
        estimator.updater = ms.AdamTensorUpdater(conf['adam_learning_rate'])
        ## model train
        model = estimator.fit(payload['train_dataset'])
        
        payload['test_result'] = model.transform(payload['test_dataset'])
        
        return payload