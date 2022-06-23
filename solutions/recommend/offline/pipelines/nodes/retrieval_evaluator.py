from .node import PipelineNode

class RetrievalEvaluatorNode(PipelineNode):
    def __call__(self, **payload) -> dict:
        conf = payload['conf']
        test_result = payload['test_result']
        user_id = conf['user_id']
        item_id = conf['item_id']
        
        from pyspark.sql import functions as F
        print('Debug -- test sample:')
        test_result.select(user_id, (F.posexplode('rec_info').alias('pos', 'rec_info'))).show(60)

        test_result[test_result[user_id]==100]\
                    .select(user_id, (F.posexplode('rec_info').alias('pos', 'rec_info'))).show(60)

        ## evaluation
        from pyspark.mllib.evaluation import RankingMetrics
        prediction_label_rdd = test_result.rdd.map(lambda x:(\
                                                [xx.name for xx in x.rec_info] if x.rec_info is not None else [], \
                                                [getattr(x, item_id)]))

        recall_metrics = RankingMetrics(prediction_label_rdd)

        print("Debug -- Precision@20: ", recall_metrics.precisionAt(20))
        print("Debug -- Recall@20: ", recall_metrics.recallAt(20))
        print("Debug -- MAP@20: ", recall_metrics.meanAveragePrecisionAt(20))
        
        return payload