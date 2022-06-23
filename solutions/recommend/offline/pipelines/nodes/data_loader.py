from .node import PipelineNode

class DataLoaderNode(PipelineNode):
    def __call__(self, **payload) -> dict:
        conf = payload['conf']
        spark = payload['spark']
        
        payload['train_dataset'] = spark.read.parquet(conf['train_path'])
        payload['test_dataset'] = spark.read.parquet(conf['test_path'])
        payload['item_dataset']  = spark.read.parquet(conf['item_path'])

        return payload