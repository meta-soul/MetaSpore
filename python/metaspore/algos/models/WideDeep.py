import torch
import metaspore as ms

class DenseModule(torch.nn.Module):
        def __init__(self, emb_output_size):
            super().__init__()
            self._dense = torch.nn.Sequential(
                ms.nn.Normalization(emb_output_size),
                torch.nn.Linear(emb_output_size, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 1),
                torch.nn.Sigmoid()
            )
        
        def forward(self, wide, deep):
            sum = torch.add(wide, deep)
            return self._dense(sum)

class DemoModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._embedding_size = 16
        self._schema_dir = 's3://dmetasoul-bucket/demo/' + 'schema/'
        self._column_name_path = self._schema_dir + 'column_name_demo.txt'
        self._combine_schema_path = self._schema_dir + 'combine_schema_demo.txt'
        self._sparse0 = ms.EmbeddingSumConcat(
            self._embedding_size, self._column_name_path, self._combine_schema_path)
        self._sparse1 = ms.EmbeddingSumConcat(
            self._embedding_size, self._column_name_path, self._combine_schema_path)
        self._sparse1.updater = ms.FTRLTensorUpdater()
        self._sparse1.initializer = ms.NormalTensorInitializer(var=0.01)
        self._sparse0.updater = ms.FTRLTensorUpdater()
        self._sparse0.initializer = ms.NormalTensorInitializer(var=0.01)
        self._dense = DenseModule(self._sparse0.feature_count * self._embedding_size)

    def forward(self, x):
        x0 = self._sparse0(x)
        x1 = self._sparse1(x)
        x3 = self._dense(x0, x1)
        return x3