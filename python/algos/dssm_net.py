import torch
import metaspore as ma
import torch.nn.functional as F

class DSSMDense(torch.nn.Module):
    def __init__(self, emb_out_size, dense_structure):
        super().__init__()
        self._emb_bn = ma.nn.Normalization(emb_out_size, momentum=0.01, eps=1e-5)
        self._d1 = torch.nn.Linear(emb_out_size, dense_structure[0])
        self._d2 = torch.nn.Linear(dense_structure[0], dense_structure[1])
        self._d3 = torch.nn.Linear(dense_structure[1], dense_structure[2])
        
    def forward(self, x):
        x = self._emb_bn(x)
        x = F.relu(self._d1(x))
        x = F.relu(self._d2(x))
        x = self._d3(x)
        return x

class SimilarityModule(torch.nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau
    
    def forward(self, x, y):
        z = torch.sum(x * y, dim=1).reshape(-1, 1)
        s = torch.sigmoid(z/self.tau)
        return s

class UserModule(torch.nn.Module):
    def __init__(self, column_name_path, combine_schema_path, emb_size, alpha, beta, l1, l2, dense_structure):
        super().__init__()
        self._embedding_size = emb_size
        self._column_name_path = column_name_path
        self._combine_schema_path = combine_schema_path
        self._sparse = ma.EmbeddingSumConcat(self._embedding_size, self._column_name_path, self._combine_schema_path)
        self._sparse.updater = ma.FTRLTensorUpdater(alpha = 0.01)
        self._sparse.initializer = ma.NormalTensorInitializer(var = 0.0001)
        self._sparse.output_batchsize1_if_only_level0 = True
        self._emb_out_size = self._sparse.feature_count * self._embedding_size
        self._dense = DSSMDense(self._emb_out_size, dense_structure)

    def forward(self, x):
        x = self._sparse(x)
        x = self._dense(x)
        return x

class ItemModule(torch.nn.Module):
    def __init__(self, column_name_path, combine_schema_path, emb_size, alpha, beta, l1, l2, dense_structure):
        super().__init__()
        self._embedding_size = emb_size
        self._column_name_path = column_name_path
        self._combine_schema_path = combine_schema_path
        self._sparse = ma.EmbeddingSumConcat(self._embedding_size, self._column_name_path, self._combine_schema_path)
        self._sparse.updater = ma.FTRLTensorUpdater(alpha = 0.01)
        self._sparse.initializer = ma.NormalTensorInitializer(var = 0.0001)
        self._emb_out_size = self._sparse.feature_count * self._embedding_size
        self._dense = DSSMDense(self._emb_out_size, dense_structure)

    def forward(self, x):
        x = self._sparse(x)
        x = self._dense(x)
        return x
