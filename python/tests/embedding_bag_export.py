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

import torch
import metaspore as ms

class EmbeddingBagModule(torch.nn.Module):
    def __init__(self, feature_count, emb_size, mode='sum'):
        super(EmbeddingBagModule, self).__init__()
        self.second_dim = feature_count * emb_size
        self.emb_size = emb_size
        self.feature_count = feature_count
        self.embedding_bag_mode=mode

    def forward(self, input, weight, offsets, batch_size):
        embs = torch.nn.functional.embedding_bag(input, weight, offsets, mode=self.embedding_bag_mode)
        return embs.reshape(batch_size, self.second_dim)

    def export_onnx(self, path):
        self.eval()
        torch_script = torch.jit.script(self)
        input = torch.tensor([i for i in range(self.feature_count)], dtype=torch.long)
        weight = torch.zeros(self.feature_count, self.emb_size)
        offsets = torch.tensor([i for i in range(self.feature_count)], dtype=torch.long)
        batch_size = torch.tensor(1, dtype=torch.long)
        # class FakeStream(object):
        #     def write(self, data):
        #         ms._metaspore.stream_write_all(ms.url_utils.use_s3(path), data)
        #     def flush(self):
        #         pass
        # ms._metaspore.ensure_local_directory(ms.url_utils.use_s3(path))
        # fout = FakeStream()
        torch.onnx.export(torch_script,
                        (input, weight, offsets, batch_size),
                        path,
                        input_names=["input", "weight", "offsets", "batch_size"],
                        output_names=["_sparse0_1"],
                        dynamic_axes={
                            "input": {0: "input_num"},
                            "weight": {0: "emb_vec_num"},
                            "offsets": {0: "offset_num"}
                        },
                        verbose=True,
                        opset_version=14)

model = EmbeddingBagModule(5, 16)
model.export_onnx("embedding_bag.onnx")

# test with dynamically shaped input, weight, offsets and dynamic batch size
# import onnxruntime as ort
# import numpy as np
# test_weight = np.ones((5, 8)).astype('f')
# test_input = np.array([0, 1, 2, 3, 4]).astype('l')
# test_offsets = np.array([0, 1, 2, 3]).astype('l')
# batch_size = np.array([2]).astype('l')
# ort_sess = ort.InferenceSession('model.onnx')
# outputs = ort_sess.run(['output'], {'input': test_input, 'weight': test_weight, 'offsets': test_offsets, 'batch_size': batch_size})

# # Print Result 
# print(f'Predicted: "{outputs}"')