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

class ExamplePreprocessor(object):
    def __init__(self, config_dir):
        print(config_dir)
        self._input_names = 'input_key1', 'input_key2'
        self._output_names = 'output_key1', 'output_key2'

    @property
    def input_names(self):
        return self._input_names

    @property
    def output_names(self):
        return self._output_names

    def _do_predict(self, x, y):
        # Implement actual preprocessing logic here.
        return y, x

    def predict(self, inputs):
        input1 = inputs['input_key1']
        input2 = inputs['input_key2']
        output1, output2 = self._do_predict(input1, input2)
        outputs = {}
        outputs['output_key1'] = output1
        outputs['output_key2'] = output2
        return outputs
