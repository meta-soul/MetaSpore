//
// Copyright 2022 DMetaSoul
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package com.dmetasoul.metaspore.demo.multimodal.service;

import com.dmetasoul.metaspore.serving.ArrowTensor;
import com.dmetasoul.metaspore.serving.FeatureTable;
import com.google.protobuf.ByteString;

import java.io.IOException;
import java.util.List;
import java.util.Map;

public interface HfPreprocessorService {
    Map<String, ByteString> predictBlocking(String modelName, List<String> texts, Map<String, String> parameters) throws IOException;
    Map<String, ArrowTensor> pbToArrow(Map<String, ByteString> payload) throws IOException;
    List<List<Float>> getFloatVectorsFromArrowTensorResult(Map<String, ArrowTensor> nspResultMap, String targetKey);
    List<List<Integer>> getIntVectorsFromArrowTensorResult(Map<String, ArrowTensor> nspResultMap, String targetKey);
    Map<String, List<List<Float>>> getFloatPredictFromArrowTensorResult(Map<String, ArrowTensor> nspResultMap);
    Map<String, List<List<Integer>>> getIntPredictFromArrowTensorResult(Map<String, ArrowTensor> nspResultMap);
}
