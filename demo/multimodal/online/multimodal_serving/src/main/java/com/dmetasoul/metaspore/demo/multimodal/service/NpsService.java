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

public interface NpsService {
    Map<String, ArrowTensor> predictBlocking(String modelName, Iterable<FeatureTable> featureTables, Map<String, String> parameters) throws IOException;
    Map<String, ArrowTensor> predictBlocking(String modelName, Map<String, ByteString> payload, Map<String, String> parameters) throws IOException;
    List<List<Float>> getFloatVectorsFromNpsResult(Map<String, ArrowTensor> nspResultMap, String targetKey);
    List<Float> getScoresFromNpsResult(Map<String, ArrowTensor> nspResultMap, String targetKey, int targetIndex);
}