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

package com.dmetasoul.metaspore.demo.multimodal.service.impl;

import com.dmetasoul.metaspore.demo.multimodal.service.MilvusService;
import com.google.common.collect.Maps;
import edu.emory.mathcs.backport.java.util.Collections;
import io.milvus.client.MilvusServiceClient;
import io.milvus.grpc.SearchResults;
import io.milvus.param.MetricType;
import io.milvus.param.R;
import io.milvus.param.dml.SearchParam;
import io.milvus.response.SearchResultsWrapper;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class MilvusServiceImpl implements MilvusService {

    private final MilvusServiceClient milvusClient;

    private Map<String, Object> milvusArgs;

    public MilvusServiceImpl(MilvusServiceClient milvusClient) {
        // Hard code those params for now.
        this.milvusArgs = new HashMap<>();
        this.milvusArgs.put("collectionName", "baike_qa_demo");
        this.milvusArgs.put("outFields", "id");
        this.milvusArgs.put("vectorField", "question_emb");
        this.milvusArgs.put("searchParams", "{\"nprobe\":128}");
        this.milvusArgs.put("metricType", MetricType.IP); // inner product is default
        this.milvusClient = milvusClient;
    }

    @Override
    public Map<String, String> getMilvusArgs() {
        List<String> names = List.of("collectionName", "outFields", "vectorField", "searchParams", "metricType");
        Map<String, String> args = new HashMap<>();
        for (String name : names) {
            args.put(name, (String) this.milvusArgs.get(name));
        }
        return args;
    }

    @Override
    public void setMilvusArgs(Map<String, String> args) {
        List<String> names = List.of("collectionName", "outFields", "vectorField", "searchParams");
        for (String name : names) {
            if (this.milvusArgs.containsKey(name) && args.containsKey(name)) {
                this.milvusArgs.put(name, args.get(name));
            }
        }
    }

    @Override
    public Map<Integer, List<SearchResultsWrapper.IDScore>> findByEmbeddingVectors(List<List<Float>> vectors, int topK) {
        return findByEmbeddingVectors(vectors, topK, 3000l);
    }

    @Override
    public Map<Integer, List<SearchResultsWrapper.IDScore>> findByEmbeddingVectors(List<List<Float>> vectors, int topK, long timeout) {
        String collectionName = (String) this.milvusArgs.get("collectionName");
        String fields = (String) this.milvusArgs.get("outFields");
        String vectorField = (String) this.milvusArgs.get("vectorField");
        String params = (String) this.milvusArgs.get("searchParams");
        MetricType metricType = (MetricType) this.milvusArgs.get("metricType");

        List<String> outFields = Collections.singletonList(fields);
        SearchParam searchParam = SearchParam.newBuilder()
                .withCollectionName(collectionName)
                .withMetricType(metricType)
                .withOutFields(outFields)
                .withTopK(topK)
                .withVectors(vectors)
                .withVectorFieldName(vectorField)
                .withExpr("")
                .withParams(params)
                .withGuaranteeTimestamp(timeout)
                .build();

        R<SearchResults> response = milvusClient.search(searchParam);
        handleResponseStatus(response);
        SearchResultsWrapper wrapper = new SearchResultsWrapper(response.getData().getResults());
        if (wrapper == null) {
            return Maps.newHashMap();
        }

        Map<Integer, List<SearchResultsWrapper.IDScore>> result = Maps.newHashMap();
        for (int i = 0; i < vectors.size(); ++i) {
            List<SearchResultsWrapper.IDScore> scores = wrapper.getIDScore(i);
            result.put(i, scores);
        }
        return result;
    }

    private void handleResponseStatus(R<?> r) {
        if (r.getStatus() != R.Status.Success.getCode()) {
            throw new RuntimeException(r.getMessage());
        }
    }
}