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

import java.util.List;
import java.util.Map;

@Service
public class MilvusServiceImpl implements MilvusService {

    private final MilvusServiceClient milvusClient;

    private final String collectionName;

    private final String outFields;

    private final String vectorField;

    private final String searchParams;

    private final MetricType metricType; // inner product is default

    public MilvusServiceImpl(MilvusServiceClient milvusClient) {
        // Hard code those params for now.
        this.collectionName = "baike_qa_demo";
        this.outFields = "id";
        this.vectorField = "question_emb";
        this.searchParams = "{\"nprobe\":128}";
        this.metricType = MetricType.IP;
        this.milvusClient = milvusClient;
    }

    @Override
    public Map<Integer, List<SearchResultsWrapper.IDScore>> findByEmbeddingVectors(List<List<Float>> vectors, int topK) {
        return findByEmbeddingVectors(vectors, topK, 3000l);
    }

    @Override
    public Map<Integer, List<SearchResultsWrapper.IDScore>> findByEmbeddingVectors(List<List<Float>> vectors, int topK, long timeout) {
        List<String> outFields = Collections.singletonList(this.outFields);
        SearchParam searchParam = SearchParam.newBuilder()
                .withCollectionName(this.collectionName)
                .withMetricType(this.metricType)
                .withOutFields(outFields)
                .withTopK(topK)
                .withVectors(vectors)
                .withVectorFieldName(this.vectorField)
                .withExpr("")
                .withParams(this.searchParams)
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