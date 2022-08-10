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
package com.dmetasoul.metaspore.recommend.dataservice;

import com.dmetasoul.metaspore.recommend.annotation.ServiceAnnotation;
import com.dmetasoul.metaspore.recommend.common.Utils;
import com.dmetasoul.metaspore.recommend.data.FieldData;
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.dmetasoul.metaspore.recommend.functions.ScatterFunction;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import io.milvus.client.MilvusServiceClient;
import io.milvus.grpc.SearchResults;
import io.milvus.param.ConnectParam;
import io.milvus.param.MetricType;
import io.milvus.param.R;
import io.milvus.param.dml.SearchParam;
import io.milvus.response.SearchResultsWrapper;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.springframework.util.Assert;

import java.util.*;

@Slf4j
@ServiceAnnotation("MilvusSearch")
public class MilvusSearchTask extends AlgoInferenceTask {
    public static final int DEFAULT_ALGO_LEVEL = 3;
    public static final int DEFAULT_MAX_RESERVATION = 50;
    private MilvusServiceClient milvusTemplate;
    private int maxReservation;
    private String collectionName;

    @Override
    public boolean initTask() {
        maxReservation = getOptionOrDefault("maxReservation", DEFAULT_MAX_RESERVATION);
        String host = getOptionOrDefault("host", "localhost");
        int port = getOptionOrDefault("port", 9000);
        ConnectParam connectParam = ConnectParam.newBuilder()
                .withHost(host)
                .withPort(port)
                .build();
        milvusTemplate = new MilvusServiceClient(connectParam);
        collectionName = getOptionOrDefault("collectionName", "");
        return true;
    }

    @Override
    public void close() {
        milvusTemplate.close();
    }

    @Override
    public void addFunctions() {
        addFunction("milvusIdScore", new ScatterFunction() {

            /**
             *  使用embedding请求milvus获取对应的topk个id和分数
             */
            @Override
            public Map<String, List<Object>> scatter(List<FieldData> fields, List<String> names, Map<String, Object> options) {
                Assert.isTrue(CollectionUtils.isNotEmpty(fields),
                        "input fields must not null");
                Assert.isTrue(fields.get(0).isMatch(DataTypeEnum.LIST_FLOAT),
                        "milvusSearch input[0] embedding is list float");
                Assert.isTrue(CollectionUtils.isNotEmpty(names) && names.size() == 2, "names should = {id, score}");
                List<List<Float>> embedding = fields.get(0).getValue();
                return searchIdScore(embedding, names, options);
            }
        });
        addFunction("milvusField", new ScatterFunction() {

            /**
             *  使用embedding请求milvus获取对应的topk个向量对应的fields数据
             */
            @Override
            public Map<String, List<Object>> scatter(List<FieldData> fields, List<String> names, Map<String, Object> options) {
                Assert.isTrue(CollectionUtils.isNotEmpty(fields),
                        "input fields must not null");
                Assert.isTrue(fields.get(0).isMatch(DataTypeEnum.LIST_FLOAT),
                        "milvusSearch input[0] embedding is list float");
                Assert.isTrue(CollectionUtils.isNotEmpty(names) && names.size() == 2, "names should = {id, score}");
                List<List<Float>> embedding = fields.get(0).getValue();
                return searchField(embedding, names, options);
            }
        });
    }

    protected SearchResultsWrapper requestMilvus(List<List<Float>> embedding, List<String> names, Map<String, Object> options) {
        String collection = Utils.getField(options, "collectionName", collectionName);
        int limit = Utils.getField(options, "maxReservation", maxReservation);
        String field = Utils.getField(options, "field", "embedding_vector");
        long timeOut = Utils.getField(options,"timeOut", 3000L);
        String searchParams = Utils.getField(options,"searchParams", "{\"nprobe\":128}");
        MetricType metricType = Utils.getMetricType(Utils.getField(options,"metricType", 2));
        SearchParam searchParam = SearchParam.newBuilder()
                .withCollectionName(collectionName)
                .withMetricType(metricType)
                .withOutFields(names)
                .withTopK(limit)
                .withVectors(embedding)
                .withVectorFieldName(field)
                .withExpr("")
                .withParams(searchParams)
                .withGuaranteeTimestamp(timeOut)
                .build();

        R<SearchResults> response = milvusTemplate.search(searchParam);
        Utils.handleResponseStatus(response);
        return new SearchResultsWrapper(response.getData().getResults());
    }

    protected Map<String, List<Object>> searchIdScore(List<List<Float>> embedding, List<String> names, Map<String, Object> options) {
        Assert.isTrue(CollectionUtils.isNotEmpty(names) && names.size() == 1, "names should = {id, score}");
        SearchResultsWrapper wrapper = requestMilvus(embedding, Lists.newArrayList(), options);
        List<Object> milvusIds = Lists.newArrayList();
        List<Object> milvusScores = Lists.newArrayList();
        for (int i = 0; i < embedding.size(); ++i) {
            Map<String, Double> idScores = Maps.newHashMap();
            List<Object> itemIds = Lists.newArrayList();
            List<Object> itemScores = Lists.newArrayList();
            List<SearchResultsWrapper.IDScore> iDScores = wrapper.getIDScore(i);
            iDScores.sort((o1, o2) -> Double.compare(o2.getScore(), o1.getScore()));
            iDScores.forEach(x->{
                itemIds.add(String.valueOf(x.getLongID()));
                itemScores.add(x.getScore());
            });
            milvusIds.add(itemIds);
            milvusScores.add(itemScores);
        }
        return Map.of(names.get(0), milvusIds, names.get(1), milvusScores);
    }

    @SuppressWarnings("unchecked")
    protected Map<String, List<Object>> searchField(List<List<Float>> embedding, List<String> names, Map<String, Object> options) {
        Assert.isTrue(CollectionUtils.isNotEmpty(names), "names should not empty");
        SearchResultsWrapper wrapper = requestMilvus(embedding, names, options);
        Map<String, List<Object>> res = Maps.newHashMap();
        for (int i = 0; i < embedding.size(); ++i) {
            for (String field : names) {
                List<Object> item = (List<Object>) wrapper.getFieldData(field, i);
                res.put(field, item);
            }
        }
        return res;
    }
}
