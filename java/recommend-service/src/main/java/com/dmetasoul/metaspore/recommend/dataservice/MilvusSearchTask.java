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
import com.dmetasoul.metaspore.recommend.data.IndexData;
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.dmetasoul.metaspore.recommend.functions.Function;
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
import java.util.stream.Collectors;

@Slf4j
@ServiceAnnotation("MilvusSearch")
public class MilvusSearchTask extends AlgoTransformTask {
    public static final int DEFAULT_ALGO_LEVEL = 3;
    public static final int DEFAULT_MAX_RESERVATION = 50;
    private MilvusServiceClient milvusTemplate;
    private int maxReservation;
    private String collectionName;

    @Override
    public boolean initTask() {
        maxReservation = getOptionOrDefault("maxReservation", DEFAULT_MAX_RESERVATION);
        String host = getOptionOrDefault("host", "localhost");
        int port = getOptionOrDefault("port", 19530);
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
        addFunction("milvusIdScore", (fields, result, options) -> {
            Assert.isTrue(CollectionUtils.isNotEmpty(fields),
                    "input fields must not null");
            Assert.isTrue(fields.get(0).isMatch(DataTypeEnum.LIST_FLOAT),
                    "milvusSearch input[0] embedding is list float");
            Assert.isTrue(CollectionUtils.isNotEmpty(result), "output fields must not empty");
            List<IndexData> embedding = fields.get(0).getIndexValue();
            return searchIdScore(embedding, result, options);
        });
        addFunction("milvusField", (fields, result, options) -> {
            Assert.isTrue(CollectionUtils.isNotEmpty(fields),
                    "input fields must not null");
            Assert.isTrue(fields.get(0).isMatch(DataTypeEnum.LIST_FLOAT),
                    "milvusSearch input[0] embedding is list float");
            Assert.isTrue(CollectionUtils.isNotEmpty(result), "output fields must not empty");
            List<IndexData> embedding = fields.get(0).getIndexValue();
            return searchField(embedding, result, options);
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
                .withCollectionName(collection)
                .withMetricType(metricType)
                .withOutFields(names)
                .withTopK(limit)
                .withVectors(embedding)
                .withVectorFieldName(field)
                .withExpr("")
                .withParams(searchParams)
                .build();

        R<SearchResults> response = milvusTemplate.search(searchParam);
        Utils.handleResponseStatus(response);
        return new SearchResultsWrapper(response.getData().getResults());
    }

    protected boolean searchIdScore(List<IndexData> embedding, List<FieldData> result, Map<String, Object> options) {
        Assert.isTrue(CollectionUtils.isNotEmpty(result), "output fields must not empty");
        boolean useStrId = Utils.getField(options,"useStrId", false);
        SearchResultsWrapper wrapper = requestMilvus(embedding.stream().map(IndexData::<List<Float>>getVal).collect(Collectors.toList()), List.of(), options);
        for (int i = 0; i < embedding.size(); ++i) {
            Map<String, Double> idScores = Maps.newHashMap();
            List<Object> itemIds = Lists.newArrayList();
            List<Object> itemScores = Lists.newArrayList();
            List<SearchResultsWrapper.IDScore> iDScores = wrapper.getIDScore(i);
            iDScores.sort((o1, o2) -> Double.compare(o2.getScore(), o1.getScore()));
            iDScores.forEach(x->{
                if (useStrId) {
                    itemIds.add(x.getStrID());
                } else {
                    itemIds.add(String.valueOf(x.getLongID()));
                }
                itemScores.add(x.getScore());
            });
            result.get(0).addIndexData(FieldData.create(embedding.get(i).getIndex(), itemIds));
            result.get(1).addIndexData(FieldData.create(embedding.get(i).getIndex(), itemScores));
        }
        return true;
    }

    @SuppressWarnings("unchecked")
    protected boolean searchField(List<IndexData> embedding, List<FieldData> result, Map<String, Object> options) {
        Assert.isTrue(CollectionUtils.isNotEmpty(result), "output fields must not empty");
        String scoreField = Utils.getField(options,"scoreField", "score");
        boolean useOrder = Utils.getField(options,"useOrder", true);
        boolean useFlat = Utils.getField(options,"useFlat", true);
        List<FieldData> output = Lists.newArrayList();
        FieldData score = null;
        for (FieldData field : result) {
            if (Objects.equals(field.getName(), scoreField)) {
                score = field;
            } else {
                output.add(field);
            }
        }
        SearchResultsWrapper wrapper = requestMilvus(embedding.stream().map(IndexData::<List<Float>>getVal).collect(Collectors.toList()),
                output.stream().map(FieldData::getName).collect(Collectors.toList()), options);
        List<Object> scores = null;
        for (int i = 0; i < embedding.size(); ++i) {
            List<Integer> ids = Lists.newArrayList();
            if (score != null) {
                List<SearchResultsWrapper.IDScore> iDScores = wrapper.getIDScore(i);
                List<Object> itemScores = iDScores.stream().map(SearchResultsWrapper.IDScore::getScore).collect(Collectors.toList());
                scores = itemScores;
                if (useOrder) {
                    for (int j = 0; j < itemScores.size(); ++j) {
                        ids.add(j);
                    }
                    ids.sort((o1, o2) -> {
                        Object val1 = itemScores.get(o1);
                        Object val2 = itemScores.get(o2);
                        if (Objects.equals(val1, val2)) return 0;
                        if (val1 == null) return -1;
                        if (val2 == null) return 1;
                        Assert.isInstanceOf(Comparable.class, val1, "itemScore col must be compareable");
                        Comparable<Object> c = (Comparable<Object>) val2;
                        return c.compareTo(val1);
                    });
                }
            }
            for (FieldData field : result) {
                List<Object> item;
                if (Objects.equals(field.getName(), scoreField)) {
                    item = scores;
                } else {
                    item = (List<Object>) wrapper.getFieldData(field.getName(), i);
                }
                if (CollectionUtils.isEmpty(item)) continue;
                if (CollectionUtils.isNotEmpty(ids) && useOrder) {
                    List<Object> orderList = Lists.newArrayList();
                    for (Integer id : ids) {
                        Object obj = Utils.get(item, id, null);
                        if (useFlat) {
                            field.addIndexData(FieldData.create(embedding.get(i).getIndex(), obj));
                            continue;
                        }
                        orderList.add(obj);
                    }
                    if (!useFlat) {
                        field.addIndexData(FieldData.create(embedding.get(i).getIndex(), orderList));
                    }
                } else {
                    if (!useFlat) {
                        field.addIndexData(FieldData.create(embedding.get(i).getIndex(), item));
                    } else {
                        for (Object obj : item) {
                            field.addIndexData(FieldData.create(embedding.get(i).getIndex(), obj));
                        }
                    }
                }
            }
        }
        return true;
    }
}
