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
import com.dmetasoul.metaspore.recommend.common.CommonUtils;
import com.dmetasoul.metaspore.recommend.common.Utils;
import com.dmetasoul.metaspore.recommend.configure.FieldInfo;
import com.dmetasoul.metaspore.recommend.data.TableData;
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
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

    @SuppressWarnings("unchecked")
    @Override
    public void addFunctions() {
        addFunction("milvusIdScore", (fieldTableData, config, taskPool) -> {
            Map<String, Object> options = config.getOptions();
            Assert.isTrue(CollectionUtils.isNotEmpty(config.getInputFields()),
                    "input fields must not null");
            Assert.isTrue(DataTypeEnum.LIST_FLOAT.equals(fieldTableData.getType(config.getInputFields().get(0))),
                    "milvusSearch input[0] embedding is list float");
            List<List<Float>> embedding = Lists.newArrayList();
            List<Object> result = fieldTableData.getValueList(config.getInputFields().get(0));
            if (CollectionUtils.isNotEmpty(result)) {
                for (Object val : result) {
                    embedding.add((List<Float>) val);
                }
            }
            return searchIdScore(embedding, fieldTableData, config.getNames(), config.getTypes(), options);
        });
        addFunction("milvusField", (fieldTableData, config, taskPool) -> {
            Map<String, Object> options = config.getOptions();
            Assert.isTrue(CollectionUtils.isNotEmpty(config.getInputFields()),
                    "input fields must not null");
            Assert.isTrue(DataTypeEnum.LIST_FLOAT.equals(fieldTableData.getType(config.getInputFields().get(0))),
                    "milvusSearch input[0] embedding is list float");
            List<List<Float>> embedding = Lists.newArrayList();
            List<Object> result = fieldTableData.getValueList(config.getInputFields().get(0));
            if (CollectionUtils.isNotEmpty(result)) {
                for (Object val : result) {
                    embedding.add((List<Float>) val);
                }
            }
            return searchField(embedding, fieldTableData, config.getNames(), config.getTypes(), options);
        });
    }

    protected SearchResultsWrapper requestMilvus(List<List<Float>> embedding, List<String> names, Map<String, Object> options) {
        String collection = CommonUtils.getField(options, "collectionName", collectionName);
        int limit = CommonUtils.getField(options, "maxReservation", maxReservation);
        String field = CommonUtils.getField(options, "vectorField", "embedding_vector");
        long timeOut = CommonUtils.getField(options,"timeOut", 3000L);
        String searchParams = CommonUtils.getField(options,"searchParams", "{\"nprobe\":128}");
        MetricType metricType = Utils.getMetricType(CommonUtils.getField(options,"metricType", 2));
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

    protected boolean searchIdScore(List<List<Float>> embedding, TableData fieldTableData,
                                    List<String> names, List<Object> types, Map<String, Object> options) {
        boolean useStrId = CommonUtils.getField(options,"useStrId", false);
        SearchResultsWrapper wrapper = requestMilvus(embedding, List.of(), options);
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
            fieldTableData.setValue(i, names.get(i), types.get(i), itemIds);
            fieldTableData.setValue(i, names.get(i), types.get(i), itemScores);
        }
        return true;
    }

    @SuppressWarnings("unchecked")
    protected boolean searchField(List<List<Float>> embedding, TableData fieldTableData,
                                  List<String> names, List<Object> types, Map<String, Object> options) {
        String scoreField = CommonUtils.getField(options,"scoreField", "score");
        String idField = CommonUtils.getField(options,"idField", "");
        boolean useStrId = CommonUtils.getField(options,"useStrId", false);
        boolean useOrder = CommonUtils.getField(options,"useOrder", true);
        boolean useFlat = CommonUtils.getField(options,"useFlat", true);
        List<String> fields = Lists.newArrayList();
        boolean useScore = false;
        boolean useId = false;
        for (String field : names) {
            if (Objects.equals(field, scoreField)) {
                useScore = true;
            } else if (Objects.equals(field, idField)) {
                useId = true;
            } else {
                fields.add(field);
            }
        }
        SearchResultsWrapper wrapper = requestMilvus(embedding, fields, options);
        List<Object> scores = null;
        List<Object> idlist = null;
        List<List<List<Object>>> res = Lists.newArrayList();
        for (int i = 0; i < embedding.size(); ++i) {
            List<Integer> ids = Lists.newArrayList();
            if (useScore) {
                List<SearchResultsWrapper.IDScore> iDScores = wrapper.getIDScore(i);
                List<Object> itemScores = iDScores.stream().map(SearchResultsWrapper.IDScore::getScore).collect(Collectors.toList());
                scores = itemScores;
                if (useId) {
                    idlist = iDScores.stream().map(x->{
                        if (useStrId) {
                            return x.getStrID();
                        } else {
                            return x.getLongID();
                        }
                    }).collect(Collectors.toList());
                }
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
            List<List<Object>> data = Lists.newArrayList();
            for (String field : names) {
                List<Object> item;
                if (Objects.equals(field, scoreField)) {
                    item = scores;
                } else if (Objects.equals(field, idField)) {
                    item = idlist;
                } else {
                    item = (List<Object>) wrapper.getFieldData(field, i);
                }
                if (CollectionUtils.isNotEmpty(ids) && useOrder) {
                    List<Object> orderList = Lists.newArrayList();
                    for (Integer id : ids) {
                        Object obj = CommonUtils.get(item, id, null);
                        orderList.add(obj);
                    }
                    data.add(orderList);
                } else {
                    data.add(item);
                }
            }
            res.add(data);
        }
        List<FieldInfo> fieldInfos = names.stream().map(FieldInfo::new).collect(Collectors.toList());
        if (useFlat) {
            fieldTableData.flatListValue(res, fieldInfos, types);
        } else {
            for (int i = 0; i < res.size(); ++i) {
                for (int k = 0; k < fieldInfos.size(); ++k) {
                    List<List<Object>> valueList = res.get(i);
                    if (valueList.size() <= k) {
                        continue;
                    }
                    fieldTableData.setValue(i, fieldInfos.get(k), types.get(k), valueList.get(k));
                }
            }
        }
        return true;
    }
}
