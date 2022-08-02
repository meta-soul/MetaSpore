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

import com.dmetasoul.metaspore.recommend.annotation.DataServiceAnnotation;
import com.dmetasoul.metaspore.recommend.common.Utils;
import com.dmetasoul.metaspore.recommend.configure.FeatureConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.google.common.collect.Maps;
import io.milvus.client.MilvusServiceClient;
import io.milvus.grpc.SearchResults;
import io.milvus.param.ConnectParam;
import io.milvus.param.MetricType;
import io.milvus.param.R;
import io.milvus.param.dml.SearchParam;
import io.milvus.response.SearchResultsWrapper;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;

import java.security.InvalidParameterException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@SuppressWarnings("rawtypes")
@Slf4j
@DataServiceAnnotation("MilvusSearch")
public class MilvusSearchTask extends AlgoTransformTask {
    public static final int DEFAULT_ALGO_LEVEL = 3;
    public static final int DEFAULT_MAX_RESERVATION = 50;
    private MilvusServiceClient milvusTemplate;
    private int algoLevel;
    private int maxReservation;
    private String milvusTask;
    private String algoName;
    private String milvusItemTask;
    private String milvusCollectionName;
    private String embeddingTask;

    private String host;
    private int port;

    @Override
    public boolean initTask() {
        algoLevel = getOptionOrDefault("algoLevel", DEFAULT_ALGO_LEVEL);
        maxReservation = getOptionOrDefault("maxReservation", DEFAULT_MAX_RESERVATION);
        algoName = getOptionOrDefault("algo-name", "itemCF");
        milvusCollectionName = getOptionOrDefault("milvusCollectionName", "");
        host = getOptionOrDefault("host", "localhost");
        port = getOptionOrDefault("port", 9000);
        ConnectParam connectParam = ConnectParam.newBuilder()
                .withHost(host)
                .withPort(port)
                .build();
        milvusTemplate = new MilvusServiceClient(connectParam);
        return !StringUtils.isEmpty(milvusItemTask) && !StringUtils.isEmpty(milvusCollectionName);
    }

    @Override
    public void close() {
        milvusTemplate.close();
    }

    @Override
    public ServiceRequest makeRequest(String depend, ServiceRequest request, DataContext context) {
        ServiceRequest req = super.makeRequest(depend, request, context);
        if (depend.equals(milvusTask)) {
            DataResult taskResult = getDataResultByName(embeddingTask, context);
            if (taskResult.getPredictResult() == null || taskResult.getPredictResult().getEmbedding() == null) {
                throw new RuntimeException("embedding gen fail at MilvusSearchTask!");
            }
            req.put("embedding", taskResult.getPredictResult().getEmbedding());
            req.put("collectionName", milvusCollectionName);
            req.setLimit(maxReservation);
        }
        if (depend.equals(milvusItemTask)) {
            DataResult taskResult = getDataResultByName(milvusTask, context);
            Map<Integer, List<SearchResultsWrapper.IDScore>> itemVectors = taskResult.getMilvusData();
            if (itemVectors.size() != 1) {
                // TODO error log
                throw new InvalidParameterException("Item vectors size must be 1. But got: " + itemVectors.size());
            }
            List<SearchResultsWrapper.IDScore> iDScores = itemVectors.values().iterator().next();
            // TODO if Milvus return list is sorted, we do not need to use sort it again.
            iDScores.sort((o1, o2) -> Double.compare(o2.getScore(), o1.getScore()));
            List<String> milvusIds = iDScores.stream().map(SearchResultsWrapper.IDScore::getLongID).map(String::valueOf).collect(Collectors.toList());
            req.put(milvusIdCol, milvusIds);
        }
        return req;
    }
    protected List<Map<String, Object>> processRequest(ServiceRequest request, DataContext context) {
        DataResult result = new DataResult();
        List<List<Float>> embedding = (List<List<Float>>) request.get("embedding");
        String collectionName = request.get("collectionName");
        int limit = request.getLimit();
        long timeOut = 30000L;
        String searchParams = "{\"nprobe\":128}";
        MetricType metricType = MetricType.IP;
        Map<String, Object> options = sourceTable.getOptions();
        String field = request.get("field", "embedding_vector");
        if (MapUtils.isNotEmpty(options)) {
            timeOut = (long) options.getOrDefault("timeOut", 3000L);
            searchParams = (String) options.getOrDefault("searchParams", "{\"nprobe\":128}");
            metricType = Utils.getMetricType((int) options.getOrDefault("metricType", 2));
        }
        SearchParam searchParam = SearchParam.newBuilder()
                .withCollectionName(collectionName)
                .withMetricType(metricType)
                .withOutFields(sourceTable.getColumnNames())
                .withTopK(limit)
                .withVectors(embedding)
                .withVectorFieldName(field)
                .withExpr("")
                .withParams(searchParams)
                .withGuaranteeTimestamp(timeOut)
                .build();

        R<SearchResults> response = dataSource.getMilvusTemplate().search(searchParam);
        Utils.handleResponseStatus(response);
        SearchResultsWrapper wrapper = new SearchResultsWrapper(response.getData().getResults());
        Map<Integer, List<SearchResultsWrapper.IDScore>> milvusData = Maps.newHashMap();
        for (int i = 0; i < embedding.size(); ++i) {
            List<SearchResultsWrapper.IDScore> scores = wrapper.getIDScore(i);
            milvusData.put(i, scores);
        }
        return result;
    }
        @Override
    public DataResult process(ServiceRequest request, DataContext context) {
            DataResult taskResult = getDataResultByName(milvusTask, context);
            Map<Integer, List<SearchResultsWrapper.IDScore>> itemVectors = taskResult.getMilvusData();
            List<SearchResultsWrapper.IDScore> iDScores = itemVectors.values().iterator().next();
            DataResult milvusItemResult = getDataResultByName(milvusItemTask, context);
            List<Map> milvusItemIds = milvusItemResult.getData();
            HashMap<String, String> milvusIdToItemIdMap = new HashMap<>();
            milvusItemIds.forEach(x -> milvusIdToItemIdMap.put(Utils.getField(x, milvusIdCol, ""), Utils.getField(x, milvusItemIdCol, "")));

            List<Map> itemModels = new ArrayList<>();
            Double maxScore = iDScores.size() > 0 ? iDScores.get(0).getScore() : 0.0;
            iDScores.forEach(x -> {
                Map<String, Object> item = Maps.newHashMap();
                if (Utils.setFieldFail(item, config.getColumnNames(), 0, milvusIdToItemIdMap.get(String.valueOf(x.getLongID())))) {
                    return;
                }
                Double score = (double) x.getScore();
                if (Utils.setFieldFail(item, config.getColumnNames(), 1, Utils.getFinalRetrievalScore(score, maxScore, algoLevel))) {
                    return;
                }
                Map<String, Object> value = Maps.newHashMap();
                value.put(algoName, score);
                if (Utils.setFieldFail(item, config.getColumnNames(), 2, value)) {
                    return;
                }
                itemModels.add(item);
            });
            DataResult result = new DataResult();
            result.setData(itemModels);
            return result;
    }
}
