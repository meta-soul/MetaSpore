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
package com.dmetasoul.metaspore.recommend.datasource;

import com.dmetasoul.metaspore.recommend.annotation.DataSourceAnnotation;
import com.dmetasoul.metaspore.recommend.common.Utils;
import com.dmetasoul.metaspore.recommend.configure.FeatureConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import io.milvus.client.MilvusServiceClient;
import io.milvus.grpc.SearchResults;
import io.milvus.param.ConnectParam;
import io.milvus.param.MetricType;
import io.milvus.param.R;
import io.milvus.param.dml.SearchParam;
import io.milvus.response.SearchResultsWrapper;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;

import java.util.List;
import java.util.Map;

@SuppressWarnings("rawtypes")
@Data
@Slf4j
@DataSourceAnnotation("milvus")
public class MilvusSource extends DataSource {
    private MilvusServiceClient milvusTemplate;

    @Override
    public boolean initService() {
        FeatureConfig.Source source = taskFlowConfig.getSources().get(name);
        if (!source.getKind().equals("milvus")) {
            log.error("config milvus fail! is not kind:{} eq milvus!", source.getKind());
            return false;
        }
        String host = (String) source.getOptions().getOrDefault("host", "localhost");
        int port = (Integer) source.getOptions().getOrDefault("port", 9000);
        ConnectParam connectParam = ConnectParam.newBuilder()
                .withHost(host)
                .withPort(port)
                .build();
        milvusTemplate = new MilvusServiceClient(connectParam);
        return true;
    }

    @Override
    public void close() {
        milvusTemplate.close();
    }

    @Override
    public boolean checkRequest(ServiceRequest request, DataContext context) {
        List<List<Float>> embedding = request.get("embedding", Lists.newArrayList());
        if (CollectionUtils.isEmpty(embedding)) {
            log.error("milvus request embedding must not be empty!");
            return false;
        }
        String field = request.get("field", "");
        String collectionName = request.get("collectionName", "");
        if (StringUtils.isEmpty(collectionName) || StringUtils.isEmpty(field)) {
            log.error("milvus request collectionName and field must not be empty!");
            return false;
        }
        return true;
    }

    private MetricType getMetricType(int index) {
        if (index < 0 || index >= MetricType.values().length) {
            index = 0;
        }
        return MetricType.values()[index];
    }

    @Override
    protected DataResult process(ServiceRequest request, DataContext context) {
        DataResult result = new DataResult();
        String parent = request.get("parent", "");
        FeatureConfig.SourceTable sourceTable = taskFlowConfig.getSourceTables().get(parent);
        List<List<Float>> embedding = (List<List<Float>>) request.get("embedding");
        String collectionName = request.get("collectionName");
        int limit = request.get("limit", 10);
        long timeOut = 30000L;
        String searchParams = "{\"nprobe\":128}";
        MetricType metricType = MetricType.IP;
        Map<String, Object> options = sourceTable.getOptions();
        String field = request.get("field", "embedding_vector");
        if (MapUtils.isNotEmpty(options)) {
            timeOut = (long) options.getOrDefault("timeOut", 3000L);
            searchParams = (String) options.getOrDefault("searchParams", "{\"nprobe\":128}");
            metricType = getMetricType((int) options.getOrDefault("metricType", 2));
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

        R<SearchResults> response = milvusTemplate.search(searchParam);
        Utils.handleResponseStatus(response);
        SearchResultsWrapper wrapper = new SearchResultsWrapper(response.getData().getResults());
        result.setMilvusData(Maps.newHashMap());
        Map<Integer, List<SearchResultsWrapper.IDScore>> milvusData = result.getMilvusData();
        for (int i = 0; i < embedding.size(); ++i) {
            List<SearchResultsWrapper.IDScore> scores = wrapper.getIDScore(i);
            milvusData.put(i, scores);
        }
        return result;
    }
}
