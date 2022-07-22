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
import com.dmetasoul.metaspore.recommend.datasource.MilvusSource;
import com.dmetasoul.metaspore.recommend.datasource.MongoDBSource;
import com.dmetasoul.metaspore.recommend.enums.ConditionTypeEnum;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.mongodb.BasicDBList;
import com.mongodb.BasicDBObject;
import io.milvus.grpc.SearchResults;
import io.milvus.param.MetricType;
import io.milvus.param.R;
import io.milvus.param.dml.SearchParam;
import io.milvus.response.SearchResultsWrapper;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.bson.Document;
import org.springframework.data.mongodb.core.query.BasicQuery;
import org.springframework.data.mongodb.core.query.Query;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static com.dmetasoul.metaspore.recommend.enums.ConditionTypeEnum.IN;
import static com.dmetasoul.metaspore.recommend.enums.ConditionTypeEnum.getEnumByName;

@SuppressWarnings("rawtypes")
@Slf4j
@DataServiceAnnotation
public class MilvusSourceTableTask extends SourceTableTask {

    private MilvusSource dataSource;
    private Document columnsObject;
    private Document queryObject;
    private Set<String> columns;

    @Override
    public boolean initService() {
        if (super.initService() && source.getKind().equals("milvus")) {
            dataSource = (MilvusSource) taskServiceRegister.getDataSources().get(sourceTable.getSource());
        }
        FeatureConfig.SourceTable sourceTable = taskFlowConfig.getSourceTables().get(name);
        columns = sourceTable.getColumnMap().keySet();
        columns.forEach(col -> columnsObject.put(col, 1));
        return true;
    }

    @Override
    protected DataResult processRequest(ServiceRequest request, DataContext context) {
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
        result.setMilvusData(Maps.newHashMap());
        Map<Integer, List<SearchResultsWrapper.IDScore>> milvusData = result.getMilvusData();
        for (int i = 0; i < embedding.size(); ++i) {
            List<SearchResultsWrapper.IDScore> scores = wrapper.getIDScore(i);
            milvusData.put(i, scores);
        }
        return result;
    }
}
