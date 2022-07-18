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
import com.dmetasoul.metaspore.recommend.configure.FeatureConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.SimpleMongoClientDatabaseFactory;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;
import org.springframework.util.Assert;
import org.apache.commons.collections4.CollectionUtils;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;

@Slf4j
@DataSourceAnnotation("mongodb")
public class MongoDBSource extends DataSource {
    private MongoTemplate mongoTemplate;
    private SimpleMongoClientDatabaseFactory simpleMongoClientDatabaseFactory;

    @Override
    public boolean initService() {
        FeatureConfig.Source source = taskFlowConfig.getSources().get(name);
        if (!source.getKind().equals("mongodb")) {
            log.error("config mongodb fail! is not kind:{} eq mongodb!", source.getKind());
            return false;
        }
        String uri = (String) source.getOptions().get("uri");
        simpleMongoClientDatabaseFactory = new SimpleMongoClientDatabaseFactory(uri);
        mongoTemplate = new MongoTemplate(simpleMongoClientDatabaseFactory);
        return true;
    }

    @Override
    public void close() {
        if (simpleMongoClientDatabaseFactory != null) {
            try {
                simpleMongoClientDatabaseFactory.destroy();
            } catch (Exception ex) {
                log.error("mongodb simpleMongoClientDatabaseFactory destroy fail! {}", ex.getMessage());
            }
        }
    }

    @Override
    public boolean checkRequest(ServiceRequest request, DataContext context) {
        if (MapUtils.isEmpty(request.getEqConditions()) && MapUtils.isEmpty(request.getInConditions())) {
            log.error("mongodb create query need condition!");
            return false;
        }
        return true;
    }

    private void setConditions(ServiceRequest request, FeatureConfig.SourceTable sourceTable, Criteria criteria) {
        if (MapUtils.isNotEmpty(request.getEqConditions())) {
            request.getEqConditions().forEach((field, queryId) -> {
                if (!sourceTable.getColumnMap().containsKey(field)) {
                    throw new RuntimeException("mongodb request Condition field is not in columns!");
                }
                criteria.and(field).is(queryId);
            });
        }
        if (MapUtils.isNotEmpty(request.getInConditions())) {
            request.getInConditions().forEach((field, queryId) -> {
                if (!sourceTable.getColumnMap().containsKey(field)) {
                    throw new RuntimeException("mongodb request Condition field is not in columns!");
                }
                if (CollectionUtils.isEmpty(queryId)) {
                    throw new RuntimeException("mongodb request inCondition queryIds is empty!");
                }
                criteria.and(field).in(queryId);
            });
        }
        List<Map<String, Map<String, Object>>> filters = Lists.newArrayList();
        if (CollectionUtils.isNotEmpty(request.getFilters())) {
            filters.addAll(request.getFilters());
        }
        if (CollectionUtils.isNotEmpty(sourceTable.getFilters())) {
            filters.addAll(sourceTable.getFilters());
        }
        if (CollectionUtils.isNotEmpty(filters)) {
            filters.forEach(x ->
                    x.forEach((key, value) -> value.forEach((key1, value1) -> {
                        switch (key1) {
                            case "regex": {
                                Assert.isInstanceOf(String.class, value1, "online filter regex value must be string");
                                criteria.and(key).regex((String) value1);
                                break;
                            }
                            case "gt":
                                criteria.and(key).gt(value1);
                                break;
                            case "lte":
                                criteria.and(key).lte(value1);
                                break;
                            case "eq":
                                criteria.and(key).is(value1);
                                break;
                            case "ne":
                                criteria.and(key).ne(value1);
                                break;
                            default:
                                log.warn("no match filter action[{}] in {}", key1, key);
                                break;
                        }
                    })));
        }
    }

    @Override
    public DataResult process(ServiceRequest request, DataContext context) {
        DataResult result = new DataResult();
        String parent = request.getParent();
        FeatureConfig.SourceTable sourceTable = taskFlowConfig.getSourceTables().get(parent);
        Query query = new Query();
        Criteria criteria = new Criteria();
        setConditions(request, sourceTable, criteria);
        query.addCriteria(criteria);
        int limit = request.getLimit();
        if (limit > 0) {
            query.limit(limit);
        }
        result.setData(mongoTemplate.find(query, Map.class, sourceTable.getTable()));
        return result;
    }
}
