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
import com.dmetasoul.metaspore.recommend.configure.FeatureConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.dmetasoul.metaspore.recommend.datasource.MongoDBSource;
import com.dmetasoul.metaspore.recommend.enums.ConditionTypeEnum;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.mongodb.BasicDBList;
import com.mongodb.BasicDBObject;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.bson.Document;
import org.springframework.data.mongodb.core.query.BasicQuery;
import org.springframework.data.mongodb.core.query.Query;

import java.util.*;

import static com.dmetasoul.metaspore.recommend.enums.ConditionTypeEnum.*;

@SuppressWarnings("rawtypes")
@Slf4j
@DataServiceAnnotation
public class MongoDBSourceTableTask extends SourceTableTask {

    private MongoDBSource dataSource;
    private Document columnsObject;
    private Document queryObject;
    private Set<String> columns;

    @Override
    public boolean initService() {
        if (super.initService() && source.getKind().equals("mongodb")) {
            dataSource = (MongoDBSource) taskServiceRegister.getDataSources().get(sourceTable.getSource());
        }
        FeatureConfig.SourceTable sourceTable = taskFlowConfig.getSourceTables().get(name);
        columns = sourceTable.getColumnMap().keySet();
        columns.forEach(col-> columnsObject.put(col, 1));
        queryObject=new Document();
        List<Map<String, Map<String, Object>>> filters = sourceTable.getFilters();
        if (CollectionUtils.isNotEmpty(filters)) {
            filters.forEach(x ->processFilters(queryObject, x));
        }
        return true;
    }

    private void processFilters(Document query, Map<String, Map<String, Object>> filters) {
        filters.forEach((key, value) -> value.forEach((key1, value1) -> {
            if (columns.contains(key)) {
                ConditionTypeEnum type = getEnumByName(key1);
                switch (type) {
                    case EQ:
                        query.put(key, value1);
                        break;
                    case GE:
                        query.put(key, new BasicDBObject("$gte", value1));
                        break;
                    case LE:
                        query.put(key, new BasicDBObject("$lte", value1));
                        break;
                    case GT:
                        query.put(key, new BasicDBObject("$gt", value1));
                        break;
                    case LT:
                        query.put(key, new BasicDBObject("$lt", value1));
                        break;
                    case IN:
                    case NIN:
                        if (value1 instanceof Collection) {
                            BasicDBList values = new BasicDBList();
                            values.addAll((Collection) value1);
                            if (type == IN) {
                                BasicDBObject in = new BasicDBObject("$in", values);
                                query.put(key, in);
                            } else {
                                BasicDBObject nin = new BasicDBObject("$nin", values);
                                query.put(key, nin);
                            }
                        }
                        break;
                    case NE:
                        query.put(key, new BasicDBObject("$ne", value1));
                        break;
                    default:
                        log.warn("no match filter action[{}] in {}", key1, key);
                        break;
                }
            }
        }));
    }

    private void fillDocument(String col, Object value) {
        if (value instanceof Collection) {
            BasicDBList values = new BasicDBList();
            values.addAll((Collection) value);
            BasicDBObject in = new BasicDBObject("$in", values);
            queryObject.put(col, in);
        } else {
            queryObject.put(col, value);
        }
    }

    @Override
    protected DataResult processRequest(ServiceRequest request, DataContext context) {
        Map<String, Object> data = request.getData();
        for (String col : columns) {
            if (MapUtils.isNotEmpty(data) && data.containsKey(col)) {
                Object value = data.get(col);
                fillDocument(col, value);
            }
        }
        Query query = new BasicQuery(queryObject, columnsObject);
        if (request.getLimit() > 0) {
            query.limit(request.getLimit());
        }
        DataResult result = new DataResult();
        List<Map> res = dataSource.getMongoTemplate().find(query, Map.class, sourceTable.getTable());
        List<Map> list = Lists.newArrayList();
        res.forEach(map -> {
            Map<String, Object> item = Maps.newHashMap();
            for (String col : columns) {
                item.put(col, map.get(col));
            }
            list.add(item);
        });
        result.setData(list);
        return result;
    }
}
