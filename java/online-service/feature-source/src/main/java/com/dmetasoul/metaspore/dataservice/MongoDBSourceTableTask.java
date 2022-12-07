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
package com.dmetasoul.metaspore.dataservice;

import com.dmetasoul.metaspore.annotation.FeatureAnnotation;
import com.dmetasoul.metaspore.data.DataContext;
import com.dmetasoul.metaspore.data.ServiceRequest;
import com.dmetasoul.metaspore.datasource.MongoDBSource;
import com.dmetasoul.metaspore.enums.ConditionTypeEnum;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.mongodb.BasicDBList;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Field;
import org.springframework.data.mongodb.core.query.Query;

import java.util.*;

import static com.dmetasoul.metaspore.enums.ConditionTypeEnum.*;

/**
 * 针对source.kind=mongodb的SourceTable的DataService的实现类
 * 调用MongoDB DataSource获取MongoDB中的数据
 * 注解DataServiceAnnotation 必须设置， value应设置为MongoDBSourceTable。
 * Created by @author qinyy907 in 14:24 22/08/01.
 */
@SuppressWarnings("rawtypes")
@Slf4j
@FeatureAnnotation("MongoDBSourceTable")
public class MongoDBSourceTableTask extends SourceTableTask {
    private MongoDBSource dataSource;
    private List<Criteria> queryObject;
    private Set<String> columns;

    @Override
    public boolean initService() {
        if (super.initService()) {
            dataSource = (MongoDBSource) serviceManager.getDataSource(sourceTable.getSource());
        }
        columns = sourceTable.getColumnMap().keySet();
        queryObject = Lists.newArrayList();
        List<Map<String, Map<String, Object>>> filters = sourceTable.getFilters();
        if (CollectionUtils.isNotEmpty(filters)) {
            filters.forEach(x -> processFilters(queryObject, x));
        }
        return true;
    }

    private void processFilters(List<Criteria> query, Map<String, Map<String, Object>> filters) {
        filters.forEach((key, value) -> value.forEach((key1, value1) -> {
            if (columns.contains(key)) {
                ConditionTypeEnum type = getEnumByName(key1);
                switch (type) {
                    case EQ:
                        query.add(Criteria.where(key).is(value1));
                        break;
                    case GE:
                        query.add(Criteria.where(key).gte(value1));
                        break;
                    case LE:
                        query.add(Criteria.where(key).lte(value1));
                        break;
                    case GT:
                        query.add(Criteria.where(key).gt(value1));
                        break;
                    case LT:
                        query.add(Criteria.where(key).lt(value1));
                        break;
                    case IN:
                    case NIN:
                        if (value1 instanceof Collection) {
                            BasicDBList values = new BasicDBList();
                            values.addAll((Collection) value1);
                            if (type == IN) {
                                query.add(Criteria.where(key).in(values));
                            } else {
                                query.add(Criteria.where(key).nin(values));
                            }
                        }
                        break;
                    case NE:
                        query.add(Criteria.where(key).ne(value1));
                        break;
                    default:
                        log.warn("no match filter action[{}] in {}", key1, key);
                        break;
                }
            }
        }));
    }

    private boolean fillDocument(Query query, String col, Object value) {
        if (value instanceof Collection) {
            BasicDBList values = new BasicDBList();
            HashSet valueSet = Sets.newHashSet((Collection) value);
            values.addAll(valueSet);
            if (values.isEmpty()) {
                return false;
            }
            query.addCriteria(Criteria.where(col).in(values));
        } else {
            query.addCriteria(Criteria.where(col).is(value));
        }
        return true;
    }

    @Override
    protected List<Map<String, Object>> processRequest(ServiceRequest request, DataContext context) {
        Query query = new Query();
        for (Criteria criteria : queryObject) {
            query.addCriteria(criteria);
        }
        Map<String, Object> data = request.getData();
        for (String col : columns) {
            if (MapUtils.isNotEmpty(data) && data.containsKey(col)) {
                Object value = data.get(col);
                if (!fillDocument(query, col, value)) {
                    return List.of();
                }
            }
        }
        if (query.getQueryObject().isEmpty()) {
            return List.of();
        }
        Field field = query.fields();
        columns.forEach(field::include);
        if (request.getLimit() > 0) {
            query.limit(request.getLimit());
        } else {
            query.limit(maxLimit);
        }
        return getDataByQuery(query);
    }

    @SuppressWarnings("unchecked")
    private List<Map<String, Object>> getDataByQuery(Query query) {
        log.debug("query mongo table {}: {}", sourceTable.getTable(), query);
        if (query == null || query.getQueryObject().isEmpty() && query.getFieldsObject().isEmpty()) {
            return List.of();
        }
        List<Map> res = dataSource.getMongoTemplate().find(query, Map.class, sourceTable.getTable());
        List<Map<String, Object>> list = Lists.newArrayList();
        res.forEach(map -> {
            Map<String, Object> item = Maps.newHashMap();
            for (String col : columns) {
                item.put(col, map.get(col));
            }
            list.add(item);
        });
        return list;
    }
}
