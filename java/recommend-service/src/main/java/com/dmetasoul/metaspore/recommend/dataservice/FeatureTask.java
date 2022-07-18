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
import com.dmetasoul.metaspore.recommend.configure.RecommendConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.dmetasoul.metaspore.recommend.enums.JoinTypeEnum;
import com.dmetasoul.metaspore.recommend.enums.TaskStatusEnum;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.tuple.Pair;

import java.util.*;
import java.util.concurrent.TimeUnit;

@SuppressWarnings("rawtypes")
@Slf4j
@DataServiceAnnotation("Feature")
public class FeatureTask extends DataService {

    private FeatureConfig.Feature feature;
    private Set<String> immediateTables;

    Map<FeatureConfig.Feature.Field, FeatureConfig.Feature.Field> rewritedField = Maps.newHashMap();

    @Override
    public boolean initService() {
        feature = taskFlowConfig.getFeatures().get(name);
        immediateTables = Sets.newHashSet(feature.getImmediateFrom());
        chains.add(new RecommendConfig.Chain(
                null, feature.getImmediateFrom(), false, 30000L, TimeUnit.MILLISECONDS));
        Set<String> visitedTables = Sets.newHashSet(feature.getImmediateFrom());
        Set<String> allTables = Sets.newHashSet(feature.getFrom());
        visitedTables.forEach(allTables::remove);
        while (!allTables.isEmpty()) {
            List<String> whenList = Lists.newArrayList();
            feature.getConditionMap().forEach((table, conditions) -> {
                if (!visitedTables.contains(table)) {
                    boolean noOtherVisited = true;
                    for (FeatureConfig.Feature.Condition cond : conditions) {
                        if (cond.getType() == JoinTypeEnum.LEFT || (cond.getType() == JoinTypeEnum.INNER && table.equals(cond.getRight().getTable()))) {
                            if (!visitedTables.contains(cond.getLeft().getTable())) {
                                noOtherVisited = false;
                                break;
                            }
                        }
                        if (cond.getType() == JoinTypeEnum.RIGHT || (cond.getType() == JoinTypeEnum.INNER && table.equals(cond.getLeft().getTable()))) {
                            if (!visitedTables.contains(cond.getRight().getTable())) {
                                noOtherVisited = false;
                                break;
                            }
                        }
                    }
                    if (noOtherVisited) {
                        whenList.add(table);
                    }
                }
            });
            if (whenList.isEmpty()) {
                whenList.addAll(allTables);
            }
            chains.add(new RecommendConfig.Chain(
                    null, whenList, false, 30000L, TimeUnit.MILLISECONDS));
            whenList.forEach(allTables::remove);
            visitedTables.addAll(whenList);
        }
        return true;
    }

    protected void otherRequest(String depend, DataResult dependResult, ServiceRequest newRequest) {
        // to do
    }

    public void setRequest(FeatureConfig.Feature.Field field1, FeatureConfig.Feature.Field field2, ServiceRequest req, DataContext context) {
        String table = field1.getTable();
        String field = field1.getFieldName();
        if (!processedTask.contains(table)) {
            log.error("depend ：{} has not executed!！", table);
            throw new RuntimeException(String.format("depend ：%s has not executed!", table));
        }
        DataResult dependResult = getDataResultByName(table, context);
        if (dependResult == null) {
            log.error("depend ：{} result get fail！", table);
            throw new RuntimeException(String.format("depend ：%s exec fail in feature！", table));
        }
        if (MapUtils.isNotEmpty(dependResult.getValues())) {
            req.putEq(field2.getFieldName(), dependResult.getValues().get(field));
        } else if (CollectionUtils.isNotEmpty(dependResult.getData())) {
            List<Object> ids = Lists.newArrayList();
            for (Map item : dependResult.getData()) {
                ids.add(item.get(field));
            }
            req.putIn(field2.getFieldName(), ids);
        } else if (dependResult.getFeatureArray() != null) {
            DataResult.FeatureArray featureArray = dependResult.getFeatureArray();
            if (featureArray.inArray(field)) {
                req.putIn(field2.getFieldName(), featureArray.getArray(field));
            }
        } else {
            otherRequest(field2.getTable(), dependResult, req);
        }
    }

    @Override
    public ServiceRequest makeRequest(String depend, ServiceRequest request, DataContext context) {
        ServiceRequest req = super.makeRequest(depend, request, context);
        if (!immediateTables.contains(depend)) {
            List<FeatureConfig.Feature.Condition> conditions = feature.getConditionMap().get(depend);
            if (CollectionUtils.isEmpty(conditions)) {
                return req;
            }
            for (FeatureConfig.Feature.Condition cond : conditions) {
                if (cond.getType() == JoinTypeEnum.INNER) {
                    if (processedTask.contains(cond.getLeft().getTable()) && !processedTask.contains(cond.getRight().getTable())) {
                        rewritedField.put(cond.getRight(), cond.getLeft());
                    }
                    if (processedTask.contains(cond.getRight().getTable()) && !processedTask.contains(cond.getLeft().getTable())) {
                        rewritedField.put(cond.getLeft(), cond.getRight());
                    }
                }
            }
            for (FeatureConfig.Feature.Condition cond : conditions) {
                if (cond.getType() == JoinTypeEnum.INNER) {
                    if (rewritedField.containsKey(cond.getLeft()) && !rewritedField.containsKey(cond.getRight()) &&
                            !cond.getRight().equals(rewritedField.get(cond.getLeft()))) {
                        rewritedField.put(cond.getRight(), rewritedField.get(cond.getLeft()));
                    }
                    if (rewritedField.containsKey(cond.getRight()) && !rewritedField.containsKey(cond.getLeft()) &&
                            !cond.getLeft().equals(rewritedField.get(cond.getRight()))) {
                        rewritedField.put(cond.getLeft(), rewritedField.get(cond.getRight()));
                    }
                }
            }
            for (FeatureConfig.Feature.Condition cond : conditions) {
                if (cond.getType() == JoinTypeEnum.LEFT ||
                        (cond.getType() == JoinTypeEnum.INNER && depend.equals(cond.getRight().getTable()))) {
                    setRequest(rewritedField.getOrDefault(cond.getLeft(), cond.getLeft()), cond.getRight(), req, context);
                }
                if (cond.getType() == JoinTypeEnum.RIGHT ||
                        (cond.getType() == JoinTypeEnum.INNER && depend.equals(cond.getLeft().getTable()))) {
                    setRequest(rewritedField.getOrDefault(cond.getRight(), cond.getRight()), cond.getLeft(), req, context);
                }
            }
        }
        if (MapUtils.isNotEmpty(feature.getFilters())) {
            req.setFilters(feature.getFilters().get(depend));
        }
        if (MapUtils.isNotEmpty(feature.getSqlFilters())) {
            req.setSqlFilters(feature.getSqlFilters().get(depend));
        }
        return req;
    }

    // 同一个table下字段数据列表长度一致
    public Map<FeatureConfig.Feature.Field, List<Object>> getTableArray(String table, DataContext context) {
        Map<FeatureConfig.Feature.Field, List<Object>> featureArray = Maps.newHashMap();
        DataResult result = getDataResultByName(table, context);
        if (result == null) {
            context.setStatus(name, TaskStatusEnum.EXEC_FAIL);
            return featureArray;
        }
        if (MapUtils.isNotEmpty(result.getValues())) {
            for (String fieldName : feature.getFromColumns().get(table)) {
                Object value = result.getValues().get(fieldName);
                featureArray.put(new FeatureConfig.Feature.Field(table, fieldName), List.of(value));
            }
        } else if (CollectionUtils.isNotEmpty(result.getData())) {
            for (String fieldName : feature.getFromColumns().get(table)) {
                List<Object> ids = Lists.newArrayList();
                for (Map item : result.getData()) {
                    ids.add(item.get(fieldName));
                }
                featureArray.put(new FeatureConfig.Feature.Field(table, fieldName), ids);
            }
        } else if (result.getFeatureArray() != null) {
            for (String fieldName : feature.getFromColumns().get(table)) {
                DataResult.FeatureArray data = result.getFeatureArray();
                featureArray.put(new FeatureConfig.Feature.Field(table, fieldName), data.getArray(fieldName));
            }
        } else {
            for (String fieldName : feature.getFromColumns().get(table)) {
                featureArray.put(new FeatureConfig.Feature.Field(table, fieldName), Lists.newArrayList());
            }
        }
        return featureArray;
    }

    public void setFeatureArray(List<FeatureConfig.Feature.Field> fields, Map<FeatureConfig.Feature.Field, List<Object>> featureArray, Map<String, List<Object>> result) {
        if (CollectionUtils.isEmpty(fields)) {
            return;
        }
        for (FeatureConfig.Feature.Field field : fields) {
            result.put(field.getFieldName(), featureArray.getOrDefault(field, Lists.newArrayList()));
        }
    }

    public void setFeatureArray(Set<String> fields, Map<String, List<FeatureConfig.Feature.Field>> fieldMap, Map<FeatureConfig.Feature.Field, List<Object>> featureArray, Map<String, List<Object>> result) {
        for (String table : fields) {
            setFeatureArray(fieldMap.get(table), featureArray, result);
        }
    }

    private boolean matchCondition(Set<String> joinedTable, String table, FeatureConfig.Feature.Condition cond) {
        if (cond == null || table == null || joinedTable == null) return false;
        return (table.equals(cond.getLeft().getTable()) && joinedTable.contains(cond.getRight().getTable())) ||
                (table.equals(cond.getRight().getTable()) && joinedTable.contains(cond.getLeft().getTable()));
    }

    public Map<FeatureConfig.Feature.Field, List<Object>> JoinFeatureArray(String table, List<FeatureConfig.Feature.Condition> conditions, Map<FeatureConfig.Feature.Field, List<Object>> data, Map<FeatureConfig.Feature.Field, List<Object>> joinTable) {
        Map<FeatureConfig.Feature.Field, List<Object>> result = Maps.newHashMap();
        Set<Pair<Integer, Integer>> indexSet = Sets.newHashSet();
        List<Pair<Integer, Integer>> indexResult = Lists.newArrayList();
        for (FeatureConfig.Feature.Condition cond : conditions) {
            FeatureConfig.Feature.Field fieldJoined = cond.getLeft().getTable().equals(table) ? cond.getRight() : cond.getLeft();
            FeatureConfig.Feature.Field fieldTable = cond.getLeft().getTable().equals(table) ? cond.getLeft() : cond.getRight();
            List<Object> joinedData = joinTable.getOrDefault(fieldJoined, Lists.newArrayList());
            List<Object> tableData = data.getOrDefault(fieldTable, Lists.newArrayList());
            List<Pair<Integer, Integer>> indexList = Lists.newArrayList();
            if (cond.getType() == JoinTypeEnum.INNER) {
                for (int i = 0; i < joinedData.size(); ++i) {
                    for (int j = 0; j < tableData.size(); ++j) {
                        if (joinedData.get(i) != null && joinedData.get(i).equals(tableData.get(j))) {
                            indexList.add(Pair.of(i, j));
                        }
                    }
                }
            } else if (cond.getType() == JoinTypeEnum.LEFT) {
                if (cond.getLeft().getTable().equals(table)) {
                    for (int j = 0; j < tableData.size(); ++j) {
                        for (int i = 0; i < joinedData.size(); ++i) {
                            if (joinedData.get(i) != null && joinedData.get(i).equals(tableData.get(j))) {
                                indexList.add(Pair.of(i, j));
                            } else {
                                indexList.add(Pair.of(null, j));
                            }
                        }
                        if (joinedData.isEmpty()) {
                            indexList.add(Pair.of(null, j));
                        }
                    }
                } else {
                    for (int i = 0; i < joinedData.size(); ++i) {
                        for (int j = 0; j < tableData.size(); ++j) {
                            if (tableData.get(j) != null && tableData.get(j).equals(joinedData.get(i))) {
                                indexList.add(Pair.of(i, j));
                            } else {
                                indexList.add(Pair.of(i, null));
                            }
                        }
                        if (tableData.isEmpty()) {
                            indexList.add(Pair.of(i, null));
                        }
                    }
                }
            } else if (cond.getType() == JoinTypeEnum.RIGHT) {
                if (cond.getLeft().getTable().equals(table)) {
                    for (int i = 0; i < joinedData.size(); ++i) {
                        for (int j = 0; j < tableData.size(); ++j) {
                            if (tableData.get(j) != null && tableData.get(j).equals(joinedData.get(i))) {
                                indexList.add(Pair.of(i, j));
                            } else {
                                indexList.add(Pair.of(i, null));
                            }
                        }
                        if (tableData.isEmpty()) {
                            indexList.add(Pair.of(i, null));
                        }
                    }
                } else {
                    for (int j = 0; j < tableData.size(); ++j) {
                        for (int i = 0; i < joinedData.size(); ++i) {
                            if (joinedData.get(i) != null && joinedData.get(i).equals(tableData.get(j))) {
                                indexList.add(Pair.of(i, j));
                            } else {
                                indexList.add(Pair.of(null, j));
                            }
                        }
                        if (joinedData.isEmpty()) {
                            indexList.add(Pair.of(null, j));
                        }
                    }
                }
            } else {
                for (int i = 0; i < joinedData.size(); ++i) {
                    for (int j = 0; j < tableData.size(); ++j) {
                        if (tableData.get(j) != null && tableData.get(j).equals(joinedData.get(i))) {
                            indexList.add(Pair.of(i, j));
                        } else {
                            indexList.add(Pair.of(i, null));
                            indexList.add(Pair.of(null, j));
                        }
                    }
                }
                if (joinedData.isEmpty()) {
                    for (int j = 0; j < tableData.size(); ++j) {
                        indexList.add(Pair.of(null, j));
                    }
                }
                if (tableData.isEmpty()) {
                    for (int i = 0; i < joinedData.size(); ++i) {
                        indexList.add(Pair.of(i, null));
                    }
                }
            }
            indexResult.clear();
            if (indexSet.isEmpty()) {
                indexResult.addAll(indexList);
            } else {
                for (Pair<Integer, Integer> item : indexList) {
                    if (indexSet.contains(item)) {
                        indexResult.add(item);
                    }
                }
            }
            indexSet.clear();
            indexSet.addAll(indexResult);
        }
        for (Pair<Integer, Integer> pair : indexResult) {
            for (Map.Entry<FeatureConfig.Feature.Field, List<Object>> entry : joinTable.entrySet()) {
                result.putIfAbsent(entry.getKey(), Lists.newArrayList());
                if (pair.getKey() == null) {
                    result.get(entry.getKey()).add(null);
                } else {
                    result.get(entry.getKey()).add(entry.getValue().get(pair.getKey()));
                }
            }
            for (Map.Entry<FeatureConfig.Feature.Field, List<Object>> entry : data.entrySet()) {
                result.putIfAbsent(entry.getKey(), Lists.newArrayList());
                if (pair.getValue() == null) {
                    result.get(entry.getKey()).add(null);
                } else {
                    result.get(entry.getKey()).add(entry.getValue().get(pair.getValue()));
                }
            }
        }
        return result;
    }


    @Override
    public DataResult process(ServiceRequest request, DataContext context) {
        DataResult dataResult = new DataResult();
        Map<String, List<Object>> featureArrays = Maps.newHashMap();
        Map<String, List<FeatureConfig.Feature.Field>> fieldMap = Maps.newHashMap();
        for (FeatureConfig.Feature.Field field : feature.getFields()) {
            String table = field.getTable();
            fieldMap.putIfAbsent(table, Lists.newArrayList());
            fieldMap.get(table).add(field);
        }
        Map<String, Map<FeatureConfig.Feature.Field, List<Object>>> data = Maps.newHashMap();
        for (String table : feature.getFrom()) {
            data.put(table, getTableArray(table, context));
        }
        if (feature.getFrom().size() == 1) {
            String table = feature.getFrom().get(0);
            setFeatureArray(fieldMap.get(table), data.get(table), featureArrays);
            dataResult.setFeatureArray(featureArrays);
            return dataResult;
        }
        Set<String> joinedTables = Sets.newHashSet();
        Set<String> noJoinTables = Sets.newHashSet(feature.getFrom());
        if (!noJoinTables.iterator().hasNext()) {
            return dataResult;
        }
        String firstTable = noJoinTables.iterator().next();
        joinedTables.add(firstTable);
        Map<FeatureConfig.Feature.Field, List<Object>> joinTable = data.get(firstTable);
        while (!noJoinTables.isEmpty()) {
            for (String table : noJoinTables) {
                List<FeatureConfig.Feature.Condition> conditions = Lists.newArrayList();
                feature.getCondition().forEach(cond -> {
                    if (matchCondition(joinedTables, table, cond)) {
                        conditions.add(cond);
                    }
                });
                if (!conditions.isEmpty()) {
                    joinTable = JoinFeatureArray(table, conditions, data.get(table), joinTable);
                    joinedTables.add(table);
                }
            }
            if (!joinedTables.isEmpty()) {
                setFeatureArray(joinedTables, fieldMap, joinTable, featureArrays);
                noJoinTables.removeAll(joinedTables);
                joinedTables.clear();
            }
            if (noJoinTables.iterator().hasNext()) {
                String nextTable = noJoinTables.iterator().next();
                joinedTables.add(nextTable);
                joinTable = data.get(firstTable);
            }
        }
        dataResult.setFeatureArray(featureArrays);
        return dataResult;
    }
}
