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
import com.dmetasoul.metaspore.recommend.common.DataTypes;
import com.dmetasoul.metaspore.recommend.configure.Chain;
import com.dmetasoul.metaspore.recommend.configure.FeatureConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.dmetasoul.metaspore.recommend.enums.ConditionTypeEnum;
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.dmetasoul.metaspore.recommend.enums.JoinTypeEnum;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.springframework.util.Assert;

import java.util.*;
import static com.dmetasoul.metaspore.recommend.enums.ConditionTypeEnum.*;
/**
 * Feature的DataService的实现类
 * Feature用于从一个或多个数据表中select相关字段，数据表之间按照join条件连接在一起，并通过where条件对连接好的数据进行过滤
 * 注解DataServiceAnnotation 必须设置
 * Created by @author qinyy907 in 14:24 22/08/01.
 */
@Slf4j
@ServiceAnnotation
public class FeatureTask extends DataService {
    /**
     * Feature相关的配置数据对象
     */
    private FeatureConfig.Feature feature;
    /**
     * 不需要设置查询条件，直接获取数据的数据表集合
     * 比如request数据，
     *    其他可直接计算的数据（参与join的feature数据被认为是直接可以计算获得数据的数据）
     */
    private Set<String> immediateTables;
    /**
     * select字段，按照from的表分类，生成表-->字段的映射，用于计算数据的辅助数据
     */
    private Map<String, List<FeatureConfig.Field>> fieldMap;
    /**
     * join条件处理：
     * 如果 A inner join B on A.a = B.b, B inner join C on B.b = C.c
     * 如果获取A表的数据，需要构建查询条件 where A.a = B.b
     *    获取B表的数据，需要构建查询条件 where B.b = A.a and B.b = C.c
     *    获取C表的数据，需要构建查询条件 where C.c = B.b
     * 假设C表的数据已经成功获取到了， 可构建rewritedField映射 B.b -> C.c, A.a -> C.c
     * 如果获取A表的数据，需要构建查询条件 where A.a = C.c
     *    获取B表的数据，需要构建查询条件 where B.b = C.c and B.b = C.c
     */
    private Map<FeatureConfig.Field, FeatureConfig.Field> rewritedField;

    /**
     * 类似于rewritedField， immediateRewritedField主要在初始化过程中构建
     */
    private Map<FeatureConfig.Field, FeatureConfig.Field> immediateRewritedField;

    /**
     * 初始化FeatureTask
     */
    @Override
    public boolean initService() {
        feature = taskFlowConfig.getFeatures().get(name);
        rewritedField = Maps.newHashMap();
        immediateRewritedField = Maps.newHashMap();
        fieldMap = Maps.newHashMap();
        executeNum = 5; // 由于每次执行都是部分构建rewritedField，所以需要多次入队执行，确保makeRequest构建正确
        for (FeatureConfig.Field field : feature.getFields()) {
            DataTypeEnum dataType = DataTypes.getDataType(feature.getColumnMap().get(field.getFieldName()));
            resFields.add(new Field(field.getFieldName(), dataType.getType(), dataType.getChildFields()));
            dataTypes.add(dataType);
            fieldMap.computeIfAbsent(field.getTable(), k->Lists.newArrayList()).add(field);
        }
        for (String table : feature.getImmediateFrom()) {
            setRewritedField(table, immediateRewritedField);
        }
        immediateTables = Sets.newHashSet();
        return true;
    }

    /**
     * 每次请求处理前，预处理
     */
    @Override
    protected void preCondition(ServiceRequest request, DataContext context) {
        immediateTables.clear();
        immediateTables.addAll(feature.getImmediateFrom());
        for (String table : feature.getImmediateFrom()) {
            DataResult result = execute(table, request, context);
            Assert.notNull(result, "immediateTables DataResult is not exist! at " + table);
        }
        rewritedField.clear();
        rewritedField.putAll(immediateRewritedField);
        List<String> executeTables = Lists.newArrayList();
        for (String table : feature.getFrom()) {
            if (!immediateTables.contains(table)) {
                if (getDataResultByName(table, context) == null) {
                    executeTables.add(table);
                } else {
                    setRewritedField(table, rewritedField);
                    immediateTables.add(table);
                }
            }
        }
        taskFlow.offer(new Chain(null, executeTables, false));
    }

    protected void setRewritedField(String depend, Map<FeatureConfig.Field, FeatureConfig.Field> rewritedField) {
        for (FeatureConfig.Feature.Condition cond : feature.getCondition()) {
            if (cond.getType() == JoinTypeEnum.RIGHT || cond.getType() == JoinTypeEnum.INNER) {
                if (cond.getRight().getTable().equals(depend)) {
                    rewritedField.put(cond.getLeft(), cond.getRight());
                }
                if (rewritedField.containsKey(cond.getRight()) && !rewritedField.containsKey(cond.getLeft())) {
                    rewritedField.put(cond.getLeft(), rewritedField.get(cond.getRight()));
                }
            }
            if (cond.getType() == JoinTypeEnum.LEFT || cond.getType() == JoinTypeEnum.INNER) {
                if (cond.getLeft().getTable().equals(depend)) {
                    rewritedField.put(cond.getRight(), cond.getLeft());
                }
                if (rewritedField.containsKey(cond.getLeft()) && !rewritedField.containsKey(cond.getRight())) {
                    rewritedField.put(cond.getRight(), rewritedField.get(cond.getLeft()));
                }
            }
        }
    }

    @Override
    public void afterProcess(String taskName, ServiceRequest request, DataContext context) {
        setRewritedField(taskName, rewritedField);
    }

    @Override
    public ServiceRequest makeRequest(String depend, ServiceRequest request, DataContext context) {
        ServiceRequest req = super.makeRequest(depend, request, context);
        // 直接获取数据的数据表集合不需要生成查询条件，不参与makeRequest计算
        if (!immediateTables.contains(depend)) {
            // 获取depend表相关的join条件， 所有条件已经经过预处理，depend位于condition的左侧
            List<FeatureConfig.Feature.Condition> conditions = feature.getConditionMap().get(depend);
            if (CollectionUtils.isEmpty(conditions)) {
                return req;
            }
            for (FeatureConfig.Feature.Condition cond : conditions) {
                if (cond.getType() == JoinTypeEnum.RIGHT || cond.getType() == JoinTypeEnum.INNER) {
                    FeatureConfig.Field field = rewritedField.getOrDefault(cond.getRight(), cond.getRight());
                    DataResult dependResult = getDataResultByName(field.getTable(), context);
                    if (dependResult == null) {
                        return null;
                    }
                    List<Object> values = dependResult.get(field.getFieldName());
                    if (CollectionUtils.isNotEmpty(values) && values.size() == 1) {
                        req.put(cond.getLeft().getFieldName(), values.get(0));
                    } else {
                        req.put(cond.getLeft().getFieldName(), values);
                    }
                }
            }
        }
        return req;
    }

    public boolean FilterTableArray(FeatureConfig.Field left, FeatureConfig.Field right, Pair<Integer, Integer> pair, List<Object> joinedData, List<Object> tableData) {
        if (MapUtils.isEmpty(feature.getFilterMap())) return true;
        Map<FeatureConfig.Field, String> fieldMap = feature.getFilterMap().get(left);
        if (MapUtils.isEmpty(fieldMap) || !fieldMap.containsKey(right)) return true;
        ConditionTypeEnum type = getEnumByName(fieldMap.get(right));
        Object leftValue = pair.getKey() != null && pair.getKey() < joinedData.size() ? joinedData.get(pair.getKey()) : null;
        Object rightValue = pair.getRight() != null && pair.getRight() < tableData.size() ? tableData.get(pair.getRight()) : null;
        return type.Op(leftValue, rightValue);
    }

    // 同一个table下字段数据列表长度一致
    public Map<FeatureConfig.Field, List<Object>> getTableArray(String table, DataContext context) {
        Map<FeatureConfig.Field, List<Object>> featureArray = Maps.newHashMap();
        DataResult result = getDataResultByName(table, context);
        if (result == null) {
            return featureArray;
        }
        for (String fieldName : feature.getFromColumns().get(table)) {
            featureArray.put(new FeatureConfig.Field(table, fieldName), result.get(fieldName));
        }
        return featureArray;
    }

    public void setFeatureArray(List<FeatureConfig.Field> fields, Map<FeatureConfig.Field, List<Object>> featureArray, Map<String, List<Object>> result) {
        if (CollectionUtils.isEmpty(fields)) {
            return;
        }
        for (FeatureConfig.Field field : fields) {
            result.put(field.getFieldName(), featureArray.getOrDefault(field, Lists.newArrayList()));
        }
    }

    public void setFeatureArray(Set<String> fields, Map<String, List<FeatureConfig.Field>> fieldMap, Map<FeatureConfig.Field, List<Object>> featureArray, Map<String, List<Object>> result) {
        for (String table : fields) {
            setFeatureArray(fieldMap.get(table), featureArray, result);
        }
    }

    private boolean matchCondition(Set<String> joinedTable, String table, FeatureConfig.Feature.Condition cond) {
        if (cond == null || table == null || joinedTable == null || joinedTable.contains(table)) return false;
        return (table.equals(cond.getLeft().getTable()) && joinedTable.contains(cond.getRight().getTable())) ||
                (table.equals(cond.getRight().getTable()) && joinedTable.contains(cond.getLeft().getTable()));
    }

    public Map<FeatureConfig.Field, List<Object>> JoinFeatureArray(List<FeatureConfig.Feature.Condition> conditions, Map<FeatureConfig.Field, List<Object>> data, Map<FeatureConfig.Field, List<Object>> joinTable) {
        Map<FeatureConfig.Field, List<Object>> result = Maps.newHashMap();
        Set<Pair<Integer, Integer>> indexSet = Sets.newHashSet();
        List<Pair<Integer, Integer>> indexResult = Lists.newArrayList();
        // conditions 中只包含一张待join表和table， indexList始终在固定的两张表之间
        for (FeatureConfig.Feature.Condition cond : conditions) {
            FeatureConfig.Field fieldJoined = cond.getRight();
            FeatureConfig.Field fieldTable = cond.getLeft();
            List<Object> joinedData = joinTable.getOrDefault(fieldJoined, Lists.newArrayList());
            List<Object> tableData = data.getOrDefault(fieldTable, Lists.newArrayList());
            List<Pair<Integer, Integer>> indexList = Lists.newArrayList();
            for (int j = 0; j < tableData.size(); ++j) {
                boolean noHit = true;
                for (int i = 0; i < joinedData.size(); ++i) {
                    if (joinedData.get(i) != null && joinedData.get(i).equals(tableData.get(j))) {
                        if (cond.getType() == JoinTypeEnum.LEFT) {
                            indexList.add(Pair.of(i, j));
                        }
                        noHit = false;
                    }
                }
                if (noHit && (cond.getType() == JoinTypeEnum.LEFT || cond.getType() == JoinTypeEnum.FULL)) {
                    indexList.add(Pair.of(null, j));
                }
            }
            for (int i = 0; i < joinedData.size(); ++i) {
                boolean noHit = true;
                for (int j = 0; j < tableData.size(); ++j) {
                    if (joinedData.get(i) != null && joinedData.get(i).equals(tableData.get(j))) {
                        if (cond.getType() != JoinTypeEnum.LEFT) {
                            indexList.add(Pair.of(i, j));
                        }
                        noHit = false;
                    }
                }
                if (noHit && (cond.getType() == JoinTypeEnum.RIGHT || cond.getType() == JoinTypeEnum.FULL)) {
                    indexList.add(Pair.of(i, null));
                }
            }
            indexResult.clear();
            if (indexSet.isEmpty()) {
                indexSet.addAll(indexList);
            }
            for (Pair<Integer, Integer> item : indexList) {
                if (indexSet.contains(item) && FilterTableArray(fieldJoined, fieldTable, item, joinedData, tableData)) {
                    indexResult.add(item);
                }
            }
            indexSet.clear();
            indexSet.addAll(indexResult);
        }
        for (Pair<Integer, Integer> pair : indexResult) {
            for (Map.Entry<FeatureConfig.Field, List<Object>> entry : joinTable.entrySet()) {
                List<Object> list = result.computeIfAbsent(entry.getKey(), key->Lists.newArrayList());
                if (pair.getKey() == null) {
                    list.add(null);
                } else {
                    list.add(entry.getValue().get(pair.getKey()));
                }
            }
            for (Map.Entry<FeatureConfig.Field, List<Object>> entry : data.entrySet()) {
                List<Object> list = result.computeIfAbsent(entry.getKey(), key->Lists.newArrayList());
                if (pair.getValue() == null) {
                    list.add(null);
                } else {
                    list.add(entry.getValue().get(pair.getValue()));
                }
            }
        }
        return result;
    }

    @Override
    public DataResult process(ServiceRequest request, DataContext context) {
        Map<String, List<Object>> featureArrays = Maps.newHashMap();
        // 按table获取每个表的数据结果
        Map<String, Map<FeatureConfig.Field, List<Object>>> data = Maps.newHashMap();
        for (String table : feature.getFrom()) {
            data.put(table, getTableArray(table, context));
        }
        // 单表特殊处理
        if (feature.getFrom().size() == 1) {
            String table = feature.getFrom().get(0);
            setFeatureArray(fieldMap.get(table), data.get(table), featureArrays);
            return setDataResult(featureArrays);
        }
        // while 每次完成一次join，已join好的表跟另一个表join，不存在join关系的表直接concat
        Set<String> joinedTables = Sets.newHashSet();
        Set<String> noJoinTables = Sets.newHashSet(feature.getFrom());
        if (!noJoinTables.iterator().hasNext()) {
            return null;
        }
        String firstTable = noJoinTables.iterator().next();
        joinedTables.add(firstTable);
        Map<FeatureConfig.Field, List<Object>> joinTable = data.get(firstTable);
        while (!noJoinTables.isEmpty()) {
            boolean updateJoinedTable = true;
            while (updateJoinedTable) {
                updateJoinedTable = false;
                for (String table : noJoinTables) {
                    List<FeatureConfig.Feature.Condition> conditions = Lists.newArrayList();
                    feature.getCondition().forEach(cond -> {
                        if (matchCondition(joinedTables, table, cond)) {
                            if (table.equals(cond.getRight().getTable())) {
                                conditions.add(FeatureConfig.Feature.Condition.reverse(cond));
                            } else {
                                conditions.add(cond);
                            }
                        }
                    });
                    if (!conditions.isEmpty()) {
                        joinTable = JoinFeatureArray(conditions, data.get(table), joinTable);
                        joinedTables.add(table);
                        updateJoinedTable = true;
                    }
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
                joinTable = data.get(nextTable);
            }
        }
        return setDataResult(featureArrays);
    }
}
