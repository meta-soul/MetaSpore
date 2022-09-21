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
package com.dmetasoul.metaspore.recommend.configure;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Queues;
import com.google.common.collect.Sets;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.cloud.context.config.annotation.RefreshScope;
import org.springframework.context.annotation.Configuration;
import org.springframework.util.Assert;

import java.util.*;

/**
 * 特征生成相关配置类
 * Created by @author qinyy907 in 14:24 22/07/15.
 */
@Slf4j
@Data
@Configuration
@RefreshScope
@ConfigurationProperties(prefix = "feature-service")
public class FeatureConfig {
    private List<Source> source;
    private List<SourceTable> sourceTable;
    private List<Feature> feature;
    private List<AlgoTransform> algoTransform;

    @Data
    public static class Source {
        private String name;
        // private String format;
        private String kind;
        private Map<String, Object> options;

        public String getKind() {
            if (StringUtils.isEmpty(kind)) return "request";
            return kind;
        }

        public boolean checkAndDefault() {
            if (StringUtils.isEmpty(name)) {
                log.error("source config name must not be empty!");
                return false;
            }
            if (options == null) {
                options = Maps.newHashMap();
            }
            if (getKind().equalsIgnoreCase("mongodb")) {
                if (!options.containsKey("uri") || !String.valueOf(options.get("uri")).startsWith("mongodb://")) {
                    log.error("source mongodb config uri error!");
                    return false;
                }
            }
            if (getKind().equalsIgnoreCase("jdbc")) {
                if (!options.containsKey("uri")) {
                    log.error("source jdbc config uri must not be empty!");
                    return false;
                }
                if (!options.containsKey("user")) options.put("user", "root");
                if (!options.containsKey("password")) options.put("password", "test");
                String uri = String.valueOf(options.containsKey("uri"));
                if (uri.startsWith("jdbc:mysql")) {
                    if (!options.containsKey("driver")) options.put("driver", "com.mysql.cj.jdbc.Driver");
                    else {
                        if (!String.valueOf(options.get("driver")).equals("com.mysql.cj.jdbc.Driver")) {
                            log.error("source jdbc mysql config driver must be com.mysql.cj.jdbc.Driver!");
                            return false;
                        }
                    }
                }
            }
            if (getKind().equalsIgnoreCase("redis")) {
                if (!options.containsKey("standalone") && !options.containsKey("sentinel")
                    && !options.containsKey("cluster")) {
                    options.put("standalone", Map.of("host", "localhost", "port", 6379));
                }
            }
            return true;
        }
    }

    @Data
    public static class SourceTable extends ColumnInfo {
        private String name;
        private String source;
        private String kind;
        private String taskName;
        private String table;

        private String prefix;

        private List<String> sqlFilters;
        private List<Map<String, Map<String, Object>>> filters;

        private Map<String, Object> options;

        public void setColumns(List<Map<String, Object>> columns) {
            super.setColumns(columns);
        }

        public String getTable() {
            if (StringUtils.isEmpty(table)) return name;
            return table;
        }

        public void setSource(String source) {
            this.source = source;
        }

        public boolean checkAndDefault() {
            if (StringUtils.isEmpty(name) || StringUtils.isEmpty(source)) {
                log.error("SourceTable config name and source must not be empty!");
                throw new IllegalStateException("SourceTable config name and source must not be empty!");
            }
            if (CollectionUtils.isEmpty(columnNames) || MapUtils.isEmpty(columnMap)) {
                log.error("SourceTable config columns must not be empty!");
                throw new IllegalStateException("SourceTable config columns must not be empty!");
            }
            if (StringUtils.isEmpty(taskName)) {
                taskName = "SourceTable";
            }
            return true;
        }
    }

    @Data
    public static class Feature extends ColumnInfo {
        private String name;
        private List<String> from;
        private List<FieldInfo> fields;
        private List<String> select;
        private List<Condition> condition;

        private List<String> immediateFrom;

        /**
         * 支持where条件  右值不支持为常数值， 需要右值为常数的过滤可以在sourcetable进行配置
         * filters:
         *   - table1.field1:
         *       ge: table2.field2
         *       ne: table3.field3
         *   - table1.field2:
         *       eq: table4.field2
         *       in: table5.field3
         */
        private List<Map<String, Map<String, Object>>> filters;
        private Map<FieldInfo, Map<FieldInfo, String>> filterMap;

        private Map<String, List<String>> fromColumns;

        private Map<String, List<Condition>> conditionMap;

        public void setFrom(List<String> from) {
            if (CollectionUtils.isEmpty(from)) return;
            this.from = from;
        }

        public void setFrom(String from) {
            if (StringUtils.isEmpty(from)) return;
            this.from = List.of(from);
        }

        public void setFields(List<String> list) {
            if (CollectionUtils.isEmpty(list)) return;
            fields = Lists.newArrayList();
            columnNames = Lists.newArrayList();
            for(String item: list) {
                FieldInfo field = FieldInfo.create(item);
                if (field == null) {
                    log.error("Feature:{} select is config error", name);
                    throw new RuntimeException(String.format("Feature:%s select is config error", name));
                }
                fields.add(field);
                columnNames.add(field.fieldName);
            }
        }

        public void setCondition(List<Map<String, String>> data) {
            if (CollectionUtils.isEmpty(data)) return;
            condition = Lists.newArrayList();
            for (Map<String, String> datum : data) {
                Condition cond = Condition.create(datum);
                if (cond == null) {
                    log.error("Feature:{} condition is config error", name);
                    throw new RuntimeException(String.format("Feature:%s condition is config error", name));
                }
                condition.add(cond);
            }
        }

        public void setFilterMap(List<Map<String, Map<String, Object>>> filters) {
            if (CollectionUtils.isEmpty(filters)) return;
            filterMap = Maps.newHashMap();
            filters.forEach(map->{
                map.forEach((k, v) -> {
                    FieldInfo field1 = FieldInfo.create(k);
                    Map<FieldInfo, String> item = filterMap.computeIfAbsent(field1, key->Maps.newHashMap());
                    v.forEach((op, value) -> {
                        FieldInfo field2 = FieldInfo.create((String) value);
                        item.put(field2, op);
                    });
                });
            });
        }

        public boolean checkAndDefault() {
            if (StringUtils.isEmpty(name)) {
                log.error("Feature config name must not be empty!");
                throw new IllegalStateException("Feature config name must not be empty!");
            }
            if (CollectionUtils.isEmpty(select)) {
                log.error("Feature config select must not be empty!");
                throw new IllegalStateException("Feature config select must not be empty!");
            }
            if (CollectionUtils.isEmpty(from)) {
                log.error("Feature config from must not be empty!");
                throw new IllegalStateException("Feature config from must not be empty!");
            }
            Set<String> fromtables = Sets.newHashSet(from);
            if (fromtables.size() != from.size()) {
                log.error("Feature config from and dependOutput must not be duplicate table!");
                throw new IllegalStateException("Feature config from and dependOutput must not be duplicate table!");
            }
            setFields(select);
            setFilterMap(filters);
            for (FieldInfo field: fields) {
                if (field.getTable() != null && !fromtables.contains(field.getTable())) {
                    log.error("Feature config the field of select, field table' must be in the rely!");
                    throw new IllegalStateException("Feature config the field of select, field table' must be in the rely!");
                }
            }
            if (CollectionUtils.isNotEmpty(condition)) {
                for (Condition cond : condition) {
                    if (cond.left.getTable() != null && !fromtables.contains(cond.left.getTable())) {
                        log.error("Feature config the field of join condition, field table' must be in the rely!");
                        throw new IllegalStateException("Feature config the field of join condition, field table' must be in the rely!");
                    }
                    if (cond.right.getTable() != null && !fromtables.contains(cond.right.getTable())) {
                        log.error("Feature config the field of join condition, field table' must be in the rely!");
                        throw new IllegalStateException("Feature config the field of join condition, field table' must be in the rely!");
                    }
                }
            }
            return true;
        }
    }

    @Data
    public static class AlgoTransform extends ColumnInfo {
        private String name;
        private String taskName;
        private List<String> feature;
        private List<String> algoTransform;
        private List<FieldAction> fieldActions;
        private List<String> output;
        private Map<String, Object> options;

        private List<FieldAction> actionList;

        protected Map<String, String> columnRel;

        public void setFeature(List<String> list) {
            feature = list;
        }

        public void setFeature(String str) {
            feature = List.of(str);
        }

        public void setAlgoTransform(List<String> list) {
            algoTransform = list;
        }

        public void setAlgoTransform(String str) {
            algoTransform = List.of(str);
        }

        public boolean checkAndDefault() {
            if (!check()) {
                return false;
            }
            columnNames = output;
            columnMap = Maps.newHashMap();
            fieldMap = Maps.newHashMap();
            columnRel = Maps.newHashMap();
            actionList = Lists.newArrayList();
            Map<String, FieldAction> fieldActionMap = Maps.newHashMap();
            Queue<FieldAction> queue = Queues.newArrayDeque();
            Set<String> actionSet = Sets.newHashSet();
            Set<String> doneSet = Sets.newHashSet();
            if (CollectionUtils.isNotEmpty(fieldActions)) {
                for (FieldAction action : fieldActions) {
                    Assert.isTrue(action.checkAndDefault(), "field action check!");
                    for (int i = 0; i < action.getNames().size(); ++i) {
                        String name = action.getNames().get(i);
                        Assert.isTrue(!fieldActionMap.containsKey(name), "name must not be duplicate!");
                        fieldActionMap.put(name, action);
                        columnMap.put(name, getType(action.getTypes().get(i)));
                        fieldMap.put(name, getField(name, action.getTypes().get(i)));
                        columnRel.put(name, action.getNames().get(0));
                    }
                }
            }
            for (String col : output) {
                if (actionSet.contains(col)) continue;
                FieldAction action = fieldActionMap.get(col);
                Assert.notNull(action, "output col must has Action colName:" + col);
                queue.offer(action);
                actionSet.addAll(action.getNames());
            }
            boolean flag;
            while (!queue.isEmpty()) {
                FieldAction action = queue.poll();
                flag = true;
                if (CollectionUtils.isNotEmpty(action.getInput())) {
                    for (String item : action.getInput()) {
                        if (doneSet.contains(item)) {
                            continue;
                        }
                        flag = false;
                        if (actionSet.contains(item)) {
                            continue;
                        }
                        FieldAction fieldAction = fieldActionMap.get(item);
                        Assert.notNull(fieldAction, "FieldAction.input item must has Action item:" + item);
                        queue.offer(fieldAction);
                        actionSet.addAll(fieldAction.getNames());
                    }
                }
                if (flag) {
                    actionList.add(action);
                    doneSet.addAll(action.getNames());
                } else {
                    queue.offer(action);
                }
            }
            if (StringUtils.isEmpty(this.taskName)) {
                this.taskName = "AlgoTransform";
            }
            return true;
        }

        protected boolean check() {
            if (StringUtils.isEmpty(name) || CollectionUtils.isEmpty(output)) {
                log.error("AlgoInference config name, fieldActions must not be empty!");
                throw new IllegalStateException("AlgoInference config name, fieldActions must not be empty!");
            }
            return true;
        }
    }
}
