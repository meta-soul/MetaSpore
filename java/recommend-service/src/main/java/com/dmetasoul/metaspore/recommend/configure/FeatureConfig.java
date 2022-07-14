package com.dmetasoul.metaspore.recommend.configure;

import com.dmetasoul.metaspore.recommend.common.DataTypes;
import com.dmetasoul.metaspore.recommend.enums.JoinTypeEnum;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.elasticsearch.common.util.set.Sets;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.cloud.context.config.annotation.RefreshScope;
import org.springframework.context.annotation.Configuration;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static com.dmetasoul.metaspore.recommend.common.DataTypes.typeIsSupport;

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
        private String format;
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
            if (getKind().equals("mongodb")) {
                if (!options.containsKey("uri") || !String.valueOf(options.get("uri")).startsWith("mongodb://")) {
                    log.error("source mongodb config uri error!");
                    return false;
                }
            }
            if (getKind().equals("jdbc")) {
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
            if (getKind().equals("redis")) {
                if (!options.containsKey("cluster")) options.put("cluster", false);
                if ((boolean)options.get("cluster")) {
                    if (!options.containsKey("nodes")) {
                        log.error("source redis cluster config nodes must not be empty!");
                        return false;
                    }
                } else {
                    if (!options.containsKey("host")) options.put("host", "localhost");
                    if (!options.containsKey("port")) options.put("port", 6379);
                }
            }
            if (getKind().equals("milvus")) {
                if (!options.containsKey("host")) options.put("host", "localhost");
                if (!options.containsKey("port")) options.put("port", 9000);
            }
            return true;
        }
    }

    @Data
    public static class SourceTable {
        private String name;
        private String source;
        private String table;

        private String prefix;

        private List<String> sqlFilters;
        private List<Map<String, Map<String, Object>>> filters;

        private Map<String, Object> options;
        private List<String> columnNames;
        private Map<String, String> columnMap;
        private List<Map<String, String>> columns;

        public String getTable() {
            if (StringUtils.isEmpty(table)) return name;
            return table;
        }

        public void setSource(String source) {
            this.source = source;
        }

        public void setColumnData() {
            if (CollectionUtils.isNotEmpty(columns)) {
                this.columnNames = Lists.newArrayList();
                this.columnMap = Maps.newHashMap();
                columns.forEach(map -> map.forEach((x, y) -> {
                    columnNames.add(x);
                    this.columnMap.put(x, y);
                }));
            }
        }

        public boolean checkAndDefault() {
            if (StringUtils.isEmpty(name) || StringUtils.isEmpty(source)) {
                log.error("SourceTable config name and source must not be empty!");
                return false;
            }
            setColumnData();
            if (CollectionUtils.isEmpty(columnNames) || MapUtils.isEmpty(columnMap)) {
                log.error("SourceTable config columns must not be empty!");
                return false;
            }
            for (Map.Entry<String, String> entry : columnMap.entrySet()) {
                if (!typeIsSupport(entry.getValue())) {
                    log.error("SourceTable config columns type:{} must be support!", entry.getValue());
                    return false;
                }
            }
            return true;
        }

        public List<String> getColumnNames() {
            return columnNames;
        }
    }

    @Data
    public static class Feature {
        private String name;
        private List<String> from;
        private List<Field> fields;
        private List<String> select;
        private List<Condition> condition;

        private List<String> immediateFrom;

        private Map<String, List<String>> sqlFilters;
        private Map<String, List<Map<String, Map<String, Object>>>> filters;

        private List<String> columnNames;
        private Map<String, String> columnMap;
        private Map<String, List<String>> fromColumns;

        private List<Map<String, String>> columns;

        private Map<String, List<Condition>> conditionMap;

        @Data
        @AllArgsConstructor
        public static class Field {
            String table;
            String fieldName;

            public static Field create(String str) {
                String[] array = str.split("\\.");
                if (array.length == 2) {
                    return new Field(array[0], array[1]);
                } else if (array.length == 1) {
                    return new Field(null, array[0]);
                }
                return null;
            }

            @Override
            public int hashCode() {
                return String.format("%s.%s", table, fieldName).hashCode();
            }

            @Override
            public boolean equals(Object obj) {
                if (obj == this) return true;
                if (obj == null) return false;
                if (obj instanceof Field) {
                    Field field = (Field) obj;
                    if ((fieldName != null && !fieldName.equals(field.getFieldName())) || (fieldName == null && field.getFieldName() != null)) {
                        return false;
                    } else
                        return (table == null || table.equals(field.getTable())) && (table != null || field.getTable() == null);
                }
                return false;
            }
        }

        @Data
        @AllArgsConstructor
        public static class Condition {
            Field left;
            Field right;
            JoinTypeEnum type;
            private Condition() {
                type = JoinTypeEnum.INNER;
            }

            public static Condition create(Map<String, String> data) {
                if (MapUtils.isEmpty(data)) {
                    log.error("feature condition config is wrong");
                    return null;
                }
                if ((data.containsKey("type") && data.size() == 2) || data.size() == 1) {
                    Condition condition = new Condition();
                    data.forEach((key, value) -> {
                        if (key.equals("type")) {
                            condition.type = JoinTypeEnum.getEnumByName(value);
                        } else {
                            condition.left = Field.create(key);
                            condition.right = Field.create(value);
                        }
                    });
                    if (condition.isInvalid()) {
                        return null;
                    }
                    return condition;
                }
                return null;
            }

            public boolean isInvalid() {
                return left == null || right == null;
            }
        }

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
                Field field = Field.create(item);
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

        public boolean checkAndDefault() {
            if (StringUtils.isEmpty(name)) {
                log.error("Feature config name must not be empty!");
                return false;
            }
            if (CollectionUtils.isEmpty(select)) {
                log.error("Feature config select must not be empty!");
                return false;
            }
            if (CollectionUtils.isEmpty(from)) {
                log.error("Feature config from must not be empty!");
                return false;
            }
            Set<String> fromtables = Sets.newHashSet(from);
            if (fromtables.size() != from.size()) {
                log.error("Feature config from and dependOutput must not be duplicate table!");
                return false;
            }
            if (CollectionUtils.isEmpty(condition) && fromtables.size() > 1) {
                log.error("Feature join table must has condition!");
                return false;
            }
            setFields(select);
            for (Field field: fields) {
                if (field.getTable() != null && !fromtables.contains(field.getTable())) {
                    log.error("Feature config the field of select, field table' must be in the rely!");
                    return false;
                }
            }
            if (CollectionUtils.isNotEmpty(condition)) {
                for (Condition cond : condition) {
                    if (cond.left.getTable() != null && !fromtables.contains(cond.left.getTable())) {
                        log.error("Feature config the field of join condition, field table' must be in the rely!");
                        return false;
                    }
                    if (cond.right.getTable() != null && !fromtables.contains(cond.right.getTable())) {
                        log.error("Feature config the field of join condition, field table' must be in the rely!");
                        return false;
                    }
                }
            }
            return true;
        }
    }

    @Data
    public static class FieldAction {
        private String name;
        private String type;
        private List<String> fields;
        private String func;
        private Map<String, Object> options;

        public void setFields(List<String> fields) {
            if (CollectionUtils.isEmpty(fields)) return;
            this.fields = fields;
        }

        public void setField(String field) {
            if (StringUtils.isEmpty(field)) return;
            this.fields = List.of(field);
        }

        public boolean checkAndDefault() {
            if (StringUtils.isEmpty(name)) {
                log.error("AlgoTransform Action name must not be empty!");
                return false;
            }
            if (CollectionUtils.isEmpty(fields)) {
                fields = List.of(name);
            }
            if (!StringUtils.isEmpty(type)) {
                if (DataTypes.getDataType(type) == null) {
                    log.error("AlgoTransform FieldAction config type:{} must be support!", type);
                    return false;
                }
            } else if (fields.size() > 1) {
                log.error("AlgoTransform Action duplicate field must define type!");
                return false;
            }
            return true;
        }
    }

    @Data
    public static class AlgoTransform {
        private String name;
        private String depend;
        private List<FieldAction> fieldActions;

        public boolean checkAndDefault() {
            if (StringUtils.isEmpty(name) || StringUtils.isEmpty(depend) || CollectionUtils.isEmpty(fieldActions)) {
                log.error("AlgoTransform config name, fieldActions and depend must not be empty!");
                return false;
            }
            for (FieldAction action : fieldActions) {
                if (!action.checkAndDefault()) {
                    log.error("AlgoTransform config action must be right!");
                    return false;
                }
            }
            return true;
        }
    }
}
