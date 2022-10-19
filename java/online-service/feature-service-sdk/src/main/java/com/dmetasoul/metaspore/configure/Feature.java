package com.dmetasoul.metaspore.configure;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;

import java.util.List;
import java.util.Map;
import java.util.Set;

@Slf4j
@Data
public class Feature extends TableInfo {
    private String name;
    private List<String> from;
    private List<FieldInfo> fields;
    private List<String> select;

    private List<Condition> condition;

    private List<String> immediateFrom;

    /**
     * 支持where条件  右值不支持为常数值， 需要右值为常数的过滤可以在sourcetable进行配置
     * filters:
     * - table1.field1:
     * ge: table2.field2
     * ne: table3.field3
     * - table1.field2:
     * eq: table4.field2
     * in: table5.field3
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
        for (String item : list) {
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
        filters.forEach(map -> {
            map.forEach((k, v) -> {
                FieldInfo field1 = FieldInfo.create(k);
                Map<FieldInfo, String> item = filterMap.computeIfAbsent(field1, key -> Maps.newHashMap());
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
        for (FieldInfo field : fields) {
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