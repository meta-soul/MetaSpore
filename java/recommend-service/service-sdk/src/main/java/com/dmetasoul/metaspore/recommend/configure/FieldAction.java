package com.dmetasoul.metaspore.recommend.configure;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import lombok.Data;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.Validate;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

@Data
public class FieldAction {
    private List<String> names;
    private List<Object> types;
    private List<FieldInfo> fields;
    private List<String> input;
    private String func;
    private Map<String, Object> options;
    private List<Map<String, List<String>>> algoColumns;
    private Map<String, FieldInfo> algoFields;
    private List<FieldInfo> inputFields;

    public void setNames(List<String> names) {
        if (CollectionUtils.isEmpty(names)) return;
        this.names = names;
    }

    public void setName(String name) {
        if (StringUtils.isEmpty(name)) return;
        this.names = List.of(name);
    }

    public void setTypes(List<Object> types) {
        if (CollectionUtils.isEmpty(types)) return;
        this.types = types;
    }

    public void setType(Object type) {
        if (type == null) return;
        this.types = List.of(type);
    }
    public void setFields(List<String> fields) {
        if (CollectionUtils.isEmpty(fields)) return;
        this.fields = FieldInfo.create(fields);
    }

    public void setField(String field) {
        if (StringUtils.isEmpty(field)) return;
        this.fields = List.of(Objects.requireNonNull(FieldInfo.create(field)));
    }

    public void processAlgoColumns(List<Map<String, List<String>>> data) {
        if (CollectionUtils.isEmpty(data)) return;
        if (fields == null) fields = Lists.newArrayList();
        if (input == null) input = Lists.newArrayList();
        Set<String> inputSet = Sets.newHashSet();
        inputSet.addAll(input);
        algoFields = Maps.newHashMap();
        for (Map<String, List<String>> item : data) {
            for (Map.Entry<String, List<String>> entry : item.entrySet()) {
                if (CollectionUtils.isEmpty(entry.getValue())) continue;
                for (String col : entry.getValue()) {
                    if (inputSet.contains(col)) continue;
                    FieldInfo field = FieldInfo.create(col);
                    fields.add(field);
                    algoFields.put(col, field);
                }
            }
        }
    }

    public void setInput(List<String> input) {
        if (CollectionUtils.isEmpty(input)) return;
        this.input = input;
    }

    public void setInput(String input) {
        if (StringUtils.isEmpty(input)) return;
        this.input = List.of(input);
    }
    public boolean checkAndDefault() {
        Validate.isTrue(CollectionUtils.isNotEmpty(names) && CollectionUtils.isNotEmpty(types) && names.size() == types.size(),
                "AlgoTransform FieldAction config name and type must be equel!");
        processAlgoColumns(algoColumns);
        if (CollectionUtils.isNotEmpty(input)) {
            Set<String> nameSet = Sets.newHashSet(names);
            for (String key : input) {
                Validate.isTrue(!nameSet.contains(key), "input field must not in names! key:" + key);
            }
        }
        inputFields = Lists.newArrayList();
        if (CollectionUtils.isNotEmpty(fields)) {
            inputFields.addAll(fields);
        }
        if (CollectionUtils.isNotEmpty(input)) {
            inputFields.addAll(input.stream().map(FieldInfo::new).collect(Collectors.toList()));
        }
        return true;
    }
}