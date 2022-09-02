package com.dmetasoul.metaspore.recommend.configure;

import com.dmetasoul.metaspore.recommend.common.DataTypes;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import lombok.Data;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.util.Assert;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

@Data
public class FieldAction {
    private List<String> names;
    private List<String> types;
    private List<FeatureConfig.Field> fields;
    private List<String> input;
    private String func;
    private Map<String, Object> options;
    private List<Map<String, List<String>>> algoColumns;
    private Map<String, FeatureConfig.Field> algoFields;

    public void setNames(List<String> names) {
        if (CollectionUtils.isEmpty(names)) return;
        this.names = names;
    }

    public void setName(String name) {
        if (StringUtils.isEmpty(name)) return;
        this.names = List.of(name);
    }

    public void setTypes(List<String> types) {
        if (CollectionUtils.isEmpty(types)) return;
        this.types = types;
    }

    public void setType(String type) {
        if (StringUtils.isEmpty(type)) return;
        this.types = List.of(type);
    }
    public void setFields(List<String> fields) {
        if (CollectionUtils.isEmpty(fields)) return;
        this.fields = FeatureConfig.Field.create(fields);
    }

    public void setField(String field) {
        if (StringUtils.isEmpty(field)) return;
        this.fields = List.of(Objects.requireNonNull(FeatureConfig.Field.create(field)));
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
                    FeatureConfig.Field field = FeatureConfig.Field.create(col);
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
        Assert.isTrue(CollectionUtils.isNotEmpty(names) && CollectionUtils.isNotEmpty(types) && names.size() == types.size(),
                "AlgoTransform FieldAction config name and type must be equel!");
        processAlgoColumns(algoColumns);
        Assert.isTrue(CollectionUtils.isNotEmpty(input) || CollectionUtils.isNotEmpty(fields),
                "fieldaction input and field must not be empty at the same time");
        for (String type : types) {
            Assert.isTrue(StringUtils.isNotEmpty(type) && DataTypes.getDataType(type) != null,
                    "AlgoTransform FieldAction config type must be support! type:" + type);
        }
        return true;
    }
}