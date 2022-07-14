package com.dmetasoul.metaspore.recommend.data;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;

@Slf4j
@Data
public class ServiceRequest implements java.io.Serializable {
    private String name;
    private String parent;

    private Set<String> parents;
    private Map<String, Object> eqConditions;
    private Map<String, List<Object>> inConditions;
    private String jdbcSql;
    private Map<String, Object> data;
    private List<String> keys;
    private List<String> sqlFilters;
    private List<Map<String, Map<String, Object>>> filters;
    private int limit;
    public ServiceRequest(String name, String parent) {
        this.name = name;
        this.parent = parent;
        limit = -1;
        this.parents = Sets.newHashSet();
        this.parents.add(parent);
    }

    public ServiceRequest(String name) {
        this.name = name;
        this.parent = name;
        limit = -1;
        this.parents = Sets.newHashSet();
        this.parents.add(parent);
    }

    public void copy(ServiceRequest req) {
        if (req == null) return;
        if (parent.equals(req.getName())) {
            this.parents.addAll(req.getParents());
        }
        if (MapUtils.isNotEmpty(req.getEqConditions())) {
            if (eqConditions == null) eqConditions = Maps.newHashMap();
            eqConditions.putAll(req.getEqConditions());
        }
        if (MapUtils.isNotEmpty(req.getInConditions())) {
            if (inConditions == null) inConditions = Maps.newHashMap();
            inConditions.putAll(req.getInConditions());
        }
        if (MapUtils.isNotEmpty(req.getData())) {
            if (data == null) data = Maps.newHashMap();
            data.putAll(req.getData());
        }
        if (CollectionUtils.isNotEmpty(req.getKeys())) {
            if (keys == null) keys = Lists.newArrayList();
            keys.addAll(req.getKeys());
        }
        if (CollectionUtils.isNotEmpty(req.getSqlFilters())) {
            if (sqlFilters == null) sqlFilters = Lists.newArrayList();
            sqlFilters.addAll(req.getSqlFilters());
        }
        if (CollectionUtils.isNotEmpty(req.getFilters())) {
            if (filters == null) filters = Lists.newArrayList();
            filters.addAll(req.getFilters());
        }
        this.jdbcSql = req.getJdbcSql();
        this.limit = req.getLimit();
    }

    public <T> void putEq(String name, T value) {
        if (eqConditions == null) eqConditions = Maps.newHashMap();
        eqConditions.put(name, value);
    }

    public <T> void putIn(String name, T value) {
        if (inConditions == null) inConditions = Maps.newHashMap();
        if (value instanceof Collection) {
            List<Object> values = Lists.newArrayList();
            values.addAll((Collection<?>) value);
            inConditions.put(name,values);
        } else {
            inConditions.put(name, List.of(value));
        }
    }

    public boolean isCircular() {
        return !parent.equals(name) && parents.contains(name);
    }

    public <T> void put(String name, T value) {
        if (data == null) data = Maps.newHashMap();
        data.put(name, value);
    }

    public <T> T get(String name, T value) {
        if (MapUtils.isEmpty(data)) return value;
        return (T)data.getOrDefault(name, value);
    }

    public <T> T get(String name) {
        return (T)data.getOrDefault(name, null);
    }
}
