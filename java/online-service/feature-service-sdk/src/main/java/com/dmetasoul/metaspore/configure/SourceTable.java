package com.dmetasoul.metaspore.configure;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;

import java.util.List;
import java.util.Map;

@Slf4j
@Data
public class SourceTable extends TableInfo {
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