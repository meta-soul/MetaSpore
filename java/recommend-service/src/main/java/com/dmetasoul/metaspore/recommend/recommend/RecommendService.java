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
package com.dmetasoul.metaspore.recommend.recommend;

import com.dmetasoul.metaspore.recommend.configure.RecommendConfig;
import com.dmetasoul.metaspore.recommend.configure.TaskFlowConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.dmetasoul.metaspore.recommend.taskService.ServiceTask;
import com.google.common.collect.Lists;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;

import java.util.List;
import java.util.Map;

@Slf4j
@Data
public abstract class RecommendService {
    protected String name;
    protected TaskFlowConfig taskFlowConfig;
    protected RecommendConfig.Service serviceConfig;
    protected ServiceTask service;

    public boolean init(String name, TaskFlowConfig taskFlowConfig, ServiceTask service) {
        if (StringUtils.isEmpty(name)) {
            log.error("name is null, init fail!");
            return false;
        }
        this.name = name;
        this.service = service;
        this.taskFlowConfig = taskFlowConfig;
        this.serviceConfig = taskFlowConfig.getServices().get(name);
        return initService();
    }

    public <T> T getOptionOrDefault(String key, T value) {
        Map<String, Object> map = serviceConfig.getOptions();
        if (MapUtils.isNotEmpty(map) && map.containsKey(key)) {
            return (T) map.getOrDefault(key, value);
        }
        return value;
    }

    public Object getField(Map map, String key) {
        if (MapUtils.isNotEmpty(map) && map.containsKey(key)) {
            return map.get(key);
        }
        return null;
    }

    public <T> T getField(Map map, String key, T value) {
        if (MapUtils.isNotEmpty(map) && map.containsKey(key)) {
            return (T) map.get(key);
        }
        return value;
    }

    public boolean setFieldFail(Map map, int index, Object value) {
        if (index < 0 || index >= serviceConfig.getColumnNames().size()) {
            return true;
        }
        map.put(serviceConfig.getColumnNames().get(index), value);
        return false;
    }

    public boolean isInvalidDepend(String depend) {
        if (getDependKey(depend, 0) == null) {
            return false;
        }
        return true;
    }

    public String getDependKey(String depend, int index) {
        if (index < 0) return null;
        if (taskFlowConfig.getSourceTables().containsKey(depend)
            && taskFlowConfig.getSourceTables().get(depend).getColumnNames().size() > index) {
            return taskFlowConfig.getSourceTables().get(depend).getColumnNames().get(index);
        }
        if (taskFlowConfig.getFeatures().containsKey(depend)
                && taskFlowConfig.getFeatures().get(depend).getColumnNames().size() > index) {
            return taskFlowConfig.getFeatures().get(depend).getColumnNames().get(index);
        }
        if (taskFlowConfig.getAlgoTransforms().containsKey(depend)
                && taskFlowConfig.getAlgoTransforms().get(depend).getFieldActions().size() > index) {
            return taskFlowConfig.getAlgoTransforms().get(depend).getFieldActions().get(index).getName();
        }
        if (taskFlowConfig.getServices().containsKey(depend)
                && taskFlowConfig.getServices().get(depend).getColumnNames().size() > index) {
            return taskFlowConfig.getServices().get(depend).getColumnNames().get(index);
        }
        if (taskFlowConfig.getChains().containsKey(depend)
                && taskFlowConfig.getChains().get(depend).getColumnNames().size() > index) {
            return taskFlowConfig.getChains().get(depend).getColumnNames().get(index);
        }
        return null;
    }

    public String getFieldType(String key) {
        return serviceConfig.getColumnMap().get(key);
    }
    protected abstract boolean initService();

    public void fillRequest(String depend, ServiceRequest request) {}

    public List<Map> getListData(List<DataResult> dataResults) {
        List<Map> result = Lists.newArrayList();
        for (DataResult dataResult: dataResults) {
            if (MapUtils.isNotEmpty(dataResult.getValues())) {
                result.add(dataResult.getValues());
            } else if (CollectionUtils.isNotEmpty(dataResult.getData())) {
                for (Map item : dataResult.getData()) {
                    result.add(item);
                }
            }
        }
        return result;
    }

    // default
    public DataResult process(ServiceRequest request, DataContext context) {
        List<DataResult> dataResults = Lists.newArrayList();
        for (String taskName : serviceConfig.getDepend()) {
            DataResult result = service.execute(taskName, request, context);
            if (result == null) {
                log.error("task:{} depend:{} exec fail status:{}!", name, taskName, context.getStatus(taskName).getName());
                continue;
            }
            if (serviceConfig.getDepend().size() == 1) {
                return process(request, result, context);
            }
            result.setName(taskName);
            dataResults.add(result);
        }
        return process(request, dataResults, context);
    }

    // default
    public DataResult process(ServiceRequest request, List<DataResult> dataResults, DataContext context) {
        return DataResult.merge(dataResults, name);
    }

    public DataResult process(ServiceRequest request, DataResult dataResult, DataContext context) {
        return dataResult;
    }
    public void close() {}
}
