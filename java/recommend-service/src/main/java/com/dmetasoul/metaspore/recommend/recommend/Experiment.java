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

import com.dmetasoul.metaspore.recommend.TaskServiceRegister;
import com.dmetasoul.metaspore.recommend.configure.Chain;
import com.dmetasoul.metaspore.recommend.configure.RecommendConfig;
import com.dmetasoul.metaspore.recommend.configure.TaskFlowConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;

import java.util.*;

@SuppressWarnings("rawtypes")
@Data
@Slf4j
public class Experiment {
    private String name;
    protected TaskServiceRegister serviceRegister;

    protected TaskFlowConfig taskFlowConfig;
    private RecommendConfig.Experiment experiment;

    protected List<Chain> chains;

    public void init(String name, TaskFlowConfig taskFlowConfig, TaskServiceRegister serviceRegister) {
        if (StringUtils.isEmpty(name)) {
            log.error("name is null, init fail!");
        }
        this.name = name;
        this.taskFlowConfig = taskFlowConfig;
        this.serviceRegister = serviceRegister;
        experiment = taskFlowConfig.getExperiments().get(name);
        chains = experiment.getChains();
    }

    public DataResult getDataResultByName(String name, DataContext context) {
        DataResult result = context.getResult(name);
        if (result == null || !result.isVaild()) {
            log.error("name ：{} result get wrong！", name);
            return null;
        }
        return result;
    }

    public List<DataResult> getDataResultByNames(List<String> names, DataContext context) {
        List<DataResult> dataResults = Lists.newArrayList();
        for (String name : names) {
            DataResult result = getDataResultByName(name, context);
            if (result != null) {
                dataResults.add(result);
            }
        }
        return dataResults;
    }
    public List<Map> getDataByColumns(DataResult dataResult, List<String> columnNames) {
        List<Map> data = Lists.newArrayList();
        if (MapUtils.isNotEmpty(dataResult.getValues())) {
            Map<String, Object> map = Maps.newHashMap();
            for (String col : columnNames) {
                Object value = dataResult.getValues().get(col);
                map.put(col, value);
            }
            data.add(map);
        } else if (CollectionUtils.isNotEmpty(dataResult.getData())) {
            for (Map item : dataResult.getData()) {
                Map<String, Object> map = Maps.newHashMap();
                for (String col : columnNames) {
                    Object value = item.get(col);
                    map.put(col, value);
                }
                data.add(map);
            }
        } else if (dataResult.getFeatureArray() != null) {
            DataResult.FeatureArray featureArray = dataResult.getFeatureArray();
            for (int index = 0; index < featureArray.getMaxIndex(); ++index) {
                Map<String, Object> map = Maps.newHashMap();
                for (String col : columnNames) {
                    Object value = featureArray.get(col, index);
                    map.put(col, value);
                }
                data.add(map);
            }
        }
        return data;
    }

    public List<Map> getTaskResultByColumns(List<String> taskNames, boolean isAny, List<String> columnNames, DataContext context) {
        List<DataResult> dataResults = getDataResultByNames(taskNames, context);
        List<Map> data = Lists.newArrayList();
        for (DataResult dataResult : dataResults) {
            data.addAll(getDataByColumns(dataResult, columnNames));
        }
        return data;
    }

    public DataResult process(ServiceRequest request, DataContext context) {
        DataResult dataResult = null;
        Chain chain = chains.get(chains.size() - 1);
        List<String> outputs = chain.getWhen();
        boolean isAny = false;
        if (CollectionUtils.isEmpty(outputs)) {
            if (CollectionUtils.isEmpty(chain.getThen())) {
                return dataResult;
            }
            int lastIndex = chain.getThen().size() - 1;
            outputs = List.of(chain.getThen().get(lastIndex));
        } else {
            isAny = chain.isAny();
        }
        dataResult = new DataResult();
        String cutField = (String) request.get("cutField");
        int maxReservation = request.getLimit();
        if (MapUtils.isNotEmpty(experiment.getOptions()) && experiment.getOptions().containsKey("cutField")) {
            cutField = (String) experiment.getOptions().get("cutField");
            maxReservation = (int) experiment.getOptions().get("maxReservation");
        }
        List<String> columns = Lists.newArrayList();
        columns.addAll(experiment.getColumnNames());
        if (StringUtils.isNotEmpty(cutField) && !experiment.getColumnMap().containsKey(cutField)) {
            columns.add(cutField);
        }
        List<Map> data = getTaskResultByColumns(outputs, isAny, columns, context);
        if (data == null) {
            log.error("experiment:{} task:{} get result fail!", name, outputs);
            throw new RuntimeException("experiment exec fail!");
        }
        if (StringUtils.isNotEmpty(cutField)) {
            for (Map map : data) {
                Object value = map.get(cutField);
                if (value != null && !Comparable.class.isAssignableFrom(value.getClass())) {
                    log.error("cutField ：{} need comparable！", cutField);
                    return null;
                }
            }
            if (maxReservation > 0 && maxReservation < data.size()) {
                String finalCutField = cutField;
                Collections.sort(data, (map1, map2) -> {
                    Object o1 = map1.get(finalCutField);
                    Object o2 = map2.get(finalCutField);
                    if (o1 == null && o2 == null) return 0;
                    else if (o1 == null) return 1;
                    else if (o2 == null) return -1;
                    return ((Comparable) o2).compareTo(o1);
                });
            }
        }
        if (maxReservation > 0 && maxReservation < data.size()) {
            data = data.subList(0, maxReservation);
        }
        dataResult.setData(data);
        return dataResult;
    }
}
