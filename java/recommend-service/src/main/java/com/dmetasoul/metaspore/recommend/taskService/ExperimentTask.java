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
package com.dmetasoul.metaspore.recommend.taskService;

import com.dmetasoul.metaspore.recommend.annotation.DataServiceAnnotation;
import com.dmetasoul.metaspore.recommend.configure.RecommendConfig;
import com.dmetasoul.metaspore.recommend.configure.TaskFlowConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.dmetasoul.metaspore.recommend.dataservice.DataService;
import com.dmetasoul.metaspore.recommend.enums.TaskStatusEnum;
import com.google.common.collect.Lists;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.*;

@SuppressWarnings("rawtypes")
@Data
@Slf4j
@DataServiceAnnotation("Experiment")
public class ExperimentTask extends TaskService {
    private RecommendConfig.Experiment experiment;

    @Override
    public boolean initService() {
        experiment = taskFlowConfig.getExperiments().get(name);
        chains = experiment.getChains();
        return true;
    }

    // last chain set output, if last chain's when is not empty, use when; else use then.get(last_index)
    @Override
    public DataResult process(ServiceRequest request, DataContext context) {
        DataResult dataResult = null;
        RecommendConfig.Chain chain = chains.get(chains.size() - 1);
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
            context.setStatus(name, TaskStatusEnum.EXEC_FAIL);
            return null;
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
