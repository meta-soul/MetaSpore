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

import com.dmetasoul.metaspore.recommend.baseservice.TaskServiceRegister;
import com.dmetasoul.metaspore.recommend.annotation.ServiceAnnotation;
import com.dmetasoul.metaspore.recommend.common.CommonUtils;
import com.dmetasoul.metaspore.recommend.configure.RecommendConfig;
import com.dmetasoul.metaspore.recommend.configure.TaskFlowConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.google.common.collect.Maps;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.commons.collections4.CollectionUtils;
import org.springframework.util.Assert;

import java.util.*;
import java.util.concurrent.CompletableFuture;

@Data
@Slf4j
@ServiceAnnotation("Experiment")
public class Experiment extends TaskFlow<Service> {
    protected RecommendConfig.Experiment experiment;
    protected List<Field> resFields;
    protected List<DataTypeEnum> dataTypes;

    public void init(String name, TaskFlowConfig taskFlowConfig, TaskServiceRegister serviceRegister) {
        super.init(name, taskFlowConfig, serviceRegister);
        experiment = taskFlowConfig.getExperiments().get(name);
        chains = experiment.getChains();
        timeout = CommonUtils.getField(experiment.getOptions(), "timeout", timeout);
    }

    public CompletableFuture<List<DataResult>> process(List<DataResult> data, DataContext context) {
        return execute(data, serviceRegister.getRecommendServices(), List.of(), experiment.getOptions(), context);
    }

    public DataResult mergeRecall(List<DataResult> data, DataResult result, Map<String, Object> option) {
        String scoreField = CommonUtils.getField(option, "score", null);
        String scoreInfoField = CommonUtils.getField(option, "scoreInfo", null);
        Assert.notNull(scoreField, "score");
        return result;
    }

    @SuppressWarnings("unchecked")
    @Override
    public void initFunctions() {
        registerMergeOperator("maxScore", (field, data, option) ->{
            if (field instanceof Number && data instanceof Number) {
                Number val1 = (Number) field;
                Number val2 = (Number) data;
                if (val1.doubleValue() < val2.doubleValue()) {
                    return data;
                }
            }
            return field;
        });
        registerMergeOperator("mergeScoreInfo", (field, data, option) ->{
            if (field instanceof Map && data instanceof Map) {
                Map<String, Object> val1 = (Map<String, Object>) field;
                Map<String, Object> val2 = (Map<String, Object>) data;
                val1.putAll(val2);
                return val1;
            }
            return field;
        });
        registerUpdateOperator("putOriginScores", (input, output, option) ->{
            Assert.isTrue(input.size() == 2 && input.get(1) instanceof Map, "originScores is map");
            Assert.isTrue(CollectionUtils.isNotEmpty(output), "output is not empty");
            String label = CommonUtils.getField(option, "label", name);
            Map<String, Object> res = Maps.newHashMap();
            Map<String, Object> map = (Map<String, Object>) input.get(1);
            map.put(label, input.get(0));
            res.put(output.get(0), map);
            return res;
        });
    }
}
