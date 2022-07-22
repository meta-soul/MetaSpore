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
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;

import java.util.List;
import java.util.Map;

@SuppressWarnings("rawtypes")
@Data
@Slf4j
public class Scene {
    private String name;
    private RecommendConfig.Scene scene;
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
        List<Map> data = null;
        dataResult.setData(data);
        return dataResult;
    }
}
