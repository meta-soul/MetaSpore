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
package com.dmetasoul.metaspore.recommend.dataservice;

import com.dmetasoul.metaspore.recommend.common.Utils;
import com.dmetasoul.metaspore.recommend.configure.FeatureConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.dmetasoul.metaspore.recommend.functions.Function;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;

import java.util.Map;
import java.util.concurrent.ExecutorService;

@SuppressWarnings("rawtypes")
@Data
@Slf4j
public abstract class AlgoTransformTask extends DataService {

    private ExecutorService taskPool;

    protected FeatureConfig.AlgoTransform config;

    private Map<String, Function> functionMap;

    @Override
    public boolean initService() {
        config = taskFlowConfig.getAlgoTransforms().get(name);
        taskFlow.offer(config.getDepend());
        return initTask();
    }

    public abstract boolean initTask();

    public <T> T getOptionOrDefault(String key, T value) {
        return Utils.getField(config.getOptions(), key, value);
    }

    @Override
    public DataResult process(ServiceRequest request, DataContext context) {
        return null;
    }
}
