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
import com.dmetasoul.metaspore.recommend.common.Utils;
import com.dmetasoul.metaspore.recommend.configure.RecommendConfig;
import com.dmetasoul.metaspore.recommend.configure.TaskFlowConfig;
import com.dmetasoul.metaspore.recommend.configure.TransformConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.google.common.collect.Lists;
import lombok.Data;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.springframework.util.StopWatch;

import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;

@Data
@Slf4j
@ServiceAnnotation("Scene")
public class Scene extends TaskFlow<Layer> {
    private RecommendConfig.Scene scene;

    public void init(String name, TaskFlowConfig taskFlowConfig, TaskServiceRegister serviceRegister) {
        super.init(name, taskFlowConfig, serviceRegister);
        scene = taskFlowConfig.getScenes().get(name);
        chains = scene.getChains();
        timeout = CommonUtils.getField(scene.getOptions(), "timeout", timeout);
        resFields = Lists.newArrayList();
        dataTypes = Lists.newArrayList();
        for (String col : scene.getColumnNames()) {
            resFields.add(scene.getFieldMap().get(col));
            dataTypes.add(scene.getColumnMap().get(col));
        }
    }

    @SneakyThrows
    public DataResult process(DataContext context) {
        StopWatch timeRecorder = new StopWatch(UUID.randomUUID().toString());
        try {
            timeRecorder.start(String.format("scene_%s_process", name));
            TransformConfig transformConfig = new TransformConfig();
            transformConfig.setName("summaryBySchema");
            CompletableFuture<DataResult> future = execute(List.of(),
                    serviceRegister.getLayerMap(), List.of(transformConfig), scene.getOptions(),
                    context).thenApplyAsync(dataResults -> {
                if (CollectionUtils.isEmpty(dataResults)) return null;
                return dataResults.get(0);
            }, taskPool);
            return future.get(timeout, timeUnit);
        } finally {
            timeRecorder.stop();
            context.updateTimeRecords(Utils.getTimeRecords(timeRecorder));
        }
    }

    public List<Map<String, Object>> output(DataContext context) {
        try (DataResult result = process(context)) {
            if (result == null || result.isNull()) return Lists.newArrayList();
            return result.output(scene.getColumnNames());
        }
    }

    @Override
    public void initFunctions() {

    }
}
