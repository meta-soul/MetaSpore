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
import com.dmetasoul.metaspore.recommend.annotation.ServiceAnnotation;
import com.dmetasoul.metaspore.recommend.common.DataTypes;
import com.dmetasoul.metaspore.recommend.common.Utils;
import com.dmetasoul.metaspore.recommend.configure.RecommendConfig;
import com.dmetasoul.metaspore.recommend.configure.TaskFlowConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.dmetasoul.metaspore.serving.ArrowAllocator;
import com.dmetasoul.metaspore.serving.FeatureTable;
import com.google.common.collect.Lists;
import lombok.Data;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.vector.types.pojo.Field;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

@Data
@Slf4j
@ServiceAnnotation
public class Scene extends TaskFlow<Layer> {
    private RecommendConfig.Scene scene;
    protected List<Field> resFields;
    protected List<DataTypeEnum> dataTypes;
    protected boolean isDup = true;
    public void init(String name, TaskFlowConfig taskFlowConfig, TaskServiceRegister serviceRegister) {
        super.init(name, taskFlowConfig, serviceRegister);
        scene = taskFlowConfig.getScenes().get(name);
        chains = scene.getChains();
        timeout = Utils.getField(scene.getOptions(), "timeout", timeout);
        resFields = Lists.newArrayList();
        dataTypes = Lists.newArrayList();
        for (String col: scene.getColumnNames()) {
            String type = scene.getColumnMap().get(col);
            DataTypeEnum dataType = DataTypes.getDataType(type);
            resFields.add(new Field(col, dataType.getType(), dataType.getChildFields()));
            dataTypes.add(dataType);
        }
        isDup = Utils.getField(scene.getOptions(), "dupOnMerge", true);
    }

    @SneakyThrows
    public DataResult process(DataContext context) {
        CompletableFuture<DataResult> future = execute(List.of(), serviceRegister.getLayerMap(), context).thenApplyAsync(dataResults -> {
            DataResult result = new DataResult();
            FeatureTable featureTable = new FeatureTable(name, resFields, ArrowAllocator.getAllocator());
            result.setFeatureTable(featureTable);
            result.mergeDataResult(dataResults, scene.getColumnMap(), isDup);
            return result;
        }, taskPool);
        return future.get(timeout, timeUnit);
    }

    public List<Map<String, Object>> output(DataContext context) {
        DataResult result = process(context);
        if (result == null || result.isNull()) return Lists.newArrayList();
        return result.output(scene.getColumnNames());
    }
}
