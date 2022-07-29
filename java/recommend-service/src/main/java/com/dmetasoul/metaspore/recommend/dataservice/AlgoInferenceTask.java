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

import com.dmetasoul.metaspore.recommend.annotation.DataServiceAnnotation;
import com.dmetasoul.metaspore.recommend.common.DataTypes;
import com.dmetasoul.metaspore.recommend.configure.Chain;
import com.dmetasoul.metaspore.recommend.configure.FeatureConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.dmetasoul.metaspore.recommend.functions.Function;
import com.dmetasoul.metaspore.serving.ArrowAllocator;
import com.dmetasoul.metaspore.serving.FeatureTable;
import com.google.common.collect.Lists;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.commons.lang3.StringUtils;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;

@SuppressWarnings("rawtypes")
@Data
@Slf4j
@DataServiceAnnotation
public class AlgoInferenceTask extends DataService {
    private ExecutorService taskPool;
    private FeatureConfig.AlgoInference algoInference;
    private FeatureConfig.Feature feature;
    private Map<String, Function> functionMap;
    private List<Field> algoFields;

    @Override
    public boolean initService() {
        algoInference = taskFlowConfig.getAlgoInferences().get(name);
        feature = taskFlowConfig.getFeatures().get(algoInference.getDepend());
        algoFields = Lists.newArrayList();
        for (FeatureConfig.FieldAction fieldAction: algoInference.getFieldActions()) {
            DataTypeEnum dataType = DataTypes.getDataType(fieldAction.getType());
            algoFields.add(Field.nullable(fieldAction.getName(), dataType.getType()));
        }
        Chain chain = new Chain();
        List<String> depends = List.of(algoInference.getDepend());
        chain.setThen(depends);
        taskFlow.offer(chain);
        this.taskPool = taskServiceRegister.getTaskPool();
        functionMap = taskServiceRegister.getFunctions();
        return true;
    }

    @Override
    public DataResult process(ServiceRequest request, DataContext context) {
        DataResult dataResult = new DataResult();
        DataResult result = getDataResultByName(algoInference.getDepend(), context);
        FeatureTable featureTable = new FeatureTable(name, algoFields, ArrowAllocator.getAllocator());
        DataResult.FeatureArray featureArray = result.getFeatureArray();
        for (FeatureConfig.FieldAction fieldAction: algoInference.getFieldActions()) {
            List<String> fields = fieldAction.getFields();
            if (StringUtils.isEmpty(fieldAction.getFunc())) {
                setFieldData(featureTable, fieldAction.getName(), fieldAction.getType(), featureArray.getArray(fields.get(0)));
                continue;
            }
            List<List<Object>> fieldValues =  Lists.newArrayList();
            List<String> fieldTypes = Lists.newArrayList();
            for (String field: fieldAction.getFields()) {
                fieldValues.add(featureArray.getArray(field));
                fieldTypes.add(feature.getColumnMap().get(field));
            }
            Function function = functionMap.get(fieldAction.getFunc());
            if (function == null) {
                log.error("function：{} get fail！", fieldAction.getFunc());
                return null;
            }
            List<Object> res = function.process(fieldValues, fieldTypes, fieldAction.getOptions());
            setFieldData(featureTable, fieldAction.getName(), fieldAction.getType(), res);
        }
        dataResult.setFeatureTable(featureTable);
        return dataResult;
    }

    public void setFieldData(FeatureTable featureTable, String col, String type, List<Object> data) {
        DataTypeEnum dataType = DataTypes.getDataType(type);
        if (!dataType.set(featureTable, col, data)) {
            log.error("set featuraTable fail!");
        }
    }
}
