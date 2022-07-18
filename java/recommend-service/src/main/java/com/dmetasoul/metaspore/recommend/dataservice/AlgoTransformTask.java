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
import com.dmetasoul.metaspore.recommend.configure.FeatureConfig;
import com.dmetasoul.metaspore.recommend.configure.RecommendConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.dmetasoul.metaspore.recommend.enums.ResultTypeEnum;
import com.dmetasoul.metaspore.recommend.enums.TaskStatusEnum;
import com.dmetasoul.metaspore.recommend.functions.Function;
import com.dmetasoul.metaspore.serving.ArrowAllocator;
import com.dmetasoul.metaspore.serving.FeatureTable;
import com.google.common.collect.Lists;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import java.util.List;
import java.util.Map;
import java.util.concurrent.*;

@SuppressWarnings("rawtypes")
@Data
@Slf4j
@DataServiceAnnotation("AlgoTransform")
public class AlgoTransformTask extends DataService {

    private Map<String, Function> functionMap;

    @Autowired
    private ExecutorService taskPool;

    private List<Field> algoFields;

    @Override
    public boolean initService() {
        FeatureConfig.AlgoTransform algoTransform = taskFlowConfig.getAlgoTransforms().get(name);
        algoFields = Lists.newArrayList();
        for (FeatureConfig.FieldAction fieldAction: algoTransform.getFieldActions()) {
            DataTypes.DataType dataType = DataTypes.getDataType(fieldAction.getType());
            algoFields.add(Field.nullablePrimitive(fieldAction.getName(), (ArrowType.PrimitiveType) dataType.getType()));
        }
        functionMap = taskServiceRegister.getFunctions();
        RecommendConfig.Chain chain = new RecommendConfig.Chain();
        List<String> depends = List.of(algoTransform.getDepend());
        chain.setThen(depends);
        chains.add(chain);
        this.taskPool = Executors.newFixedThreadPool(2);  //default
        return true;
    }

    @Override
    public DataResult process(ServiceRequest request, DataContext context) {
        DataResult dataResult;
        FeatureConfig.AlgoTransform algoTransform = taskFlowConfig.getAlgoTransforms().get(name);
        String table = algoTransform.getDepend();
        FeatureConfig.Feature feature = taskFlowConfig.getFeatures().get(name);
        if (context.getStatus(table) != TaskStatusEnum.SUCCESS) {
            log.error("depend ：{} result get fail！", table);
            return null;
        }
        DataResult result = context.getResult(name);
        if (result == null || result.getResultType() != ResultTypeEnum.FEATUREARRAYS) {
            log.error("depend ：{} result get wrong！", table);
            return null;
        }
        FeatureTable featureTable = new FeatureTable(name, algoFields, ArrowAllocator.getAllocator());
        DataResult.FeatureArray featureArray = result.getFeatureArray();
        dataResult = new DataResult();
        for (FeatureConfig.FieldAction fieldAction: algoTransform.getFieldActions()) {
            List<String> fields = fieldAction.getFields();
            if (StringUtils.isEmpty(fieldAction.getFunc())) {
                setFieldData(featureTable, fieldAction.getName(), fieldAction.getType(), featureArray.get(fields.get(0)));
                continue;
            }
            List<Object> fieldValues =  Lists.newArrayList();
            List<String> fieldTypes = Lists.newArrayList();
            for (String field: fieldAction.getFields()) {
                fieldValues.add(featureArray.get(field));
                fieldTypes.add(feature.getColumnMap().get(field));
            }
            Function function = functionMap.get(fieldAction.getFunc());
            if (function == null) {
                log.error("function：{} get fail！", fieldAction.getFunc());
                return null;
            }
            Object res = function.process(fieldValues, fieldTypes, fieldAction.getOptions());
            setFieldData(featureTable, fieldAction.getName(), fieldAction.getType(), res);
        }
        dataResult.setFeatureTable(featureTable);
        return dataResult;
    }

    public void setFieldData(FeatureTable featureTable, String col, String type, Object data) {
    }
}
