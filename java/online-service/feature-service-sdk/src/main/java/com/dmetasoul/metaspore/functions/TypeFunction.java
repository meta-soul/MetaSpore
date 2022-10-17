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
package com.dmetasoul.metaspore.functions;

import com.dmetasoul.metaspore.annotation.FeatureAnnotation;
import com.dmetasoul.metaspore.configure.TableInfo;
import com.dmetasoul.metaspore.configure.FieldAction;
import com.dmetasoul.metaspore.configure.FieldInfo;
import com.dmetasoul.metaspore.data.TableData;
import com.dmetasoul.metaspore.enums.DataTypeEnum;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.Validate;

import java.util.Map;
import java.util.concurrent.ExecutorService;

import static com.dmetasoul.metaspore.common.ConvTools.*;

@Slf4j
@FeatureAnnotation("typeTransform")
public class TypeFunction implements Function {
    @Override
    public boolean process(@NonNull TableData fieldTableData, @NonNull FieldAction config, @NonNull ExecutorService taskPool) {
        Map<String, Object> options = config.getOptions();
        if (CollectionUtils.isNotEmpty(config.getNames())) {
            Validate.isTrue(config.getInputFields() != null && config.getInputFields().size() == config.getNames().size(),
                    "input and output must same size");
            for (int i = 0; i < config.getNames().size(); ++i) {
                processField(fieldTableData, config.getInputFields().get(i), config.getNames().get(i), config.getTypes().get(i));
            }
        }
        return true;
    }

    private void processField(TableData fieldTableData, FieldInfo input, String name, Object type) {
        FieldInfo output = new FieldInfo(name);
        DataTypeEnum outType = TableInfo.getType(type);
        for (int i = 0; i < fieldTableData.getData().size(); ++i) {
            Map<FieldInfo, Object> item = fieldTableData.getData().get(i);
            Object value = item.get(input);
            Object data = null;
            if (outType.equals(fieldTableData.getDataSchema().get(input))) {
                data = value;
            } else if (outType.equals(DataTypeEnum.STRING)) {
                data = parseString(value);
            } else if (outType.equals(DataTypeEnum.LONG)) {
                data = parseLong(value);
            } else if (outType.equals(DataTypeEnum.INT)) {
                data = parseInteger(value);
            } else if (outType.equals(DataTypeEnum.DOUBLE)) {
                data = parseDouble(value);
            } else if (outType.equals(DataTypeEnum.BOOL)) {
                data = parseBoolean(value);
            } else if (outType.equals(DataTypeEnum.DATE)) {
                data = parseLocalDateTime(value);
            } else if (outType.equals(DataTypeEnum.TIMESTAMP)) {
                data = parseTimestamp(value);
            } else if (outType.equals(DataTypeEnum.DECIMAL)) {
                data = parseBigDecimal(value);
            } else if (outType.equals(DataTypeEnum.FLOAT)) {
                data = parseFloat(value);
            } else if (outType.equals(DataTypeEnum.TIME)) {
                data = parseLocalTime(value);
            }
            if (value != null && data == null) {
                log.error("typeTransform type not match, transform fail, output null at inType: {}, outType: {}, value: {}", fieldTableData.getDataSchema().get(input), outType, value);
            }
            fieldTableData.setValue(i, name, data);
        }
    }
}
