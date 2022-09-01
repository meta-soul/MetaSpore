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
package com.dmetasoul.metaspore.recommend.functions;

import com.dmetasoul.metaspore.recommend.annotation.FunctionAnnotation;
import com.dmetasoul.metaspore.recommend.data.FieldData;
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.springframework.util.Assert;

import javax.validation.constraints.NotEmpty;
import java.util.List;
import java.util.Map;

import static com.dmetasoul.metaspore.recommend.common.ConvTools.*;
@Slf4j
@FunctionAnnotation("typeTransform")
public class TypeFunction implements Function{
    @Override
    public boolean process(@NotEmpty List<FieldData> fields, @NotEmpty List<FieldData> result, Map<String, Object> options) {
        FieldData input = fields.get(0);
        FieldData output = result.get(0);
        DataTypeEnum outType = output.getType();
        Assert.notNull(outType, "output type should be set!");
        for (int i = 0; i < input.getValue().size(); ++i) {
            Object value = input.getValue().get(i);
            Object data = null;
            if (outType.equals(input.getType())) {
                data = value;
            }else if (outType.equals(DataTypeEnum.STRING)) {
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
                log.error("typeTransform type not match, transform fail, output null at inType: {}, outType: {}, value: {}", input.getType(), outType, value);
            }
            output.addIndexData(FieldData.create(i, data));
        }
        return true;
    }
}
