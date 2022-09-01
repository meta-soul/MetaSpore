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
import org.apache.commons.collections4.CollectionUtils;
import org.springframework.util.Assert;

import javax.validation.constraints.NotEmpty;
import java.util.List;
import java.util.Map;

import static com.dmetasoul.metaspore.recommend.common.ConvTools.*;

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
            if (outType.equals(DataTypeEnum.STRING)) {
                output.addIndexData(FieldData.create(i, parseString(value)));
            }
            if (outType.equals(DataTypeEnum.LONG)) {
                output.addIndexData(FieldData.create(i, parseLong(value)));
            }
            if (outType.equals(DataTypeEnum.INT)) {
                output.addIndexData(FieldData.create(i, parseInteger(value)));
            }
            if (outType.equals(DataTypeEnum.DOUBLE)) {
                output.addIndexData(FieldData.create(i, parseDouble(value)));
            }
            if (outType.equals(DataTypeEnum.BOOL)) {
                output.addIndexData(FieldData.create(i, parseBoolean(value)));
            }
            if (outType.equals(DataTypeEnum.DATE)) {
                output.addIndexData(FieldData.create(i, parseLocalDateTime(value)));
            }
            if (outType.equals(DataTypeEnum.TIMESTAMP)) {
                output.addIndexData(FieldData.create(i, parseTimestamp(value)));
            }
            if (outType.equals(DataTypeEnum.DECIMAL)) {
                output.addIndexData(FieldData.create(i, parseBigDecimal(value)));
            }
            if (outType.equals(DataTypeEnum.FLOAT)) {
                output.addIndexData(FieldData.create(i, parseFloat(value)));
            }
            if (outType.equals(DataTypeEnum.TIME)) {
                output.addIndexData(FieldData.create(i, parseLocalTime(value)));
            }
        }
        return true;
    }
}
