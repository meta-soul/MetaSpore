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
import com.dmetasoul.metaspore.recommend.common.CommonUtils;
import com.dmetasoul.metaspore.recommend.configure.FieldAction;
import com.dmetasoul.metaspore.recommend.data.FieldData;
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.springframework.util.Assert;

import javax.validation.constraints.NotEmpty;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutorService;

import static com.dmetasoul.metaspore.recommend.common.ConvTools.*;

@Slf4j
@FunctionAnnotation("randomGenerator")
public class RandomGeneratorFunction implements Function {
    @Override
    public boolean process(List<FieldData> fields, @NotEmpty List<FieldData> result,
                           @NonNull FieldAction config, @NonNull ExecutorService taskPool) {
        Map<String, Object> options = config.getOptions();
        if (CollectionUtils.isNotEmpty(result)) {
            FieldData output = result.get(0);
            DataTypeEnum outType = output.getType();
            Object data = null;
            Random random = new Random();
            if (outType.equals(DataTypeEnum.LONG)) {
                data = random.nextLong();
            } else if (outType.equals(DataTypeEnum.INT)) {
                int bound = CommonUtils.getField(options, "bound", 0, Integer.class);
                if (bound > 0) {
                    data = random.nextInt(bound);
                } else {
                    data = random.nextInt();
                }
            } else if (outType.equals(DataTypeEnum.DOUBLE)) {
                data = random.nextDouble();
            } else if (outType.equals(DataTypeEnum.FLOAT)) {
                data = random.nextFloat();
            }
            if (data == null) {
                log.error("RandomGenerator only support long, int, double, float but {} ", outType);
            }
            output.addIndexData(FieldData.create(0, data));
        }
        return true;
    }
}
