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
import com.dmetasoul.metaspore.recommend.configure.FieldInfo;
import com.dmetasoul.metaspore.recommend.data.TableData;
import com.google.common.collect.Lists;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;

@Slf4j
@FunctionAnnotation("concatField")
public class ConcatFunction implements Function {
    @Override
    public boolean process(@NonNull TableData fieldTableData,
                           @NonNull FieldAction config, @NonNull ExecutorService taskPool) {
        Map<String, Object> options = config.getOptions();
        String joinStr = CommonUtils.getField(config.getOptions(), "join", "#");
        if (CollectionUtils.isNotEmpty(config.getNames())) {
            List<Object> result = Lists.newArrayList();
            for (int i = 0; i < fieldTableData.getData().size(); ++i) {
                List<Object> data = Lists.newArrayList();
                if (CollectionUtils.isNotEmpty(config.getInputFields())) {
                    for (FieldInfo fieldInfo : config.getInputFields()) {
                        data.add(fieldTableData.getValue(i, fieldInfo));
                    }
                }
                result.add(StringUtils.join(data, joinStr));
            }
            fieldTableData.addValueList(config.getNames().get(0), result);
        }
        return true;
    }
}
