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
import com.dmetasoul.metaspore.recommend.common.ConvTools;
import com.dmetasoul.metaspore.recommend.configure.FieldAction;
import com.dmetasoul.metaspore.recommend.data.FieldData;
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.google.common.collect.Lists;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.util.Strings;
import org.springframework.util.Assert;

import javax.validation.constraints.NotEmpty;
import java.util.List;
import java.util.Map;

import static com.dmetasoul.metaspore.recommend.common.ConvTools.*;

@Slf4j
@FunctionAnnotation("concatField")
public class ConcatFunction implements Function {
    @Override
    public boolean process(@NotEmpty List<FieldData> fields, @NotEmpty List<FieldData> result, @NonNull FieldAction config) {
        Map<String, Object> options = config.getOptions();
        Assert.isTrue(1 == result.size() && result.get(0).getType().equals(DataTypeEnum.STRING),
                "output must be one str!");
        String joinStr = CommonUtils.getField(config.getOptions(), "join", "#");
        FieldData output = result.get(0);
        List<List<String>> array = Lists.newArrayList();
        for (FieldData input : fields) {
            for (int i = 0; i < input.getValue().size(); ++i) {
                if (array.size() <= i) {
                    array.add(Lists.newArrayList());
                }
                Object value = input.getValue().get(i);
                String str = ConvTools.parseString(value);
                if (str == null) str = "";
                array.get(i).add(str);
            }
        }
        for (int i = 0; i < array.size(); ++i) {
            List<String> list = array.get(i);
            Object data = StringUtils.join(list, joinStr);
            output.addIndexData(FieldData.create(i, data));
        }
        return true;
    }
}
