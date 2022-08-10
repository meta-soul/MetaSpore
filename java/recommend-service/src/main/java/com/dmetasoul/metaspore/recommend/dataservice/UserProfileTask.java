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

import com.dmetasoul.metaspore.recommend.annotation.ServiceAnnotation;
import com.dmetasoul.metaspore.recommend.common.Utils;
import com.dmetasoul.metaspore.recommend.data.FieldData;
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.dmetasoul.metaspore.recommend.functions.FlatFunction;
import com.google.common.collect.Lists;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.springframework.util.Assert;

import java.util.Collection;
import java.util.List;

@Data
@Slf4j
@ServiceAnnotation("UserProfile")
public class UserProfileTask extends AlgoTransformTask {
    private double alpha;
    private String colRecentItemIds;
    private static String splitor = "\u0001";

    @Override
    public boolean initTask() {
        alpha = getOptionOrDefault("alpha", 1.0);
        splitor = getOptionOrDefault("splitor", splitor);
        return true;
    }
    @Override
    public void addFunctions() {
        addFunction("splitRecentIds", (fields, options) -> {
            Assert.isTrue(CollectionUtils.isNotEmpty(fields) && fields.size() == 1, "input values size must eq 1");
            FieldData fieldData = fields.get(0);
            Assert.isTrue(fieldData.isMatch(DataTypeEnum.STRING), "split input must string!");
            String split = Utils.getField(options, "splitor", splitor);
            List<Object> input = fieldData.getValue();
            List<Object> res = Lists.newArrayList();
            for (Object o : input) {
                Assert.isTrue(o instanceof String, "value must string!");
                String value = (String) o;
                res.add(List.of(value.split(split)));
            }
            return res;
        });
        addFunction("recentWeight", (FlatFunction) (indexs, fields, options) -> {
            Assert.isTrue(CollectionUtils.isNotEmpty(fields) && indexs != null, "input data is not null");
            List<Object> res = Lists.newArrayList();
            List<Object> input = fields.get(0).getValue();
            int num = 0;
            for (int i = 0; i < input.size(); ++i) {
                Object item = input.get(i);
                Assert.isInstanceOf(Collection.class, item);
                Collection<?> list = (Collection<?>) item;
                for (Object o : list) {
                    num += 1;
                    indexs.add(i);
                    res.add(1 / (1 + Math.pow((list.size() - i - 1), alpha)));
                }
            }
            return res;
        });
    }
}
