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

import com.dmetasoul.metaspore.recommend.annotation.TransformFunction;
import com.dmetasoul.metaspore.recommend.common.Utils;
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.google.common.collect.Lists;
import org.apache.commons.collections4.CollectionUtils;
import org.springframework.util.Assert;

import java.util.List;
import java.util.Map;

@TransformFunction("split")
public class SplitFunction extends Function{
    private static final String SPLITOR = "\u0001";

    @Override
    public List<Object> process(List<List<Object>> values, List<DataTypeEnum> types, Map<String, Object> options) {
        Assert.isTrue(CollectionUtils.isNotEmpty(values) && values.size() == 1, "input values size must eq 1");
        Assert.isTrue(CollectionUtils.isNotEmpty(types) && types.get(0).equals(DataTypeEnum.STRING), "split input must string!");
        String splitor = Utils.getField(options, "splitor", SPLITOR);
        List<Object> input = values.get(0);
        List<Object> res = Lists.newArrayList();
        for (Object o : input) {
            Assert.isTrue(o instanceof String, "value must string!");
            String value = (String) o;
            res.add(List.of(value.split(splitor)));
        }
        return res;
    }
}
