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
import com.dmetasoul.metaspore.recommend.common.CommonUtils;
import com.dmetasoul.metaspore.recommend.data.FieldData;
import com.dmetasoul.metaspore.recommend.data.IndexData;
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.springframework.util.Assert;

import java.util.List;
import java.util.Map;

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
        addFunction("splitRecentIds", (fields, result, config, taskPool) -> {
            Map<String, Object> options = config.getOptions();
            Assert.isTrue(CollectionUtils.isNotEmpty(fields) && fields.size() == 1, "input values size must eq 1");
            Assert.isTrue(CollectionUtils.isNotEmpty(result), "output fields must not empty");
            FieldData fieldData = fields.get(0);
            Assert.isTrue(DataTypeEnum.STRING.isMatch(fieldData), "split input must string and not empty!");
            String split = CommonUtils.getField(options, "splitor", splitor);
            List<IndexData> input = fieldData.getIndexValue();
            for (IndexData o : input) {
                Assert.isTrue(o.getVal() instanceof String, "value must string! value:" + o.getVal());
                String value = o.getVal();
                result.get(0).addIndexData(FieldData.create(o.getIndex(), List.of(value.split(split))));
            }
            return true;
        });
        addFunction("recentWeight", (fields, result, config, taskPool) -> {
            Map<String, Object> options = config.getOptions();
            Assert.isTrue(CollectionUtils.isNotEmpty(fields), "input data is not null");
            Assert.isTrue(CollectionUtils.isNotEmpty(result), "output fields must not empty");
            List<IndexData> input = fields.get(0).getIndexValue();
            int num = 0;
            for (IndexData item : input) {
                List<String> list = item.getVal();
                for (int k = 0; k < list.size(); ++k) {
                    result.get(0).addIndexData(FieldData.create(item.getIndex(), list.get(k)));
                    result.get(1).addIndexData(FieldData.create(item.getIndex(), 1 / (1 + Math.pow((list.size() - k - 1), alpha))));
                }
            }
            return true;
        });
    }
}
