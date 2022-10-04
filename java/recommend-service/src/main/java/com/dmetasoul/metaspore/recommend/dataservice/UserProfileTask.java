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
import com.dmetasoul.metaspore.recommend.configure.FieldInfo;
import com.dmetasoul.metaspore.recommend.enums.DataTypeEnum;
import com.dmetasoul.metaspore.recommend.data.TableData;
import com.google.common.collect.Lists;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.springframework.util.Assert;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

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
    @SuppressWarnings("unchecked")
    @Override
    public void addFunctions() {
        addFunction("splitRecentIds", (fieldTableData, config, taskPool) -> {
            Map<String, Object> options = config.getOptions();
            List<FieldInfo> fields = config.getInputFields();
            Assert.isTrue(CollectionUtils.isNotEmpty(fields) && fields.size() == 1, "input values size must eq 1");
            Assert.isTrue(DataTypeEnum.STRING.equals(fieldTableData.getType(fields.get(0))), "split input must string and not empty!");
            String split = CommonUtils.getField(options, "splitor", splitor);
            for (int i = 0; i < fieldTableData.getData().size(); ++i) {
                String value = (String) fieldTableData.getValue(i, fields.get(0));
                if (value == null) {
                    continue;
                }
                fieldTableData.setValue(i, config.getNames().get(0), config.getTypes().get(0), List.of(value.split(split)));
            }
            return true;
        });
        addFunction("recentWeight", (fieldTableData, config, taskPool) -> {
            Map<String, Object> options = config.getOptions();
            List<FieldInfo> fields = config.getInputFields();
            Assert.isTrue(CollectionUtils.isNotEmpty(fields), "input data is not null");
            TableData recallData = new TableData();
            List<String> names = config.getNames();
            List<Object> types = config.getTypes();
            Assert.isTrue(names.size() > 2, "output has 3 fields");
	    int num = 0;
            for (int i = 0; i < fieldTableData.getData().size(); ++i) {
                List<Object> list = (List<Object>) fieldTableData.getValue(i, fields.get(1));
                if (list == null) {
                    continue;
                }
                for (int k = 0; k < list.size(); ++k) {
                    double weight = 1 / (1 + Math.pow((list.size() - k - 1), alpha));
		    recallData.setValue(num, names.get(0), types.get(0), fieldTableData.getValue(i, fields.get(0)));
                    recallData.setValue(num, config.getNames().get(1), config.getTypes().get(1), list.get(k));
                    recallData.setValue(num, config.getNames().get(2), config.getTypes().get(2), weight);
		    num += 1;
                }
            }
            fieldTableData.reset(recallData);
            // fieldTableData.flatListValue(data, fieldInfos, config.getTypes());
            return true;
        });
    }
}
