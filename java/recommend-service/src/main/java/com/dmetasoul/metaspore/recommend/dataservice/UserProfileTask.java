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

import com.dmetasoul.metaspore.recommend.annotation.DataServiceAnnotation;
import com.dmetasoul.metaspore.recommend.common.Utils;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;

import java.util.List;
import java.util.Map;

import static com.dmetasoul.metaspore.recommend.common.Utils.getField;

@SuppressWarnings("rawtypes")
@Data
@Slf4j
@DataServiceAnnotation("UserProfile")
public class UserProfileTask extends AlgoTransformTask {
    private double alpha;
    private String colRecentItemIds;
    private static String splitor = "\u0001";

    @Override
    public boolean initTask() {
        alpha = getOptionOrDefault("alpha", 1.0);
        return true;
    }

    @Override
    public DataResult process(ServiceRequest request, DataContext context) {
        DataResult taskResult = getDataResultByName(config.getDepend().getThen().get(0), context);
        Object object = getField(taskResult.getValues(), colRecentItemIds, null);
        if (object == null) {
            return null;
        }
        List<String> recentMovieArr = null;
        if (object instanceof String) {
            String recentIdsStr = (String)object;
            recentMovieArr = List.of(recentIdsStr.split(splitor));
        }
        if (object instanceof List && config.getColumnMap().get(colRecentItemIds).equals("str[]")) {
            recentMovieArr = (List)object;
        }
        List<Map> data = Lists.newArrayList();
        for (int i = 0; recentMovieArr != null && i < recentMovieArr.size(); i++) {
            Map<String, Object> item = Maps.newHashMap();
            if (Utils.setFieldFail(item, config.getColumnNames(), 0, recentMovieArr.get(i))) {
                continue;
            }
            if (Utils.setFieldFail(item, config.getColumnNames(), 1, 1 / (1 + Math.pow((recentMovieArr.size() - i - 1), alpha)))) {
                continue;
            }
            data.add(item);
        }
        DataResult result = new DataResult();
        result.setData(data);
        return result;
    }
}
