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

package com.dmetasoul.metaspore.recommend.recommend;


import com.dmetasoul.metaspore.recommend.annotation.RecommendAnnotation;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import lombok.extern.slf4j.Slf4j;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Slf4j
@RecommendAnnotation("userService")
public class UserProfile extends RecommendService{
    private double alpha;
    private String colRecentItemIds;
    private static String splitor = "\u0001";

    @Override
    protected boolean initService() {
        alpha = getOptionOrDefault("alpha", 1.0);
        colRecentItemIds = getOptionOrDefault("colRecentItemIds", "recentItemIds");
        return true;
    }

    @Override
    public DataResult process(ServiceRequest request, DataResult dataResult, DataContext context) {
        Map<String, Object> userData = dataResult.getValues();
        Object object = getField(userData, colRecentItemIds);
        if (object == null) {
            return null;
        }
        List<String> recentMovieArr = null;
        if (object instanceof String) {
            String recentIdsStr = (String)object;
            recentMovieArr = List.of(recentIdsStr.split(splitor));
        }
        if (object instanceof List && getFieldType(colRecentItemIds).equals("str[]")) {
            recentMovieArr = (List)object;
        }
        List<Map> data = Lists.newArrayList();
        for (int i = 0; recentMovieArr != null && i < recentMovieArr.size(); i++) {
            Map<String, Object> item = Maps.newHashMap();
            if (setFieldFail(item, 0, recentMovieArr.get(i))) {
                continue;
            }
            if (setFieldFail(item, 1, 1 / (1 + Math.pow((recentMovieArr.size() - i - 1), alpha)))) {
                continue;
            }
            data.add(item);
        }
        DataResult result = new DataResult();
        result.setData(data);
        return result;
    }
}