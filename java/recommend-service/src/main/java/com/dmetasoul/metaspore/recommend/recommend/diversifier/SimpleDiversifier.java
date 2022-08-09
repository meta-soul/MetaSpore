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

package com.dmetasoul.metaspore.recommend.recommend.diversifier;

import com.dmetasoul.metaspore.recommend.annotation.RecommendAnnotation;
import com.dmetasoul.metaspore.recommend.common.Utils;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.dmetasoul.metaspore.recommend.functions.TransformFunction;
import com.dmetasoul.metaspore.recommend.recommend.Service;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import lombok.extern.slf4j.Slf4j;

import java.util.*;

@Slf4j
@RecommendAnnotation("diversifyBaseService")
public class SimpleDiversifier extends Service {
    protected boolean useDiversify = true;
    protected int window = 4;
    protected int tolerance = 4;

    protected String groupType = "";

    @Override
    protected boolean initService() {
        this.useDiversify = Utils.getField(serviceConfig.getOptions(), "useDiversify", true);
        this.window = Utils.getField(serviceConfig.getOptions(),"window", 4);
        this.tolerance = Utils.getField(serviceConfig.getOptions(),"tolerance", 4);
        this.groupType = Utils.getField(serviceConfig.getOptions(),"groupType", "");
        return true;
    }

    public Map<String, List<Map>> groupByType(List<Map> numbers) {
        Map<String, List<Map>> map = Maps.newHashMap();
        for (Map item : numbers) {
            Object type = item.get(groupType);
            if (!(type instanceof String)) {
                log.warn("diversifyBaseService:{} groupType:{} is not string!", name, groupType);
                continue;
            }
            if (map.containsKey(type)) {
                map.get(type).add(item);
            } else {
                List<Map> genreItemList = Lists.newArrayList();
                genreItemList.add(item);
                map.put((String) type, genreItemList);
            }
        }
        return map;
    }

    @Override
    public void addFunctions() {
        addFunction("diversifyBase", new TransformFunction() {
            @Override
            public boolean transform(List<DataResult> data, DataContext context, Map<String, Object> options) {
                return false;
            }
        });
    }

}