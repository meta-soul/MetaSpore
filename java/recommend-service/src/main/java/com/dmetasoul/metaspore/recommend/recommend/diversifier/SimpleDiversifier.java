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
import com.dmetasoul.metaspore.recommend.configure.RecommendConfig;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.dmetasoul.metaspore.recommend.recommend.RecommendService;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.MapUtils;

import java.util.*;

@Slf4j
@RecommendAnnotation("diversifyBaseService")
public class SimpleDiversifier extends RecommendService {
    protected boolean useDiversify = true;
    protected int window = 4;
    protected int tolerance = 4;

    protected String groupType = "";

    @Override
    protected boolean initService() {
        this.useDiversify = getOptionOrDefault("useDiversify", Boolean.TRUE);
        this.window = getOptionOrDefault("window", 4);
        this.tolerance = getOptionOrDefault("tolerance", 4);
        this.groupType = getOptionOrDefault("groupType", "");
        return true;
    }

    @Override
    public DataResult process(ServiceRequest request, DataContext context) {
        return null;
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
    public DataResult process(ServiceRequest request, List<DataResult> dataResults, DataContext context) {
        List<Map> items = getListData(dataResults);
        LinkedList<Map> itemLinked = new LinkedList(items);
        List<Map> diverseResult = Lists.newArrayList();
        //compute count of genre
        int genreCount = groupByType(items).size();
        if (window > genreCount) {
            window = genreCount;
        }
        HashMap<String, Integer> genreInWindow = new HashMap<>();
        int stepCount = 0;
        while (!itemLinked.isEmpty()) {
            int slide = 0;

            if (itemLinked.size() != items.size()) {
                slide = window - 1;
            }

            if (stepCount != 0) {
                String genreUseless = (String) diverseResult.get(stepCount - 1).get(groupType);
                genreInWindow.put(genreUseless, genreInWindow.get(genreUseless) - 1);
                if (genreInWindow.get((String)diverseResult.get(stepCount - 1).get(groupType)) == 0) {
                    genreInWindow.remove((String)diverseResult.get(stepCount - 1).get(groupType));
                }
            }
            while (slide < window) {
                Map te = itemLinked.peek();
                if (genreInWindow.containsKey((String)te.get(groupType))) {
                    int toleranceTemp = window - slide;
                    Map itemStart = Maps.newHashMap();
                    Iterator<Map> itemModelIterator = itemLinked.iterator();
                    if (itemModelIterator.hasNext()) {
                        itemStart = itemModelIterator.next();
                    }
                    while (toleranceTemp > 0 && itemModelIterator.hasNext()) {
                        itemStart = itemModelIterator.next();
                        toleranceTemp--;
                    }
                    int startFound = window - slide;

                    while (startFound < Math.min(tolerance + window - slide, itemLinked.size())
                            && genreInWindow.containsKey((String)itemStart.get(groupType))
                            && itemModelIterator.hasNext()) {
                        startFound++;
                        itemStart = itemModelIterator.next();
                    }
                    if (toleranceTemp == itemLinked.size() || toleranceTemp == tolerance + window - slide) {
                        diverseResult.add(itemLinked.peek());
                        genreInWindow.put((String) itemLinked.peek().get(groupType), genreInWindow.get((String)itemLinked.peek().get(groupType)) + 1);
                        itemLinked.remove();
                        slide++;
                        continue;
                    }
                    String targetGenre = (String) itemStart.get(groupType);
                    int value = genreInWindow.containsKey(targetGenre) ? genreInWindow.get(targetGenre) + 1 : 1;
                    diverseResult.add(itemStart);
                    genreInWindow.put(targetGenre, value);
                    itemModelIterator.remove();
                } else {
                    genreInWindow.put((String) te.get(groupType), 1);
                    diverseResult.add(te);
                    itemLinked.remove();
                }
                slide++;
            }
            stepCount++;
        }
        DataResult result = new DataResult();
        result.setData(diverseResult);
        return result;
    }

}