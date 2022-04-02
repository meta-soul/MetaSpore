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

package com.dmetasoul.metaspore.demo.movielens.diversify.diversifier.impl;

import com.google.common.collect.Lists;
import com.dmetasoul.metaspore.demo.movielens.diversify.diversifier.Diversifier;
import com.dmetasoul.metaspore.demo.movielens.model.ItemModel;
import org.springframework.stereotype.Service;
import org.springframework.util.CollectionUtils;

import java.util.*;

@Service
public class SimpleDiversifier implements Diversifier {

    public List<ItemModel> diverse(List<ItemModel> itemmodels, Integer window, Integer tolerance) {
        LinkedList<ItemModel> itemLinked = new LinkedList(itemmodels);
        List<ItemModel> diverseResult = new ArrayList<>();
        //compute count of genre
        int genreCount = groupByType(itemmodels).size();
        if (window == null || window > genreCount) {
            window = genreCount;
        }
        HashMap<String, Integer> keys = new HashMap<>();
        int stepCount = 0;
        while (!itemLinked.isEmpty()) {
            int slide = 0;

            if (itemLinked.size() != itemmodels.size()) {
                slide = window - 1;
            }

            if (stepCount != 0) {
                String genreUseless = diverseResult.get(stepCount - 1).getGenre();
                keys.put(genreUseless, keys.get(genreUseless) - 1);
                if (keys.get(diverseResult.get(stepCount - 1).getGenre()) == 0) {
                    keys.remove(diverseResult.get(stepCount - 1).getGenre());
                }
            }
            while (slide < window) {
                ItemModel te = itemLinked.peek();
                if (keys.containsKey(te.getGenre())) {
                    int toleranceTemp = window - slide;
                    ItemModel itemStart = new ItemModel();
                    Iterator<ItemModel> itemModelIterator = itemLinked.iterator();
                    if (itemModelIterator.hasNext()) {
                        itemStart = itemModelIterator.next();
                    }
                    while (toleranceTemp > 0 && itemModelIterator.hasNext()) {
                        itemStart = itemModelIterator.next();
                        toleranceTemp--;
                    }
                    int startFound = window - slide;

                    while (startFound < Math.min(tolerance + window - slide, itemLinked.size())
                            && keys.containsKey(itemStart.getGenre())
                            && itemModelIterator.hasNext()) {
                        startFound++;
                        itemStart = itemModelIterator.next();
                    }
                    if (toleranceTemp == itemLinked.size() || toleranceTemp == tolerance + window - slide) {
                        diverseResult.add(itemLinked.peek());
                        keys.put(itemLinked.peek().getGenre(), keys.get(itemLinked.peek().getGenre()) + 1);
                        itemLinked.remove();
                        slide++;
                        continue;
                    }
                    String targetGenre = itemStart.getGenre();
                    int value = keys.containsKey(targetGenre) ? keys.get(targetGenre) + 1 : 1;
                    diverseResult.add(itemStart);
                    keys.put(targetGenre, value);
                    itemModelIterator.remove();
                } else {
                    keys.put(te.getGenre(), 1);
                    diverseResult.add(te);
                    itemLinked.remove();
                }
                slide++;
            }
            stepCount++;
        }
        return diverseResult;
    }

    public static Map<String, Integer> groupByType(List<ItemModel> numbers) {
        Map<String, Integer> map = new HashMap<>();
        for (ItemModel item : numbers) {
            if (map.containsKey(item.getGenre())) {
                continue;
            } else {
                map.put(item.getGenre(), 1);
            }
        }
        return map;
    }
}