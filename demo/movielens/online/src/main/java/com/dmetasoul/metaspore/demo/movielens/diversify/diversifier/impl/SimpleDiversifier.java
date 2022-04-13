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

import com.dmetasoul.metaspore.demo.movielens.diversify.diversifier.Diversifier;
import com.dmetasoul.metaspore.demo.movielens.model.ItemModel;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendContext;
import org.springframework.stereotype.Service;
import com.dmetasoul.metaspore.demo.movielens.diversify.diversifier.Utils;

import java.util.*;

@Service
public class SimpleDiversifier implements Diversifier {
    public static final String DIVERSIFIER_NAME = "SimpleDiersifier";

    public List<ItemModel> diverse(RecommendContext recommendContext,
                                   List<ItemModel> itemmodels,
                                   Integer window,
                                   Integer tolerance) {
        LinkedList<ItemModel> itemLinked = new LinkedList(itemmodels);
        List<ItemModel> diverseResult = new ArrayList<>();
        //compute count of genre
        int genreCount = Utils.groupByType(itemmodels).size();
        if (window == null || window > genreCount) {
            window = genreCount;
        }
        HashMap<String, Integer> genreInWindow = new HashMap<>();
        int stepCount = 0;
        while (!itemLinked.isEmpty()) {
            int slide = 0;

            if (itemLinked.size() != itemmodels.size()) {
                slide = window - 1;
            }

            if (stepCount != 0) {
                String genreUseless = diverseResult.get(stepCount - 1).getGenre();
                genreInWindow.put(genreUseless, genreInWindow.get(genreUseless) - 1);
                if (genreInWindow.get(diverseResult.get(stepCount - 1).getGenre()) == 0) {
                    genreInWindow.remove(diverseResult.get(stepCount - 1).getGenre());
                }
            }
            while (slide < window) {
                ItemModel te = itemLinked.peek();
                if (genreInWindow.containsKey(te.getGenre())) {
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
                            && genreInWindow.containsKey(itemStart.getGenre())
                            && itemModelIterator.hasNext()) {
                        startFound++;
                        itemStart = itemModelIterator.next();
                    }
                    if (toleranceTemp == itemLinked.size() || toleranceTemp == tolerance + window - slide) {
                        diverseResult.add(itemLinked.peek());
                        genreInWindow.put(itemLinked.peek().getGenre(), genreInWindow.get(itemLinked.peek().getGenre()) + 1);
                        itemLinked.remove();
                        slide++;
                        continue;
                    }
                    String targetGenre = itemStart.getGenre();
                    int value = genreInWindow.containsKey(targetGenre) ? genreInWindow.get(targetGenre) + 1 : 1;
                    diverseResult.add(itemStart);
                    genreInWindow.put(targetGenre, value);
                    itemModelIterator.remove();
                } else {
                    genreInWindow.put(te.getGenre(), 1);
                    diverseResult.add(te);
                    itemLinked.remove();
                }
                slide++;
            }
            stepCount++;
        }
        return diverseResult;
    }
}