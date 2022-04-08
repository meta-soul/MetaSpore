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
import org.springframework.data.annotation.Reference;
import org.springframework.stereotype.Service;

import java.util.*;

@Service
public class MaximalMarginalRelevanceDiversifier implements Diversifier{
    public static final String DIVERSIFIER_NAME = "MMRDiersifier";
    public static final double LAMADA = 0.7;

    @Reference
    public List<ItemModel> diverse(RecommendContext recommendContext,
                                   List<ItemModel> itemModels,
                                   Integer window,
                                   Integer tolerance
    ) {
        Double lamada = recommendContext.getLamada();
        if (lamada == null) {
            lamada = LAMADA;
        }
        int genreCount = groupByType(itemModels).size();
        if (window == null || window > genreCount) {
            window = genreCount;
        }

        //label the visited
        HashMap<ItemModel, Integer> itemVisited = new HashMap<>();
        for (int i = 0; i < itemModels.size(); i++) {
            itemVisited.put(itemModels.get(i), 0);
        }
        //compute the genre in the window
        HashMap<String, Integer> genreInWindow = new HashMap();
        HashMap<String, Integer> genreSplitedInWindow = new HashMap<>();

        //start diverse
        for (int i = 0; i < itemModels.size(); i++) {
            //copmpute the count of genre in window
            int genreInWindowNum = 0;
            if (!genreInWindow.isEmpty()) {
                for (String genre : genreInWindow.keySet()) {
                    genreInWindowNum += genreInWindow.get(genre);
                }
            }

            if (genreInWindow.containsKey(itemModels.get(i).getGenre())) {
                int maxIndex = i;
                double maxMMR = Double.MIN_VALUE;

                for (int startFound = i; startFound < Math.min(i + tolerance, itemModels.size()); startFound++) {
                    if (itemVisited.get(itemModels.get(startFound)) != 0) {
                        continue;
                    }
                    //comput rate
                    double rankingScore = itemModels.get(startFound).getFinalRankingScore() * lamada;
                    double simScore = getSimScore(itemModels.get(startFound), genreSplitedInWindow) * (1 - lamada);
                    if ((rankingScore - simScore) > maxMMR) {
                        maxIndex = startFound;
                        maxMMR = rankingScore - simScore;
                    }
                }
                String minGenre = itemModels.get(maxIndex).getGenre();
                int defaults = genreInWindow.containsKey(minGenre) ? genreInWindow.get(minGenre) + 1 : 1;
                genreInWindow.put(minGenre, defaults);
                //renew genreSplitedWindow;
                List<String> genreList = itemModels.get(maxIndex).getGenreList();
                for (String genre : genreList) {
                    int value = genreSplitedInWindow.containsKey(genre) ? genreSplitedInWindow.get(genre) + 1 : 1;
                    genreSplitedInWindow.put(genre, value);
                }
                //exchange location
                ItemModel needDiverse = itemModels.get(maxIndex);
                itemVisited.put(itemModels.get(maxIndex), 1);
                for (int j = maxIndex; j > i; j--) {
                    itemModels.set(j, itemModels.get(j - 1));
                }
                itemModels.set(i, needDiverse);
            } else {
                genreInWindow.put(itemModels.get(i).getGenre(), 1);
                itemVisited.put(itemModels.get(i), 1);
                List<String> genreList = itemModels.get(i).getGenreList();
                for (String genre : genreList) {
                    int value = genreSplitedInWindow.containsKey(genre) ? genreSplitedInWindow.get(genre) + 1 : 1;
                    genreSplitedInWindow.put(genre, value);
                }
            }
            if (genreInWindowNum == window) {
                ItemModel itemDelete = itemModels.get(i - window + 1);
                List<String> itemGenreDelete = itemDelete.getGenreList();
                for (String genre : itemGenreDelete) {
                    genreSplitedInWindow.put(genre, genreSplitedInWindow.get(genre) - 1);
                    if (genreSplitedInWindow.get(genre) == 0) {
                        genreSplitedInWindow.remove(genre);
                    }
                }
                String deleteGenre = itemDelete.getGenre();
                genreInWindow.put(deleteGenre, genreInWindow.get(deleteGenre) - 1);
                if (genreInWindow.get(deleteGenre) == 0) {
                    genreInWindow.remove(deleteGenre);
                }
            }
        }

        return itemModels;
    }

    public static Double getSimScore(ItemModel item, HashMap<String, Integer> itemInWindow) {
        HashSet<String> genreSet = new HashSet<>();
        List<String> itemGenre = item.getGenreList();
        double intersection = 0;
        double differenSet = 0;
        for (String i : itemInWindow.keySet()) {
            intersection += itemInWindow.get(i);
        }
        for (String i : itemGenre) {
            if (itemInWindow.containsKey(i)) {
                differenSet += itemInWindow.get(i);
                intersection -= itemInWindow.get(i);
            }
        }
        return differenSet / (intersection + itemGenre.size());
    }

    public static Map<String, List<ItemModel>> groupByType(List<ItemModel> numbers) {
        Map<String, List<ItemModel>> map = new HashMap<>();
        for (ItemModel item : numbers) {
            if (map.containsKey(item.getGenre())) {
                map.get(item.getGenre()).add(item);
            } else {
                List<ItemModel> ls = new ArrayList<>();
                ls.add(item);
                map.put(item.getGenre(), ls);
            }
        }
        return map;
    }
}


