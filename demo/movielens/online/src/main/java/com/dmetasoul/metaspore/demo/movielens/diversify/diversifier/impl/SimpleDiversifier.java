package com.dmetasoul.metaspore.demo.movielens.diversify.diversifier.impl;

import com.google.common.collect.Lists;
import com.dmetasoul.metaspore.demo.movielens.diversify.diversifier.Diversifier;
import com.dmetasoul.metaspore.demo.movielens.model.ItemModel;
import org.springframework.stereotype.Service;
import org.springframework.util.CollectionUtils;

import java.util.*;

@Service
public class SimpleDiversifier implements Diversifier {

    public List<ItemModel> diverse(List<ItemModel> itemModels, Integer window, Integer tolerance) {
        if (CollectionUtils.isEmpty(itemModels)) {
            return itemModels;
        }
        if (itemModels instanceof ArrayList) {
            itemModels = Lists.newArrayList(itemModels);
        }
        if (window == null || window > groupByType(itemModels).size()) {
            window = groupByType(itemModels).size();
        }
        HashMap<String, Integer> keys = new HashMap<>();
        for (int i = 0; i < itemModels.size() - window; i++) {
            List<ItemModel> subList = itemModels.subList(i, i + window);

            if (i >= 1) {
                String movie_genre = itemModels.get(i - 1).getGenre();
                keys.put(movie_genre, keys.get(itemModels.get(i - 1).getGenre()) - 1);
                if (keys.get(itemModels.get(i - 1).getGenre()) == 0) keys.remove(itemModels.get(i - 1).getGenre());
            }
            //int j = window + 1;
            int m = 0;
            if (i > 0) {
                m = window - 1;
            }
            for (; m < window; m++) {
                ItemModel item = subList.get(m);
                if (keys.containsKey(item.getGenre())) {
                    int findTarget = i + window;
                    while (findTarget < Math.min(itemModels.size(),
                            i + window + tolerance) && keys.containsKey(itemModels.get(findTarget).getGenre())) {
                        findTarget++;
                    }
                    if (findTarget == Math.min(itemModels.size(), i + window + tolerance)) {
                        findTarget = i + window;
                    }
                    itemModels.set(i + m, itemModels.get(findTarget));
                    itemModels.set(findTarget, item);
                    int value = keys.containsKey(itemModels.get(i + m).getGenre()) ? keys.get(itemModels.get(i + m).getGenre()) + 1 : 1;
                    keys.put(itemModels.get(i + m).getGenre(), value);
                } else {
                    keys.put(item.getGenre(), 1);
                }
            }
        }
        return itemModels;
    }

    public static HashSet<String> groupByType(List<ItemModel> numbers) {
        HashSet<String> hashset = new HashSet<>();
        Map<String, List<ItemModel>> map = new HashMap<>();
        for (ItemModel item : numbers) {
            if (map.containsKey(item.getGenre())) {
                continue;
            } else {
                hashset.add(item.getGenre());
            }
        }
        return hashset;
    }
}



