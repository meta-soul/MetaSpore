package com.dmetasoul.metaspore.demo.movielens.diversify.diversifier;

import com.dmetasoul.metaspore.demo.movielens.model.ItemModel;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Utils {
    public static Map<String, List<ItemModel>> groupByType(List<ItemModel> numbers) {
        Map<String, List<ItemModel>> map = new HashMap<>();
        for (ItemModel item : numbers) {
            if (map.containsKey(item.getGenre())) {
                map.get(item.getGenre()).add(item);
            } else {
                List<ItemModel> genreItemList = new ArrayList<>();
                genreItemList.add(item);
                map.put(item.getGenre(), genreItemList);
            }
        }
        return map;
    }
}
