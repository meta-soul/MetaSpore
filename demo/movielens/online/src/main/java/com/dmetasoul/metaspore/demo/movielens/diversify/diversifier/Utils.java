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
