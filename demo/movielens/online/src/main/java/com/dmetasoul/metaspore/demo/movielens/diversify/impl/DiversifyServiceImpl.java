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

package com.dmetasoul.metaspore.demo.movielens.diversify.impl;

import com.dmetasoul.metaspore.demo.movielens.diversify.DiversifierService;
import com.dmetasoul.metaspore.demo.movielens.diversify.diversifier.DiverseProvider;
import com.dmetasoul.metaspore.demo.movielens.diversify.diversifier.Diversifier;
import com.dmetasoul.metaspore.demo.movielens.diversify.diversifier.impl.MaximalMarginalRelevanceDiversifier;
import com.dmetasoul.metaspore.demo.movielens.diversify.diversifier.impl.SimpleDiversifier;
import com.dmetasoul.metaspore.demo.movielens.model.ItemModel;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendContext;
import org.springframework.stereotype.Service;

import java.util.*;

@Service
public class DiversifyServiceImpl implements DiversifierService {
    public static final String DIVERSIFIER_NAMES = "SimpleDiversifier";

    @Override
    public List<ItemModel> diverse(RecommendContext recommendContext,
                                   List<ItemModel> itemModels,
                                   Integer window,
                                   Integer tolerance) {
        List<Diversifier> diversifierList = new ArrayList<>();
        SimpleDiversifier t = new SimpleDiversifier();
        diversifierList.add(t);
        MaximalMarginalRelevanceDiversifier d = new MaximalMarginalRelevanceDiversifier();
        diversifierList.add(d);
        DiverseProvider diverseProvider = new DiverseProvider(diversifierList);

        if (recommendContext.getDiversifierName() == null) {
            recommendContext.setDiversifierName(DIVERSIFIER_NAMES);
        }
        String diversifierMethod = recommendContext.getDiversifierName().toLowerCase();
        List<ItemModel> finalItemModelDiverse = diverseProvider.getDiversifiers(diversifierMethod).diverse(recommendContext, itemModels, window, tolerance);
        return finalItemModelDiverse;
    }
}