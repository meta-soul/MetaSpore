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
import com.dmetasoul.metaspore.demo.movielens.model.ItemModel;
import org.springframework.stereotype.Service;

import java.util.*;

@Service
public class DiversifyServiceImpl implements DiversifierService {
    public static final List<String> DIVERSIFIER_NAMES = List.of("SimpleDiversifier");

    private final DiverseProvider diverseProvider;

    public DiversifyServiceImpl(DiverseProvider diverseProvider) {
        this.diverseProvider = diverseProvider;
    }

    @Override
    public List<ItemModel> diverse(List<ItemModel> itemModels, Integer window, Integer tolerance) {
        List<ItemModel> finalItemModelDiverse = new ArrayList<>();

        for (Diversifier m : diverseProvider.getDiversifiers(DIVERSIFIER_NAMES)) {
            finalItemModelDiverse = m.diverse(itemModels, window, tolerance);
        }
        return finalItemModelDiverse;
    }
}