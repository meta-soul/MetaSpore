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

package com.dmetasoul.metaspore.demo.movielens.ranking.ranker.impl;

import com.dmetasoul.metaspore.demo.movielens.model.ItemModel;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendContext;
import com.dmetasoul.metaspore.demo.movielens.model.UserModel;
import com.dmetasoul.metaspore.demo.movielens.ranking.ranker.Ranker;
import com.dmetasoul.metaspore.serving.ArrowAllocator;
import com.dmetasoul.metaspore.serving.ArrowTensor;
import com.dmetasoul.metaspore.serving.FeatureTable;
import com.dmetasoul.metaspore.demo.movielens.service.NpsService;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.*;

@Service
public class WideAndDeepRanker implements Ranker {
    public static final String ALGO_NAME = "WideAndDeep";
    public static final String DEFAULT_MODEL_NAME = "movie_lens_wdl";
    public static final String TARGET_KEY = "output";
    public static final int TARGET_INDEX = 0;

    private final NpsService npsService;

    public WideAndDeepRanker(NpsService npsService) {
        this.npsService = npsService;
    }

    // TODO maybe we should add a parent class for all ranker implements like LightGBMRanker and WideAndDeepRanker.
    @Override
    public List<ItemModel> rank(RecommendContext recommendContext, UserModel userModel, List<ItemModel> itemModels) throws IOException {
        String modelName = recommendContext.getWideAndDeepModelName();
        if (modelName == null || "".equals(modelName)) {
            modelName = DEFAULT_MODEL_NAME;
        }

        List<FeatureTable> featureTables = constructNpsFeatureTables(userModel, itemModels);
        Map<String, ArrowTensor> npsResultMap = npsService.predictBlocking(modelName, featureTables, Collections.emptyMap());
        List<Float> scores = npsService.getScoresFromNpsResult(npsResultMap, TARGET_KEY, TARGET_INDEX);

        Iterator<ItemModel> itemModelIt = itemModels.iterator();
        Iterator<Float> scoreIt = scores.iterator();
        while (itemModelIt.hasNext() && scoreIt.hasNext()) {
            ItemModel itemModel = itemModelIt.next();
            Float score = scoreIt.next();
            itemModel.getOriginalRankingScoreMap().put(ALGO_NAME, (double) score);
        }

        return itemModels;
    }

    private List<FeatureTable> constructNpsFeatureTables(UserModel userModel, List<ItemModel> itemModels) {
        List<Field> tableFields = List.of(
                Field.nullablePrimitive("movie_id", ArrowType.Utf8.INSTANCE)
        );

        FeatureTable lrLayerTable = new FeatureTable("lr_sparse", tableFields, ArrowAllocator.getAllocator());
        FeatureTable dnnLayerTable = new FeatureTable("dnn_sparse", tableFields, ArrowAllocator.getAllocator());

        Iterator<ItemModel> itemModelIt = itemModels.iterator();
        int i = 0;
        while(itemModelIt.hasNext()) {
            ItemModel itemModel = itemModelIt.next();
            lrLayerTable.setString(i, itemModel.getId(), lrLayerTable.getVector(0));
            dnnLayerTable.setString(i, itemModel.getId(), dnnLayerTable.getVector(0));
            i++;
        }

        return List.of(lrLayerTable, dnnLayerTable);
    }
}