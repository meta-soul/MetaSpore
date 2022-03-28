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

package com.dmetasoul.metaspore.demo.movielens.retrieval.matcher.impl;

import com.dmetasoul.metaspore.demo.movielens.retrieval.matcher.Matcher;
import com.dmetasoul.metaspore.serving.ArrowAllocator;
import com.dmetasoul.metaspore.serving.ArrowTensor;
import com.dmetasoul.metaspore.serving.FeatureTable;
import com.dmetasoul.metaspore.demo.movielens.domain.MilvusItemId;
import com.dmetasoul.metaspore.demo.movielens.model.ItemModel;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendContext;
import com.dmetasoul.metaspore.demo.movielens.model.UserModel;
import com.dmetasoul.metaspore.demo.movielens.repository.MilvusItemIdRepository;
import com.dmetasoul.metaspore.demo.movielens.service.MilvusService;
import com.dmetasoul.metaspore.demo.movielens.service.NpsService;
import io.milvus.response.SearchResultsWrapper;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.security.InvalidParameterException;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class TwoTowersMatcher implements Matcher {

    public static final String ALGO_NAME = "TwoTowersSimpleX";
    public static final String DEFAULT_MODEL_NAME = "two_towers_simplex";
    public static final String TARGET_KEY = "output";
    public static final int DEFAULT_ALGO_LEVEL = 3;
    public static final int DEFAULT_MAX_RESERVATION = 50;

    private final NpsService npsService;
    private final MilvusService milvusService;
    private final MilvusItemIdRepository milvusItemIdRepository;

    public TwoTowersMatcher(NpsService npsService, MilvusService milvusService, MilvusItemIdRepository milvusItemIdRepository) {
        this.npsService = npsService;
        this.milvusService = milvusService;
        this.milvusItemIdRepository = milvusItemIdRepository;
    }

    @Override
    public List<ItemModel> match(RecommendContext recommendContext, UserModel userModel) throws IOException {

        Integer maxReservation = recommendContext.getTwoTowersSimpleXMaxReservation();
        if (maxReservation == null || maxReservation < 0) {
            maxReservation = DEFAULT_MAX_RESERVATION;
        }
        Integer algoLevel = recommendContext.getTwoTowersSimpleXAlgoLevel();
        if (algoLevel == null || algoLevel < 0) {
            algoLevel = DEFAULT_ALGO_LEVEL;
        }
        String modelName = recommendContext.getTwoTowersSimpleXModelName();
        if (modelName == null || "".equals(modelName)) {
            modelName = DEFAULT_MODEL_NAME;
        }

        // 1. Prepare features.
        List<FeatureTable> featureTables = constructNpsFeatureTables(userModel);

        // 2. Call NSP service to get user vector.
        List<List<Float>> userVectors = getUserVectorsFromNps(modelName, featureTables);

        // 3. Call Milvus service to get top k item milvus ids.
        List<SearchResultsWrapper.IDScore> iDScores = getIDScoresFromMilvus(userVectors, maxReservation);

        // 4. Call MongoDB to get item ids from milvus ids.
        Collection<MilvusItemId> milvusItemIds = getItemIdsFromCache(iDScores);

        // 5. Return ItemModel list as result.
        return getItemModels(iDScores, milvusItemIds, algoLevel);
    }

    private List<List<Float>> getUserVectorsFromNps(String modelName, List<FeatureTable> featureTables) throws IOException {
        Map<String, ArrowTensor> npsResultMap = npsService.predictBlocking(modelName, featureTables, Collections.emptyMap());
        return npsService.getVectorsFromNpsResult(npsResultMap, TARGET_KEY);
    }

    private List<SearchResultsWrapper.IDScore> getIDScoresFromMilvus(List<List<Float>> vectors, Integer maxReservation) {
        Map<Integer, List<SearchResultsWrapper.IDScore>> itemVectors = milvusService.findByEmbeddingVectors(vectors, maxReservation);
        if (itemVectors.size() != 1) {
            // TODO error log
            throw new InvalidParameterException("Item vectors size must be 1. But got: " + itemVectors.size());
        }
        List<SearchResultsWrapper.IDScore> iDScores = itemVectors.values().iterator().next();
        // TODO if Milvus return list is sorted, we do not need to use sort it again.
        iDScores.sort((o1, o2) -> Double.compare(o2.getScore(), o1.getScore()));
        return iDScores;
    }

    private Collection<MilvusItemId> getItemIdsFromCache(List<SearchResultsWrapper.IDScore> iDScores) {
        List<String> milvusIds = iDScores.stream().map(SearchResultsWrapper.IDScore::getLongID).map(String::valueOf).collect(Collectors.toList());
        return milvusItemIdRepository.findByQueryidIn(milvusIds);
    }

    private List<ItemModel> getItemModels(List<SearchResultsWrapper.IDScore> iDScores, Collection<MilvusItemId> milvusItemIds, Integer algoLevel) {
        // TODO if MongoDB return list is sequentially stable, we do not need to use HashMap here.
        HashMap<String, String> milvusIdToItemIdMap = new HashMap<>();
        milvusItemIds.forEach(x -> milvusIdToItemIdMap.put(x.getMilvusId(), x.getItemId()));

        List<ItemModel> itemModels = new ArrayList<>();
        Double maxScore = iDScores.size() > 0 ? iDScores.get(0).getScore() : 0.0;
        iDScores.forEach(x -> {
            ItemModel itemModel = new ItemModel();
            itemModel.setId(milvusIdToItemIdMap.get(String.valueOf(x.getLongID())));
            Double score = (double) x.getScore();
            itemModel.getOriginalRetrievalScoreMap().put(ALGO_NAME, score);
            itemModel.setFinalRetrievalScore(Matcher.getFinalRetrievalScore(score, maxScore, algoLevel));
            itemModels.add(itemModel);
        });

        return itemModels;
    }

    private List<FeatureTable> constructNpsFeatureTables(UserModel userModel) {
        List<Field> userFields = List.of(
                Field.nullablePrimitive("user_id", ArrowType.Utf8.INSTANCE)
        );

        List<Field> interactedItemsFields = List.of(
                new Field("recent_movie_ids", FieldType.nullable(ArrowType.List.INSTANCE), List.of(Field.nullable("item", new ArrowType.Utf8())))
        );

        FeatureTable userTable = new FeatureTable("_sparse_user", userFields, ArrowAllocator.getAllocator());
        FeatureTable interactedItemsTable = new FeatureTable("_sparse_interacted_items", interactedItemsFields, ArrowAllocator.getAllocator());

        userTable.setString(0, userModel.getUserId(), userTable.getVector(0));
        interactedItemsTable.setStringList(0, Arrays.asList(userModel.getRecentMovieArr()), interactedItemsTable.getVector(0));

        return List.of(userTable, interactedItemsTable);
    }
}