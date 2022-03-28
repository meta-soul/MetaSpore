package com.dmetasoul.metaspore.demo.movielens.retrieval.matcher.impl;

import com.dmetasoul.metaspore.demo.movielens.retrieval.matcher.Matcher;
import com.dmetasoul.metaspore.demo.movielens.domain.Itemcf;
import com.dmetasoul.metaspore.demo.movielens.model.ItemModel;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendContext;
import com.dmetasoul.metaspore.demo.movielens.model.UserModel;
import com.dmetasoul.metaspore.demo.movielens.repository.ItemcfRepository;
import org.springframework.stereotype.Service;

import java.util.*;

@Service
public class ItemCfMatcher implements Matcher {
    public static final String ALGO_NAME = "ItemCF";
    public static final int DEFAULT_ALGO_LEVEL = 3;
    public static final int DEFAULT_MAX_RESERVATION = 50;

    private final ItemcfRepository itemcfRepository;

    public ItemCfMatcher(ItemcfRepository itemcfRepository) {
        this.itemcfRepository = itemcfRepository;
    }

    @Override
    public List<ItemModel> match(RecommendContext recommendContext, UserModel userModel) {
        Integer maxReservation = recommendContext.getItemCfMaxReservation();
        if (maxReservation == null || maxReservation < 0) {
            maxReservation = DEFAULT_MAX_RESERVATION;
        }
        Integer algoLevel = recommendContext.getItemCfAlgoLevel();
        if (algoLevel == null || algoLevel < 0) {
            algoLevel = DEFAULT_ALGO_LEVEL;
        }

        ArrayList<ItemModel> itemModels = new ArrayList<>();
        Map<String, Double> triggerWeightMap = userModel.getTriggerWeightMap();
        String[] recentMovieArr = userModel.getRecentMovieArr();
        Collection<Itemcf> itemCf = itemcfRepository.findByQueryidIn(Arrays.asList(recentMovieArr));
        Iterator<String> recentMovieIt = Arrays.asList(recentMovieArr).iterator();
        Iterator<Itemcf> itemCfIterator = itemCf.iterator();
        HashMap<String, Double> itemToItemScore = new HashMap<>();
        while (itemCfIterator.hasNext() && recentMovieIt.hasNext()) {
            Itemcf itemcf = itemCfIterator.next();
            List itemCfValue = itemcf.getValue();
            String recentMovie = recentMovieIt.next();
            itemCfValue.forEach(x -> {
                Map<String, Object> map = (Map<String, Object>) x;
                String itemId = map.get("_1").toString();
                Double itemScore = (Double) map.get("_2") * triggerWeightMap.get(recentMovie);
                if (!itemToItemScore.containsKey(itemId) || itemScore > itemToItemScore.get(itemId)) {
                    itemToItemScore.put(itemId, itemScore);
                }
            });
        }
        ArrayList<Map.Entry<String, Double>> entryList = new ArrayList<>(itemToItemScore.entrySet());
        entryList.sort((o1, o2) -> o2.getValue().compareTo(o1.getValue()));
        Double maxScore = entryList.size() > 0 ? entryList.get(0).getValue() : 0.0;
        Integer finalAlgoLevel = algoLevel;
        entryList.stream().limit(maxReservation).forEach(x -> {
            ItemModel itemModel = new ItemModel();
            itemModel.setId(x.getKey());
            itemModel.setFinalRetrievalScore(Matcher.getFinalRetrievalScore(x.getValue(), maxScore, finalAlgoLevel));
            itemModel.getOriginalRetrievalScoreMap().put(ALGO_NAME, x.getValue());
            itemModels.add(itemModel);
        });
        return itemModels;
    }
}


