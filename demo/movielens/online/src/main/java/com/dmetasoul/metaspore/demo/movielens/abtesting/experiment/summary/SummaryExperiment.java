package com.dmetasoul.metaspore.demo.movielens.abtesting.experiment.summary;

import com.dmetasoul.metaspore.demo.movielens.domain.Item;
import com.dmetasoul.metaspore.demo.movielens.model.ItemModel;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendResult;
import com.dmetasoul.metaspore.demo.movielens.repository.ItemRepository;
import com.dmetasoul.metaspore.pipeline.BaseExperiment;
import com.dmetasoul.metaspore.pipeline.annotation.ExperimentAnnotation;
import com.dmetasoul.metaspore.pipeline.pojo.Context;
import org.springframework.stereotype.Component;
import org.springframework.util.CollectionUtils;

import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@ExperimentAnnotation(name = "summary.base")
@Component
public class SummaryExperiment implements BaseExperiment<RecommendResult, RecommendResult>  {
    private final ItemRepository itemRepository;

    public SummaryExperiment(ItemRepository itemRepository) {
        this.itemRepository = itemRepository;
    }

    @Override
    public void initialize(Map<String, Object> map) {
        System.out.println("match.base initialize... " + map);
    }

    @Override
    public RecommendResult run(Context context, RecommendResult recommendResult) {
        System.out.println("summary.base experiment, userModel:" + recommendResult.getUserId());
        List<ItemModel> rankingItemModels = recommendResult.getRecommendItemModels();
        if (CollectionUtils.isEmpty(rankingItemModels)) {
            System.out.println("summary.base experiment, match result is null");
            return recommendResult;
        }

        Collection<Item> summarizedItems = itemRepository.findByQueryidIn(rankingItemModels.stream().map(ItemModel::getId).collect(Collectors.toList()));
        // TODO if MongoDB return list is sequentially stable, we do not need to use HashMap here.
        HashMap<String, Item> itemMap = new HashMap<>();
        summarizedItems.forEach(x -> itemMap.put(x.getQueryid(), x));
        rankingItemModels.forEach(x -> x.fillSummary(itemMap.get(x.getId())));
        recommendResult.setRecommendItemModels(rankingItemModels);
        return recommendResult;
    }
}
