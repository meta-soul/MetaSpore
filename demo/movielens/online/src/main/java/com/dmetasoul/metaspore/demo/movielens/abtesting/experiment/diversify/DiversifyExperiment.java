package com.dmetasoul.metaspore.demo.movielens.abtesting.experiment.diversify;

import com.dmetasoul.metaspore.demo.movielens.diversify.DiversifierService;
import com.dmetasoul.metaspore.demo.movielens.model.ItemModel;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendResult;
import com.dmetasoul.metaspore.pipeline.BaseExperiment;
import com.dmetasoul.metaspore.pipeline.annotation.ExperimentAnnotation;
import com.dmetasoul.metaspore.pipeline.pojo.Context;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

@ExperimentAnnotation(name = "diversify.base")
@Component

public class DiversifyExperiment implements BaseExperiment<RecommendResult, RecommendResult>  {
    private final DiversifierService diversifierService;

    protected boolean useDiversify = true;

    protected int window;

    protected int tolerance;

    public DiversifyExperiment(DiversifierService diversifierService) {
        this.diversifierService = diversifierService;
    }

    @Override
    public void initialize(Map<String, Object> map) {
        this.useDiversify = (Boolean) map.getOrDefault("useDiversify", Boolean.TRUE);
        this.window = (int) map.getOrDefault("window", 4);
        this.tolerance = (int) map.getOrDefault("tolerance", 4);
        System.out.println("diversify.base initialize, useDiversify:" + this.useDiversify
                                                   +", window:" + this.window
                                                   +", tolerance:" + this.tolerance);
    }

    @Override
    public RecommendResult run(Context context, RecommendResult recommendResult) {
        List<ItemModel> itemModel = recommendResult.getRecommendItemModels();
        if (!useDiversify) {
            System.out.println("diversify.base experiment, turn off diversify");
            return recommendResult;
        }
        List<ItemModel> diverseItemModels = diversifierService.diverse(itemModel, this.window, this.tolerance);

        recommendResult.setRecommendItemModels(diverseItemModels);
        return recommendResult;
    }
}
