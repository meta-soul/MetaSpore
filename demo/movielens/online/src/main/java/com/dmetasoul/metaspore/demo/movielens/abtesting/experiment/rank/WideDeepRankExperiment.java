package com.dmetasoul.metaspore.demo.movielens.abtesting.experiment.rank;

import com.dmetasoul.metaspore.demo.movielens.model.ItemModel;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendContext;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendResult;
import com.dmetasoul.metaspore.demo.movielens.model.UserModel;
import com.dmetasoul.metaspore.demo.movielens.ranking.RankingService;
import com.dmetasoul.metaspore.pipeline.annotation.ExperimentAnnotation;
import com.dmetasoul.metaspore.pipeline.pojo.Context;
import lombok.SneakyThrows;
import org.springframework.stereotype.Component;
import org.springframework.util.CollectionUtils;

import java.util.List;
import java.util.Map;

@ExperimentAnnotation(name = "rank.wideDeep")
@Component
public class WideDeepRankExperiment extends RankExperiment{
    private final RankingService rankingService;

    public WideDeepRankExperiment(RankingService rankingService) {
        super(rankingService);
        this.rankingService = rankingService;
    }

    @Override
    public void initialize(Map<String, Object> map) {
        System.out.println("rank.wideDeep initialize... " + map);
        super.initialize(map);
    }

    @SneakyThrows
    @Override
    public RecommendResult run(Context context, RecommendResult recommendResult) {
        System.out.println("rank.wideDeep experiment, userModel:" + recommendResult.getUserId());
        UserModel userModel = recommendResult.getUserModel();
        List<ItemModel> recommendItemModels = recommendResult.getRecommendItemModels();
        if (userModel == null) {
            System.out.println("rank.wideDeep experiment, user model is null");
            return recommendResult;
        }
        if (CollectionUtils.isEmpty(recommendItemModels)) {
            System.out.println("rank.wideDeep experiment, match result is null");
            return recommendResult;
        }

        // TODO set different ranking service according to the experiment settings
        RecommendContext recommendContext = recommendResult.getRecommendContext();
        recommendContext.setRankerName(this.rankerName);
        recommendContext.setRankingSortStrategyType(this.sortStrategyType);
        recommendContext.setRankingSortStrategyAlpha(this.sortStrategyAlpha);
        recommendContext.setRankingSortStrategyBeta(this.sortStrategyBeta);
        recommendContext.setRankingMaxReservation(this.maxReservation);
        recommendContext.setWideAndDeepModelName(this.modelName);

        List<ItemModel> rankingItemModels = rankingService.rank(recommendContext, userModel, recommendItemModels);
        recommendResult.setRecommendItemModels(rankingItemModels);
        return recommendResult;
    }
}
