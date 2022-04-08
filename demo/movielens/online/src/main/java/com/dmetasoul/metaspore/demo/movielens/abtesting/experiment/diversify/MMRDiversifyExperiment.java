package com.dmetasoul.metaspore.demo.movielens.abtesting.experiment.diversify;

import com.dmetasoul.metaspore.demo.movielens.diversify.DiversifierService;
import com.dmetasoul.metaspore.demo.movielens.model.ItemModel;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendResult;
import com.dmetasoul.metaspore.pipeline.annotation.ExperimentAnnotation;
import com.dmetasoul.metaspore.pipeline.pojo.Context;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

@ExperimentAnnotation(name = "diversify.MMR")
@Component

public class MMRDiversifyExperiment extends DiversifyExperiment{

    public MMRDiversifyExperiment(DiversifierService diversifierService) {
        super(diversifierService);
    }
    @Override
    public void initialize(Map<String,Object> map){
        super.initialize(map);
        recommendContext.setLamada(this.lamada);
        recommendContext.setDiversifierName("MMRDiersifier");
    }

    @Override
    public RecommendResult run(Context context,RecommendResult recommendResult){
        List<ItemModel> itemModel =recommendResult.getRecommendItemModels();
        if(!useDiversify){
            System.out.println("diversify.base experiment, turn off diversify");
            return recommendResult;
        }
        List<ItemModel> diverseItemModels=diversifierService.diverse(recommendContext,itemModel,this.window,this.tolerance);
        recommendResult.setRecommendItemModels(diverseItemModels);
        return recommendResult;
    }

}
