package com.dmetasoul.metaspore.example.experiment;

import com.dmetasoul.metaspore.example.layer.SecondLayerPojo;
import com.dmetasoul.metaspore.pipeline.BaseExperiment;
import com.dmetasoul.metaspore.pipeline.annotation.ExperimentAnnotation;
import com.dmetasoul.metaspore.pipeline.impl.Context;
import org.springframework.stereotype.Component;

import java.util.Map;

@ExperimentAnnotation(name = "RankExperimentTwo")
@Component
public class RankExperimentTwo implements BaseExperiment<SecondLayerPojo, SecondLayerPojo> {

    @Override
    public void initialize(Map<String, Object> args) {
        System.out.println("RankExperimentTwo initialize ... ");
        System.out.println("RankExperimentTwo args: " + args);
    }

    @Override
    public SecondLayerPojo run(Context context, SecondLayerPojo input) {
        System.out.println("RankExperimentTwo running ... ");
        input.setMilvus3("execute experiment RankExperimentTwo");
        return input;
    }
}