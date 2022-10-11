package com.dmetasoul.metaspore.example.experiment;

import com.dmetasoul.metaspore.example.layer.SecondLayerPojo;
import com.dmetasoul.metaspore.pipeline.BaseExperiment;
import com.dmetasoul.metaspore.pipeline.annotation.ExperimentAnnotation;
import com.dmetasoul.metaspore.pipeline.impl.Context;
import org.springframework.beans.BeanUtils;
import org.springframework.stereotype.Component;

import java.util.Map;

@ExperimentAnnotation(name = "RankExperimentOne")
@Component
public class RankExperimentOne implements BaseExperiment<SecondLayerPojo, SecondLayerPojo> {

    private Map<String, Object> extraExperimentArgs;

    @Override
    public void initialize(Map<String, Object> args) {
        System.out.println("RankExperimentOne initialize ... ");
        System.out.println("RankExperimentOne args: " + args);
        System.out.println("RecallExperimentOne args: " + args);
        this.extraExperimentArgs = args;
    }

    @Override
    public SecondLayerPojo run(Context context, SecondLayerPojo input) {
        System.out.println("RankExperimentOne running ... ");
        System.out.println("extraExperimentArgs: " + extraExperimentArgs);
        SecondLayerPojo output = new SecondLayerPojo();
        BeanUtils.copyProperties(input, output);
        output.setMilvus4("execute experiment RankExperimentOne");
        return output;
    }
}