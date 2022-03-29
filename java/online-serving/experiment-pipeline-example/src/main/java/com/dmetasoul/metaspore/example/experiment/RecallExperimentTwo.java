package com.dmetasoul.metaspore.example.experiment;

import com.dmetasoul.metaspore.example.layer.FirstLayerPojo;
import com.dmetasoul.metaspore.example.layer.SecondLayerPojo;
import com.dmetasoul.metaspore.pipeline.BaseExperiment;
import com.dmetasoul.metaspore.pipeline.annotation.ExperimentAnnotation;
import com.dmetasoul.metaspore.pipeline.impl.Context;
import org.springframework.beans.BeanUtils;
import org.springframework.stereotype.Component;

import java.util.Map;

@ExperimentAnnotation(name = "RecallExperimentTwo")
@Component
public class RecallExperimentTwo implements BaseExperiment<FirstLayerPojo, SecondLayerPojo> {


    @Override
    public void initialize(Map<String, Object> args) {
        System.out.println("RecallExperimentTwo initialize ... ");
        System.out.println("RecallExperimentTwo args: " + args);

    }

    @Override
    public SecondLayerPojo run(Context context, FirstLayerPojo input) {
        System.out.println("RecallExperimentTwo running ... ");
        SecondLayerPojo output = new SecondLayerPojo();
        BeanUtils.copyProperties(input, output);
        output.setMilvus2("execute experiment RecallExperimentTwo");
        context.setCustomData("RecallExperimentTwo");
        return output;
    }
}