package com.dmetasoul.metaspore.example.experiment;

import com.dmetasoul.metaspore.example.layer.FirstLayerPojo;
import com.dmetasoul.metaspore.example.layer.SecondLayerPojo;
import com.dmetasoul.metaspore.pipeline.BaseExperiment;
import com.dmetasoul.metaspore.pipeline.annotation.ExperimentAnnotation;
import com.dmetasoul.metaspore.pipeline.impl.Context;
import org.springframework.stereotype.Component;

import java.util.Map;

@ExperimentAnnotation(name = "RecallExperimentOne")
@Component
public class RecallExperimentOne implements BaseExperiment<FirstLayerPojo, SecondLayerPojo> {

    private Map<String, Object> extraExperimentArgs;

    @Override
    public void initialize(Map<String, Object> args) {
        System.out.println("RecallExperimentOne initialize ... ");
        System.out.println("RecallExperimentOne args: " + args);
        this.extraExperimentArgs = args;
    }

    @Override
    public SecondLayerPojo run(Context context, FirstLayerPojo input) {
        System.out.println("RecallExperimentOne running  ... ");
        System.out.println("extraExperimentArgs: " + extraExperimentArgs);
        SecondLayerPojo output = new SecondLayerPojo();
        output.setMilvusAdd("different pojo");
        output.setMilvus(input.getMilvus());
        output.setMilvus3("execute experiment milvus3");
        output.setMilvus("excute experiment milvus");
        context.setCustomData("RecallExperimentOne");
        return output;
    }
}