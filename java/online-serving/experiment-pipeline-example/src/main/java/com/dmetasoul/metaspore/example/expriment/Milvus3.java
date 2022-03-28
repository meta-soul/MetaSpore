package com.dmetasoul.metaspore.example.expriment;

import com.dmetasoul.metaspore.example.layer.MyExperimentPojo2;
import com.dmetasoul.metaspore.pipeline.BaseExperiment;
import com.dmetasoul.metaspore.pipeline.annotation.ExperimentAnnotation;
import com.dmetasoul.metaspore.pipeline.pojo.Context;
import org.springframework.stereotype.Component;

import java.util.Map;

@ExperimentAnnotation(name = "milvus3")
@Component
class Milvus3 implements BaseExperiment<MyExperimentPojo2,MyExperimentPojo2> {

    @Override
    public void initialize(Map<String, Object> args) {
        System.out.println("Milvus3 initialize ... ");
        System.out.println("Milvus3 args: " + args);
    }

    @Override
    public MyExperimentPojo2 run(Context context, MyExperimentPojo2 input) {
        System.out.println("milvus3 running ... ");
        input.setMilvus3("execute experiment milvus3");
        return input;
    }
}