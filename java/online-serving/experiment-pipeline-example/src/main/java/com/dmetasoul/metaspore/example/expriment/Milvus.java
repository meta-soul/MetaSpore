package com.dmetasoul.metaspore.example.expriment;

import com.dmetasoul.metaspore.example.layer.MyExperimentPojo;
import com.dmetasoul.metaspore.example.layer.MyExperimentPojo2;
import com.dmetasoul.metaspore.pipeline.BaseExperiment;
import com.dmetasoul.metaspore.pipeline.annotation.ExperimentAnnotation;
import com.dmetasoul.metaspore.pipeline.pojo.Context;
import org.springframework.stereotype.Component;

import java.util.Map;

@ExperimentAnnotation(name = "milvus")
@Component
class Milvus implements BaseExperiment<MyExperimentPojo, MyExperimentPojo2> {

    private String myExtraArg;

    @Override
    public void initialize(Map<String, Object> args) {
        System.out.println("Milvus initialize ... ");
        System.out.println("Milvus args: " + args);
        String extraArg = (String) args.get("extraArg1");
        this.myExtraArg = extraArg;
    }

    @Override
    public MyExperimentPojo2 run(Context context, MyExperimentPojo input) {
        System.out.println("Milvus running  ... ");
        System.out.println("myExtraArg: " + myExtraArg);
        MyExperimentPojo2 output = new MyExperimentPojo2();
        output.setMilvusAdd("different pojo");
        output.setMilvus(input.getMilvus());
        output.setMilvus3("execute experiment milvus3");
        output.setMilvus("excute experiment milvus");
        System.out.println("global sceneArgs: " + context.getSceneArgs());
        return output;
    }
}