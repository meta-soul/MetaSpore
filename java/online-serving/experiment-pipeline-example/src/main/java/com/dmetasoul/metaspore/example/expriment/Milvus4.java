package com.dmetasoul.metaspore.example.expriment;

import com.dmetasoul.metaspore.example.layer.MyExperimentPojo;
import com.dmetasoul.metaspore.pipeline.BaseExperiment;
import com.dmetasoul.metaspore.pipeline.annotation.ExperimentAnnotation;
import com.dmetasoul.metaspore.pipeline.pojo.Context;
import org.springframework.beans.BeanUtils;
import org.springframework.stereotype.Component;

import java.util.Map;

@ExperimentAnnotation(name = "milvus3")
@Component
class Milvus4 implements BaseExperiment<MyExperimentPojo, MyExperimentPojo> {


    @Override
    public void initialize(Map<String, Object> args) {
        System.out.println("Milvus4 initialize ... ");
        System.out.println("Milvus4 args: " + args);

    }

    @Override
    public MyExperimentPojo run(Context context, MyExperimentPojo input) {
        System.out.println("milvus4 running ... ");
        MyExperimentPojo output = new MyExperimentPojo();
        BeanUtils.copyProperties(input, output);
        output.setMilvus4("execute experiment milvus4");


        return output;
    }
}