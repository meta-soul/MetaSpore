package com.dmetasoul.metaspore.example.expriment;

import com.dmetasoul.metaspore.example.layer.MyExperimentPojo;
import com.dmetasoul.metaspore.example.layer.MyExperimentPojo2;
import com.dmetasoul.metaspore.pipeline.BaseExperiment;
import com.dmetasoul.metaspore.pipeline.annotation.ExperimentAnnotation;
import com.dmetasoul.metaspore.pipeline.pojo.Context;
import org.springframework.beans.BeanUtils;
import org.springframework.stereotype.Component;

import java.util.Map;

@ExperimentAnnotation(name = "milvus2")
@Component
class Milvus2 implements BaseExperiment<MyExperimentPojo, MyExperimentPojo2> {


    @Override
    public void initialize(Map<String, Object> args) {
        System.out.println("Milvus2 initialize ... ");
        System.out.println("Milvus2 args: " + args);

    }

    @Override
    public MyExperimentPojo2 run(Context context, MyExperimentPojo input) {
        System.out.println("milvus2 running ... ");
        MyExperimentPojo2 output = new MyExperimentPojo2();
        BeanUtils.copyProperties(input, output);
        output.setMilvus2("execute experiment milvus2");
        return output;
    }
}