package com.dmetasoul.metaspore.udf.example;

import com.dmetasoul.metaspore.data.DataContext;
import com.dmetasoul.metaspore.data.DataResult;
import com.dmetasoul.metaspore.recommend.interfaces.TransformFunction;
import lombok.extern.slf4j.Slf4j;

import javax.validation.constraints.NotEmpty;
import java.util.List;
import java.util.Map;

@Slf4j
public class DataResultFunction implements TransformFunction {
    @Override
    public boolean transform(@NotEmpty List<DataResult> list, @NotEmpty List<DataResult> list1, DataContext context, Map<String, Object> options) {
        log.info("******************this is example dataResult transform process function!*******************");
        list1.addAll(list);
        return true;
    }
}
