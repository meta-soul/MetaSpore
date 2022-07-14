package com.dmetasoul.metaspore.recommend.functions;

import com.dmetasoul.metaspore.recommend.annotation.TransformFunction;

import java.util.List;
import java.util.Map;
@TransformFunction("logscale")
public class LogScaleFunction extends Function {

    @Override
    public void init(Map<String, Object> params) {

    }

    @Override
    public Object process(List<Object> values, List<String> types, Map<String, Object> options) {
        return values.get(0);
    }
}
