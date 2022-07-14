package com.dmetasoul.metaspore.recommend.functions;

import java.util.List;
import java.util.Map;

public abstract class Function {

    public Object process(List<Object> values, List<String> types, Map<String, Object> options) {
        return values.get(0);
    }

    public abstract void init(Map<String, Object> params);
}
