package com.dmetasoul.metaspore.recommend.functions;

import com.dmetasoul.metaspore.recommend.annotation.TransformFunction;
import lombok.extern.slf4j.Slf4j;

import java.util.List;
import java.util.Map;

import static com.dmetasoul.metaspore.recommend.common.Utils.parseIntFromString;

@Slf4j
@TransformFunction("minmax")
public class MinMaxFunction extends Function {
    private final static String NAMEMIN = "min";
    private final static String NAMEMAX = "max";

    private int min = 0;
    private int max = 120;

    @Override
    public void init(Map<String, Object> params) {
    }

    @Override
    public Object process(List<Object> values, List<String> types, Map<String, Object> options) {
        return values.get(0);
    }
}
