package com.dmetasoul.metaspore.pipeline;

import com.dmetasoul.metaspore.pipeline.pojo.Context;

import java.util.Map;

public interface BaseExperiment<T, R> {

    void initialize(Map<String, Object> args);

    R run(Context ctx, T in);
}
