package com.dmetasoul.metaspore.pipeline;

import com.dmetasoul.metaspore.pipeline.pojo.Context;
import com.dmetasoul.metaspore.pipeline.pojo.LayerArgs;

public interface BaseLayer<T> {

    void intitialize(LayerArgs layerArgs);

    // 切流
    String split(Context ctx, T in);
}
