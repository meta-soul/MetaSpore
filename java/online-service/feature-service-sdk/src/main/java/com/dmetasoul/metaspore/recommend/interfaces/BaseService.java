package com.dmetasoul.metaspore.recommend.interfaces;

import com.dmetasoul.metaspore.data.DataContext;
import com.dmetasoul.metaspore.data.DataResult;

import java.util.List;
import java.util.concurrent.CompletableFuture;

public interface BaseService {
    CompletableFuture<List<DataResult>> execute(DataResult data, DataContext context);

    CompletableFuture<List<DataResult>> execute(List<DataResult> data, DataContext context);

    CompletableFuture<List<DataResult>> execute(DataContext context);

    void close();
}
