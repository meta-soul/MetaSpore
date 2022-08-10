package com.dmetasoul.metaspore.recommend.recommend;

import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;

import java.util.List;
import java.util.concurrent.*;

public interface BaseService {
    CompletableFuture<List<DataResult>> execute(DataResult data, DataContext context);
    CompletableFuture<List<DataResult>> execute(List<DataResult> data, DataContext context);
    CompletableFuture<List<DataResult>> execute(DataContext context);
    void close();
}
