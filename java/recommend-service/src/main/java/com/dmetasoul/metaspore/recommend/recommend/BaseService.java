package com.dmetasoul.metaspore.recommend.recommend;

import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;

import java.util.List;
import java.util.concurrent.*;

@Slf4j
@Data
public interface BaseService {
    CompletableFuture<List<DataResult>> execute(DataResult data, DataContext context);
    CompletableFuture<List<DataResult>> execute(List<DataResult> data, DataContext context);
    CompletableFuture<List<DataResult>> execute(DataContext context);
    void close();
}
