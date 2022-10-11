package com.dmetasoul.metaspore.udf.example;

import com.dmetasoul.metaspore.recommend.configure.FieldAction;
import com.dmetasoul.metaspore.recommend.data.TableData;
import com.dmetasoul.metaspore.recommend.functions.Function;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;

import java.util.concurrent.ExecutorService;

@Slf4j
public class FieldFunction implements Function {
    @Override
    public boolean process(@NonNull TableData fieldTableData, @NonNull FieldAction config, @NonNull ExecutorService taskPool) {
        log.info("******************this is example field process function!*******************");
        return true;
    }
}
