package com.dmetasoul.metaspore.udf.example;

import com.dmetasoul.metaspore.configure.FieldAction;
import com.dmetasoul.metaspore.data.TableData;
import com.dmetasoul.metaspore.functions.Function;
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
