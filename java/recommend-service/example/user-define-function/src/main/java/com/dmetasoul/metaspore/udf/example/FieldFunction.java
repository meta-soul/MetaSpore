package com.dmetasoul.metaspore.udf.example;

import com.dmetasoul.metaspore.recommend.configure.FieldAction;
import com.dmetasoul.metaspore.recommend.data.FieldData;
import com.dmetasoul.metaspore.recommend.functions.Function;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.Validate;

import javax.validation.constraints.NotEmpty;
import java.util.List;

@Slf4j
public class FieldFunction implements Function {
    @Override
    public boolean process(@NonNull TableData fieldTableData, @NonNull FieldAction config, @NonNull ExecutorService taskPool) {
        log.info("******************this is example field process function!*******************");
        return true;
    }
}
