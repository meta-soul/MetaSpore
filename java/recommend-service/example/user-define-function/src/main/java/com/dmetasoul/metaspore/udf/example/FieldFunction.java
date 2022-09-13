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
    public boolean process(@NotEmpty List<FieldData> list, @NotEmpty List<FieldData> list1, @NonNull FieldAction fieldAction) {
        Validate.isTrue(list.size() == list1.size(), "in and out must be same size");
        log.info("******************this is example field process function!*******************");
        for (int i = 0; i < list.size(); ++i) {
            FieldData item = list.get(i);
            list1.get(0).setIndexValue(item.getIndexValue());
        }
        return true;
    }
}
