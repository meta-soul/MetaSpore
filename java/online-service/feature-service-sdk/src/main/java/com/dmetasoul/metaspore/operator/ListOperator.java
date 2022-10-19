package com.dmetasoul.metaspore.operator;

import com.dmetasoul.metaspore.serving.FeatureTable;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.vector.complex.ListVector;
import org.apache.arrow.vector.complex.impl.UnionListWriter;
import org.springframework.util.Assert;

import java.util.List;

@Slf4j
@Data
public class ListOperator<T> extends ArrowOperator {
    @Override
    public boolean set(FeatureTable featureTable, int index, String col, Object value) {
        if (value != null && !(value instanceof List)) {
            log.error("set featureTable fail! value type is not match!");
            return false;
        }
        ListVector listVector = featureTable.getVector(col);
        Assert.notNull(listVector, "listvector is not null at col:" + col);
        if (value == null) {
            listVector.setNull(index);
        } else {
            @SuppressWarnings("unchecked") List<T> data = (List<T>) value;
            UnionListWriter writer = listVector.getWriter();
            writer.setPosition(index);
            writeList(writer, data, listVector.getField().getChildren(), featureTable, listVector.getAllocator());
        }
        featureTable.setRowCount(index + 1);
        return true;
    }
}
