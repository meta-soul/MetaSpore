package com.dmetasoul.metaspore.operator;

import com.dmetasoul.metaspore.serving.FeatureTable;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.arrow.vector.complex.StructVector;
import org.apache.arrow.vector.complex.impl.NullableStructWriter;
import org.springframework.util.Assert;

@Slf4j
@Data
public class StructOperator extends ArrowOperator {
    @Override
    public boolean set(FeatureTable featureTable, int index, String col, Object value) {
        StructVector structVector = featureTable.getVector(col);
        Assert.notNull(structVector, "structVector is not null at col:" + col);
        if (value == null) {
            structVector.setNull(index);
        } else {
            NullableStructWriter writer = structVector.getWriter();
            writer.setPosition(index);
            writeStruct(writer, value, structVector.getField().getChildren(), featureTable, structVector.getAllocator());
        }
        featureTable.setRowCount(index + 1);
        return true;
    }
}
