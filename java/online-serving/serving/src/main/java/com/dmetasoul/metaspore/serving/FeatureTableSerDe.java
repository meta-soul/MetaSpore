package com.dmetasoul.metaspore.serving;

import com.google.protobuf.ByteString;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowFileWriter;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;

import java.io.IOException;
import java.nio.channels.Channels;

public class FeatureTableSerDe {

    public static void serializeTo(FeatureTable table, PredictRequest.Builder builder) throws IOException {
        String name = table.getName();
        ByteString.Output out = ByteString.newOutput();
        table.finish();
        ArrowFileWriter writer = new ArrowFileWriter(table.getRoot(),
                /*DictionaryProvider=*/null, Channels.newChannel(out));
        writer.start();
        writer.writeBatch();
        writer.end();
        ByteString payload = out.toByteString();
        builder.putPayload(name, payload);
        writer.close();
    }

    public static FeatureTable deserializeFrom(String name, ByteString bytes) throws IOException {
        BufferAllocator allocator = ArrowAllocator.getAllocator();
        ArrowStreamReader reader = new ArrowStreamReader(bytes.newInput(), allocator);
        if (!reader.loadNextBatch()) {
            throw new IOException("Empty record batch received");
        }
        VectorSchemaRoot root = reader.getVectorSchemaRoot();
        reader.close();
        return new FeatureTable(name, root);
    }
}
