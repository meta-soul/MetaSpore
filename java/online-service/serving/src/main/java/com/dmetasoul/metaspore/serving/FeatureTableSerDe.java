//
// Copyright 2022 DMetaSoul
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package com.dmetasoul.metaspore.serving;

import com.google.protobuf.ByteString;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowFileWriter;
import org.apache.arrow.vector.ipc.ArrowStreamReader;

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

    public static FeatureTable deserializeFrom(String name, ByteString bytes, ArrowAllocator alloc) throws IOException {
        ArrowStreamReader reader = new ArrowStreamReader(bytes.newInput(), alloc.getAlloc());
        if (!reader.loadNextBatch()) {
            throw new IOException("Empty record batch received");
        }
        VectorSchemaRoot root = reader.getVectorSchemaRoot();
        reader.close();
        return new FeatureTable(name, root);
    }
}