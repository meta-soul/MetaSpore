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
import org.apache.arrow.flatbuf.Message;
import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.vector.ipc.ReadChannel;
import org.apache.arrow.vector.ipc.message.MessageChannelReader;
import org.apache.arrow.vector.ipc.message.MessageResult;

import java.io.IOException;
import java.nio.channels.Channels;

public class ArrowMessage {

    public static ArrowMessage readFromByteString(ByteString bs, ArrowAllocator alloc) throws IOException {
        MessageChannelReader reader = new MessageChannelReader(new ReadChannel(Channels.newChannel(bs.newInput())),
                alloc.getAlloc());
        MessageResult mr = reader.readNext();
        ArrowMessage m = new ArrowMessage();
        m.message = mr.getMessage();
        m.body = mr.getBodyBuffer();
        alloc.addBuffer(m.body);
        return m;
    }

    public Message message;
    public ArrowBuf body;
}