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

    public static ArrowMessage readFromByteString(ByteString bs) throws IOException {
        MessageChannelReader reader = new MessageChannelReader(new ReadChannel(Channels.newChannel(bs.newInput())),
                ArrowAllocator.getAllocator());
        MessageResult mr = reader.readNext();
        ArrowMessage m = new ArrowMessage();
        m.message = mr.getMessage();
        m.body = mr.getBodyBuffer();
        return m;
    }

    public Message message;
    public ArrowBuf body;
}
