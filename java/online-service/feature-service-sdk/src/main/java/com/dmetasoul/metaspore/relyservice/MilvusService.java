package com.dmetasoul.metaspore.relyservice;

import com.dmetasoul.metaspore.common.CommonUtils;
import io.milvus.client.MilvusServiceClient;
import io.milvus.param.ConnectParam;
import lombok.Data;

import java.util.Map;

@Data
public class MilvusService implements RelyService {

    private MilvusServiceClient milvusTemplate;

    public static String genKey(Map<String, Object> option) {
        String host = CommonUtils.getField(option, "host", "127.0.0.1");
        int port = CommonUtils.getField(option, "port", 19530, Integer.class);
        if (host.equalsIgnoreCase("localhost")) {
            host = "127.0.0.1";
        }
        return String.format("milvus_%s:%d", host, port);
    }

    @Override
    public void init(Map<String, Object> option) {
        String host = CommonUtils.getField(option, "host", "localhost");
        int port = CommonUtils.getField(option, "port", 19530);
        ConnectParam connectParam = ConnectParam.newBuilder()
                .withHost(host)
                .withPort(port)
                .build();
        milvusTemplate = new MilvusServiceClient(connectParam);
    }

    @Override
    public void close() {
        if (milvusTemplate != null) {
            milvusTemplate.close();
        }
    }
}
