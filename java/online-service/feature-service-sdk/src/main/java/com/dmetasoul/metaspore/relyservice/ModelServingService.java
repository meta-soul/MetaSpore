package com.dmetasoul.metaspore.relyservice;

import com.dmetasoul.metaspore.common.CommonUtils;
import com.dmetasoul.metaspore.serving.LoadGrpc;
import com.dmetasoul.metaspore.serving.ServingClient;
import io.grpc.ManagedChannel;
import io.grpc.netty.shaded.io.grpc.netty.NegotiationType;
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.util.Strings;

import java.util.Map;
import java.util.concurrent.TimeUnit;

@Slf4j
public class ModelServingService implements RelyService {
    protected ManagedChannel channel;
    @Override
    public void init(Map<String, Object> option) {
        this.channel = initManagedChannel(option);
    }

    public static String genKey(Map<String, Object> option) {
        String host = CommonUtils.getField(option, "host", "127.0.0.1");
        int port = CommonUtils.getField(option, "port", 50000, Integer.class);
        if (host.equalsIgnoreCase("localhost")) {
            host = "127.0.0.1";
        }
        return String.format("modelserving_%s:%d", host, port);
    }

    public static String genKey(String host, int port) {
        return String.format("modelserving_%s:%d", host, port);
    }

    public ManagedChannel initManagedChannel(Map<String, Object> option) {
        String host = CommonUtils.getField(option, "host", "127.0.0.1");
        int port = CommonUtils.getField(option, "port", 50000, Integer.class);
        NegotiationType negotiationType = NegotiationType.valueOf(Strings.toRootUpperCase((String) option.getOrDefault("negotiationType", "plaintext")));
        NettyChannelBuilder channelBuilder = NettyChannelBuilder.forAddress(host, port)
                .keepAliveWithoutCalls((Boolean) option.getOrDefault("enableKeepAliveWithoutCalls", false))
                .negotiationType(negotiationType)
                .keepAliveTime((Long) option.getOrDefault("keepAliveTime", 300L), TimeUnit.SECONDS)
                .keepAliveTimeout((Long) option.getOrDefault("keepAliveTimeout", 10L), TimeUnit.SECONDS);
        return channelBuilder.build();
    }

    public boolean LoadModel(String modelName, String version, String dirPath) {
        if (StringUtils.isEmpty(modelName) || StringUtils.isEmpty(version) || StringUtils.isEmpty(dirPath)) {
            log.error("load model request is loss, current req: model name: {}, version: {}, dir: {}", modelName, version, dirPath);
            return false;
        }
        LoadGrpc.LoadBlockingStub client = LoadGrpc.newBlockingStub(channel);
        return ServingClient.loadModel(client, modelName, version, dirPath);
    }

    @SneakyThrows
    @Override
    public void close() {
        if (channel == null || channel.isShutdown()) return;
        channel.shutdown().awaitTermination(1, TimeUnit.SECONDS);
    }
}
