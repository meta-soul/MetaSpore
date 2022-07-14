package com.dmetasoul.metaspore.recommend.data;

import com.dmetasoul.metaspore.recommend.enums.TaskStatusEnum;
import com.google.gson.Gson;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

import static com.dmetasoul.metaspore.recommend.common.Utils.getObjectToMap;

@Slf4j
@Data
public class DataContext {
    private String id;
    private Map<String, Object> request;
    private ConcurrentHashMap<String, DataResult> results = new ConcurrentHashMap<>();
    private ConcurrentHashMap<String, TaskStatusEnum> taskStatus = new ConcurrentHashMap<>();

    public DataContext() {
        this.id = UUID.randomUUID().toString().replaceAll("-", "");
    }

    public DataContext(String id) {
        this.id = id;
    }

    public DataContext(Map<String, Object> request) {
        this.request = request;
    }

    public void setRequest(Object req) {
        if (req != null) {
            try {
                request = getObjectToMap(req);
            } catch (IllegalAccessException e) {
                log.error("req ObjectToMap fail!");
                throw new RuntimeException(e);
            }
        }
    }

    public void setRequest(String req) {
        if (StringUtils.isNotEmpty(req)) {
            request = new Gson().fromJson(req, Map.class);
        }
    }

    public void setRequest(Map<String, Object> req) {
        request = req;
    }

    public String genContextKey(String name, String parent) {
        return String.format("%s_%s", name, parent);
    }

    public String genRequestSign(ServiceRequest request) {
        if (request == null) return "";
        ByteArrayOutputStream output = new ByteArrayOutputStream(10240);
        try {
            ObjectOutputStream oos = new ObjectOutputStream(output);
            oos.writeObject(request);
            MessageDigest md = MessageDigest.getInstance("md5");
            byte[] data =  md.digest(output.toByteArray());
            StringBuilder sb = new StringBuilder();
            for (byte b : data) {
                sb.append(Integer.toHexString(b & 0xff));
            }
            oos.close();
            output.close();
            return sb.toString();
        } catch (IOException | NoSuchAlgorithmException e) {
            throw new RuntimeException(e);
        }
    }

    public DataResult getResult(String name, String parent) {
        return results.get(genContextKey(name, parent));
    }

    public void setResult(String name, String parent, DataResult result) {
        results.put(genContextKey(name, parent), result);
    }

    public TaskStatusEnum getStatus(String name, String parent) {
        return taskStatus.get(genContextKey(name, parent));
    }

    public void setStatus(String name, String parent, TaskStatusEnum statusEnum) {
        taskStatus.put(genContextKey(name, parent), statusEnum);
    }

    public void setStatus(String name, TaskStatusEnum statusEnum) {
        taskStatus.put(name, statusEnum);
    }

    public DataResult getResult(String name) {
        return results.get(name);
    }

    public void setResult(String name, DataResult result) {
        results.put(name, result);
    }

    public TaskStatusEnum getStatus(String name) {
        return taskStatus.get(name);
    }


}
