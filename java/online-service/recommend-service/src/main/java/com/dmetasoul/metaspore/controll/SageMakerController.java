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
package com.dmetasoul.metaspore.controll;

import com.dmetasoul.metaspore.actuator.PullContextRefresher;
import com.dmetasoul.metaspore.baseservice.RecommendService;
import com.dmetasoul.metaspore.baseservice.TaskServiceRegister;
import com.dmetasoul.metaspore.common.CommonUtils;
import com.dmetasoul.metaspore.common.S3Client;
import com.dmetasoul.metaspore.common.ServicePropertySource;
import com.dmetasoul.metaspore.data.ServiceResult;
import com.dmetasoul.metaspore.recommend.Scene;
import com.dmetasoul.metaspore.relyservice.ModelServingService;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.nio.file.Path;
import java.util.Map;
import java.util.Set;

import static org.springframework.web.bind.annotation.RequestMethod.GET;
import static org.springframework.web.bind.annotation.RequestMethod.POST;

/**
 * 用于实现restfull接口
 * Created by @author qinyy907 in 14:24 22/07/15.
 */
@Slf4j
@RestController
@RequestMapping
public class SageMakerController {
    public static final String SPRING_CONFIG_NAME = "recommend-config";
    public static final String MODEL_DATA_PATH;

    static {
        String osName = System.getProperty("os.name").toLowerCase();
        if (osName.contains("windows")) {
            MODEL_DATA_PATH = Path.of(System.getProperty("user.dir"), "/data/models").toString();
        } else {
            MODEL_DATA_PATH = "/data/models";
        }
    }

    @Autowired
    private RecommendService recommendService;

    @Autowired
    public TaskServiceRegister taskServiceRegister;

    @Autowired
    private PullContextRefresher pullContextRefresher;

    @RequestMapping(value = "/ping", method = GET, produces = "application/json")
    public ServiceResult ping() {
        return ServiceResult.of(0, "recommend service is ok!");
    }

    @RequestMapping(value = "/invocations", method = POST, produces = "application/json")
    public ServiceResult invocations(@RequestBody Map<String, Object> req) {
        String operator = CommonUtils.getField(req, "operator", "recommend");
        if (operator.equalsIgnoreCase("updateconfig")) {
            String config = CommonUtils.getField(req, "config");
            if (config == null) {
                return ServiceResult.of(-1, "no config arg in updateconfig!");
            }
            String configName = CommonUtils.getField(req, "configName", SPRING_CONFIG_NAME);
            String configFormat = CommonUtils.getField(req, "configFormat", "yaml");
            if (StringUtils.isEmpty(configName) || StringUtils.isEmpty(configFormat)) {
                return ServiceResult.of(-1, String.format("config name:%s or format:%s must not empty in updateconfig!", configName, configFormat));
            }
            ServicePropertySource.Format format = null;
            if (configFormat.equalsIgnoreCase("yaml") || configFormat.equalsIgnoreCase("yml")) {
                format = ServicePropertySource.Format.YAML;
            } else if (configFormat.equalsIgnoreCase("properties") || configFormat.equalsIgnoreCase("prop")) {
                format = ServicePropertySource.Format.PROPERTIES;
            }
            Set<String> keys = this.pullContextRefresher.updateConfig(configName, config, format);
            return ServiceResult.of(0, "update config successfully!")
                    .setInfo("config", config)
                    .setInfo("updateKeys", keys);
        } else if (operator.equalsIgnoreCase("loadmodel")) {
            String modelName = CommonUtils.getField(req, "modelName");
            String version = CommonUtils.getField(req, "version");
            String dirPath = CommonUtils.getField(req, "dirPath");
            if (StringUtils.isEmpty(modelName) || StringUtils.isEmpty(version) || StringUtils.isEmpty(dirPath)) {
                return ServiceResult.of(-1, "no modelName or version or dirPath arg in recommend!"
                        + String.format("modelName:[%s] version:[%s] dirPath:[%s]", modelName, version, dirPath));
            }
            String servingName = CommonUtils.getField(req, "servingName");
            if (StringUtils.isEmpty(servingName)) {
                String host = CommonUtils.getField(req, "host", "127.0.0.1");
                int port = CommonUtils.getField(req, "port", 50000);
                servingName = ModelServingService.genKey(host, port);
            }
            Map<String, Object> info = new java.util.HashMap<>(Map.of(
                    "servingName", servingName,
                    "modelName", modelName,
                    "version", version,
                    "dirPath", dirPath
            ));
            if (dirPath.startsWith("s3://")) {
                String localDir = CommonUtils.getField(req, "localDir", MODEL_DATA_PATH);
                if (StringUtils.isEmpty(localDir)) localDir = MODEL_DATA_PATH;
                // to do aws sdk download later
                // dirPath = S3Client.downloadModel(modelName, version, dirPath, localDir);
                dirPath = S3Client.downloadModelByShell(modelName, version, dirPath, localDir);
                info.put("localDirPath", dirPath);
            }
            ModelServingService modelServingService = taskServiceRegister.getRelyService(servingName, ModelServingService.class);
            if (!modelServingService.LoadModel(modelName, version, dirPath)) {
                return ServiceResult.of(-1, "serving load model resp fail!").setInfo(info);
            }
            return ServiceResult.of(0, "serving load model resp successfully!").setInfo(info);
        } else if (operator.equalsIgnoreCase("feature")) {
            Map<String, Object> request = CommonUtils.getField(req, "request");
            if (MapUtils.isEmpty(request)) {
                return ServiceResult.of(-1, "no request arg in feature!");
            }
            String task = CommonUtils.getField(request, "task");
            if (StringUtils.isEmpty(task)) {
                return ServiceResult.of(-1, String.format("get feature task result need teskName: %s!", task));
            }
            return recommendService.getDataServiceResult(task, request);
        } else if (operator.equalsIgnoreCase("recommend")) {
            Map<String, Object> request = CommonUtils.getField(req, "request");
            if (MapUtils.isEmpty(request)) {
                return ServiceResult.of(-1, "no request arg in recommend!");
            }
            String scene = CommonUtils.getField(request, "scene");
            if (StringUtils.isEmpty(scene)) {
                return ServiceResult.of(-1, String.format("get recommend result need scene: %s!", scene));
            }
            Scene sceneService = taskServiceRegister.getScene(scene);
            if (sceneService == null) {
                return ServiceResult.of(-1, String.format("scene:%s is not support!", scene));
            }
            return recommendService.getDataServiceResult(scene, request);
        } else if (operator.equalsIgnoreCase("itemSummary")) {
            Map<String, Object> request = CommonUtils.getField(req, "request");
            if (MapUtils.isEmpty(request)) {
                return ServiceResult.of(-1, "no request arg in recommend!");
            }
            String item_key = CommonUtils.getField(request, "item_key", "item_id");
            String id = CommonUtils.getField(request, item_key);
            if (StringUtils.isEmpty(id)) {
                return ServiceResult.of(-1, String.format("itemSummary need %s: %s!", item_key, id));
            }
            return recommendService.itemSummary(item_key, id, request);
        }
        return ServiceResult.of(-1, "no method support to " + operator);
    }
}
