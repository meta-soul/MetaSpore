package com.dmetasoul.metaspore.pipeline.utils;

import com.dmetasoul.metaspore.pipeline.pojo.SceneConfig;
import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.constructor.Constructor;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;

public class SceneUtil {

    public static SceneConfig loadFromYmlFile(String ymlFilePath) throws FileNotFoundException {
        InputStream inputStream = new FileInputStream(new File(ymlFilePath));
        Yaml yaml = new Yaml(new Constructor(SceneConfig.class));
        SceneConfig globalConfig = yaml.load(inputStream);
        System.out.println("datas: " + globalConfig.getScenes().toString());
        return globalConfig;
    }
}
