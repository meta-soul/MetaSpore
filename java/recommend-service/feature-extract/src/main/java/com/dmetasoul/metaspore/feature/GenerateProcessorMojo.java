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

package com.dmetasoul.metaspore.feature;

import com.dmetasoul.metaspore.feature.dao.TableAttributes;
import com.dmetasoul.metaspore.feature.javapoet.GeneratorDriver;
import com.dmetasoul.metaspore.feature.javapoet.PackageInfo;
import com.dmetasoul.metaspore.feature.utils.ParserHelper;
import org.apache.maven.plugin.AbstractMojo;
import org.apache.maven.plugins.annotations.LifecyclePhase;
import org.apache.maven.plugins.annotations.Mojo;
import org.apache.maven.plugins.annotations.Parameter;
import org.springframework.util.ResourceUtils;

import java.io.File;
import java.io.FileNotFoundException;

import org.apache.maven.project.MavenProject;

@Mojo(name = "generate", defaultPhase = LifecyclePhase.GENERATE_SOURCES)
public class GenerateProcessorMojo extends AbstractMojo {

    @Parameter(defaultValue = "${project}", readonly = true)
    protected MavenProject project;

    @Parameter(defaultValue = "${project.basedir}/src/main/resources/tables")
    protected String tablesDirPath;

    @Parameter(property = "packageName")
    private String packageName;

    @Override
    public void execute() {
        try {
            getLog().info("packageName: " + packageName);
            getLog().info("tablesDirPath: " + tablesDirPath);
            PackageInfo packageInfo = getPackageInfo(packageName);
            getLog().info("packageInfo: " + packageInfo);
            generateEntries(tablesDirPath, packageInfo);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void setPackageName(String packageName) {
        this.packageName = packageName;
    }

    public void setTablesDirPath(String tablesDirPath) {
        this.tablesDirPath = tablesDirPath;
    }

    private PackageInfo getPackageInfo(String packageName) {
        return new PackageInfo(packageName);
    }

    private void generateEntries(String tablesPath, PackageInfo packageInfo) throws FileNotFoundException {

        File filesPath = ResourceUtils.getFile(tablesPath);
        if (!filesPath.exists()) {
            getLog().info(filesPath.getPath());
            getLog().info("table directory path error");
        }

        File[] files = filesPath.listFiles();
        for (File file : files) {
            TableAttributes table = ParserHelper.getTableFromYaml(file.getPath());
            getLog().info(table.toString());
            GeneratorDriver generatorDriver = new GeneratorDriver();
            generatorDriver.createDao(table, packageInfo);
        }
    }

}