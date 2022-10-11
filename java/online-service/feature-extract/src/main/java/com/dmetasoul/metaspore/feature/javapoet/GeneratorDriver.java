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

package com.dmetasoul.metaspore.feature.javapoet;

import com.squareup.javapoet.JavaFile;
import com.dmetasoul.metaspore.feature.dao.TableAttributes;
import com.dmetasoul.metaspore.feature.javapoet.enums.DbTypesEnum;
import com.dmetasoul.metaspore.feature.javapoet.mongodb.MongoClassGenerator;
import com.dmetasoul.metaspore.feature.javapoet.mysql.MySqlClassGenerator;
import com.dmetasoul.metaspore.feature.utils.GeneratorUtil;

import java.util.Locale;
import java.util.stream.Stream;

public class GeneratorDriver {
    private BaseClassGenerator baseClassGenerator;

    public void createDao(TableAttributes table, PackageInfo packageInfo) {

        DbTypesEnum dbType = DbTypesEnum.valueOf(table.getDbType().toUpperCase(Locale.ENGLISH));
        switch (dbType) {
            case MONGODB:
                baseClassGenerator = new MongoClassGenerator();
                break;
            case MYSQL:
                baseClassGenerator = new MySqlClassGenerator();
                break;
            default:
                throw new IllegalStateException("Unexpected value: " + table.getDbType());
        }

        JavaFile daoFile = baseClassGenerator.generateDomain(table, packageInfo);
        JavaFile repositoryFile = baseClassGenerator.generateRepository(table.getTableName(), packageInfo
                , daoFile, table);
        Stream.of(daoFile, repositoryFile).forEach(
                GeneratorUtil::writeJavaFile);

    }
}