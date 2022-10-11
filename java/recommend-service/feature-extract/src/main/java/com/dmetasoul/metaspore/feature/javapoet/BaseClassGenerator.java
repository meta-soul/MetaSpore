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

public interface BaseClassGenerator {
    JavaFile generateDomain(TableAttributes table, PackageInfo packageInfo);

    JavaFile generateRepository(String repoPrefixName, PackageInfo packageInfo,
                                JavaFile daoClass, TableAttributes table);

}