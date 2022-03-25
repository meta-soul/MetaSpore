package com.dmetasoul.metaspore.feature.javapoet;

import com.squareup.javapoet.JavaFile;
import com.dmetasoul.metaspore.feature.dao.TableAttributes;

public interface BaseClassGenerator {
    JavaFile generateDomain(TableAttributes table, PackageInfo packageInfo);

    JavaFile generateRepository(String repoPrefixName, PackageInfo packageInfo,
                                JavaFile daoClass, TableAttributes table);

}
