package com.dmetasoul.metaspore.feature;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class GenerateProcessorMojoTest {
    private GenerateProcessorMojo generateProcessorMojo;


    @BeforeEach()
    void setUp() {
        generateProcessorMojo = new GenerateProcessorMojo();
        generateProcessorMojo.setPackageName("com.dmetasoul.metaspore.feature");
        generateProcessorMojo.setTablesDirPath("src/main/resources/tables");
    }


    @Test
    void execute() {
        generateProcessorMojo.execute();
    }

}