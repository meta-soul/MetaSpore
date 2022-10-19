package com.dmetasoul.metaspore.example;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication(scanBasePackages = {"com.dmetasoul.metaspore"})
public class ExperimentPipelineExampleApplication {

    public static void main(String[] args) {
        SpringApplication.run(ExperimentPipelineExampleApplication.class, args);
    }

}
