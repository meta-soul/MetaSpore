package com.dmetasoul.metaspore.demo.movielens;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration;
@SpringBootApplication(scanBasePackages = {"com.dmetasoul.metaspore"}, exclude = {DataSourceAutoConfiguration.class})
public class MovielensRecommendApplication {

    public static void main(String[] args) {
        SpringApplication.run(MovielensRecommendApplication.class, args);
    }
}
