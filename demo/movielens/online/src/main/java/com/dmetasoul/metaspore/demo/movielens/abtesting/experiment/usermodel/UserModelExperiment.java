package com.dmetasoul.metaspore.demo.movielens.abtesting.experiment.usermodel;

import com.dmetasoul.metaspore.demo.movielens.domain.User;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendContext;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendResult;
import com.dmetasoul.metaspore.demo.movielens.model.UserModel;
import com.dmetasoul.metaspore.demo.movielens.repository.UserRepository;
import com.dmetasoul.metaspore.pipeline.BaseExperiment;
import com.dmetasoul.metaspore.pipeline.annotation.ExperimentAnnotation;
import com.dmetasoul.metaspore.pipeline.pojo.Context;
import lombok.SneakyThrows;
import org.springframework.stereotype.Component;

import java.security.InvalidParameterException;
import java.util.Map;
import java.util.Optional;

@ExperimentAnnotation(name = "userModel.base")
@Component
public class UserModelExperiment implements BaseExperiment<RecommendContext, RecommendResult> {
    private final UserRepository userRepository;

    public UserModelExperiment(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    @Override
    public void initialize(Map<String, Object> args) {
        System.out.println("userModel.base initialize... " + args);
    }

    @SneakyThrows
    @Override
    public RecommendResult run(Context context, RecommendContext recommendContext) {
        Optional<User> user = userRepository.findByQueryid(recommendContext.getUserId());
        if (user.isEmpty()) {
            throw new InvalidParameterException("No such user with id: " + recommendContext.getUserId());
        }
        UserModel userModel = new UserModel(user.get());
        RecommendResult result = new RecommendResult();
        result.setUserId(recommendContext.getUserId());
        result.setUserModel(userModel);
        result.setRecommendContext(recommendContext);
        System.out.println("userModel.base experiment, userModel:" + userModel.getUserId());
        return result;
    }
}