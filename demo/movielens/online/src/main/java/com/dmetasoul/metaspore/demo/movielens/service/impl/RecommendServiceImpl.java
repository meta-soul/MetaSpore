package com.dmetasoul.metaspore.demo.movielens.service.impl;

import com.dmetasoul.metaspore.demo.movielens.model.ItemModel;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendContext;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendResult;
import com.dmetasoul.metaspore.demo.movielens.model.UserModel;
import com.dmetasoul.metaspore.demo.movielens.ranking.RankingService;
import com.dmetasoul.metaspore.demo.movielens.retrieval.RetrievalService;
import com.dmetasoul.metaspore.demo.movielens.service.RecommendService;
import com.dmetasoul.metaspore.demo.movielens.domain.Item;
import com.dmetasoul.metaspore.demo.movielens.domain.User;
import com.dmetasoul.metaspore.demo.movielens.repository.ItemRepository;
import com.dmetasoul.metaspore.demo.movielens.repository.UserRepository;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class RecommendServiceImpl implements RecommendService {
    private final UserRepository userRepository;
    private final ItemRepository itemRepository;
    private final RetrievalService retrievalService;
    private final RankingService rankingService;

    public RecommendServiceImpl(UserRepository userRepository, ItemRepository itemRepository, RetrievalService retrievalService, RankingService rankingService) {
        this.userRepository = userRepository;
        this.itemRepository = itemRepository;
        this.retrievalService = retrievalService;
        this.rankingService = rankingService;
    }

    @Override
    public RecommendResult recommend(RecommendContext recommendContext) throws IOException {
        // 1. User model.
        Optional<User> user = userRepository.findByQueryid(recommendContext.getUserId());
        UserModel userModel = new UserModel(user.orElseGet(User::new));

        // 2. Retrieval.
        List<ItemModel> retrievalItemModels = retrievalService.match(recommendContext, userModel);

        // 3. Rank.
        List<ItemModel> rankingItemModels = rankingService.rank(recommendContext, userModel, retrievalItemModels);

        // 4. Summary.
        Collection<Item> summarizedItems = itemRepository.findByQueryidIn(rankingItemModels.stream().map(ItemModel::getId).collect(Collectors.toList()));
        // TODO if MongoDB return list is sequentially stable, we do not need to use HashMap here.
        HashMap<String, Item> itemMap = new HashMap<>();
        summarizedItems.forEach(x -> itemMap.put(x.getQueryid(), x));
        rankingItemModels.forEach(x -> x.fillSummary(itemMap.get(x.getId())));

        return new RecommendResult(userModel, rankingItemModels);
    }
}
