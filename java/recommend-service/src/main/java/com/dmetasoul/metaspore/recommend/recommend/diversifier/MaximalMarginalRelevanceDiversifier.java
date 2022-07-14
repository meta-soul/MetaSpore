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
package com.dmetasoul.metaspore.recommend.recommend.diversifier;

import com.dmetasoul.metaspore.recommend.annotation.RecommendAnnotation;
import com.dmetasoul.metaspore.recommend.data.DataContext;
import com.dmetasoul.metaspore.recommend.data.DataResult;
import com.dmetasoul.metaspore.recommend.data.ServiceRequest;
import com.google.common.collect.Lists;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.MapUtils;
import org.springframework.context.annotation.Primary;

import java.util.*;

// References:
// * The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries

@Slf4j
@Primary
@RecommendAnnotation("diversifyMMRService")
public class MaximalMarginalRelevanceDiversifier extends SimpleDiversifier {
    public static final double DEFAULT_LAMBDA = 0.7;
    public static final String SEQUENCE_FEATURE_SPLITTER = "'\u0001'";

    @Override
    public DataResult process(ServiceRequest request, List<DataResult> dataResults, DataContext context) {
        DataResult result = new DataResult();
        List<Map> itemModels = getListData(dataResults);
        if (itemModels.size() <= window || itemModels.size() == 0) {
            result.setData(itemModels);
            return result;
        }
        Double lambda = DEFAULT_LAMBDA;
        String colRankingScore = "rankingScore";
        String colTypeList = "genreList";
        Map<String, Object> map = serviceConfig.getOptions();
        if (MapUtils.isNotEmpty(map)) {
            lambda = (Double) map.getOrDefault("lambda", DEFAULT_LAMBDA);
            colRankingScore = (String) map.getOrDefault("rankingScore", colRankingScore);
            colTypeList = (String) map.getOrDefault("genreList", colTypeList);
        }

        LinkedList itemLinkedList = new LinkedList(itemModels);

        if (itemLinkedList.isEmpty()) {
            result.setData(itemModels);
            return result;
        }

        int genreCount = groupByType(itemModels).size();
        if (window > genreCount) {
            window = genreCount;
        }
        // compute the genre in the window
        HashMap<String, Integer> genreSplitedInWindow = new HashMap<>();
        HashMap<String, Integer> genreInWindow = new HashMap<>();
        Queue<String> listNodeInWindow = new ArrayDeque<>();

        ListNode itemNode = itemLinkedList.head.next;
        ListNode itemNodePrev = itemLinkedList.head;
        // start diverse
        while (itemNode.next != null) {
            if (genreInWindow.containsKey((String)itemNode.itemModel.get(groupType))) {
                double maxMMR = Double.MIN_VALUE;
                ListNode itemNodefind = itemNode.next;
                ListNode itemNodefindPrev = itemNode;
                ListNode itemMaxMMR = new ListNode();
                ListNode itemMaxMMRPrev = new ListNode();
                for (int startFound = 0; startFound < tolerance && itemNodefind != null; startFound++) {
                    // MMR rate=ArgMax[lambda*sim(Di,Q)-(i-lambda)*SimScore]
                    // SimScore:itemModel's final simscore
                    // sim(Di,Q):the jaccard Coefficient between itemModel and the genres that were already in the window
                    double rankingScore = (double)itemNodefind.itemModel.getOrDefault(colRankingScore, 0.01) * lambda;
                    double simScore = getSimScore(itemNodefind.itemModel, colTypeList, genreSplitedInWindow) * (1 - lambda);
                    if ((rankingScore - simScore) > maxMMR) {
                        itemMaxMMR = itemNodefind;
                        itemMaxMMRPrev = itemNodefindPrev;
                        maxMMR = rankingScore - simScore;
                    }
                    itemNodefind = itemNodefind.next;
                    itemNodefindPrev = itemNodefindPrev.next;
                }

                String maxGenre = (String)itemMaxMMR.itemModel.get(groupType);
                int genreValue = genreInWindow.containsKey(maxGenre) ? genreInWindow.get(maxGenre) + 1 : 1;
                genreInWindow.put(maxGenre, genreValue);
                listNodeInWindow.offer(maxGenre);
                // renew genreSplitedWindow;
                List<String> genreList = getGenreList(itemMaxMMR.itemModel, colTypeList);
                for (String genre : genreList) {
                    int defaultcount = genreSplitedInWindow.containsKey(genre) ? genreSplitedInWindow.get(genre) + 1 : 1;
                    genreSplitedInWindow.put(genre, defaultcount);
                }

                itemLinkedList.insertBefore(itemNode, itemNodePrev, itemMaxMMR, itemMaxMMRPrev);

                itemNodePrev = itemMaxMMR;
                itemNode = itemNodePrev.next;
            } else {
                String dibersifyGenre = (String)itemNode.itemModel.get(groupType);
                listNodeInWindow.offer(dibersifyGenre);
                genreInWindow.put(dibersifyGenre, 1);
                List<String> genreList = getGenreList(itemNode.itemModel, colTypeList);
                for (String genre : genreList) {
                    int defaultcount = genreSplitedInWindow.containsKey(genre) ? genreSplitedInWindow.get(genre) + 1 : 1;
                    genreSplitedInWindow.put(genre, defaultcount);
                }
                itemNodePrev = itemNode;
                itemNode = itemNode.next;
            }
            if (listNodeInWindow.size() == window) {
                String deleteItemGenre = listNodeInWindow.poll();
                genreInWindow.put(deleteItemGenre, genreInWindow.get(deleteItemGenre) - 1);
                if (genreInWindow.get(deleteItemGenre) == 0) genreInWindow.remove(deleteItemGenre);

                List<String> itemGenreDelete = List.of(deleteItemGenre.split(SEQUENCE_FEATURE_SPLITTER));
                for (String genre : itemGenreDelete) {
                    genreSplitedInWindow.put(genre, genreSplitedInWindow.get(genre) - 1);
                    if (genreSplitedInWindow.get(genre) == 0) genreSplitedInWindow.remove(genre);
                }
            }
        }
        ListNode getAnsList = itemLinkedList.head.next;
        List<Map> itemDiverdified = new ArrayList<>();
        while (getAnsList != null) {
            itemDiverdified.add(getAnsList.itemModel);
            getAnsList = getAnsList.next;
        }
        result.setData(itemDiverdified);
        return result;
    }


    // simScore= \frac{A \cup B}{A \cup B}
    public static Double getSimScore(Map item, String colTypeList, HashMap<String, Integer> itemInWindow) {
        List<String> itemGenre = (List<String>) item.getOrDefault(colTypeList, Lists.newArrayList());
        double intersection = 0;
        double differentSet = 0;
        for (String i : itemInWindow.keySet()) {
            differentSet += itemInWindow.get(i);
        }
        for (String i : itemGenre) {
            if (itemInWindow.containsKey(i)) {
                differentSet -= itemInWindow.get(i);
                intersection += itemInWindow.get(i);
            }
        }
        return intersection / (differentSet + itemGenre.size());
    }

    public static List getGenreList(Map itemModel, String colTypeList) {
        List<String> genreList = (List<String>) itemModel.getOrDefault(colTypeList, Lists.newArrayList());
        if (genreList.size() == 0) genreList.add("null");
        return genreList;
    }

    public class ListNode {
        Map itemModel;
        ListNode next;

        ListNode() {
        }

        ListNode(Map itemModel) {
            this.itemModel = itemModel;
        }
    }

    class LinkedList {
        int size;
        ListNode head;

        public LinkedList() {
            size = 0;
            head = new ListNode(null);
        }

        public LinkedList(List<Map> itemList) {
            size = itemList.size();
            head = new ListNode(null);
            for (int i = itemList.size() - 1; i >= 0; i--) {
                if (((String)itemList.get(i).get(groupType)).length() == 0) itemList.get(i).put(groupType, "null");
                ListNode insertHead = new ListNode(itemList.get(i));
                insertHead.next = head.next;
                head.next = insertHead;
            }
        }

        public void insertBefore(ListNode originNode, ListNode originPrevNode, ListNode swapNode, ListNode swapNodePrev) {
            swapNodePrev.next = swapNode.next;
            originPrevNode.next = swapNode;
            swapNode.next = originNode;
        }

        public boolean isEmpty() {
            return size == 0;
        }
    }

}


