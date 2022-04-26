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
package com.dmetasoul.metaspore.demo.movielens.diversify.diversifier.impl;

import com.dmetasoul.metaspore.demo.movielens.diversify.diversifier.Diversifier;
import com.dmetasoul.metaspore.demo.movielens.model.ItemModel;
import com.dmetasoul.metaspore.demo.movielens.model.RecommendContext;
import org.springframework.stereotype.Service;
import com.dmetasoul.metaspore.demo.movielens.diversify.Utils;

import java.util.*;

import com.dmetasoul.metaspore.demo.movielens.common.Constants;

// References:
// * The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries

@Service
public class MaximalMarginalRelevanceDiversifier implements Diversifier {
    public static final String ALGO_NAME = "MaximalMarginalRelevanceDiversifier";
    public static final double DEFAULT_LAMBDA = 0.7;

    public List<ItemModel> diverse(RecommendContext recommendContext,
                                   List<ItemModel> itemModels,
                                   Integer window,
                                   Integer tolerance
    ) {
        if (itemModels.size() <= window || itemModels.size() == 0) return itemModels;

        Double lambda = recommendContext.getLambda();
        if (lambda == null) {
            lambda = DEFAULT_LAMBDA;
        }

        LinkedList itemLinkedList = new LinkedList(itemModels);
        if (itemLinkedList.isEmpty()) return itemModels;

        int genreCount = Utils.groupByType(itemModels).size();
        if (window == null || window > genreCount) {
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
            if (genreInWindow.containsKey(itemNode.itemModel.getGenre())) {
                double maxMMR = Double.MIN_VALUE;
                ListNode itemNodefind = itemNode.next;
                ListNode itemNodefindPrev = itemNode;
                ListNode itemMaxMMR = new ListNode();
                ListNode itemMaxMMRPrev = new ListNode();
                for (int startFound = 0; startFound < tolerance && itemNodefind != null; startFound++) {
                    // MMR rate=ArgMax[lambda*sim(Di,Q)-(i-lambda)*SimScore]
                    // SimScore:itemModel's final simscore
                    // sim(Di,Q):the jaccard Coefficient between itemModel and the genres that were already in the window
                    double rankingScore = itemNodefind.itemModel.getFinalRankingScore() * lambda;
                    double simScore = getSimScore(itemNodefind.itemModel, genreSplitedInWindow) * (1 - lambda);
                    if ((rankingScore - simScore) > maxMMR) {
                        itemMaxMMR = itemNodefind;
                        itemMaxMMRPrev = itemNodefindPrev;
                        maxMMR = rankingScore - simScore;
                    }
                    itemNodefind = itemNodefind.next;
                    itemNodefindPrev = itemNodefindPrev.next;
                }

                String maxGenre = itemMaxMMR.itemModel.getGenre();
                int genreValue = genreInWindow.containsKey(maxGenre) ? genreInWindow.get(maxGenre) + 1 : 1;
                genreInWindow.put(maxGenre, genreValue);
                listNodeInWindow.offer(maxGenre);
                // renew genreSplitedWindow;
                List<String> genreList=getGenreList(itemMaxMMR.itemModel);
                for (String genre : genreList) {
                    int defaultcount = genreSplitedInWindow.containsKey(genre) ? genreSplitedInWindow.get(genre) + 1 : 1;
                    genreSplitedInWindow.put(genre, defaultcount);
                }

                itemLinkedList.swap(itemNode, itemNodePrev, itemMaxMMR, itemMaxMMRPrev);

                itemNodePrev = itemMaxMMR;
                itemNode = itemNodePrev.next;
            } else {
                listNodeInWindow.offer(itemNode.itemModel.getGenre());
                genreInWindow.put(itemNode.itemModel.getGenre(), 1);
                List<String> genreList=getGenreList(itemNode.itemModel);
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

                List<String> itemGenreDelete = List.of(deleteItemGenre.split(Constants.SEQUENCE_FEATURE_SPLITTER));
                for (String genre : itemGenreDelete) {
                    genreSplitedInWindow.put(genre, genreSplitedInWindow.get(genre) - 1);
                    if (genreSplitedInWindow.get(genre) == 0) genreSplitedInWindow.remove(genre);
                }
            }
        }
        ListNode getAnsList = itemLinkedList.head.next;
        List<ItemModel> itemDiverdified = new ArrayList<>();
        while (getAnsList != null) {
            itemDiverdified.add(getAnsList.itemModel);
            getAnsList = getAnsList.next;
        }
        return itemDiverdified;
    }


    // simScore= \frac{A \cup B}{A \cup B}
    public static Double getSimScore(ItemModel item, HashMap<String, Integer> itemInWindow) {
        List<String> itemGenre = item.getGenreList();
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
    public static List getGenreList(ItemModel itemModel){
        List<String> genreList=itemModel.getGenreList();
        if(genreList.size()==0)genreList.add("null");
        return genreList;
    }

    public class ListNode {
        ItemModel itemModel;
        ListNode next;

        ListNode() {
        }

        ListNode(ItemModel itemModel) {
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

        public LinkedList(List<ItemModel> itemList) {
            size = itemList.size();
            head = new ListNode(null);
            for (int i = itemList.size() - 1; i >= 0; i--) {
                ListNode insertHead = new ListNode(itemList.get(i));
                insertHead.next = head.next;
                head.next = insertHead;
            }
        }

        public void swap(ListNode raw, ListNode rawPrev, ListNode swapNode, ListNode swapNodePrev) {
            swapNodePrev.next = swapNode.next;
            rawPrev.next = swapNode;
            swapNode.next = raw;
        }

        public boolean isEmpty() {
            return size == 0;
        }
    }

}


