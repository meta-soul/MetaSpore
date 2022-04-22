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
        if (itemModels.size() <= window) return itemModels;
        Double lambda = recommendContext.getLambda();
        if (lambda == null) {
            lambda = DEFAULT_LAMBDA;
        }
        LinkList itemLinkList = new LinkList();
        int genreCount = Utils.groupByType(itemModels).size();
        if (window == null || window > genreCount) {
            window = genreCount;
        }

        itemLinkList.addAll(itemModels);

        // label the visited
        HashMap<ItemModel, Integer> itemVisited = new HashMap<>();
        for (int i = 0; i < itemModels.size(); i++) {
            itemVisited.put(itemModels.get(i), 0);
        }
        // compute the genre in the window
        HashMap<String, Integer> genreInWindow = new HashMap();
        HashMap<String, Integer> genreSplitedInWindow = new HashMap<>();
        // start diverse
        ListNode itemNode = itemLinkList.head.next;
        for (int i = 0; i < itemModels.size(); i++) {
            //compute the count of genre in window
            int genreInWindowNum = 0;
            if (!genreInWindow.isEmpty()) {
                for (String genre : genreInWindow.keySet()) {
                    genreInWindowNum += genreInWindow.get(genre);
                }
            }

            if (genreInWindow.containsKey(itemNode.itemModel.getGenre())) {
                int maxIndex = i;
                double maxMMR = Double.MIN_VALUE;
                ListNode itemNodefind = itemNode;
                ListNode itemMaxMMR = new ListNode();
                for (int startFound = 0; startFound < tolerance && itemNodefind != null; startFound++) {
                    // MMR rate=ArgMax[lambda*sim(Di,Q)-(i-lambda)*SimScore]
                    // SimScore:itemModel's final simscore
                    // sim(Di,Q):the jaccard Coefficient between itemModel and the genres that were already in the window

                    double rankingScore = itemNodefind.itemModel.getFinalRankingScore() * lambda;
                    double simScore = getSimScore(itemNodefind.itemModel, genreSplitedInWindow) * (1 - lambda);
                    if ((rankingScore - simScore) > maxMMR) {
                        itemMaxMMR = itemNodefind;
                        maxMMR = rankingScore - simScore;
                    }
                }
                String minGenre = itemModels.get(maxIndex).getGenre();
                renewHashMap(genreInWindow, minGenre);
                // renew genreSplitedWindow;
                List<String> genreList = itemModels.get(maxIndex).getGenreList();
                for (String genre : genreList) {
                    renewHashMap(genreSplitedInWindow, genre);
                }
                // exchange location
                itemLinkList.swap(itemNode, itemMaxMMR);
            } else {
                genreInWindow.put(itemModels.get(i).getGenre(), 1);
                itemVisited.put(itemModels.get(i), 1);
                List<String> genreList = itemModels.get(i).getGenreList();
                for (String genre : genreList) {
                    renewHashMap(genreSplitedInWindow, genre);
                }
                itemNode = itemNode.next;
            }
            if (genreInWindowNum == window) {
                ItemModel itemDelete = itemModels.get(i - window + 1);
                List<String> itemGenreDelete = itemDelete.getGenreList();
                for (String genre : itemGenreDelete) {
                    genreSplitedInWindow.put(genre, genreSplitedInWindow.get(genre) - 1);
                    if (genreSplitedInWindow.get(genre) == 0) {
                        genreSplitedInWindow.remove(genre);
                    }
                }
                String deleteGenre = itemDelete.getGenre();
                genreInWindow.put(deleteGenre, genreInWindow.get(deleteGenre) - 1);
                if (genreInWindow.get(deleteGenre) == 0) {
                    genreInWindow.remove(deleteGenre);
                }
            }
        }
        ListNode getList = itemLinkList.head.next;
        List<ItemModel> itemDiverdified = new ArrayList<>();
        while (getList != null) {
            itemDiverdified.add(getList.itemModel);
            getList = getList.next;
        }
        return itemDiverdified;
    }

    public static void renewHashMap(HashMap<String, Integer> genreMap, String genre) {
        int defaultcount = genreMap.containsKey(genre) ? genreMap.get(genre) + 1 : 1;
        genreMap.put(genre, defaultcount);
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

    public class ListNode {
        ItemModel itemModel;
        ListNode next;
        ListNode prev;

        ListNode() {
        }

        ListNode(ItemModel itemModel) {
            this.itemModel = itemModel;
        }
    }

    class LinkList {
        int size;
        ListNode head;

        public LinkList() {
            size = 0;
            head = new ListNode(null);
        }
        public void addAll(List<ItemModel> itemList){
            int length=itemList.size();
            for(int i=length-1;i>=0;i--){
                ListNode addToHead=new ListNode(itemList.get(i));
                addToHead.next=head.next;
                head.next.prev=addToHead;
                head.next=addToHead;
                addToHead.prev=head;
                size++;
            }
        }
        public void add(ItemModel itemModel) {
            ListNode addToHead = new ListNode(itemModel);
            addToHead.next = head.next;
            head.next.prev = addToHead;
            head.next = addToHead;
            addToHead.prev = head;
            size++;
        }

        public void swap(ListNode i, ListNode j) {
            ListNode tempNext = j.next;
            ListNode tempPrev = j.prev;
            i.prev.next = j;
            i.prev = j.prev;
            j.next = i;
            j.prev = i.prev;
            i.prev = j;
            tempNext.prev = tempPrev;
            tempPrev.next = tempNext;
        }

        public boolean isEmpty() {
            return size == 0;
        }
    }

}


