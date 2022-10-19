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
package com.dmetasoul.metaspore.recommend.diversifier;

import com.dmetasoul.metaspore.annotation.ServiceAnnotation;
import com.google.common.collect.Lists;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Primary;

import java.util.*;

// References:
// * The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries

@Slf4j
@Primary
@ServiceAnnotation("diversifyMMRService")
public class MaximalMarginalRelevanceDiversifier extends SimpleDiversifier {
    public static final double DEFAULT_LAMBDA = 0.7;
    public static final String SEQUENCE_FEATURE_SPLITTER = "'\u0001'";

    @Override
    public void addFunctions() {
        addFunction("diversifyMMR", (data, resultList, context, options) -> false);
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
                if (((String) itemList.get(i).get(groupType)).length() == 0) itemList.get(i).put(groupType, "null");
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


