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

package com.dmetasoul.metaspore.tracking.controller;

import com.dmetasoul.metaspore.tracking.domain.TrackingEvent;
import com.dmetasoul.metaspore.tracking.producer.TrackingEventProducer;
import com.fasterxml.jackson.core.JsonProcessingException;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.kafka.support.SendResult;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import javax.validation.Valid;
import java.util.concurrent.ExecutionException;

@RestController
@Slf4j
public class TrackingEventsController {

    @Autowired
    TrackingEventProducer trackingEventProducer;

    @PostMapping("/v1/tracking")
    public ResponseEntity<TrackingEvent> postLibraryEvent(@RequestBody @Valid TrackingEvent trackingEvent) throws JsonProcessingException, ExecutionException, InterruptedException {

        //invoke kafka producer
        trackingEventProducer.sendTrackingEventWithTopic(trackingEvent);
        return ResponseEntity.status(HttpStatus.CREATED).body(trackingEvent);
    }

//    //PUT
//    @PutMapping("/v1/trackinag")
//    public ResponseEntity<?> putLibraryEvent(@RequestBody @Valid LibraryEvent libraryEvent) throws JsonProcessingException, ExecutionException, InterruptedException {
//
//
//        if(libraryEvent.getLibraryEventId()==null){
//            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body("Please pass the LibraryEventId");
//        }
//
//        libraryEvent.setLibraryEventType(TrackingEventType.UPDATE);
//        libraryEventProducer.sendLibraryEvent_Approach2(libraryEvent);
//        return ResponseEntity.status(HttpStatus.OK).body(libraryEvent);
//    }
}