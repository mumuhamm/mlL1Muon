/*
 * EventsGeneratorBase.cpp
 *
 *  Created on: Mar 13, 2020
 *      Author: kbunkow
 */

#include "lutNN/lutNN2/interface/EventsGeneratorBase.h"

#include <iomanip>
#include <iostream>
#include <algorithm>


namespace lutNN {

using namespace std;
template <typename EventType>
EventsGeneratorBase<EventType>::~EventsGeneratorBase() {
    std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<std::endl;
    clear();
    std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<std::endl;
}

template <typename EventType>
void EventsGeneratorBase<EventType>::clear() {
    for(auto& event : events)
        delete event;

    events.clear();
}

template <typename EventType>
void EventsGeneratorBase<EventType>::shuffle() {
    std::shuffle (events.begin(), events.end(), generator);
}


template <typename EventType>
void EventsGeneratorBase<EventType>::getNextMiniBatch(vector<EventType*>& miniBatchEvents) {
    if(miniBatchBegin + miniBatchEvents.size() >= events.end() ) {
        shuffle();
        shuffle();
        miniBatchBegin = events.begin();
    }

/*
    std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" miniBatchEvents.size() "<<miniBatchEvents.size()<<" events.size() "<<events.size()
            <<" miniBatchBegin "<<*miniBatchBegin
            <<" events.begin "<<*(events.begin())<<" end "<<*(events.end()-1)<<std::endl;
*/

    std::copy(miniBatchBegin, miniBatchBegin + miniBatchEvents.size(), miniBatchEvents.begin());
    miniBatchBegin = miniBatchBegin + miniBatchEvents.size();
}

template class EventsGeneratorBase<EventInt>;
template class EventsGeneratorBase<EventFloat>;

}
