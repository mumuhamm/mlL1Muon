/*
 * EventsGeneratorBase.h
 *
 *  Created on: Mar 13, 2020
 *      Author: kbunkow
 */

#ifndef INTERFACE_EVENTSGENERATORBASE_H_
#define INTERFACE_EVENTSGENERATORBASE_H_

#include <random>
#include <vector>

#include "lutNN/lutNN2/interface/Event.h"

namespace lutNN {
template <typename EventType>
class EventsGeneratorBase {
public:
    EventsGeneratorBase(std::default_random_engine& generator): miniBatchBegin(events.begin()), generator(generator) {};
    //miniBatchBegin must be set to  events.begin() after reading the events!!!!

    virtual ~EventsGeneratorBase();

    //virtual void generateEvents(std::vector<Event*>& events, int mode) = 0;

    //virtual std::vector<EventType*>& getEvents() = 0;

    virtual std::vector<EventType*>& getEvents() {
        return events;
    }

    virtual void getRandomEvents(std::vector<EventType*>& miniBatchEvents) = 0;

    virtual void shuffle();

    virtual void getNextMiniBatch(std::vector<EventType*>& miniBatchEvents);

    //deletes the events
    void clear();

protected:
    std::vector<EventType*> events;

    typename std::vector<EventType*>::iterator miniBatchBegin;

    std::default_random_engine& generator;
};
}

#endif /* INTERFACE_EVENTSGENERATORBASE_H_ */
