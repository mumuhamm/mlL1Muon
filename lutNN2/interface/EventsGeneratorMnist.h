/*
 * EventsGeneratorMnist.h
 *
 *  Created on: Jul 10, 2019
 *      Author: kbunkow
 */

#ifndef INTERFACE_EVENTSGENERATORMNIST_H_
#define INTERFACE_EVENTSGENERATORMNIST_H_

#include <random>
#include <vector>
#include <iostream>
#include <sstream>

#include "lutNN/lutNN2/interface/EventsGeneratorBase.h"
#include "lutNN/lutNN2/interface/Event.h"
#include "lutNN/lutNN2/interface/MNISTParser.h"

namespace lutNN {

template <typename EventType>
class EventsGeneratorMnist: public EventsGeneratorBase<EventType> {
private:
    std::string imageFile;
    std::string labelsFile;

    std::string filename;
    std::vector<std::vector<uint8_t> > images;
    std::vector<uint8_t> labels;

    unsigned int rowCnt = 0;
    unsigned int columnCnt = 0;

public:
    EventsGeneratorMnist(std::string imageFile, std::string labelsFile, std::default_random_engine& generator);

    virtual ~EventsGeneratorMnist();

    void printImage(unsigned int num, std::ostream& ostr = std::cout);

    void printEvent(EventType* event, unsigned int columnCnt, unsigned int rowCnt, std::ostream& ostr = std::cout);

    //generates one cell in event->inputs[] for every pixel
    virtual void generateEvents(unsigned int inputCnt, unsigned int outputCnt, int xShift = 0, int yShift = 0, float scaleFactor = 1.);

    //version where each input is a number composed of bits from a 3x3 pixels square, the squares are side by side (like tiles or a chessboard)
    virtual void generateEventsSquerTiles(unsigned int inputCnt, unsigned int outputCnt, int xShift, int yShift);

    virtual void getRandomEvents(std::vector<EventType*>& events);

//    virtual std::vector<EventType*>& getEvents() {
//        return events;
//    }

    int gelPixelCnt() {
        return rowCnt * columnCnt;
    }
};


} /* namespace lutNN */

#endif /* INTERFACE_EVENTSGENERATORMNIST_H_ */
