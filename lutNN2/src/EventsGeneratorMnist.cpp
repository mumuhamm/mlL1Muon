/*
 * EventsGeneratorMnist.cpp
 *
 *  Created on: Jul 10, 2019
 *      Author: kbunkow
 */

#include <iomanip>
#include <iostream>
#include <algorithm>

#include "lutNN/lutNN2/interface/EventsGeneratorMnist.h"

namespace lutNN {

using namespace std;

template <typename EventType>
EventsGeneratorMnist<EventType>::EventsGeneratorMnist(std::string imageFile, std::string labelsFile, std::default_random_engine& generator):
    EventsGeneratorBase<EventType>(generator), imageFile(imageFile), labelsFile(labelsFile) {

    readMnistImages(imageFile, images, rowCnt, columnCnt);

    readMnistLabel(labelsFile, labels);

    std::cout<<"read imageFile "<<imageFile<<" size "<<images.size()<<" rowCnt "<<rowCnt<<" columnCnt "<<columnCnt<<std::endl;
}

template <typename EventType>
EventsGeneratorMnist<EventType>::~EventsGeneratorMnist() {

}

template <typename EventType>
void EventsGeneratorMnist<EventType>::printImage(unsigned int num, ostream& ostr) {
    std::vector<uint8_t> image = images.at(num);
    for(unsigned int column = 0; column < columnCnt; column++)
        ostr<<"#";
    ostr<<"#"<<endl;
    for(unsigned int row = 0; row < rowCnt; row++) {
        for(unsigned int column = 0; column < columnCnt; column++) {
            int indx= row * columnCnt + column;
            if(image.at(indx) == 0)
                ostr<<(" ");
            else if(image[indx] < 64)
                ostr<<(".");
            else if(image[indx] < 128)
                ostr<<("o");
            else
                ostr<<("@");
        }
        ostr<<"#"<<endl;
    }
    for(unsigned int column = 0; column < columnCnt; column++)
        ostr<<"#";
    ostr<<"#"<<endl;
    ostr<<"label "<<(unsigned int)labels[num]<<endl;
}

template <typename EventType>
void EventsGeneratorMnist<EventType>::printEvent(EventType* event, unsigned int columnCnt, unsigned int rowCnt, ostream& ostr) {
    for(unsigned int row = 0; row < rowCnt; row++) {
        for(unsigned int column = 0; column < columnCnt; column++) {
            ostr<<setw(3)<<hex<<event->inputs.at(row * columnCnt + column);
        }
        ostr<<endl;
    }
    ostr<<"expected out: "<<endl;
    for(unsigned int iOut = 0; iOut < event->expextedResult.size(); iOut++) {
        ostr<<iOut<<": "<<event->expextedResult[iOut]<<"  ";
    }
    ostr<<endl<<endl;;
}

template <typename EventType>
void EventsGeneratorMnist<EventType>::generateEvents(unsigned int inputCnt, unsigned int outputCnt, int xShift, int yShift, float scaleFactor) {
    this->events.resize( (2*xShift + 1) * (2*yShift + 1) * images.size());
    unsigned int iEvent = 0;
    for(unsigned int iImage = 0; iImage < images.size(); iImage++) {
        std::vector<uint8_t> image = images.at(iImage);

        for(int xS = -xShift; xS <= xShift; xS++) {
            for(int yS = -yShift; yS <= yShift; yS++) {
                EventType* event = new EventType(inputCnt, outputCnt);

                //hot one
                for(unsigned int iRes = 0; iRes < event->expextedResult.size(); iRes++) {
                    if(iRes == (unsigned int)labels[iImage])
                        event->expextedResult[iRes] = 1;
                    else
                        event->expextedResult[iRes] = 0;
                }

                event->classLabel = labels[iImage];

                for(int row = 0; row < (int)rowCnt; row++) {
                    for(int column = 0; column < (int)columnCnt; column++) {
                        if( (row + yS < 0) || (row + yS >= (int)rowCnt) || (column + xS < 0) || (column + xS >= (int)columnCnt) )
                            continue;
                        event->inputs[(row) * columnCnt + (column)] = image[ (row + yS) * columnCnt + (column + xS) ] * scaleFactor;
                    }
                }

                event->number = iEvent;
                if(xS == 0 && yS == 0)
                    event->weight = 4.;
                else {
                    event->weight = 1. / (xS * xS + yS * yS);
                }
                this->events.at(iEvent++) = event;
            }
        }
    }

    this->miniBatchBegin = this->events.begin();
}

//version where each input is a number composed of bits from a 3x3 pixels square, the squares are side by side (like tiles or a chessboard),
template <typename EventType>
void EventsGeneratorMnist<EventType>::generateEventsSquerTiles(unsigned int inputCnt, unsigned int outputCnt, int xShift, int yShift) {
    this->events.resize( (2*xShift + 1) * (2*yShift + 1) * images.size());
    unsigned int iEvent = 0;
    for(unsigned int iImage = 0; iImage < images.size(); iImage++) {
        std::vector<uint8_t> image = images.at(iImage);

        for(int xS = -xShift; xS <= xShift; xS++) {
            for(int yS = -yShift; yS <= yShift; yS++) {
                EventType* event = new EventType(inputCnt, outputCnt);

                //hot one
                for(unsigned int iRes = 0; iRes < event->expextedResult.size(); iRes++) {
                    if(iRes == (unsigned int)labels[iImage])
                        event->expextedResult[iRes] = 1;
                    else
                        event->expextedResult[iRes] = 0;
                }

                event->classLabel = labels[iImage];

                unsigned int a = 3;

                for(unsigned int iInput = 0; iInput < event->inputs.size(); iInput++) {
                    event->inputs[iInput] = 0;
                }
                for(int row = 0; row < (int)rowCnt; row++) {
                    for(int column = 0; column < (int)columnCnt; column++) {
                        if( (row + yS < 0) || (row + yS >= (int)rowCnt ) ||
                         (column + xS < 0) || column + xS >= (int)columnCnt)
                            continue;
                        if(image[ (row + yS) * columnCnt + (column + xS) ] >= 64) {
                            unsigned int iInput =  column / a + row / a * (columnCnt +2)/ a; //TODO +2 is if columnCnt%a != 0
                            unsigned int iBit = column % a + row%a * a;
                            int val = event->inputs[iInput];
                            val |= (1<<iBit);
                            event->inputs[iInput] = val;
                            //cout<<column<<" "<<row<<" "<<iInput<<" "<<" "<<iBit<<" "<<event->inputs[iInput]<<endl;
                        }
                    }
                }

                event->number = iEvent;
                this->events.at(iEvent++) = event;
            }
        }
    }

    this->miniBatchBegin = this->events.begin();
}

template <typename EventType>
void EventsGeneratorMnist<EventType>::getRandomEvents(vector<EventType*>& rndEvents) {
    std::uniform_int_distribution<int> randomEventDist(0, this->events.size() -1);
    for(auto& event : rndEvents) {
        int num = randomEventDist(this->generator);
        event = this->events[num];
    }
}


template class EventsGeneratorMnist<EventInt>;
template class EventsGeneratorMnist<EventFloat>;

} /* namespace lutNN */
