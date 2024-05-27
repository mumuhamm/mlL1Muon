/*
 * Event.h
 *
 *  Created on: Jul 10, 2019
 *      Author: kbunkow
 */

#ifndef INTERFACE_EVENT_H_
#define INTERFACE_EVENT_H_

#include <vector>
#include <iostream>
#include <iomanip>

namespace lutNN {

template <typename InputType>
class Event {
public:
    Event(unsigned int inputCnt, unsigned int ouptutCnt, InputType defInVal = 0): inputs(inputCnt, defInVal), expextedResult(ouptutCnt, 0), nnResult(ouptutCnt, 0), noHitVal(defInVal) {

    }

    virtual ~Event() {};

    std::vector<InputType> inputs;
    std::vector<double> expextedResult;

    unsigned short classLabel = 0;

    std::vector<double> nnResult;

    void print();

    unsigned int number = 0;

    float weight = 1.;

    InputType noHitVal = 0;
};


template <typename InputType>
void Event<InputType>::print() {
    for(unsigned int i = 0; i < inputs.size(); i++) {
        std::cout<<"input "<<i<<" val "<<inputs[i]<<std::endl;
    }

    for(unsigned int i = 0; i < expextedResult.size(); i++) {
        std::cout<<"result "<<i<<" exp "<<std::setw(5)<<expextedResult[i]<<" nnOut "<<nnResult[i]<<std::endl;
    }
}

typedef Event<int> EventInt;

typedef Event<float> EventFloat;

/*class EventInt: public Event<int> {

};

class EventFloat: public Event<float> {

};*/

} /* namespace lutNN */

#endif /* INTERFACE_EVENT_H_ */
