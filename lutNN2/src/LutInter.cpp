/*
 * LutInter.cpp
 *
 *  Created on: Dec 13, 2019
 *      Author: kbunkow
 */

#include "lutNN/lutNN2/interface/LutInter.h"

#include <cmath>
#include <cassert>
#include <iostream>
#include <iomanip>

namespace lutNN {

LutInter::LutInter():
        LutNode(0, 0, true)
{
    name = "LutInter_";
}

LutInter::LutInter(unsigned int number, unsigned int bitsPerInput, bool propagateGradient, unsigned int rangesCnt, bool interpolate):
                LutNode(number, 1<<bitsPerInput, propagateGradient), //bitsPerInput(bitsPerInput),
                rangesCnt(rangesCnt),
                interpolate(interpolate),
                //entries(1<<bitsPerInput),
                //gradients(1<<bitsPerInput),
                momementum(1<<bitsPerInput, 0)

{
    name = "LutInter_" + std::to_string(number);

}

LutInter::~LutInter() {

}


void LutInter::run(float eventWeight) {
    float inVal = inputNode->getOutValue();
    if(inVal < 0) {
        lastAddr = 0;
        derivative = 0;
        outValue = floatValues[lastAddr];

        entries[lastAddr] += eventWeight;
    }
    else if(inVal >= (floatValues.size() - 1) ) {
        lastAddr = floatValues.size() -1;
        derivative = 0;
        outValue = floatValues[lastAddr];

        entries[lastAddr] += eventWeight;
    }
    else  {
        lastAddr = floor(inVal);
        derivative = floatValues[lastAddr +1] - floatValues[lastAddr];

        double deltaInVal =  inVal - lastAddr;
        if (interpolate)
            outValue = floatValues[lastAddr] + deltaInVal * derivative;
        else
            outValue = floatValues[round(inVal)]; //FIXME the case with no interpolation should also handle in another way the lastAddr and entries

        entries[lastAddr] += (1. -deltaInVal) * eventWeight;
        entries[lastAddr + 1] += deltaInVal * eventWeight;
    }

/*    if(inVal < 0) { //in principle can be merged with last else
        lastAddr = 0;
        derivative = floatValues[1] - floatValues[0];
        outValue = floatValues[0] + inVal * derivative;

        //outValue = floatValues[0];
    }
    else if(lastAddr >= (floatValues.size() - 1) ) {
        lastAddr = floatValues.size() -1;
        derivative = floatValues[lastAddr] - floatValues[lastAddr -1];
        outValue = floatValues[lastAddr] + (inVal - lastAddr) * derivative;
    }
    //TODO add a case when lastAddr == inVal, because if this place is minimum (maximu,?) then the side in which gradient should go should be found
    else {
        derivative = floatValues[lastAddr +1] - floatValues[lastAddr];
        outValue = floatValues[lastAddr] + (inVal - lastAddr) * derivative;
    }*/

    //entries[lastAddr]++; //TODO cannot be both here and in the updateGradient

    /*
    std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" "<<getName()<<" inputNode "<<inputNode->getName()<<" inVal "<<inVal<<" outVal "<<outValue<<" derivative "<<derivative
            <<" lastAddr "<<lastAddr<<" floatValues[lastAddr] "<<floatValues[lastAddr]<<" floatValues[lastAddr-1] "
            <<(lastAddr > 0 ? floatValues[lastAddr-1] : 0 )<<std::endl;*/

    if(std::isinf( outValue ) || std::isnan( outValue ) || fabs(outValue) > 1024) { //FIXME cannot be 1024 in the last layer
        std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" "<<getName()<<" inputNode "<<inputNode->getName()<<" inVal "<<inVal<<" outVal "<<outValue<<" derivative "<<derivative
                <<" lastAddr "<<lastAddr<<" floatValues[lastAddr] "<<floatValues[lastAddr]<<" floatValues[lastAddr-1] "
                <<(lastAddr > 0 ? floatValues[lastAddr-1] : 0 )<<" breaking !!!!!!!!!!!!!!!"<<std::endl;
        exit(1);
    }
    //std::cout<<"LutInter::run:"<<__LINE__<<" "<<getName()<<" inVal "<<std::setw(8)<<inVal<<" lastAddr "<<std::setw(4)<<lastAddr<<" outValue "<<std::setw(10)<<outValue<<" lastGradient "<<std::setw(10)<<lastGradient<<std::endl;
}

void LutInter::updateGradient() {
    if(lastGradient == 0)
        return;

    //std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" inputNode->getName "<<inputNode->getName()<<" gradients.size "<<gradients.size()<<std::endl;
    double inVal = inputNode->getOutValue();
    double deltaInVal =  inVal - lastAddr;
    //std::cout<<__FUNCTION__<<":"<<__LINE__<<" "<<getName()<<" inputNode "<<inputNode->getName()<<" lastAddr "<<lastAddr<<" inVal "<<inVal<<" lastGradient "<<lastGradient<<std::endl;
    //lastGradient is the gradient back-propagated from the output node(s)


    if(inVal <= 0) {
        gradients[lastAddr] += lastGradient;
          //TODO propagate back not 0, but some positive value
    }
    else if(inVal >= (floatValues.size() - 1) ) {
        gradients[lastAddr] += lastGradient;
          //TODO propagate back not 0, but some negative value
    }
    else  {
        double grad1 = lastGradient * (1. -deltaInVal);

        gradients[lastAddr] += grad1;

        double grad2 = lastGradient * deltaInVal;

        gradients[lastAddr + 1] += grad2;

        if(std::isinf( grad1 ) || std::isnan( grad1 ) || std::isinf( grad2 ) || std::isnan( grad2 ) ) {
            std::cout<<__FUNCTION__<<":"<<__LINE__<<" "<<getName()<<" inputNode->getName "<<inputNode->getName()<<" lastAddr "<<lastAddr<<" lastGradient "<<lastGradient<<" deltaInVal "<<deltaInVal<<" grad1 "<<grad1<<" grad2 "<<grad2<<std::endl;
        }
    }

    if(propagateGradient) {
        lastGradient = lastGradient * derivative;

        inputNode->getLastGradient() += lastGradient;
    }

    lastGradient = 0;
}

void LutInter::reset() {
    lastAddr = 0;
    lastGradient = 0;
    derivative = 0;

    for(auto& grad : gradients)
        grad = 0;

    for(auto& entry : entries)
        entry = 0;
}
} /* namespace lutNN */
