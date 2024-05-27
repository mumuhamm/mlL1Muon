/*
 * LutNode.cpp
 *
 *  Created on: Jan 16, 2022
 *      Author: kbunkow
 */

#include "lutNN/lutNN2/interface/LutNode.h"

namespace lutNN {

void LutNode::run(float eventWeight) {
    lastAddr = 0;
    unsigned int pos = 0;
    for(auto& inputNode : inputNodes) {
        lastAddr |= (inputNode->getOutValueInt() << pos );
        pos += 1;//bitsPerInput; TODO handle bitsPerInput
    }

    //this->outValue = floatValues.at(lastAddr);
    this->outValue = floatValues[lastAddr];

    if(dither) {
        /*this->outValue += dither;
        this->outValue = (int)this->outValue & ((1<<outputBits)-1);*/

        this->outValue = 0; //TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1 take it from layerCOnf
    }

    entries[lastAddr] += eventWeight;

    lastGradient = 0;
}

void LutNode::updateGradient() {
    //updating gradient of this node
    gradients[lastAddr] += lastGradient; //eventWeight is included in the lastGradient

    if(!propagateGradient) {
        lastGradient = 0;
        return;
    }

    if(lastGradient == 0)
        return;

    //propagating gradient to do the input nodes
    for(unsigned int iInputNode = 0; iInputNode < getInputNodes().size(); iInputNode++) {
        Node* inputNode = getInputNodes()[iInputNode];

        unsigned int newAddr = getLastAddr() ^ (1UL << iInputNode); //flip the bit corresponding to the inputLutNode

        //attempt to avoid gradient vanishig but looks it does not improve
        //gradients[newAddr] += lastGradient * eventWeight / 10.;

        float outValueChange = getFloatValues()[newAddr] - getOutValue();
        inputNode->getLastGradient() += (lastGradient * outValueChange);
        inputNode->updateStat(1, 0, 0);
    }

    lastGradient = 0;
}

void LutNode::reset() {
    //here is something wrong the, the gradients are not cleaned
    /*for(auto it = gradients.origin(); it < (gradients.origin() + gradients.num_elements()); ++it) {
        (*it) = 0;
    }*/

    for(auto& gradient : gradients)
        gradient = 0;


    //this cannot be here, must be done after every run
    /*for(auto& grad : lastGradient)
        grad = 0;*/

    for(auto& entry : entries)
        entry = 0;

    //std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<std::endl;
}


}


