/*
 * LutNode.cpp
 *
 *  Created on: Jul 8, 2019
 *      Author: kbunkow
 */

#include "lutNN/lutNN2/interface/LutBinary.h"
#include <iostream>
#include <iomanip>
#include <sstream>

namespace lutNN {

LutBinary::LutBinary(): LutNode(0, 0, false) {
    name = "LutBinary_empty";
}

LutBinary::LutBinary(unsigned int number, unsigned int bitsPerInput, unsigned int inputCnt, unsigned int outputBits, bool propagateGradient):
        LutNode(number, inputCnt, 1<<(bitsPerInput * inputCnt), propagateGradient), values(1<<(bitsPerInput * inputCnt), 0), bitsPerInput(bitsPerInput), outputBits(outputBits)
        //gradients(boost::extents[1<<(bitsPerInput * inputCnt)][1<<outputBits]),
		//gradients(1<<(bitsPerInput * inputCnt))
        //lastGradient(1<<outputBits),
		//entries(1<<(bitsPerInput * inputCnt))
{
	/*momentum.resize(boost::extents[1<<(bitsPerInput * inputCnt)][2]);
	for(unsigned int i = 0; i < momentum.size(); i++) {
		for(unsigned int j = 0; j < momentum[i].size(); j++) {
			momentum[i][j] = 0;
		}
	}*/
    name = "LutBinary_" + std::to_string(number);
}

LutBinary::~LutBinary() {
    //std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<std::endl;
}

void LutBinary::run(float eventWeight) {
    lastAddr = 0;
    unsigned int pos = 0;
    for(auto& inputNode : inputNodes) {
        lastAddr |= (inputNode->getOutValueInt() << pos );
        pos += bitsPerInput;
    }

    /* debug
    if(lastAddr >= values.size()) {
        std::cout<<" LutBinary::run lastAddr >= values.size() "<<getName()<<" lastAddr "<<lastAddr<<std::endl;
        for(auto& inputNode : inputNodes) {
            std::cout<<inputNode->getName()<<" "<< inputNode->getOutValueInt()<<std::endl;
        }
    }*/
    //this->outValue = values.at(lastAddr);
    this->outValue = values[lastAddr];

    /*
    if(dither) {
    	this->outValue += dither;
    	this->outValue = (int)this->outValue & ((1<<outputBits)-1);

    	this->outValue = 1;
    }*/

    entries[lastAddr] += eventWeight;
}

//TODO include eventWeight
void LutBinary::updateGradient() {
    if(lastGradient == 0)
        return;

    //updating gradient of this node
    if(getOutValueInt() == 0) //getOutValueInt == 0,  inputNodeNewValue = 1, so the gradient comes with plus
        getGradients()[getLastAddr()] += lastGradient;
    else                 //getOutValueInt == 1,  inputNodeNewValue = 0, so the gradient comes with minus
        getGradients()[getLastAddr()] -= lastGradient;

    if(!propagateGradient) {
        lastGradient = 0;
        return;
    }

    //propagating gradient to do the input nodes
    for(unsigned int iInputNode = 0; iInputNode < getInputNodes().size(); iInputNode++) {
        unsigned int newAddr = lastAddr ^ (1UL << iInputNode); //flip the bit corresponding to the inputLutNode

        /*attempt to avoid gradient vanishig but looks it does not improve
        if(getIntValues()[newAddr] == 0) //getOutValueInt == 0,  inputNodeNewValue = 1, so the gradient comes with plus
            getGradients()[getLastAddr()] += lastGradient / 10.;
        else                 //getOutValueInt == 1,  inputNodeNewValue = 0, so the gradient comes with minus
            getGradients()[getLastAddr()] -= lastGradient / 10.; */

        if(values[newAddr] != this->outValue ) {
            getInputNodes()[iInputNode]->getLastGradient() += lastGradient;
        }
    }

    lastGradient = 0;
}

void LutBinary::updateStat(unsigned int eventWeihgt, int outVal, float cost) {

}

//void LutBinary::resetStat() {
//    //here is something wrong the, the gradients are not cleaned
//    /*for(auto it = gradients.origin(); it < (gradients.origin() + gradients.num_elements()); ++it) {
//        (*it) = 0;
//    }*/
//
//    for(auto& gradient : gradients)
//    	gradient = 0;
//
//
//    //this cannot be here, must be done after every run
//    /*for(auto& grad : lastGradient)
//    	grad = 0;*/
//
//    for(auto& entry : entries)
//    	entry = 0;
//
//    //std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<std::endl;
//}

} /* namespace lutNN */
