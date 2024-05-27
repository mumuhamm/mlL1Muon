/*
 * LutNeuron.cpp
 *
 *  Created on: Dec 13, 2019
 *      Author: kbunkow
 */

#include "lutNN/lutNN2/interface/SumNode.h"

namespace lutNN {

SumNode::SumNode(): Node(0, 0), outValOffset(0) {
    name = "SumNode_empty";
}

SumNode::SumNode(unsigned int number, unsigned int inputCnt, float outValOffset, unsigned int biasShift, bool shiftLastGradient):
        Node(number, inputCnt), outValOffset(outValOffset), biasShift(biasShift), shiftLastGradient(shiftLastGradient) {
    name = "SumNode_";// + std::to_string(number);
}

SumNode::~SumNode() {

}

void SumNode::run(float eventWeight) {
    inValSum = 0;
    for(auto& inNode : inputNodes) {
        inValSum += inNode->getOutValue();
    }

    outValue = inValSum;
    //std::cout<<getName()<<" outValue "<<outValue;

    //outValue *= batchNormScale;
    //outValue += batchNormOffset;

    if( (outValOffset != 0) ) {
        if(outValue < (-outValOffset) ) {
        std::cout<<getName()<<" outValue "<<outValue<<" outValOffset "<<outValOffset<<" not good out outValue !!!!!!!!!!!!!!!!!!!!"<<std::endl;
            outValue = (-outValOffset);
        }
        else if (outValue >= (outValOffset-1) ) {
            std::cout<<getName()<<" outValue "<<outValue<<" outValOffset "<<outValOffset<<" not good out outValue !!!!!!!!!!!!!!!!!!!!"<<std::endl;
            outValue = (outValOffset-1) - 0.0001;
        }
    }

    //outValue from the previous LUT layer should be centered around 0,
    //so the outValOffset shifts the outValue such that they are in the middle of the LUT in the next layer, so the out values are in range 0...2*outValOffset
    outValue += outValOffset;

    if(biasNode) {
        outValue += (biasNode->getOutValueInt() << biasShift); //shifts outValue to different ranges of the LUT in the next layer,
                                                               //depending on the   biasNode->getOutValueInt()
        //std::cout<<" outValue "<<outValue<<" outValOffset "<<outValOffset<<" biasNode->getOutValueInt "<<biasNode->getOutValueInt()<<" biasShift "<<biasShift<<std::endl;
            //<<" outValSum "<<outValSum<<std::endl;
    }

    lastGradient = 0;
}

void SumNode::updateGradient() {
    if(shiftLastGradient) {
        //the gradient changes the LUT values with minus,
        //so shiftFactor should be positive if it should shift the  OutValues of the inputNodes into left (i.e. decrease)

        lastGradient = lastGradient + shiftFactor * inValSum * fabs(lastGradient);
    }

    //protection against going out of bounds
    /*if( outValOffset != 0) {
        if(inValSum < (-outValOffset)) {
            if(lastGradient > 0)
                lastGradient = -lastGradient;
        }
        else if(inValSum > (outValOffset-1) ) {
            if(lastGradient < 0)
                lastGradient = -lastGradient;
        }
    }*/

    for(auto& inputNode : inputNodes) {
        inputNode->getLastGradient() += lastGradient;
    }
    lastGradient = 0;

    //updates of the maxInValSum etc must be here and not in run
    //because run is executed also for the validation sample, and we want to have this values from the last batch
    if(inValSum > maxInValSum)
        maxInValSum = inValSum;

    if(inValSum < minInValSum)
        minInValSum = inValSum;

    outValSum  += inValSum;
    outVal2Sum += inValSum * inValSum;
}

void SumNode::updateParamaters(LearnigParams& learnigParams) {
    float w = (maxInValSum - minInValSum) / 2.;

    float margin = 0.9;

    float maxW = margin * outValOffset;

    shiftFactor = (maxW - w) / (maxW * maxW);

    //std::cout<<getName()<<" maxInValSum "<<maxInValSum<<" minInValSum "<<minInValSum<<" shiftFactor "<<shiftFactor<<std::endl;
}

void SumNode::reset() {
    lastGradient = 0;

    outValSum = 0;
    outVal2Sum = 0;

    maxInValSum = std::numeric_limits<float>::min();
    minInValSum = std::numeric_limits<float>::max();;
}

std::string SumNode::print() {
    std::ostringstream ostr;
    ostr<<"SumNode: ";
    if(biasNode)
        ostr<<" biasNode "<<biasNode->getName();
    else
        ostr<<" biasNode - nullptr";

    ostr<<" outValOffset "<<outValOffset<<" biasShift "<<biasShift;
    return ostr.str();
}

} /* namespace lutNN */
