/*
 * LutLayer.cpp
 *
 *  Created on: Feb 6, 2022
 *      Author: kbunkow
 */

#include "lutNN/lutNN2/interface/LutLayer.h"

namespace lutNN {

template<typename OutValuesType>
void LutLayer<OutValuesType>::run(boost::dynamic_bitset<>& inputs) {
    calulateAddresses(inputs);

    for(size_t iOut = 0; iOut < lastOutVals.size(); iOut++) {
        auto& lastAddress = lastAddresses[iOut % lastAddresses.size()];
        lastOutVals[iOut] = floatLuts(iOut % lutCnt , lastAddress );
    }
}

template void LutFloatLayer::run(boost::dynamic_bitset<>& inputs);

void LutBinaryLayer::run(boost::dynamic_bitset<>& inputs) {
    calulateAddresses(inputs);

    for(size_t iOut = 0; iOut < lastOutVals.size(); iOut++) {
        auto& lastAddress = lastAddresses[iOut % lastAddresses.size()];
        lastOutVals[iOut] = binaryLuts[iOut % lutCnt ][lastAddress];
    }
}

template<typename OutValuesType>
void LutLayer<OutValuesType>::updateGradients(std::vector<float>& prevLayerLastGradients) {
/*
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
*/

}


} /* namespace lutNN */
