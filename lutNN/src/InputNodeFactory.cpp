/*
 * InputNodeFactory.cpp
 *
 *  Created on: Sep 18, 2018
 *      Author: kbunkow
 */

#include "lutNN/interface/InputNodeFactory.h"

namespace lutNN {

InputNodeSelInputsFactoryFlatDist::InputNodeSelInputsFactoryFlatDist(unsigned int inputCnt, unsigned int selInputCnt, std::default_random_engine& generator):
        inputCnt(inputCnt), selInputCnt(selInputCnt), generator(generator) {
    // TODO Auto-generated constructor stub

}

InputNodeSelInputsFactoryFlatDist::~InputNodeSelInputsFactoryFlatDist() {
    // TODO Auto-generated destructor stub
}

InputNode* InputNodeSelInputsFactoryFlatDist::get(unsigned int number) {
    std::vector<int> selectedInputs(selInputCnt, 0);
    std::vector<int> selectedInputThresholds(selInputCnt, 0);

    //std::normal_distribution<int> randomOutDist(0, radious);
    std::uniform_int_distribution<> flatDist(0, inputCnt-1);
    for(auto& selectedInput : selectedInputs) {
        selectedInput = flatDist(generator);
    }

    for(auto& selectedInputThreshold : selectedInputThresholds) {
        selectedInputThreshold = 64; //TODO maybe add random dist
    }

    return new InputNodeSelInputs(number, selectedInputs, selectedInputThresholds);
}

InputNodeSelBinaryInputsFactoryFlatDist::InputNodeSelBinaryInputsFactoryFlatDist(unsigned int inputCnt, unsigned int selInputCnt, std::default_random_engine& generator):
        inputCnt(inputCnt), selInputCnt(selInputCnt), generator(generator) {
    // TODO Auto-generated constructor stub

}

InputNode* InputNodeSelBinaryInputsFactoryFlatDist::get(unsigned int number) {
    std::vector<int> selectedInputs(selInputCnt, 0);

    //std::normal_distribution<int> randomOutDist(0, radious);
    std::uniform_int_distribution<> flatDist(0, inputCnt-1);
    for(auto& selectedInput : selectedInputs) {
        selectedInput = flatDist(generator);
    }
    return new InputNodeSelBinaryInputs(number, selectedInputs);
}

} /* namespace lutNN */
