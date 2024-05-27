/*
 * ClassifierOutNode.cpp
 *
 *  Created on: Sep 15, 2019
 *      Author: kbunkow
 */

#include "lutNN/lutNN2/interface/NetworkOutNode.h"
#include <math.h>
#include <iostream>
#include <limits>

namespace lutNN {

NetworkOutNode::NetworkOutNode(): Node(0, 0), outputValues(0, 0) {

}

NetworkOutNode::NetworkOutNode(unsigned int classesCnt): Node(0, classesCnt), outputValues(classesCnt, 0) {
    std::cout<<" NetworkOutNode constructor"<<std::endl;
}

NetworkOutNode::~NetworkOutNode() {
    // TODO Auto-generated destructor stub
}

void NetworkOutNode::run(float eventWeight) {
    for(unsigned int i = 0; i < inputNodes.size(); i++) {
        outputValues.at(i) = inputNodes[i]->getOutValue();
        //std::cout<<"NetworkOutNode::run() "<<i<<" outputValues "<<outputValues.at(i) <<" inputNodes->getOutValue "<<inputNodes[i]->getOutValue()<<std::endl;
    }

}

void NetworkOutNode::calcualteGradient(CostFunction& costFunction, std::vector<double> expextedResults, float eventWeight) {
    for(unsigned int iClass = 0; iClass < outputValues.size(); iClass++) {
        inputNodes[iClass]->getLastGradient() = costFunction.getDerivative()[iClass] * eventWeight;
    }
}

void NetworkOutNode::calcualteGradient(CostFunction& costFunction, unsigned short& expectedClassLabel, float eventWeight) {
    //std::cout<<"NetworkOutNode::calcualteGradient"<<std::endl;
    for(unsigned int iClass = 0; iClass < outputValues.size(); iClass++) {
        inputNodes[iClass]->getLastGradient() = costFunction.getDerivative()[iClass] * eventWeight;
        //std::cout<<" iClass "<<iClass<<" "<<costFunction.getDerivative()[iClass]<<" getLastGradient "<<static_cast<LutNode*>(inputNodes[iClass])->getLastGradient()<<std::endl;
    }
}

const std::vector<double>& NetworkOutNode::updateInput(unsigned int iClass, unsigned int iSubClass, double newInVal) {
    modifiedOutputValues = outputValues;

    modifiedOutputValues.at(iClass) = newInVal;
    return modifiedOutputValues;
}

void NetworkOutNode::print(std::ostream& ostr) const {
    ostr<<"NetworkOutNode"<<std::endl;
    for(unsigned int iClass = 0; iClass < outputValues.size(); iClass++) {
        ostr<<" iClass "<<iClass<<inputNodes[iClass]->getName()<<std::endl;
    }
}

SoftMax::SoftMax(): NetworkOutNode(0) {
    name = "SoftMaxNode";
}

SoftMax::SoftMax(unsigned int classesCnt): NetworkOutNode(classesCnt), inputValues(classesCnt, 0) {
    name = "SoftMaxNode";
}

void SoftMax::run(float eventWeight) {
    double maxOutVal = std::numeric_limits<double>::lowest();

    auto inputValue = inputValues.begin();
    for(auto& inputNode : inputNodes) {
        (*inputValue) = inputNode->getOutValue(); //copying the output values from the input nodes
        if(maxOutVal < (*inputValue) )
            maxOutVal = (*inputValue);
        inputValue++;
    }

    softMax(inputValues, maxOutVal, outputValues);
}


void SoftMax::softMax(std::vector<double>& inputValues, double& maxInVal, std::vector<double >& outputValues) {
    double denominator = 0;
    auto outValue  = outputValues.begin();
    for(auto& inputValue : inputValues) {
        double expVal = exp(inputValue - maxInVal);
        *outValue = expVal;
        denominator += expVal;

       /* if(isinf( denominator ) || isnan( denominator )) {
            std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" outValue "<<*outValue<<" denominator "<<denominator
                     <<" getOutValUpdated "<<inputValue<<std::endl;
            exit(1);
        }*/

        outValue++;
    }
    if(denominator == 0) {
        std::cout<<__FUNCTION__<<":"<<__LINE__<<" denominator == 0 !!!!!!!!!!! "<<*outValue<<" denominator "<<denominator<<std::endl;
        return; //TODO what todo then
    }
    for(auto& outValue : outputValues) {
        outValue /= denominator;

        if(isnan( outValue )) {
            std::cout<<__FUNCTION__<<":"<<__LINE__<<" outValue "<<outValue<<" denominator "<<denominator<<std::endl;
            exit(1);
        }
    }
}

void SoftMax::calcualteGradient(CostFunction& costFunction, std::vector<double> expextedResults, float eventWeight) {
    auto expextedResult = expextedResults.begin();
    auto inputNode = inputNodes.begin();
    for(auto& outValue : outputValues) {
        double gradient = outValue - *expextedResult;
        (*inputNode)->getLastGradient() = gradient * eventWeight;

        expextedResult++;
        inputNode++;
    }
}


void SoftMax::calcualteGradient(CostFunction& costFunction, unsigned short& expectedClassLabel, float eventWeight) {
    for(unsigned int iClass = 0; iClass < outputValues.size(); iClass++) {
        double expextedResult = (iClass == expectedClassLabel ? 1 : 0 );
        double gradient = outputValues[iClass] - expextedResult;
        inputNodes[iClass]->getLastGradient() = gradient * eventWeight;
    }

}

SoftMaxWithSubClasses::SoftMaxWithSubClasses(): NetworkOutNode(0) {
    name = "SoftMaxWithSubClassesNode";
}

SoftMaxWithSubClasses::SoftMaxWithSubClasses(unsigned int classesCnt): NetworkOutNode(classesCnt), inputNodesBySubClasses(classesCnt), inputValues(classesCnt) {
    name = "SoftMaxWithSubClassesNode";
}

void SoftMaxWithSubClasses::softMax(std::vector<std::vector<double> >& inputValues, double maxInVal, std::vector<double >& outputValues) {
    double denominator = 0;

    for(auto& outValue : outputValues ) {
        outValue = 0;
    }

    auto outValueIt  = outputValues.begin();
    for(auto& classValues : inputValues) {
        for(auto& inputValue : classValues) {
            double expVal = exp(inputValue - maxInVal);
            *outValueIt += expVal;
            denominator += expVal;
        }
        outValueIt++;
    }

    if(denominator == 0) {
        std::cout<<__FUNCTION__<<":"<<__LINE__<<" denominator == 0 !!!!!!!!!!! "<<" denominator "<<denominator<<std::endl;
        return; //TODO what todo then
    }
    for(auto& outVal : outputValues) {
        outVal /= denominator;

        if(isnan( outVal )) {
            std::cout<<__FUNCTION__<<":"<<__LINE__<<" outValue "<<outVal<<" denominator "<<denominator<<std::endl;
            exit(1);
        }
    }
}


void SoftMaxWithSubClasses::run(float eventWeight) {
/*    double maxOutVal = std::numeric_limits<double>::lowest();
    for(auto& inVal : inputValues) {
        if(maxOutVal < inVal )
            maxOutVal = inVal;
    }*/

    //copying the output values from the input nodes
/*    auto classValues = inputValues.begin();
    for(auto& classNodes : inputNodeSubClass) {
        auto inputValue = classValues->begin();
        for(auto& inputNode : classNodes) {
            (*inputValue) = inputNode->getOutValue();
            inputValue++;
        }
        classValues++;
    }*/

    for(unsigned int iClass = 0; iClass < inputValues.size(); iClass++) {
        for(unsigned int iSubClass = 0; iSubClass < inputValues[iClass].size(); iSubClass++) {
            inputValues[iClass][iSubClass] = inputNodesBySubClasses[iClass][iSubClass]->getOutValue();
        }
    }
    softMax(inputValues, maxInVal,outputValues);
}

const std::vector<double>& SoftMaxWithSubClasses::updateInput(unsigned int iClass, unsigned int iSubClass, double newInVal) {
    auto& modifiedVal = inputValues.at(iClass).at(iSubClass);
    double orgValue = modifiedVal;
    modifiedVal = newInVal;
    softMax(inputValues, maxInVal, outputValues);
    modifiedVal = orgValue;
    return outputValues;
}

void SoftMaxWithSubClasses::calcualteGradient(CostFunction& costFunction, std::vector<double> expextedResults, float eventWeight) {

}

void SoftMaxWithSubClasses::calcualteGradient(CostFunction& costFunction, unsigned short& expectedClassLabel, float eventWeight) {
    double denominator = 0;
    double expectedClassDenominator = 0;

    for(unsigned int iClass = 0; iClass < inputValues.size(); iClass++) {
        for(unsigned int iSubClass = 0; iSubClass < inputValues[iClass].size(); iSubClass++) {
            double expVal = exp(inputValues[iClass][iSubClass] - maxInVal);
            denominator += expVal;
            if(iClass == expectedClassLabel) {
                expectedClassDenominator += expVal;
            }
        }
    }

    if(denominator == 0 || expectedClassDenominator == 0) {
        std::cout<<__FUNCTION__<<":"<<__LINE__<<" denominator == 0 !!!!!!!!!!! "<<" denominator "<<denominator<<" expectedClassDenominator "<<expectedClassDenominator
                <<" expectedClassLabel "<<expectedClassLabel<<std::endl;
        return; //TODO what todo then
    }

    for(unsigned int iClass = 0; iClass < inputValues.size(); iClass++) {
        for(unsigned int iSubClass = 0; iSubClass < inputValues[iClass].size(); iSubClass++) {
            double expVal = exp(inputValues[iClass][iSubClass] - maxInVal);

            double gradient = expVal / denominator;

            if(iClass == expectedClassLabel) {
                gradient -= (expVal / expectedClassDenominator);
            }

            inputNodesBySubClasses[iClass][iSubClass]->getLastGradient() = gradient * eventWeight;
        }
    }
}

void SoftMaxWithSubClasses::print(std::ostream& ostr) const {
    ostr<<"SoftMaxWithSubClasses"<<std::endl;
    for(unsigned int iClass = 0; iClass < inputValues.size(); iClass++) {
        for(unsigned int iSubClass = 0; iSubClass < inputValues[iClass].size(); iSubClass++) {
            ostr<<" iClass "<<iClass<<" iSubClass "<<iSubClass<<" "<<inputNodesBySubClasses[iClass][iSubClass]->getName()<<std::endl;
        }
    }
}

} /* namespace lutNN */
