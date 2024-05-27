/*
 * LutNetwork.cpp
 *
 *  Created on: May 1, 2018
 *      Author: Karol Bunkowski
 */

#include <math.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <exception>
#include <unordered_set>
#include <cassert>

///timer/timer.hpp>
#include "lutNN/lutNN2/interface/LutNetworkBase.h"
#include "lutNN/lutNN2/interface/SumNode.h"

namespace lutNN {

LayerConfig::~LayerConfig() {
    std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<std::endl;
}

LutNetworkBase::LutNetworkBase(NetworkOutNode* outputNode, bool useNoHitCntNode): outputNode(outputNode), layersStat(layersConf.size()) {
    //initLuts();
	beta1 = 0.8;
    beta2 = 0.999;

    if(useNoHitCntNode) {
        noHitCntNode = new InputNode(0);
        noHitCntNode->setName("noHitCntNode");
    }
}

/*
void LutNetwork::initLuts() {
    //filing LUTs with some initial values
    for(unsigned int iLayer = 0; iLayer < layers.size(); iLayer++ ) {
        for(unsigned int iNode = 0; iNode < layers[iLayer].size(); iNode++) {
            for(unsigned int iLut = 0; iLut < layers[iLayer][iNode]->getLuts().size(); iLut++) {
                for(unsigned int iAddr = 0; iAddr < layers[iLayer][iNode]->getLuts()[iLut]->getValues().size(); iAddr++) {
                    //double x =
                    //neuronLayers[iLayer][iNode]->getLuts()[iLut]->getLut()[iAddr] = 20. - 0.0005 * pow(iAddr - (iLayer + iNode + iLut)*20. - 64, 3);
                    //neuronLayers[iLayer][iNode]->getLuts()[iLut]->getLut()[iAddr] =  iAddr - (iLayer * 10. + iNode * 10. + iLut *10.)*1. - 64; //pow(-1, iLut) *
                    if(iLayer == 0 && iAddr == 32)
                        neuronLayers[iLayer][iNode]->getLuts()[iLut]->getValues()[iAddr] = 16;
                    if(iLayer == 1 && iAddr == 64 + 32)
                        neuronLayers[iLayer][iNode]->getLuts()[iLut]->getValues()[iAddr] = 1;

                    if(iLayer == layers.size() -1 )
                        layers[iLayer][iNode]->getLuts()[iLut]->getValues()[iAddr] = ((int)iAddr)/(2.+iLut) + iLut * 10 - iNode * 20;// - (int)layersDef[iLayer]->lutAddrOffset;
                    else {
                        //neuronLayers[iLayer][iNode]->getLuts()[iLut]->getValues()[iAddr] = (pow(-1, iLut) * ((int)iAddr) - pow(-1, iLut +1 ) * ((int)layersDef[iLayer]->lutAddrOffset/2  - 50. * iNode)) * 0.1;
                        layers[iLayer][iNode]->getLuts()[iLut]->getValues()[iAddr] = 0.5 * pow(-0.8, iLut + iNode) * ( (int)iAddr - (int)layersDef[iLayer]->nextLayerLutSize/2. - 11. * iNode + 15. * iLut);

                    }
                }
            }
        }
    }
}*/


void LutNetworkBase::initLuts2(std::default_random_engine& generator) {
    //filing LUTs with some initial values
    for(unsigned int iLayer = 0; iLayer < layers.size(); iLayer++ ) {
        if(layersConf.at(iLayer)->nodeType != LayerConfig::lutInter && layersConf.at(iLayer)->nodeType != LayerConfig::lutNode)
            continue;

        std::uniform_real_distribution<> b0dist(layersConf.at(iLayer)->minLutVal/4, layersConf.at(iLayer)->maxLutVal/4); //offset
        std::uniform_real_distribution<> b1dist(layersConf.at(iLayer)->initSlopeMin, layersConf.at(iLayer)->initSlopeMax );
        std::uniform_real_distribution<> b2dist(0, 20);

        std::uniform_int_distribution<> binrydist(0, 1);

        for(unsigned int iNode = 0; iNode < layers[iLayer].size(); iNode++) {
            auto lutNode = static_cast<LutNode*>(layers[iLayer][iNode].get() );
            unsigned int lutSize = lutNode->getFloatValues().size();

            //std::uniform_real_distribution<> randomOutDist(-300./layers[iLayer][iNode]->getLuts().size(), 300./layers[iLayer][iNode]->getLuts().size() );
            //std::normal_distribution<double> randomOutDist(0, 2.0);

            double b0 = b0dist(generator);
            double b1 = b1dist(generator);
            double b2 = b2dist(generator);

            if(binrydist(generator))
                b1 *= -1;

            b0 = - b1 * lutSize / layersConf.at(iLayer)->lutRangesCnt / 2;

           /* double b0_1 = b0dist(generator);
            double b1_1 = b1dist(generator);
            double b2_1 = b2dist(generator);*/
           /* if(iLayer != neuronLayers.size() -1)
                b1 = b1 * neuronLayers[iLayer][iNode]->getLuts().size() * 4.;*/

            float offset = b0;
            if(offset * b1 > 0)
                offset = -offset;

            float val = offset;
            unsigned int range = lutSize / layersConf.at(iLayer)->lutRangesCnt;
            for(unsigned int iAddr = 0; iAddr < lutSize; iAddr++) {
                if(iAddr % range == 0)
                    val = offset;

                lutNode->getFloatValues()[iAddr] = val;
                if(val > layersConf.at(iLayer)->maxLutVal)
                    lutNode->getFloatValues()[iAddr] = layersConf.at(iLayer)->maxLutVal;
                else if (val < layersConf.at(iLayer)->minLutVal)
                    lutNode->getFloatValues()[iAddr] = layersConf.at(iLayer)->minLutVal;

                val += b1;

            }

            if(iLayer == 0 && layersConf.at(iLayer)->noHitValue) {
                lutNode->getFloatValues().back() = 0;//TODO the value for the address meaning no hit !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

				/* for version with no separation of distributions for different hits count
                for(unsigned int iRange = 0; iRange < layersConf.at(iLayer)->lutRangesCnt; iRange++) {
                    lutNode->getFloatValues()[(iRange+1) * lutNode->getFloatValues().size() / layersConf.at(iLayer)->lutRangesCnt -1] = 1024/16;
                } */
            }
        }
    }
}

void LutNetworkBase::initLuts(std::default_random_engine& generator) {
    //filing LUTs with some initial values
    for(unsigned int iLayer = 0; iLayer < layers.size(); iLayer++ ) {
        if(layersConf.at(iLayer)->nodeType != LayerConfig::lutInter && layersConf.at(iLayer)->nodeType != LayerConfig::lutNode)
            continue;

        std::uniform_real_distribution<> b0dist( -layersConf.at(iLayer)->maxLutVal/4, layersConf.at(iLayer)->maxLutVal/4); //offset
        std::uniform_real_distribution<> b1dist(-layersConf.at(iLayer)->maxLutVal / (1024./layersConf.at(iLayer)->lutRangesCnt), layersConf.at(iLayer)->maxLutVal /(1024./layersConf.at(iLayer)->lutRangesCnt) );
        std::uniform_real_distribution<> b2dist(0, 20);

        std::uniform_int_distribution<> binrydist(0, 1);

        double stdDev = layersConf.at(iLayer)->maxLutVal/50.;
        if(iLayer == 4)
            stdDev = 0;
        std::normal_distribution<double> normalDist(0, stdDev);

        double offset = 0;

        for(unsigned int iNode = 0; iNode < layers[iLayer].size(); iNode++) {
            auto lutNode = static_cast<LutNode*>(layers[iLayer][iNode].get() );
            unsigned int lutSize = lutNode->getFloatValues().size();

            //std::uniform_real_distribution<> randomOutDist(-300./layers[iLayer][iNode]->getLuts().size(), 300./layers[iLayer][iNode]->getLuts().size() );
            //std::normal_distribution<double> randomOutDist(0, 2.0);

            double b0 = b0dist(generator);
            double b1 = b1dist(generator);
            double b2 = b2dist(generator);

            //if(binrydist(generator))
            //    b1 *= -1;

            b0 = - b1 * lutSize / layersConf.at(iLayer)->lutRangesCnt / 2;

           /* double b0_1 = b0dist(generator);
            double b1_1 = b1dist(generator);
            double b2_1 = b2dist(generator);*/
           /* if(iLayer != neuronLayers.size() -1)
                b1 = b1 * neuronLayers[iLayer][iNode]->getLuts().size() * 4.;*/

            int dir = binrydist(generator);
            float offset = b0;
            if(offset * b1 > 0)
                offset = -offset;

            float val = offset;
            unsigned int range = lutSize / layersConf.at(iLayer)->lutRangesCnt;
            for(unsigned int iAddr = 0; iAddr < lutSize; iAddr++) {
                if(iAddr % range == 0)
                    val = offset;


                lutNode->getFloatValues()[iAddr] = val;

                if(val > layersConf.at(iLayer)->maxLutVal)
                    lutNode->getFloatValues()[iAddr] = layersConf.at(iLayer)->maxLutVal;
                else if (val < layersConf.at(iLayer)->minLutVal)
                    lutNode->getFloatValues()[iAddr] = layersConf.at(iLayer)->minLutVal;


                //std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" "<<lutNode->getName()<<" iAddr "<<iAddr<<" val "<<val<<std::endl;

                val += b1;


                /*if(iLayer == 0) {
                    layers[iLayer][iNode]->getLuts()[iLut]->getValues()[iAddr] = 0; //randomOutDist(generator);
                }
                else {//if(iLayer = neuronLayers.size() -1) {
                    if(dir == 0)
                        dir = -1;
                    if(b1_1 < 8)
                        b1 = 0;

                    lutNode->getFloatValues()[iAddr] = b1 * 0.001 * (iAddr  - lutSize/2. + b2) * dir;
                }*/
                /*else {
                    b1 = 2.;
                    neuronLayers[iLayer][iNode]->getLuts()[iLut]->getValues()[iAddr] = b1 * sin( b0 * iAddr / lutSize + b2)
                    + b1_1 * sin( b0_1 * iAddr / lutSize + b2_1) + offset;
                    //double b1 = b0dist(generator);
                    //neuronLayers[iLayer][iNode]->getLuts()[iLut]->getValues()[iAddr] = b1;
                }*/
            }

            /*if(iLayer == 0) {
                lutNode->getFloatValues().back() = 0; //TODO the value for the address meaning no hit !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            }*/
        }
    }
}

LutNetworkBase::~LutNetworkBase() {
    std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<std::endl;
}

/*
void LutNetworkBase::calcualteOutputValues() {
    if(outputType == simple) {
        auto outValue  = outputValues.begin();
        for(auto& outNode : layers.back()) {
 *outValue = outNode->getOutValue();
            outValue++;
        }
    }
    else if(outputType == softMax) {
        double maxOutVal = std::numeric_limits<double>::lowest();
        for(auto& outNode : layers.back()) {
            if(maxOutVal < outNode->getOutValue() )
                maxOutVal = outNode->getOutValue();
        }

        double denominator = 0;
        auto outValue  = outputValues.begin();
        for(auto& outNode : layers.back()) {
 *outValue = exp( outNode->getOutValue() - maxOutVal); //in run() outValUpdated = outVal
            denominator += *outValue;

            if(isinf( denominator ) || isnan( denominator )) {
                std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" outValue "<<*outValue<<" denominator "<<denominator
                         <<" getOutValUpdated "<<outNode->getOutValue()<<std::endl;
                exit(1);
            }

            outValue++;
        }
        if(denominator == 0) {
            std::cout<<__FUNCTION__<<":"<<__LINE__<<" denominator == 0 !!!!!!!!!!! "<<*outValue<<" denominator "<<denominator<<std::endl;
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

}

void softMaxFunction(std::vector<double>& inputValues, std::vector<double>& outputValues) {
    double maxOutVal = std::numeric_limits<double>::lowest();
    for(auto& inVal : inputValues) {
        if(maxOutVal < inVal )
            maxOutVal = inVal;
    }

    double denominator = 0;
    auto outValue  = outputValues.begin();
    for(auto& inVal : inputValues) {
 *outValue = exp( inVal - maxOutVal); //in run() outValUpdated = outVal
        denominator += *outValue;

        if(isinf( denominator ) || isnan( denominator )) {
            std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" outValue "<<*outValue<<" denominator "<<denominator
                     <<" getOutValUpdated "<<inVal<<std::endl;
            exit(1);
        }

        outValue++;
    }
    if(denominator == 0) {
        std::cout<<__FUNCTION__<<":"<<__LINE__<<" denominator == 0 !!!!!!!!!!! "<<*outValue<<" denominator "<<denominator<<std::endl;
        return; //TODO what todo then
    }
    for(auto& outVal : outputValues) {
        outVal /= denominator;

        if(isnan( outVal )) {
            std::cout<<__FUNCTION__<<":"<<__LINE__<<" outValue "<<outVal<<" denominator "<<denominator<<std::endl;
            exit(1);
        }
    }

}*/
/*
template <typename InputType>
void LutNetworkBase::setInputs(Event<InputType>* event, int efficiency, unsigned int minHitCnt,
        std::default_random_engine& rndGenerator)
{
    std::uniform_int_distribution<> rndDist(0, 100);
    unsigned int hitCnt = 0;
    for(unsigned int inputNode = 0; inputNode < getInputNodes().size(); inputNode++) {
        if(event->inputs[inputNode] != event->noHitVal) {
            if(inputNode != 7 && rndDist(rndGenerator) > efficiency) { //TODO!!!!!!!!!!!!!!!!!!!! 7 is the CSC ME2/2, for some reason we dont want to drop this input, remove when use for other purpose then OMTF
                getInputNodes()[inputNode]->setInput(event->noHitVal);
            }
            else {
                getInputNodes()[inputNode]->setInput(event->inputs[inputNode]);
                hitCnt++;
            }
        }
        else {
            getInputNodes()[inputNode]->setInput(event->inputs[inputNode]); //clear the value from the previous run
        }
    }

    if(getNoHitCntNode() != 0)
        getNoHitCntNode()->setInput(getInputNodes().size() - hitCnt);

    //TODO debug begin remove
    std::cout<<"-------------------"<<std::endl;
    for(unsigned int inputNode = 0; inputNode < getInputNodes().size(); inputNode++) {
        std::cout<<inputNode<<" "<<getInputNodes()[inputNode]->getOutValue()<<std::endl;
    }

    EventFloatOmtf* eventFloatOmtf = dynamic_cast<EventFloatOmtf*>(event);
    std::cout<<(*eventFloatOmtf)<<std::endl;
    //TODO debug end

    //if too few hits left, setting the original event
    if(hitCnt < minHitCnt)
    {
        setInputs(event);
    }
}*/

/*
template void LutNetworkBase::setInputs(EventInt* event, int efficiency, unsigned int minHitCnt, std::default_random_engine& rndGenerator);
template void LutNetworkBase::setInputs(EventFloat* event, int efficiency, unsigned int minHitCnt, std::default_random_engine& rndGenerator);

template <typename InputType>
void LutNetworkBase::run(Event<InputType>* event,  int inputEfficiency, unsigned int minHitCnt, std::default_random_engine& rndGenerator) {
    setInputs<InputType>(event, inputEfficiency, minHitCnt, rndGenerator);
    run();
    for(unsigned int iOut = 0;iOut < getOutputValues().size();iOut++) {
        event->nnResult[iOut] = getOutputValues()[iOut];
    }
}

template void LutNetworkBase::run(EventInt* event, int efficiency, unsigned int minHitCnt, std::default_random_engine& rndGenerator);
template void LutNetworkBase::run(EventFloat* event, int efficiency, unsigned int minHitCnt, std::default_random_engine& rndGenerator);*/

void LutNetworkBase::run(float eventWeight) {
    for(auto& layer : layers) {
        for(auto& node : layer) {
            node->run(eventWeight);
        }
    }
    //calcualteOutputValues();
    outputNode->run();
}

//TODO maerge the two versions of  runTraining using tempalte
template <typename InputType>
void LutNetworkBase::runTraining(const std::vector<InputType>& inputs, unsigned short classLabel,
        float eventWeight, CostFunction& costFunction) {
    //std::cout<<__FUNCTION__<<":"<<__LINE__<<" calling run "<<std::endl;
    run(inputs, eventWeight);

    double cost = costFunction.get(classLabel, getOutputValues() );

    totalCost += cost;
    eventCnt++;
    //std::cout<<__FUNCTION__<<":"<<__LINE__<<" totalCost "<<totalCost<<std::endl;

    outputNode->calcualteGradient(costFunction, classLabel, eventWeight);
    //std::cout<<__FUNCTION__<<":"<<__LINE__<<std::endl;
    for(int iLayer = layers.size() -1; iLayer >= 0 ; iLayer-- ) {//TODO from which layer it should start?
        for(auto& node : layers.at(iLayer)) {
            node->updateGradient(); //updates gradient of this node, and propagates gradient back
        }
    }
}

template void LutNetworkBase::runTraining(const std::vector<int>& inputs, unsigned short classLabel, float eventWeight, CostFunction& costFunction);
template void LutNetworkBase::runTraining(const std::vector<float>& inputs, unsigned short classLabel, float eventWeight, CostFunction& costFunction);

template <typename InputType>
void LutNetworkBase::runTraining(const std::vector<InputType>& inputs, const std::vector<double>& expextedResult,
        float eventWeight, CostFunction& costFunction) {
    //std::cout<<__FUNCTION__<<":"<<__LINE__<<" calling run "<<std::endl;
    run(inputs, eventWeight);

    double cost = costFunction(expextedResult, getOutputValues() );

    totalCost += cost;
    eventCnt++;
    //std::cout<<__FUNCTION__<<":"<<__LINE__<<" totalCost "<<totalCost<<std::endl;

    outputNode->calcualteGradient(costFunction, expextedResult, eventWeight);

    for(int iLayer = layers.size() -1; iLayer >= 0 ; iLayer-- ) {//TODO from which layer it should start?
        for(auto& node : layers.at(iLayer)) {
            node->updateGradient(); //updates gradient of this node, and propagates gradient back
        }
    }
}

template void LutNetworkBase::runTraining(const std::vector<int>& inputs, const std::vector<double>& expextedResult, float eventWeight, CostFunction& costFunction);
template void LutNetworkBase::runTraining(const std::vector<float>& inputs, const std::vector<double>& expextedResult, float eventWeight, CostFunction& costFunction);


//template void LutInterNetwork::runTraining(EventInt* event, CostFunction& costFunction);
//template void LutInterNetwork::runTraining(EventFloat* event, CostFunction& costFunction);

/*

template <typename InputType>
void LutInterNetwork::runTraining(Event<InputType>* event, CostFunction& costFunction,
        int efficiency, unsigned int minHitCnt, std::default_random_engine& rndGenerator) {
    //std::cout<<__FUNCTION__<<":"<<__LINE__<<" calling run "<<std::endl;
    run(event, efficiency, minHitCnt, rndGenerator);

    double cost = costFunction(event->expextedResult, getOutputValues() );

    totalCost += cost;
    eventCnt++;
    //std::cout<<__FUNCTION__<<":"<<__LINE__<<" totalCost "<<totalCost<<std::endl;

    outputNode->calcualteGradient(costFunction, event->expextedResult);

    for(int iLayer = layers.size() -1; iLayer >= 0 ; iLayer-- ) {//TODO from which layer it should start?
        for(auto& node : layers.at(iLayer)) {
            node->updateGradient();
        }
    }
}

template void LutInterNetwork::runTraining(EventInt* event, CostFunction& costFunction, int efficiency, unsigned int minHitCnt, std::default_random_engine& rndGenerator);
template void LutInterNetwork::runTraining(EventFloat* event, CostFunction& costFunction, int efficiency, unsigned int minHitCnt, std::default_random_engine& rndGenerator);
*/



void LutNetworkBase::calcualteAdamBiasCorr() {
    beta1ToT *= beta1;
    beta2ToT *= beta2;

    biasCorr0 =  1. / (1 - beta1ToT);
    biasCorr1 =  1. / (1 - beta2ToT);
}


double LutNetworkBase::getLutChangeAdam(LutNode* lutNode, unsigned int addr, double gradient, double alpha) {
	lutNode->getMomentum()[addr][0] = beta1 * lutNode->getMomentum()[addr][0] + (1 - beta1) * gradient;
	lutNode->getMomentum()[addr][1] = beta2 * lutNode->getMomentum()[addr][1] + (1 - beta2) * gradient * gradient;

	double m0 = lutNode->getMomentum()[addr][0] * biasCorr0;
	double m1 = lutNode->getMomentum()[addr][1] * biasCorr1;

	double change = alpha * m0 /( sqrt(m1) + 1e-8);

	return change;
}


void LutNetworkBase::reset() {
    //meanCost =
    //std::cout<<__FUNCTION__<<":"<<__LINE__<<" totalCost "<<totalCost<<" eventCnt "<<eventCnt<<" totalCost/eventCnt "<<totalCost/eventCnt<<std::endl; //<<" notUpdatedNodes/evnCnt "<<((float)notUpdatedNodes)/eventCnt
    totalCost = 0;

    wellClassifiedEvents = 0;

    eventCnt = 0;
    notUpdatedNodes = 0;
    for(auto& layer : layers) {
        for(auto& node : layer) {
            node->reset();
        }
    }

    /*    for(unsigned int iLayer = 0; iLayer < layersStat.size(); iLayer++) {
        layersStat[iLayer].meanDeltaAddr = 0;
    }*/
}


std::string LayerConfig::nodeTypeToStr(LayerConfig::NodeType nodeType) {
    /*    softMax,
        sumNode,
        lutNode,
        lutInter,
        neuron,*/

    if(nodeType == LayerConfig::lutNode)
        return "lutNode";

    if(nodeType == LayerConfig::lutInter)
        return "lutInter";

    if(nodeType == LayerConfig::sumNode)
        return "sumNode";

    if(nodeType == LayerConfig::lutBinary)
        return "lutBinary";

    return "unknown";

}

std::ostream & operator << (std::ostream &out, LutNetworkBase& net) {
    out<<"LutNetwork: inputNodes: "<<net.inputNodes.size()<<" layers: "<<net.layers.size()<<std::endl;

    for(unsigned int iNode = 0; iNode < net.inputNodes.size(); iNode++) {
        out<<net.inputNodes[iNode]->getName()<<std::endl;
    }

    for(unsigned int iLayer = 0; iLayer < net.layers.size(); iLayer++ ) {
        for(unsigned int iNode = 0; iNode < net.layers[iLayer].size(); iNode++) {
            //out<<*(net.layers[iLayer][iNode])<<std::endl;
            out<<"layer "<<iLayer<<" node "<<std::setw(3)<<iNode<<" "<<net.layers[iLayer][iNode]->getName()<<", inputNodes: ";
            for(auto& inputNode : net.layers[iLayer][iNode]->getInputNodes())
                out<<inputNode->getName()<<" ";
            SumNode* sumNode = dynamic_cast<SumNode*> (net.layers[iLayer][iNode].get());
            if(sumNode) {
                out<<sumNode->print();
            }

            out<<std::endl;
        }
        out<<std::endl;
    }


    out<<"outputNode: "<<net.outputNode->getName()<<", inputNodes: ";
    for(auto& inputNode : net.outputNode->getInputNodes())
        out<<inputNode->getName()<<" ";
    out<<std::endl;
    if(net.noHitCntNode)
        out<<"noHitCntNode "<<net.noHitCntNode->getName();
    else
        out<<"noHitCntNode null";

    out<<"\nnetwork size summary"<<std::endl;
    int total = 0;
    int totalParameters = 0;
    for(unsigned int iLayer = 0; iLayer < net.layers.size(); iLayer++ ) {
        out<<"iLayer "<<iLayer<<" size "<<std::setw(5)<<net.layers[iLayer].size()<<" inputs size "<<std::setw(5)<<net.layers[iLayer].at(0)->getInputNodes().size();
        LutNode* lutNode = dynamic_cast<LutNode*>  (net.layers[iLayer].at(0).get());
        if(lutNode) {
            out<<" lutSize "<<std::setw(4)<<lutNode->getFloatValues().size();
            totalParameters += lutNode->getFloatValues().size() * net.layers[iLayer].size();
        }
        out<<std::endl;
        total += net.layers[iLayer].size();
    }
    out<<"total nodes count "<<total<<std::endl;
    out<<"totalParameters "<<totalParameters<<std::endl;

    out<<std::endl;
    return out;
}

void LutNetworkBase::print() {
    std::ostream& ostr = std::cout;
    ostr<<"LutNetworkBase::print()"<<std::endl;
    for(unsigned int iLayer = 0; iLayer < layers.size(); iLayer++) {
        ostr<<"iLayer "<<iLayer<<" nodesInLayer: "<<layersConf[iLayer]->nodesInLayer<<std::endl;
        for(auto& node : layers[iLayer]) {
            ostr<<"iLayer "<<iLayer<<" "<<node->getName()<<std::endl;
            for(unsigned int iInputNode = 0; iInputNode < node->getInputNodes().size(); iInputNode++) {
            	ostr<<"    input "<<iInputNode<<" "<<node->getInputNodes()[iInputNode]->getName()<<std::endl;
            }
        }
    }

    outputNode->print(ostr);
}

void LutNetworkBase::printLayerStat() {
    for(unsigned int iLayer = 0; iLayer < layersStat.size(); iLayer++) {
        double meanDeltaAddr = layersStat[iLayer].meanDeltaAddr/(double)eventCnt/layers[iLayer].size();
        std::cout<<"iLayer "<<iLayer
                <<" averageAbsDeltaLutValue "<<layersStat[iLayer].averageAbsDeltaLutValue
                <<" averageRelativeDeltaLutValue "<<layersStat[iLayer].averageRelativeDeltaLutValue
                <<" meanDeltaAddr "<<meanDeltaAddr
                <<" averageAbsDeltaLutInSmooth "<<layersStat[iLayer].averageAbsDeltaLutInSmooth
                <<" averageRelativeDeltaLutInSmooth "<<layersStat[iLayer].averageRelativeDeltaLutInSmooth
                <<std::endl;
    }
}

} /* namespace lutNN */
