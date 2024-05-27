/*
 * LutNetwork.cpp
 *
 *  Created on: May 1, 2018
 *      Author: Karol Bunkowski
 */

#include "lutNN/interface/LutNetwork.h"
#include <math.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <exception>
#include <unordered_set>
#include <cassert>

namespace lutNN {

void Event::print() {
    for(unsigned int i = 0; i < inputs.size(); i++) {
        std::cout<<"input "<<i<<" val "<<inputs[i]<<std::endl;
    }

    for(unsigned int i = 0; i < expextedResult.size(); i++) {
        std::cout<<"result "<<i<<" exp "<<std::setw(5)<<expextedResult[i]<<" nnOut "<<nnResult[i]<<std::endl;
    }
}

double CostFunctionMeanSquaredError::get(std::vector<double> expextedResults, std::vector<double> nnResults)  {
    double cost = 0;
    auto expextedResult = expextedResults.begin();
    for(auto& nnResult : nnResults) {
        cost += pow(*expextedResult - nnResult, 2);
        expextedResult++;
    }
    return cost;
}

double CostFunctionCrossEntropy::get(std::vector<double> expextedResults, std::vector<double> nnResults)  {
    auto expextedResult = expextedResults.begin();
    for(auto& nnResult : nnResults) {
        if(*expextedResult) {
            return -log(nnResult);
        }
        expextedResult++;
    }
    return 0;
}

LutNetwork::LutNetwork(std::vector<ConfigParametersPtr >& layersDef, OutputType outputType, InputNodeFactory* inputNodeFactory): layersDef(layersDef), outputType(outputType), layersStat(layersDef.size()),
        outputValues(layersDef.back()->neuronsInLayer, 0), inputNodeFactory(inputNodeFactory) {

    unsigned int inputNodesCnt = layersDef.at(0)->neuronInputCnt; //Default is ConfigParameters::fullyConnected
    if(layersDef[0]->layerType == ConfigParameters::oneToOne) {
        inputNodesCnt = layersDef.at(0)->neuronInputCnt * layersDef.at(0)->neuronsInLayer;
    }

    for(unsigned int iNode = 0; iNode < inputNodesCnt; iNode++) {
        inputNodes.emplace_back(std::unique_ptr<InputNode>(inputNodeFactory->get(iNode)));
    }

    for(unsigned int iLayer = 0; iLayer < layersDef.size(); iLayer++ ) {
        layers.emplace_back( NeuronLayer(layersDef[iLayer]->neuronsInLayer ) );
        for(unsigned int iNode = 0; iNode < layersDef[iLayer]->neuronsInLayer; iNode++) {
            layers.back()[iNode].reset(new NeuronNode(layersDef[iLayer], iLayer, iNode));
        }

    }

    //connections
    if(layersDef[0]->layerType == ConfigParameters::fullyConnected) {
        for(unsigned int iNode = 0; iNode < layers[0].size(); iNode++) {
            layers[0][iNode]->connectInputs(inputNodes);
        }
    }
    else if(layersDef[0]->layerType == ConfigParameters::oneToOne) {
        for(unsigned int iNode = 0; iNode < layers[0].size(); iNode++) {
            unsigned int neuronInputCnt = layers.at(0).at(iNode)->getLuts().size();
            for(unsigned int iLut = 0; iLut < neuronInputCnt; iLut++) {
                layers.at(0).at(iNode)->connectInput(inputNodes[iNode * neuronInputCnt + iLut].get(), iLut);
            }
        }
    }

    for(unsigned int iLayer = 1; iLayer < layers.size(); iLayer++ ) {
        if(layersDef[iLayer]->layerType == ConfigParameters::fullyConnected) {
            for(unsigned int iNode = 0; iNode < layers[iLayer].size(); iNode++) {
                layers[iLayer][iNode]->connectInputs(layers[iLayer -1]);
            }
        }
        else if(layersDef[iLayer]->layerType == ConfigParameters::singleLut) {
            for(unsigned int iNode = 0; iNode < layers[iLayer].size(); iNode++) {
                layers.at(iLayer).at(iNode)->connectInput(layers[iLayer -1][iNode], 0);
            }
        }
        else if(layersDef[iLayer]->layerType == ConfigParameters::oneToOne) {
            for(unsigned int iNode = 0; iNode < layers[iLayer].size(); iNode++) {
                unsigned int neuronInputCnt = layers.at(0).at(iNode)->getLuts().size();
                for(unsigned int iLut = 0; iLut < neuronInputCnt; iLut++) {
                    layers.at(0).at(iNode)->connectInput(layers[iLayer -1][iNode * neuronInputCnt + iLut], iLut);
                }
            }
        }
    }

    //connectOneNetPerOut();

    //connect1();

    //initLuts();
}
void LutNetwork::connectOneNetPerOut() {
    unsigned int outputCnt = layers.back().size();

    for(unsigned int iNode = 0; iNode < layers[0].size(); iNode++) {
        layers[0][iNode]->connectInputs(inputNodes);
    }

    for(unsigned int iOut = 0; iOut < outputCnt; iOut++) {
        if( (layers[0].size() % outputCnt) != 0)
            throw std::invalid_argument("LutNetwork::: wrong input layer size");

        for(unsigned int iLayer = 1; iLayer < layers.size(); iLayer++ ) {
            unsigned int nodesPerOut = layers[iLayer].size() / outputCnt;
            unsigned int prevLayNodesPerOut = layers[iLayer-1].size() / outputCnt;
            for(unsigned int iNode = iOut * nodesPerOut; iNode < (iOut +1) * nodesPerOut; iNode++) {
                int iLut = 0;
                for(unsigned int prevLayerNode = iOut * prevLayNodesPerOut ; prevLayerNode < (iOut +1)* prevLayNodesPerOut; prevLayerNode++) {
                    layers.at(iLayer).at(iNode)->connectInput(layers.at(iLayer -1).at(prevLayerNode), iLut++);
                }
            }
        }
    }
}

void LutNetwork::connect() {
    //connections
    for(unsigned int iNode = 0; iNode < layers[0].size(); iNode++) {
        layers[0][iNode]->connectInputs(inputNodes);
    }

    unsigned int iLayer = 1;
    unsigned int nodesCnt = layers[iLayer].size() / layers[iLayer-1].size();
    {

        for(unsigned int iNode = 0; iNode < layers[iLayer].size(); iNode++) {
            unsigned int prevLayerNode = iNode / nodesCnt;
            layers[iLayer][iNode]->connectInput(layers[iLayer -1][prevLayerNode], 0);
            layers[iLayer][iNode]->connectInput(inputNodes[inputNodes.size() -1].get(), 1); //fired planes layer
        }
    }

    iLayer = 2;
    {
        for(unsigned int iNode = 0; iNode < layers[iLayer].size(); iNode++) {
            assert(nodesCnt == layers[iLayer][iNode]->getLuts().size() );
            for(unsigned int iInput = 0; iInput < nodesCnt; iInput++ ) {
                unsigned int prevLayerNode = iNode * nodesCnt + iInput;
                layers[iLayer][iNode]->connectInput(layers[iLayer -1][prevLayerNode], iInput);
            }
        }
    }

    iLayer = 3;
    {
        for(unsigned int iNode = 0; iNode < layers[iLayer].size(); iNode++) {
            layers[iLayer][iNode]->connectInputs(layers[iLayer -1]);
        }
    }
}

void LutNetwork::connect1() {
    //connections

    for(unsigned int iNode = 0; iNode < layers[0].size(); iNode++) {
        layers[0][iNode]->connectInputs(inputNodes);
    }

    for(unsigned int iLayer = 1; iLayer < layers.size(); iLayer++ ) {
        for(unsigned int iNode = 0; iNode < layers[iLayer].size(); iNode++) {
            unsigned int prevLayerNode = 0;
            for(; prevLayerNode < layers[iLayer -1].size(); prevLayerNode++) {
                layers[iLayer][iNode]->connectInput(layers[iLayer -1][prevLayerNode], prevLayerNode);
            }
            layers[iLayer][iNode]->connectInput(inputNodes[inputNodes.size() -1].get(), prevLayerNode); //fired planes layer
        }

    }
}

void LutNetwork::initLuts() {
    //filing LUTs with some initial values
    for(unsigned int iLayer = 0; iLayer < layers.size(); iLayer++ ) {
        for(unsigned int iNode = 0; iNode < layers[iLayer].size(); iNode++) {
            for(unsigned int iLut = 0; iLut < layers[iLayer][iNode]->getLuts().size(); iLut++) {
                for(unsigned int iAddr = 0; iAddr < layers[iLayer][iNode]->getLuts()[iLut]->getValues().size(); iAddr++) {
                    //double x =
                    //neuronLayers[iLayer][iNode]->getLuts()[iLut]->getLut()[iAddr] = 20. - 0.0005 * pow(iAddr - (iLayer + iNode + iLut)*20. - 64, 3);
                    //neuronLayers[iLayer][iNode]->getLuts()[iLut]->getLut()[iAddr] =  iAddr - (iLayer * 10. + iNode * 10. + iLut *10.)*1. - 64; //pow(-1, iLut) *
                    /*if(iLayer == 0 && iAddr == 32)
                        neuronLayers[iLayer][iNode]->getLuts()[iLut]->getValues()[iAddr] = 16;
                    if(iLayer == 1 && iAddr == 64 + 32)
                        neuronLayers[iLayer][iNode]->getLuts()[iLut]->getValues()[iAddr] = 1;*/

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
}


void LutNetwork::initLuts(std::default_random_engine& generator) {
    //filing LUTs with some initial values
    for(unsigned int iLayer = 0; iLayer < layers.size(); iLayer++ ) {
        std::uniform_real_distribution<> b0dist(5, 10);
        std::uniform_real_distribution<> b1dist(0, 10);
        std::uniform_real_distribution<> b2dist(0, 20);

        std::uniform_int_distribution<> binrydist(0, 1);

        double offset = 0;
        /*if(iLayer == neuronLayers.size() -1) {
            b1dist = std::uniform_real_distribution<>(0.2, 0.8); //0.01, 0.3 was good for  8bit LUTs
            offset = 0.4 / neuronLayers[iLayer][0]->getLuts().size(); //1.4 good for 8bit LUTs
        }*/
        for(unsigned int iNode = 0; iNode < layers[iLayer].size(); iNode++) {
            for(unsigned int iLut = 0; iLut < layers[iLayer][iNode]->getLuts().size(); iLut++) {
                std::uniform_real_distribution<> randomOutDist(-300./layers[iLayer][iNode]->getLuts().size(), 300./layers[iLayer][iNode]->getLuts().size() );
                //std::normal_distribution<double> randomOutDist(0, 2.0);

                double b0 = b0dist(generator);
                double b1 = b1dist(generator);
                double b2 = b2dist(generator);

                double b0_1 = b0dist(generator);
                double b1_1 = b1dist(generator);
                double b2_1 = b2dist(generator);
                /*if(iLayer != neuronLayers.size() -1)
                    b1 = b1 * neuronLayers[iLayer][iNode]->getLuts().size() * 4.;
*/
                unsigned int lutSize = layers[iLayer][iNode]->getLuts()[iLut]->getValues().size();
                int dir = binrydist(generator);
                for(unsigned int iAddr = 0; iAddr < lutSize; iAddr++) {
                    if(iLayer == 0) {
                        layers[iLayer][iNode]->getLuts()[iLut]->getValues()[iAddr] = 0; //randomOutDist(generator);
                    }
                    else {//if(iLayer = neuronLayers.size() -1) {
                        if(dir == 0)
                            dir = -1;
                        if(b1_1 < 8)
                            b1 = 0;

                        layers[iLayer][iNode]->getLuts()[iLut]->getValues()[iAddr] = b1 * 0.001 * (iAddr  - lutSize/2. + b2) * dir;
                    }
                    /*else {
                        b1 = 2.;
                        neuronLayers[iLayer][iNode]->getLuts()[iLut]->getValues()[iAddr] = b1 * sin( b0 * iAddr / lutSize + b2)
                        + b1_1 * sin( b0_1 * iAddr / lutSize + b2_1) + offset;
                        //double b1 = b0dist(generator);
                        //neuronLayers[iLayer][iNode]->getLuts()[iLut]->getValues()[iAddr] = b1;
                    }*/
                }
            }
        }
    }
}

LutNetwork::~LutNetwork() {
    // TODO Auto-generated destructor stub
}


void LutNetwork::calcualteOutputValues() {
    if(outputType == simple) {
        auto outValue  = outputValues.begin();
        for(auto& outLayer : layers.back()) {
            *outValue = outLayer->getOutValUpdated();
            outValue++;
        }
    }
    else if(outputType == softMax) {
        double maxOutVal = std::numeric_limits<double>::lowest();
        for(auto& outLayer : layers.back()) {
            if(maxOutVal < outLayer->getOutValUpdated() )
                maxOutVal = outLayer->getOutValUpdated();
        }

        double denominator = 0;
         auto outValue  = outputValues.begin();
        for(auto& outLayer : layers.back()) {
            *outValue = exp( outLayer->getOutValUpdated() - maxOutVal); //in run() outValUpdated = outVal
            denominator += *outValue;

            if(isinf( denominator ) || isnan( denominator )) {
                std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" outValue "<<*outValue<<" denominator "<<denominator
                         <<" getOutValUpdated "<<outLayer->getOutValUpdated()<<std::endl;
                exit(1);
            }

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

}

void LutNetwork::run() {
    for(NeuronLayer& layer : layers) {
        for(NeuronNodePtr& neuron : layer) {
            neuron->run();
        }
    }
    calcualteOutputValues();
}

void LutNetwork::runFromLayer(unsigned int fromLayer) {
    //std::cout<<__FUNCTION__<<":"<<__LINE__<<" fromLayer "<<fromLayer<<std::endl;
    for(unsigned int iLayer = fromLayer; iLayer < layers.size(); iLayer++ ) {
        for(NeuronNodePtr& neuron : layers[iLayer]) {
            neuron->run();
        }
    }
    calcualteOutputValues();
}

void LutNetwork::runFromNeuron(NeuronNode* neuronNode) {
    //std::cout<<__FUNCTION__<<":"<<__LINE__<<" fromLayer "<<fromLayer<<std::endl;

    for(unsigned int iLayer = neuronNode->getLayer() + 1; iLayer < layers.size(); iLayer++ ) {
        for(NeuronNodePtr& neuron : layers[iLayer]) {
            neuron->resetUpdate();
        }
    }

    std::unordered_set<NeuronNode*> updatedNodes;
    for(auto& lut : neuronNode->getChildLuts() ) {
        //lut->getOwnerNode()->resetUpdate();
        bool outAddrWasChanged = lut->getOwnerNode()->update(lut);
        if(outAddrWasChanged) //for the last layer it has no sense, but it is not a problem, as further there is break in the inner for
            updatedNodes.insert(lut->getOwnerNode() ); //not good here
    }

    while(updatedNodes.size() ) {
        std::unordered_set<NeuronNode*> nextLayerUpdatedNodes;
        for(NeuronNode* neuronNode :  updatedNodes) {
            if(neuronNode->getLayer() == layers.size() - 1) {
                break; //we are in the last layer now
            }
            //neuronNode->resetUpdate(); todo reset apropretly
            for(auto& lut : neuronNode->getChildLuts() ) {
                bool outAddrWasChanged = lut->getOwnerNode()->update(lut);
                if(outAddrWasChanged)
                    nextLayerUpdatedNodes.insert(lut->getOwnerNode() );
            }
        }
        updatedNodes = nextLayerUpdatedNodes;
    }

    calcualteOutputValues();
}

/*void LutNetwork::runFromNeurons(std::unordered_set<NeuronNode*> updatedNodes) {
    //std::cout<<__FUNCTION__<<":"<<__LINE__<<" fromLayer "<<fromLayer<<std::endl;

    std::unordered_set<NeuronNode*> updatedNodes;
    for(auto& lut : neuronNode->getChildLuts() ) {
        lut->getOwnerNode()->update(lut);
        updatedNodes.insert(lut->getOwnerNode() );
    }
}*/

//in mode=1 only the network final outputs are trained
 void LutNetwork::runTraining(std::vector<double> expextedResults, CostFunction& costFunction) {
     int eventWeight = 1;
     //std::cout<<__FUNCTION__<<":"<<__LINE__<<" calling run "<<std::endl;
     run();
     double cost = costFunction(expextedResults, getOutputValues() );
     totalCost += cost;
     eventCnt++;

     //std::cout<<__FUNCTION__<<":"<<__LINE__<<" totalCost "<<totalCost<<std::endl;

     //last layer needs special threading
     for(NeuronNodePtr& neuron : layers.back() ) {
         //double deltaCost = -2. * (expextedResults[neuron->getNumber() ] - getOutputValues()[neuron->getNumber()] );
         //double deltaCost = getOutputValues()[neuron->getNumber()] - expextedResults[neuron->getNumber() ]; //for cross entropy cost  //todo move  to the cost fucntion
         double deltaCost = costFunction.derivative(expextedResults, getOutputValues(), neuron->getNumber());
         //deltaOut /= neuronLayers.back().size(); this is not needed, i.e. wrong
         //deltaOut /= neuron->getLuts().size(); //TODO - is this needed?
         //std::cout<<__FUNCTION__<<":"<<__LINE__<<" "<<neuron->name()<<" expextedResults "<<expextedResults[neuron->getNumber() ]<<" neuron->getOutVal() "<<neuron->getOutVal()<<" deltaOut/luts.size() "<<deltaOut<<std::endl;
         for(auto& lut : neuron->getLuts() ) {
             if(! lut->getInputNode()->isEnabled() )
                 continue;
             auto& lutStat = lut->getLutStat()[lut->getAddr() ];
             lutStat.gradientSumPos += (deltaCost * eventWeight);
             lutStat.gradientSumNeg -= (deltaCost * eventWeight);
             lutStat.entries += eventWeight;
             lutStat.absGradientSum += abs(deltaCost);
             //std::cout<<__FUNCTION__<<":"<<__LINE__<<" last layer "<<neuron->name()<<" updating stat for "<<lut->name()<<" lut->getAddr() "<<lut->getAddr()<<std::endl;
         }
     }

     for(int iLayer = layers.size() -2; iLayer >= 0 ; iLayer-- ) {
         int nextLayerLutSize = layersDef[iLayer+1]->lutSize;

         for(NeuronNodePtr& neuron : layers[iLayer]) {
             if(!neuron->enabled )
                 continue;

             int orgOutAddr = neuron->getOutAddr();

             double outOfBoundCostMod = 0.1;

             double deltaCostPos = 0;
             double deltaCostNeg = 0;

             if(orgOutAddr < 0) {
                 deltaCostPos = -outOfBoundCostMod * cost;
                 deltaCostNeg =  outOfBoundCostMod * cost;
             }
             else if(orgOutAddr >= nextLayerLutSize) {
                 deltaCostPos =  outOfBoundCostMod * cost;
                 deltaCostNeg = -outOfBoundCostMod * cost;
             }
             else {
                 auto checkCost = [&](int deltaOutAddr) {
                     /*if(orgOutAddr < 0) { this is a mistake
                                      deltaOutAddr = -orgOutAddr + deltaOutAddr; //changing deltaOutAddr such that it is >= 0,
                                  }
                                  else if(orgOutAddr >= maxOutAddr) {
                                      deltaOutAddr = maxOutAddr - orgOutAddr - deltaOutAddr -1; //changing deltaOutAddr such that it is < maxOutAddr
                                  }*/

                     int newOutAddr = orgOutAddr + deltaOutAddr;

                     if(newOutAddr < 0 || newOutAddr >= nextLayerLutSize) {
                         //std::cout<<__FUNCTION__<<":"<<__LINE__<<" "<<neuron->name()<<" orgOutAddr "<<orgOutAddr<<" cost "<<cost<<" deltaOutAddr "<<deltaOutAddr<<" betterCost "<<betterCost<<" deltaOutAddrForBetterCost "<<deltaOutAddrForBetterCost<<std::endl;
                         return cost * (outOfBoundCostMod + 1);
                     }

                     neuron->setOutAddrUpdated(newOutAddr);
                     runFromNeuron(neuron.get() );
                     neuron->resetUpdate();

                     /*neuron->setOutAddr(newOutAddr);
                     runFromLayer(iLayer +1);
                     neuron->setOutAddr(orgOutAddr);*/  //cleaning the change now, be careful if it does not affects this what is done after

                     double newCost = costFunction(expextedResults, getOutputValues() );
                     //runFromLayer(iLayer +1); //todo only for testing

                     return newCost;
                     /*if( (newOutAddr < 0 || newOutAddr >= maxOutAddr) && newCost < betterCost ) {
                                                   std::cout<<__FUNCTION__<<":"<<__LINE__<<" "<<neuron->name()<<" orgOutAddr "<<orgOutAddr<<" newOutAddr "<<newOutAddr<<" cost "<<cost<<" deltaOutAddr "
                                                           <<deltaOutAddr<<" newCost "<<newCost<<" betterCost "<<betterCost<<" deltaOutAddrForBetterCost "<<deltaOutAddrForBetterCost<<std::endl;

                                                   runFromLayer(iLayer +1);
                                                   std::cout<<__FUNCTION__<<":"<<__LINE__<<" "<<neuron->name()<<" orgOutAddr "<<orgOutAddr<<" newOutAddr "<<newOutAddr<<" result "<<getOutputLayer()[0]->getOutVal()<<std::endl;

                                                   neuron->setOutAddr(orgOutAddr-1);
                                                   runFromLayer(iLayer +1);
                                                   std::cout<<__FUNCTION__<<":"<<__LINE__<<" "<<neuron->name()<<" orgOutAddr "<<orgOutAddr<<" newOutAddr "<<newOutAddr<<" result "<<getOutputLayer()[0]->getOutVal()<<std::endl;

                                                   neuron->setOutAddr(orgOutAddr);
                                                   runFromLayer(iLayer +1);
                                                   std::cout<<__FUNCTION__<<":"<<__LINE__<<" "<<neuron->name()<<" orgOutAddr "<<orgOutAddr<<" newOutAddr "<<newOutAddr<<" result "<<getOutputLayer()[0]->getOutVal()<<std::endl;
                                               }*/
                 };

                 double deltaOutAddr = 0;
                 for(int i = 1; i < 20; i++) {
                     double newCost = checkCost(i);
                     deltaCostPos = newCost - cost;
                     if(deltaCostPos != 0) {
                         deltaOutAddr = orgOutAddr + i - neuron->getOutVal(); //TODO choose option i
                         deltaCostPos /= deltaOutAddr; //TODO can be deltaOutAddr = 0????
                         break;
                     }
                 }
                 for(int i = 1; i < 20; i++) {
                     double newCost = checkCost(-i);
                     deltaCostNeg = newCost - cost;
                     if(deltaCostNeg != 0) {
                         deltaOutAddr = orgOutAddr - i - neuron->getOutVal(); //TODO choose option -i
                         deltaCostNeg /= (-deltaOutAddr);
                         break;
                     }
                 }
                 layersStat[iLayer].meanDeltaAddr += deltaOutAddr;
             }

             if(neuron->minOutAddr > orgOutAddr ) {
                 neuron->minOutAddr = orgOutAddr;
             }

             if(neuron->maxOutAddr < orgOutAddr) {
                 neuron->maxOutAddr = orgOutAddr;
             }

             /*if(orgOutAddr < 0) {
                 neuron->underflowOutCnt++;
             }
             else if(orgOutAddr >= maxOutAddr) {
                 neuron->overflowOutCnt++;
             }*/


             //deltaCost /= neuron->getLuts().size(); //TODO for each LUT the change of the value is equal - is this needed? It looks that not. If this is used, then the layer with more LUTs are less updated, which is not good

             /*if(deltaOutAddrForBetterCost)
                 deltaCost /= deltaOutAddrForBetterCost;*/
             /*if(deltaOutAddrForBetterCost < 0) //dont divide by deltaOutAddrForBetterCost, just change the sign if needed
                 deltaCost = -deltaCostPos;*/

             //deltaCostPos *= deltaOutAddrForBetterCost;

             if(deltaCostPos >= 0 && deltaCostNeg >= 0) {
                 notUpdatedNodes++;
                //std::cout<<__FUNCTION__<<":"<<__LINE__<<" "<<neuron->name()<<" orgOutAddr "<<orgOutAddr<<" cost "<<cost<<" betterCost "<<betterCost<<" deltaOutAddrForBetterCost "<<deltaOutAddrForBetterCost<<" mod "<<mod<<" deltaCost "<<deltaCost<<std::endl;
             }

             //std::cout<<__FUNCTION__<<":"<<__LINE__<<" "<<neuron->name()<<" orgOutAddr "<<orgOutAddr<<" cost "<<cost<<" betterCost "<<betterCost<<" deltaOutAddrForBetterCost "<<deltaOutAddrForBetterCost<<std::endl;
             for(auto& lut : neuron->getLuts() ) {
                 auto& lutStat = lut->getLutStat()[lut->getAddr() ];
                 lutStat.gradientSumPos += (deltaCostPos * eventWeight);
                 lutStat.gradientSumNeg += (deltaCostNeg * eventWeight);
                 lutStat.entries += eventWeight;
                 lutStat.absGradientSum += abs(deltaCostPos - deltaCostNeg); //TODO has not much sense now
             }
         }
     }
 }

 void LutNetwork::runTrainingInter(std::vector<double> expextedResults, CostFunction& costFunction) {
     //std::cout<<__FUNCTION__<<":"<<__LINE__<<" calling run "<<std::endl;
     run();
     double cost = costFunction(expextedResults, getOutputValues() );

     totalCost += cost;
     eventCnt++;
     //std::cout<<__FUNCTION__<<":"<<__LINE__<<" totalCost "<<totalCost<<std::endl;

     for(int iLayer = layers.size() -1; iLayer >= 0 ; iLayer-- ) {
         for(NeuronNodePtr& neuron : layers[iLayer]) {
             neuron->derivative = 0;
             if(iLayer == layers.size() -1) {
                 neuron->derivative = costFunction.derivative(expextedResults, getOutputValues(), neuron->getNumber());
             }

             //neuron in the laset layer has no childLuts, so else is not needed
             for(auto& childLut : neuron->getChildLuts() ) {
                 neuron->derivative += childLut->getOwnerNode()->derivative * childLut->getDerivative();
             }

             if(neuron->derivative == 0) {
                 notUpdatedNodes++;
                //std::cout<<__FUNCTION__<<":"<<__LINE__<<" "<<neuron->name()<<" orgOutVal "<<orgOutVal<<" cost "<<cost<<" betterCost "<<betterCost<<" deltaOutAddrForBetterCost "<<deltaOutAddrForBetterCost<<" mod "<<mod<<" deltaCost "<<deltaCost<<std::endl;
             }

             //std::cout<<__FUNCTION__<<":"<<__LINE__<<" "<<neuron->name()<<" orgOutVal "<<orgOutVal<<" cost "<<cost<<" betterCost "<<betterCost<<" deltaOutAddrForBetterCost "<<deltaOutAddrForBetterCost<<std::endl;
             if(layersDef[iLayer]->lutType == ConfigParameters::interpolated) {
                 for(auto& lut : neuron->getLuts() ) {
                     double inVal = lut->getInputNode()->getOutVal();
                     double deltaInVal =  inVal - lut->getAddr();
                     if( inVal < lut->getValues().size() -1) {
                         auto& lutStat1 = lut->getLutStat()[lut->getAddr() ];
                         double grad1 = neuron->derivative * (1 -deltaInVal);

                         lutStat1.gradientSumPos += grad1;
                         lutStat1.absGradientSum += abs(grad1);
                         lutStat1.entries++;

                         auto& lutStat2 = lut->getLutStat()[lut->getAddr() + 1];
                         double grad2 = neuron->derivative * deltaInVal;

                         lutStat2.gradientSumPos += grad2;
                         lutStat2.absGradientSum += abs(grad2);
                         lutStat2.entries++;
                     }
                     else {
                         auto& lutStat1 = lut->getLutStat()[lut->getAddr() ];
                         double grad1 = neuron->derivative * (1 +deltaInVal);

                         lutStat1.gradientSumPos += grad1;
                         lutStat1.absGradientSum += abs(grad1);
                         lutStat1.entries++;

                         auto& lutStat2 = lut->getLutStat()[lut->getAddr() - 1];
                         double grad2 = neuron->derivative * (-deltaInVal);

                         lutStat2.gradientSumPos += grad2;
                         lutStat2.absGradientSum += abs(grad2);
                         lutStat2.entries++;
                     }
                 }
             }
             else if(layersDef[iLayer]->lutType == ConfigParameters::discrete) {
                 for(auto& lut : neuron->getLuts() ) {
                     auto& lutStat1 = lut->getLutStat()[lut->getAddr() ];
                     double grad1 = neuron->derivative ;

                     lutStat1.gradientSumPos += grad1;
                     lutStat1.absGradientSum += abs(grad1);
                     lutStat1.entries++;
                 }
             }
         }
     }
 }

 void LutNetwork::weightByEvents(unsigned int iLayer) {
     NeuronLayer& layer = layers[iLayer];
          if(iLayer >= layers.size() -1)
              throw std::invalid_argument("LutNetwork::shiftAndStretch: iLayer >= neuronLayers.size() -1");

          for(NeuronNodePtr& neuron : layer) {
              for(auto& lut : neuron->getLuts() ) {
                  for(unsigned int iAddr = 0; iAddr < lut->getValues().size(); iAddr++) {
                      lut->getValues()[iAddr] *= 1.5*sqrt((lut->getLutStat()[iAddr].entries) / ((double)eventCnt) );
                      //lut->getValues()[iAddr] = log(lut->getLutStat()[iAddr].entries) -  ((double)eventCnt) );
                  }
              }
          }
 }
 void LutNetwork::shiftAndRescale(unsigned int iLayer, double shiftRatio, double scaleRatio) {
     NeuronLayer& layer = layers[iLayer];
     if(iLayer >= layers.size() -1)
         throw std::invalid_argument("LutNetwork::shiftAndStretch: iLayer >= neuronLayers.size() -1");

     for(NeuronNodePtr& neuron : layer) {
         double scale = 0;
         double shift = 0;
         int nextLayerLutSize = 0;

         //calculation of the size of the distribution of the neuron output values (which is stored in nextLayerLut->getLutStat() )
         //it is needed to calculate the shift and scaling factors(scale and shift)
         /*Lut* nextLayerLut = 0;
         if(layersDef[iLayer + 1]->layerType == ConfigParameters::fullyConnected)
             nextLayerLut = neuronLayers[iLayer + 1].front()->getLuts()[neuron->getNumber()].get();
         if(layersDef[iLayer + 1]->layerType == ConfigParameters::singleLut)
             nextLayerLut = neuronLayers[iLayer + 1][neuron->getNumber()]->getLuts()[0].get();

         nextLayerLutSize = nextLayerLut->getValues().size();*/

         nextLayerLutSize =  neuron->getChildLuts()[0]->getValues().size();

         /*double iAddrFirst = 0; //first address with non zero entries
         double iAddrLast = nextLayerLutSize -1;  //last address with non zero entries*/

         double iAddrFirst = neuron->minOutAddr; //first address with non zero entries
         double iAddrLast = neuron->maxOutAddr;  //last address with non zero entrie

         scale = scaleRatio * ( (nextLayerLutSize - 11)/ (iAddrLast - iAddrFirst + 1) ); //leaving margin of 5 bins

         shift = shiftRatio * (nextLayerLutSize -1 - iAddrLast - iAddrFirst) / 2.;
         /*if( abs(shift) < 1.) //TODO in principle there is no sense to shift if the shift is smaller the 1, but it looks that then it works worse
             shift = 0;
         else*/
             shift = shift/neuron->getLuts().size(); //shift goes equally to each LUT

         std::cout<<std::dec<<"LutNetwork::shiftAndRescale:"<<__LINE__<<" iLayer "<<iLayer<<" neuron "<<neuron->getNumber()<<" iAddrFirst "<<iAddrFirst<<" iAddrLast "<<iAddrLast
                 <<" shift "<<shift<<" scale "<<scale<<std::endl;
         for(auto& lut : neuron->getLuts() ) {
             for(unsigned int iAddr = 0; iAddr < lut->getValues().size(); iAddr++) {
                 //shifting such that middle is in the nextLayerLutSize / 2.
                 lut->getValues()[iAddr] += shift;
                 //re-scaling such that the neuron output values fill fully the next layer input LUTs
                 lut->getValues()[iAddr] = scale * lut->getValues()[iAddr];
             }
         }
     }

 }

 void LutNetwork::updateLuts(std::vector<LearnigParams> learnigParamsVec) {
     unsigned int iLayer = 0;
     for(NeuronLayer& layer : layers) {
         layersStat[iLayer].averageAbsDeltaLutValue = 0;
         layersStat[iLayer].averageRelativeDeltaLutValue = 0;

         double learnigRate = learnigParamsVec[iLayer].learnigRate;
         double beta = learnigParamsVec[iLayer].beta;
         double lambda = 1-learnigParamsVec[iLayer].lambda;

         for(NeuronNodePtr& neuron : layer) {
             if(!neuron->enabled )
                continue;
             for(auto& lut : neuron->getLuts() ) {
                 if(! lut->getInputNode()->isEnabled() )
                     continue;
                 //int totalEntries = 0;
                 for(unsigned int iAddr = 0; iAddr < lut->getValues().size(); iAddr++) {
                     auto& lutStat = lut->getLutStat()[iAddr];
                     double deltaLutValue = 0;
                     if(lutStat.entries)
                     {
                         if(lutStat.gradientSumPos <= lutStat.gradientSumNeg) {
                             if(lutStat.gradientSumPos < 0) {
                                 deltaLutValue = lutStat.gradientSumPos;
                             }
                         }
                         else {
                             if(lutStat.gradientSumNeg < 0) {
                                 deltaLutValue = -lutStat.gradientSumNeg;
                             }
                         }

                         deltaLutValue /= eventCnt;
                         lutStat.absGradientSum /= eventCnt;

                         if(deltaLutValue < -1)
                             deltaLutValue = -1;
                         else if(deltaLutValue > 1)
                             deltaLutValue = 1;

                     }

                     //lut->getValues()[iAddr] += deltaLutValue;
                     lutStat.gradientSumPos = deltaLutValue;
                 }

                 for(unsigned int iAddr = 0; iAddr < lut->getValues().size(); iAddr++) {
                     //if(lut->getLutStat()[iAddr].entries)
                     {
                         lut->getLutStat()[iAddr].momentum = beta * lut->getLutStat()[iAddr].momentum + learnigRate * lut->getLutStat()[iAddr].gradientSumPos;
                         lut->getValues()[iAddr] -= lut->getLutStat()[iAddr].momentum ;

                         lut->getValues()[iAddr] *= lambda; //l2 regulariazation //TODO should it be inn the if or outside?
                     }
                     /*if(lut->getValues()[iAddr] > 0) //l1 regulariazation
                         lut->getValues()[iAddr] -= lambda;
                     else
                         lut->getValues()[iAddr] += lambda;*/

                     layersStat[iLayer].averageAbsDeltaLutValue += abs(lut->getLutStat()[iAddr].momentum);
                     if(lut->getValues()[iAddr])
                         layersStat[iLayer].averageRelativeDeltaLutValue -= lut->getLutStat()[iAddr].momentum / lut->getValues()[iAddr];

                 }

             }
         }
         layersStat[iLayer].averageAbsDeltaLutValue /= (layer.size() * layer[0]->getLuts().size() * layersDef[iLayer]->lutSize);
         layersStat[iLayer].averageRelativeDeltaLutValue /= (layer.size() * layer[0]->getLuts().size() * layersDef[iLayer]->lutSize);
         iLayer++;
     }
 }

 void LutNetwork::updateLutsInter(std::vector<LearnigParams> learnigParamsVec) {
     unsigned int iLayer = 0;
     for(NeuronLayer& layer : layers) {
         layersStat[iLayer].averageAbsDeltaLutValue = 0;
         layersStat[iLayer].averageRelativeDeltaLutValue = 0;

         double learnigRate = learnigParamsVec[iLayer].learnigRate;
         double beta = learnigParamsVec[iLayer].beta;
         //double lambda = 1-learnigParamsVec[iLayer].lambda;
         double lambda = learnigParamsVec[iLayer].lambda;
         float maxLutVal = learnigParamsVec[iLayer].maxLutVal;

         int nonEmptyLutsCnt = 0;
         for(NeuronNodePtr& neuron : layer) {
             if(!neuron->enabled )
                continue;
             for(auto& lut : neuron->getLuts() ) {
                 if(! lut->getInputNode()->isEnabled() )
                     continue;

                 for(unsigned int iAddr = 0; iAddr < lut->getValues().size(); iAddr++) {
                     lut->getLutStat()[iAddr].gradientSumPos /= eventCnt;

                     if(isnan(lut->getLutStat()[iAddr].gradientSumPos)) {
                         std::cout<<(*lut)<<__FUNCTION__<<":"<<__LINE__<<" lut->getLutStat()[iAddr].gradientSumPos "<<lut->getLutStat()[iAddr].gradientSumPos
                                 <<" eventCnt "<<eventCnt<<std::endl;
                         exit(1);
                     }

                     //todo include learning rate?
                    /* if(lut->getLutStat()[iAddr].gradientSumPos  < -1)
                         lut->getLutStat()[iAddr].gradientSumPos  = -1;
                     else if(lut->getLutStat()[iAddr].gradientSumPos  > 1)
                         lut->getLutStat()[iAddr].gradientSumPos  = 1;*/

                     lut->getLutStat()[iAddr].momentum = beta * lut->getLutStat()[iAddr].momentum
                             + learnigRate * (lut->getLutStat()[iAddr].gradientSumPos + lut->getValues()[iAddr] * lambda);

                     //lut->getValues()[iAddr] *= lambda; //l2 regulariazation //TODO should be done in the above line i.e. sshluld be included in gradient and momentum

                     lut->getValues()[iAddr] -= lut->getLutStat()[iAddr].momentum ;

                     if(isnan(lut->getValues()[iAddr])) {
                         std::cout<<(*lut)<<__FUNCTION__<<":"<<__LINE__<<" lut->getValues()[iAddr] "<<lut->getValues()[iAddr]
                                 <<" lut->getLutStat()[iAddr].momentum "<<lut->getLutStat()[iAddr].momentum<<std::endl;
                         exit(1);
                     }

                     /*if( abs(lut->getValues()[iAddr]) > 4) {
                         std::cout<<__FUNCTION__<<":"<<__LINE__<<" "<<std::dec<<neuron->name()<<" "<<lut->name()<<" addr "<<iAddr<<" abs(val) > 4 "
                                 <<" lut val "<<lut->getValues()[iAddr]
                                 <<" momentum "<<lut->getLutStat()[iAddr].momentum<<" gradientSumPos "
                                 <<lut->getLutStat()[iAddr].gradientSumPos <<" entries "<<lut->getLutStat()[iAddr].entries<<" !!!!!!!!!!"<<std::endl;
                     }*/

                     if(lut->getValues()[iAddr] > maxLutVal)
                         lut->getValues()[iAddr] = maxLutVal;
                     else if(lut->getValues()[iAddr] < -maxLutVal)
                         lut->getValues()[iAddr] = -maxLutVal;

                     lut->getLutStat()[iAddr].absGradientSum /= eventCnt;

                     /*if(lut->getValues()[iAddr] > 0) //l1 regulariazation
                         lut->getValues()[iAddr] -= lambda;
                     else
                         lut->getValues()[iAddr] += lambda;*/

                     if(lut->getLutStat()[iAddr].entries) {
                         nonEmptyLutsCnt++;
                         layersStat[iLayer].averageAbsDeltaLutValue += abs(lut->getLutStat()[iAddr].momentum);
                         if(lut->getValues()[iAddr])
                             layersStat[iLayer].averageRelativeDeltaLutValue -= lut->getLutStat()[iAddr].momentum / lut->getValues()[iAddr];
                     }
                 }

             }
         }
         //layersStat[iLayer].averageAbsDeltaLutValue /= (layer.size() * layer[0]->getLuts().size() * layersDef[iLayer]->lutSize);
         //layersStat[iLayer].averageRelativeDeltaLutValue /= (layer.size() * layer[0]->getLuts().size() * layersDef[iLayer]->lutSize);
         layersStat[iLayer].averageAbsDeltaLutValue /= (nonEmptyLutsCnt);
         layersStat[iLayer].averageRelativeDeltaLutValue /= (nonEmptyLutsCnt);
         iLayer++;
     }
 }

void LutNetwork::smoothLuts(std::vector<LearnigParams> learnigParamsVec) {
     unsigned int iLayer = 0;
     for(NeuronLayer& layer : layers) {
         double smoothWeight = learnigParamsVec[iLayer].smoothWeight;
         if(smoothWeight) {
             for(NeuronNodePtr& neuron : layer) {
                 if(!neuron->enabled )
                     continue;
                 for(auto& lut : neuron->getLuts() ) {
                     if(! lut->getInputNode()->isEnabled() )
                         continue;

                     if(lut->getInputNode()->getNodeType() != Node::INPUT_NODE) {
                         for(unsigned int iAddr = 0; iAddr < lut->getValues().size(); iAddr++) {
                             double meanLutValue = 0;
                             if(iAddr == 0) {
                                 meanLutValue =  (lut->getValues()[iAddr +1] + smoothWeight * lut->getValues()[iAddr] ) / (smoothWeight +1.);
                             }
                             else if(iAddr == lut->getValues().size() -1)
                                 meanLutValue =  (lut->getValues()[iAddr -1] + smoothWeight * lut->getValues()[iAddr] ) / (smoothWeight +1);
                             else {
                                 //meanLutValue =  (lut->getValues()[iAddr -1] + lut->getValues()[iAddr +1] + smoothWeight * lut->getValues()[iAddr] ) / (smoothWeight + 2);

                                /* meanLutValue = ( lut->getValues()[iAddr -1] * lut->getLutStat()[iAddr -1].entries
                                                + lut->getValues()[iAddr +1] * lut->getLutStat()[iAddr +1].entries
                                                + lut->getValues()[iAddr]    * lut->getLutStat()[iAddr].entries* smoothWeight)
                                                        / (lut->getLutStat()[iAddr -1].entries + lut->getLutStat()[iAddr +1].entries + lut->getLutStat()[iAddr].entries * smoothWeight + 0.001);*/

                                 int weigt_n1 = (lut->getLutStat()[iAddr -1].entries ? 1 : 0);
                                 int weigt_p1 = (lut->getLutStat()[iAddr +1].entries ? 1 : 0);
                                 if(lut->getLutStat()[iAddr].entries == 0) {
                                     weigt_n1 = 1;
                                     weigt_p1 = 1;
                                 }

                                 meanLutValue =  (weigt_n1 * lut->getValues()[iAddr -1] + weigt_p1 * lut->getValues()[iAddr +1] + smoothWeight * lut->getValues()[iAddr] ) / (smoothWeight + weigt_n1 + weigt_p1);
                             }
                             lut->getLutStat()[iAddr].gradientSumNeg = meanLutValue; //using gradientSumNeg as a buffer
                         }
                         for(unsigned int iAddr = 0; iAddr < lut->getValues().size(); iAddr++) {
                             lut->getValues()[iAddr] =  lut->getLutStat()[iAddr].gradientSumNeg;
                         }
                     }
                 }
             }
         }
         iLayer++;
     }
 }

/*void LutNetwork::smoothLuts1(std::vector<LearnigParams> learnigParamsVec) {
    unsigned int iLayer = 0;
    for(NeuronLayer& layer : neuronLayers) {
        double smoothWeight = learnigParamsVec[iLayer].smoothWeight;
        if(smoothWeight) {
            for(NeuronNodePtr& neuron : layer) {
                if(!neuron->enabled )
                    continue;
                for(auto& lut : neuron->getLuts() ) {
                    if(! lut->getInputNode()->isEnabled() )
                        continue;

                    if(lut->getInputNode()->getNodeType() == Node::INPUT_NODE) {
                        for(unsigned int iAddr = 0; iAddr < lut->getValues().size(); iAddr++) {
                            unsigned int bitCnt = log2(lut->getValues().size() +1);
                            double meanLutValue = 0; //mean LUT value for all addresses different by only one bit from the current one
                            double weightSum = 0;
                            for(unsigned int iBit = 0; iBit < bitCnt; iBit++) {
                                unsigned int newAddr = iAddr ^ (1 << iBit);
                                meanLutValue += lut->getValues()[newAddr] * lut->getLutStat()[newAddr].entries;
                                weightSum += lut->getLutStat()[newAddr].entries;
                            }
                            if(weightSum) {
                                meanLutValue /= weightSum;
                                //calculating mean from the current LUT value and meanLutValue
                                if( abs(lut->getValues()[iAddr]) > 0.3 && abs(meanLutValue) > 10 * abs(lut->getValues()[iAddr]) ) {
                                    std::cout<<(*lut)<<__FUNCTION__<<":"<<__LINE__<<" meanLutValue "<<meanLutValue<<" lut->getValues()[iAddr] "<<lut->getValues()[iAddr]<<std::endl;
                                }
                                if(isnan(meanLutValue)) {
                                    std::cout<<(*lut)<<__FUNCTION__<<":"<<__LINE__<<" meanLutValue "<<meanLutValue<<" lut->getValues()[iAddr] "<<lut->getValues()[iAddr]<<std::endl;
                                }
                                lut->getLutStat()[iAddr].gradientSumNeg = (meanLutValue + smoothWeight * lut->getValues()[iAddr]) / (smoothWeight + 1);
                            }
                            else {
                                lut->getLutStat()[iAddr].gradientSumNeg = lut->getValues()[iAddr];
                            }

                        }
                        for(unsigned int iAddr = 0; iAddr < lut->getValues().size(); iAddr++) {
                            lut->getValues()[iAddr] =  lut->getLutStat()[iAddr].gradientSumNeg;
                        }
                    }
                }
            }
        }
        iLayer++;
    }
}*/

void LutNetwork::smoothLuts1(std::vector<LearnigParams> learnigParamsVec) {
    unsigned int iLayer = 0;
    for(NeuronLayer& layer : layers) {
        double smoothWeight = learnigParamsVec[iLayer].smoothWeight;

        layersStat[iLayer].averageAbsDeltaLutInSmooth = 0;
        layersStat[iLayer].averageRelativeDeltaLutInSmooth = 0;
        int nonEmptyLutsCnt = 0;
        if(smoothWeight) {
            for(NeuronNodePtr& neuron : layer) {
                if(!neuron->enabled )
                    continue;
                for(auto& lut : neuron->getLuts() ) {
                    if(! lut->getInputNode()->isEnabled() )
                        continue;

                    if(lut->getInputNode()->getNodeType() == Node::INPUT_NODE) {
                        for(unsigned int iAddr = 0; iAddr < lut->getValues().size(); iAddr++) {
                            unsigned int bitCnt = log2(lut->getValues().size() +1);
                            double sumOverNeighbours = 0; //mean LUT value for all addresses different by only one bit from the current one
                            for(unsigned int iBit = 0; iBit < bitCnt; iBit++) {
                                unsigned int newAddr = iAddr ^ (1 << iBit);
                                sumOverNeighbours += lut->getValues()[newAddr];
                            }

                            //calculating mean from the current LUT value and sumOverNeighbours
                            /*if( abs(lut->getValues()[iAddr]) > 0.3 && abs(sumOverNeighbours) > 10 * abs(lut->getValues()[iAddr]) ) {
                                std::cout<<(*lut)<<__FUNCTION__<<":"<<__LINE__<<" sumOverNeighbours "<<sumOverNeighbours<<" lut->getValues()[iAddr] "<<lut->getValues()[iAddr]<<std::endl;
                            }*/
                            if(isnan(sumOverNeighbours)) {
                                std::cout<<(*lut)<<__FUNCTION__<<":"<<__LINE__<<" sumOverNeighbours "<<sumOverNeighbours<<" lut->getValues()[iAddr] "<<lut->getValues()[iAddr]<<std::endl;
                                exit(1);
                            }

                            lut->getLutStat()[iAddr].gradientSumNeg = (sumOverNeighbours + smoothWeight * lut->getValues()[iAddr]) / (smoothWeight + bitCnt);


                        }
                        for(unsigned int iAddr = 0; iAddr < lut->getValues().size(); iAddr++) {
                            if(lut->getLutStat()[iAddr].entries) {
                                double delta = lut->getValues()[iAddr] -  lut->getLutStat()[iAddr].gradientSumNeg;
                                nonEmptyLutsCnt++;
                                layersStat[iLayer].averageAbsDeltaLutInSmooth += abs(delta);

                                if(lut->getValues()[iAddr])
                                    layersStat[iLayer].averageRelativeDeltaLutInSmooth += abs(delta / lut->getValues()[iAddr]);
                            }
                            lut->getValues()[iAddr] =  lut->getLutStat()[iAddr].gradientSumNeg;
                        }
                    }
                }
            }
        }

        layersStat[iLayer].averageAbsDeltaLutInSmooth /= nonEmptyLutsCnt;
        layersStat[iLayer].averageRelativeDeltaLutInSmooth /=  nonEmptyLutsCnt;
        iLayer++;
    }
}
 void LutNetwork::resetStats() {
     //meanCost =
     //std::cout<<__FUNCTION__<<":"<<__LINE__<<" totalCost "<<totalCost<<" eventCnt "<<eventCnt<<" totalCost/eventCnt "<<totalCost/eventCnt<<" notUpdatedNodes/evnCnt "<<((float)notUpdatedNodes)/eventCnt<<std::endl;
     totalCost = 0;
     eventCnt = 0;
     notUpdatedNodes = 0;
     for(NeuronLayer& layer : layers) {
         for(NeuronNodePtr& neuron : layer) {
             neuron->outValOffsetStat = 0;

             neuron->underflowOutCnt = 0;
             neuron->overflowOutCnt = 0;
             neuron->underflowOutValSum = 0;
             neuron->overflowOutValSum = 0;

             neuron->minOutAddr = neuron->config->nextLayerLutSize;
             neuron->maxOutAddr = -1;

             for(auto& lut : neuron->getLuts() ) {
                 for(unsigned int iAddr = 0; iAddr < lut->getValues().size(); iAddr++) {
                     auto& lutStat = lut->getLutStat()[iAddr ];
                     lutStat.gradientSumPos = 0;
                     lutStat.gradientSumNeg = 0;
                     lutStat.entries = 0;

                     lutStat.absGradientSum = 0;
                 }
             }
         }
     }

     for(unsigned int iLayer = 0; iLayer < layersStat.size(); iLayer++) {
         layersStat[iLayer].meanDeltaAddr = 0;
     }
 }

 //to be used during training, the Lut::outWeight is set to 1 with probability retainProb, otherwise to 0
void LutNetwork::setDropOut(double retainProb, std::default_random_engine& generator) {
    std::uniform_real_distribution<> dropOutdist(0, 1);
    for(NeuronLayer& layer : layers) {
        for(NeuronNodePtr& neuron : layer) {
            if(neuron->getLayer() == 0) { //TODO!!!!!!!!!!!!!!!!!!!!!!!!
                for(auto& lut : neuron->getLuts() ) {
                    if(dropOutdist(generator) <= retainProb)
                        lut->setOutWeight(1);
                    else
                        lut->setOutWeight(0);
                }
            }
        }
    }
}

 //to be used during testing, Lut::outWeight is set to the given lutOutWeight, lutOutWeight should be the same as retainProb set in setDropOut
void LutNetwork::setLutOutWeight(double lutOutWeight) {
    for(NeuronLayer& layer : layers) {
        for(NeuronNodePtr& neuron : layer) {
            for(auto& lut : neuron->getLuts() ) {
                if(neuron->getLayer() == 0)
                    lut->setOutWeight(lutOutWeight);
                else
                    lut->setOutWeight(1);
            }
        }
    }
}

std::ostream & operator << (std::ostream &out, LutNetwork& net) {
    out<<"LutNetwork: inputNodes: "<<net.inputNodes.size()<<" neuronLayers: "<<net.layers.size()<<std::endl;
    for(unsigned int iLayer = 0; iLayer < net.layers.size(); iLayer++ ) {
        for(unsigned int iNode = 0; iNode < net.layers[iLayer].size(); iNode++) {
            out<<*(net.layers[iLayer][iNode])<<std::endl;
        }
    }
    return out;
}


void LutNetwork::printLayerStat() {
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
