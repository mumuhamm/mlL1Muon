/*
 * LutInterNetwork.cpp
 *
 *  Created on: Dec 13, 2019
 *      Author: kbunkow
 */

#include "lutNN/lutNN2/interface/LutInterNetwork.h"
#include "lutNN/lutNN2/interface/LutInter.h"
#include "lutNN/lutNN2/interface/SumNode.h"

#include <math.h>

namespace lutNN {

LutInterNetwork::LutInterNetwork(NetworkOutNode* outputNode, bool useNoHitCntNode): LutNetworkBase(outputNode, useNoHitCntNode)  {
    /*    if(outputType == OutputType::softMax)
        outNode.reset(new SoftMaxNode(layersConf.back()->nodesInLayer) );*/

}

LutInterNetwork::~LutInterNetwork() {
}

void LutInterNetwork::updateLuts(std::vector<LearnigParams>& learnigParamsVec) {
    for(unsigned int iLayer = 0; iLayer < layers.size(); iLayer++ ) {
        auto& layer = layers[iLayer];
        if(getLayersConf()[iLayer]->nodeType == LayerConfig::lutInter) {
            //layersStat[iLayer].averageAbsDeltaLutValue = 0;
            //layersStat[iLayer].averageRelativeDeltaLutValue = 0;

            double learnigRate = learnigParamsVec[iLayer].learnigRate;
            double beta = learnigParamsVec[iLayer].beta;
            //double lambda = 1-learnigParamsVec[iLayer].lambda;
            double lambda = learnigParamsVec[iLayer].lambda;
            float maxLutVal = layersConf[iLayer]->maxLutVal;
            float minLutVal = layersConf[iLayer]->minLutVal;
            float maxLutValChange = layersConf[iLayer]->maxLutValChange;

            int nonEmptyLutsCnt = 0;

            double maxAbsLayerChange = 0;
            for(auto& node : layer) {
                //std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" "<<node->getName()<<std::endl;
                LutInter* lutNode = static_cast<LutInter*> (node.get());

                double maxAbsNodeChange = 0;

                double entriesValSum = 0;
                double entriesSum = 0;
                for(unsigned int iAddr = 0; iAddr < lutNode->getFloatValues().size(); iAddr++) {
                    if(iLayer == 0  && (iAddr == lutNode->getFloatValues().size() -1) && layersConf[iLayer]->noHitValue) //TODO!!!!!!!!!!! in the first layer, the last address is for the no-hit value
                        continue;

                    if(lutNode->getEntries()[iAddr])
                        lutNode->getGradients()[iAddr] /= lutNode->getEntries()[iAddr]; //eventCnt; //TODO maybe rather divide by number of events in a given bin

                    lutNode->getMomementum()[iAddr] = beta * lutNode->getMomementum()[iAddr] //momentum
                                                     + learnigRate * (lutNode->getGradients()[iAddr] + lutNode->getFloatValues()[iAddr] * lambda); //last gradient with L2 regularization


                    if(lutNode->getEntries()[iAddr]) { //monitoring the change in the LUTs
                        nonEmptyLutsCnt++;
                        if(lutNode->getMomementum()[iAddr] > maxAbsNodeChange )
                            maxAbsNodeChange = fabs(lutNode->getMomementum()[iAddr]);

                        /*if(fabs(lutNode->getMomementum()[iAddr]) > maxLutValChange) {
                                        std::cout<<"LutInterNetwork::updateLuts layer "<<iLayer<<" "<<lutNode->getName()<<" iAddr "<<iAddr<<" change "<<lutNode->getMomementum()[iAddr]<<" value "<<lutNode->getFloatValues()[iAddr]<<" entries "<<lutNode->getEntries()[iAddr]<<std::endl;
                                    }*/

                        //layersStat[iLayer].averageAbsDeltaLutValue += fabs(lutNode->getMomementum()[iAddr]);
                        //if(lutNode->getFloatValues()[iAddr])
                        //	layersStat[iLayer].averageRelativeDeltaLutValue -= lutNode->getMomementum()[iAddr] / lutNode->getFloatValues()[iAddr];

                        //std::cout<<"LutInterNetwork::updateLuts layer "<<iLayer<<" "<<lutNode->getName()<<" iAddr "<<iAddr<<" change "<<lutNode->getMomementum()[iAddr]<<" value "<<lutNode->getFloatValues()[iAddr]<<" entries "<<lutNode->getEntries()[iAddr]<<std::endl;
                    }

                    entriesValSum += lutNode->getEntries()[iAddr] * iAddr; //needed only for monitoring
                    entriesSum    += lutNode->getEntries()[iAddr];         //needed only for monitoring

                    //limiting the LUT change
                    if(fabs(lutNode->getMomementum()[iAddr]) > maxLutValChange) {
                        std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" "<<lutNode->getName()<<" iAddr "<<iAddr
                                <<" FloatValues()[iAddr] "<<lutNode->getFloatValues()[iAddr]
                                <<" Gradients "<<lutNode->getGradients()[iAddr]
                                <<" Entries "<<lutNode->getEntries()[iAddr]
                                <<" Momementum "<<lutNode->getMomementum()[iAddr]
                                <<std::endl;

                        lutNode->getMomementum()[iAddr] = std::copysign(maxLutValChange, lutNode->getMomementum()[iAddr]);
                    }

                    lutNode->getFloatValues()[iAddr] -= lutNode->getMomementum()[iAddr];

                    if(isinf( lutNode->getFloatValues()[iAddr] ) || isnan( lutNode->getFloatValues()[iAddr] ) )  {
                        std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" "<<lutNode->getName()<<" iAddr "<<iAddr<<" lutNode->getFloatValues()[iAddr] "<<lutNode->getFloatValues()[iAddr]
                        <<" !!!!!!!!!!!!!!!!!"<<std::endl;
                        exit(1);
                    }

                    //limiting the max and min LutVal
                    if(lutNode->getFloatValues()[iAddr] > maxLutVal)
                        lutNode->getFloatValues()[iAddr] = maxLutVal;
                    else if(lutNode->getFloatValues()[iAddr] < minLutVal)
                        lutNode->getFloatValues()[iAddr] = minLutVal;

                    //lut->getLutStat()[iAddr].absGradientSum /= eventCnt;

                    /*if(lut->getValues()[iAddr] > 0) //l1 regulariazation
                        lut->getValues()[iAddr] -= lambda;
                    else
                        lut->getValues()[iAddr] += lambda;*/
                }

                //std::cout<<lutNode->getName()<<" entriesValSum "<<entriesValSum<<" entriesSum "<< entriesSum<<" average "<<entriesValSum / entriesSum<<std::endl;;

                if(maxAbsNodeChange > 0.2)
                {
                    //std::cout<<"LutInterNetwork::updateLuts layer "<<iLayer<<" "<<lutNode->getName()<<" maxAbsChange "<<maxAbsNodeChange<<std::endl;
                }

                if(maxAbsLayerChange < maxAbsNodeChange)
                    maxAbsLayerChange = maxAbsNodeChange;

            }

        } //end of if(getLayersConf()[iLayer]->nodeType == LayerConfig::lutInter
        else if(getLayersConf()[iLayer]->nodeType == LayerConfig::sumNode) {
            //kind of batch normalization, does not work yet, TODO - fix
            for(auto& node : layer) {
                node->updateParamaters(learnigParamsVec[iLayer]);

/*
                if(iLayer != layers.size() -1) {
                    //std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" "<<node->getName()<<std::endl;
                    SumNode* sumNode = static_cast<SumNode*> (node.get());

                    double meanOutVal = sumNode->getOutValSum() / eventCnt;
                    double meanOuVal2 = sumNode->getOutVal2Sum() / eventCnt;

                    double stdDev = sqrt(meanOuVal2 - meanOutVal * meanOutVal);
                    //FIXME include event weight!!!!

                    double batchNormOffset = -meanOutVal;
                    double expectedStdDev = 2 * layersConf[iLayer]->outValOffset / sqrt(12) * 0.1;// - i.e. 3/4 of the flat distribution stdDev
                    double batchNormScale = expectedStdDev / stdDev;


                    if(iLayer == 3) {
                        std::cout<<"LutInterNetwork::updateLuts layer "<<iLayer<<" "<<sumNode->getName()<<" meanOutVal "<<meanOutVal<<" meanOuVal2 "<<meanOuVal2
                            <<" stdDev "<<stdDev<<" batchNormOffset "<<batchNormOffset<<" batchNormScale "<<batchNormScale
                            <<" outValOffset "<<layersConf[iLayer]->outValOffset<<" expectedStdDev "<<expectedStdDev<<" eventCnt "<<eventCnt<<std::endl;


                        sumNode->setBatchNormOffset(batchNormOffset);
                        sumNode->setBatchNormScale(batchNormScale);
                    }
                }*/
            }
        }
    }
}


void LutInterNetwork::smoothLuts(std::vector<LearnigParams>& learnigParamsVec) {

    auto getWeight = [](double entries) {
        if(entries)
            return 1.;
        else
            return 0.;

        //return entries;
    };

    for(unsigned int iLayer = 0; iLayer < layers.size(); iLayer++ ) {
        auto& layer = layers[iLayer];
        if(getLayersConf()[iLayer]->nodeType != LayerConfig::lutInter)
            continue;

        for(auto& node : layer) {
            //std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" "<<node->getName()<<std::endl;
            LutInter* lutNode = static_cast<LutInter*> (node.get());

            for(unsigned int iAddr = 0; iAddr < lutNode->getFloatValues().size() -1; iAddr++) { //TODO!!!!!!!!!!! last address is for the no-hit value, butit should be only int the first layer
                double sum = 0;
                double weight = 0;
                if(iAddr) {
                    sum    += getWeight(lutNode->getEntries()[iAddr-1]) * lutNode->getFloatValues()[iAddr-1];
                    weight += getWeight(lutNode->getEntries()[iAddr-1]);
                }
                if(iAddr < lutNode->getFloatValues().size() -2)  {//-2 becasue the last bin is reserved, should not be changed not included in the smooth
                    sum    += getWeight(lutNode->getEntries()[iAddr+1]) * lutNode->getFloatValues()[iAddr+1];
                    weight += getWeight(lutNode->getEntries()[iAddr+1]);
                }

                sum    += getWeight(lutNode->getEntries()[iAddr]) * lutNode->getFloatValues()[iAddr];
                weight += getWeight(lutNode->getEntries()[iAddr]);

                if(weight)
                    lutNode->getGradients()[iAddr] = sum / weight; //here we store the weighted average
                else
                    lutNode->getGradients()[iAddr] = lutNode->getFloatValues()[iAddr]; //then we don't change
            }

            for(unsigned int iAddr = 0; iAddr < lutNode->getFloatValues().size(); iAddr++) {
                if(iLayer == 0  && (iAddr == lutNode->getFloatValues().size() -1) ) //TODO!!!!!!!!!!! in the first layer, the last address is for the no-hit value
                    continue;

                lutNode->getFloatValues()[iAddr] +=  learnigParamsVec[iLayer].smoothWeight *(lutNode->getGradients()[iAddr] - lutNode->getFloatValues()[iAddr]);
            }

        }
    }
}

} /* namespace lutNN */
