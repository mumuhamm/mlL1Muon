/*
 * BinaryLutNetwork.cpp
 *
 *  Created on: Jul 10, 2019
 *      Author: kbunkow
 */

#include "lutNN/lutNN2/interface/BinaryLutNetwork.h"
#include <iomanip>

namespace lutNN {

BinaryLutNetwork::BinaryLutNetwork(): LutNetworkBase()
{

}

BinaryLutNetwork::BinaryLutNetwork(NetworkOutNode* outputNode, std::default_random_engine* rndGenerator):
                        LutNetworkBase(outputNode, false), rndGenerator(rndGenerator)
{
    std::cout<<"outputNode->getOutputValues().size() "<<outputNode->getOutputValues().size()
        <<" outputNode->getInputNodes().size() "<<outputNode->getInputNodes().size()<<std::endl;
}

BinaryLutNetwork::~BinaryLutNetwork() {
    // TODO Auto-generated destructor stub
}

void BinaryLutNetwork::initLutsRnd(std::default_random_engine& generator) {
    std::cout<<"BinaryLutNetwork::initLutsRnd, line "<<std::dec<<__LINE__<<std::endl;
    for(unsigned int iLayer = 0; iLayer < layers.size(); iLayer++ ) {
        if(layersConf.at(iLayer)->nodeType == LayerConfig::lutNode ||
           layersConf.at(iLayer)->nodeType == LayerConfig::lutInter ||
           layersConf.at(iLayer)->nodeType == LayerConfig::lutBinary) {

            //std::uniform_real_distribution<> b0dist(0, layersConf.at(iLayer)->maxLutVal); //offset

            double stdDev = layersConf.at(iLayer)->maxLutVal / 50. ;
            if(iLayer == layers.size() - 2)
                stdDev = (layersConf.at(iLayer)->maxLutVal - layersConf.at(iLayer)->minLutVal)/20.; //TODO !!!!!!!!!!!!!!!!!!

            std::normal_distribution<> b0dist(layersConf.at(iLayer)->middleLutVal , stdDev);

            for(unsigned int iNode = 0; iNode < layers[iLayer].size(); iNode++) {
                auto lutNode = static_cast<LutNode*>(layers[iLayer][iNode].get() );
                unsigned int lutSize = lutNode->getFloatValues().size();

                for(unsigned int iAddr = 0; iAddr < lutSize; iAddr++) {
                    float val = b0dist(generator);
                    if(val < layersConf.at(iLayer)->minLutVal)
                        val = layersConf.at(iLayer)->minLutVal;
                    else if (val > layersConf.at(iLayer)->maxLutVal)
                        val = layersConf.at(iLayer)->maxLutVal;

                    if(iLayer < layers.size() - 2)
                        static_cast<LutBinary*>(lutNode)->getIntValues()[iAddr] = round(val);

                    lutNode->getFloatValues()[iAddr] = val;
                }
            }
        }
    }

    /*std::normal_distribution<> b0dist(64, 32);
	for(unsigned int inputNode = 0; inputNode < getInputNodes().size(); inputNode++) {
	    InputNodeBinary* inputNodeBinary = static_cast<InputNodeBinary*>(getInputNodes()[inputNode].get());
	    float val = b0dist(generator);
	    if(val < 0)
	        val = 0;
	    else if (val > 256)
	        val = 256;
	    inputNodeBinary->getThreshold() = val ;
	}*/


    std::cout<<"BinaryLutNetwork::initLutsRnd, line "<<std::dec<<__LINE__<<std::endl;
}

void BinaryLutNetwork::initLutsAnd(std::default_random_engine& generator) {
    for(unsigned int iLayer = 0; iLayer < layers.size(); iLayer++ ) {
        if(layersConf.at(iLayer)->nodeType == LayerConfig::sumIntNode)
            continue;

        for(unsigned int iNode = 0; iNode < layers[iLayer].size(); iNode++) {
            auto lutNode = static_cast<LutNode*>(layers[iLayer][iNode].get() );
            unsigned int lutSize = lutNode->getFloatValues().size();

            if(iLayer == 0) {
                for(unsigned int iAddr = 0; iAddr < lutSize; iAddr++) {
                    lutNode->getFloatValues()[iAddr] = layersConf[iLayer]->middleLutVal - 0.01;
                }
            }
            else {
                for(unsigned int iAddr = 0; iAddr < lutSize; iAddr++) {
                    lutNode->getFloatValues()[iAddr] = layersConf[iLayer]->middleLutVal - 0.01;
                }
                unsigned int addressBitCnt = (layersConf[iLayer]->bitsPerNodeInput * layersConf[iLayer]->nodeInputCnt);
                unsigned int lastAddr = lutSize -1;
                lutNode->getFloatValues().at(lastAddr) = layersConf[iLayer]->middleLutVal + 0.05;
                for(unsigned int iBit = 0; iBit < addressBitCnt; iBit++) {
                    unsigned int address = lastAddr ^ (1UL<<iBit);
                    lutNode->getFloatValues().at(address) = layersConf[iLayer]->middleLutVal + 0.01;
                    std::cout<<lutNode->getName()<<" iBit "<<iBit<<" "<<lutNode->getFloatValues().at(address)<<std::endl;
                }
            }
        }
    }

}

/*void BinaryLutNetwork::calcualteOutputValues() {
    outputNode->run();
}*/

void BinaryLutNetwork::run(float eventWeight) {
  std::uniform_int_distribution<> flatDist(0, 100);
    //this dithering is not good, better add the noise to the event
    /*for(unsigned int inputNode = 0; inputNode < getInputNodes().size(); inputNode++) {
	    if(flatDist(rndGenerator) < layersConf[0]->ditherRate)
		    getInputNodes()[inputNode]->InputNode::setInput(getInputNodes()[inputNode]->getOutValueInt()  ^ 1UL); //watch out works reasonably only for binary input
	}*/

    unsigned int iLayer = 0;
    for(auto& layer : layers) {
        /*if(layersConf[iLayer]->ditherRate) { TODO restore if needed
            for(auto& node : layer) {
                if(flatDist(*rndGenerator) < layersConf[iLayer]->ditherRate) {
                    node->setDither(1);
                    node->getOutValue() = 0;
                }
                else {
                    node->setDither(0);
                    node->run();
                    node->getOutValue() *= 1. / (1. -  layersConf[iLayer]->ditherRate/100.); //change ditherRate in % to fraction
                }

            }
        }
        else*/
        {
            for(auto& node : layer) {
                node->setDither(0); //clearing, as it could be set in the previous run, it is important for the testing run after training run
                node->run(eventWeight);
            }
        }
        iLayer++;
    }
    //calcualteOutputValues();
    outputNode->run();


    /*
    //debug
    bool was = false;
    for(unsigned int iOut = 0;iOut < outputNode->getOutputValues().size();iOut++) {
        if(outputNode->getOutputValues()[iOut] >= 1.)
            was = true;
    }

    if(was) {
        std::cout<<"outputValue >= 1 !!!!"<<std::endl;
        for(unsigned int iOut = 0;iOut < outputNode->getOutputValues().size();iOut++) {
            std::cout<<"iOut "<<iOut<<" "<<outputNode->getOutputValues()[iOut]<<std::endl;
        }

        for(unsigned int iClass = 0; iClass < layers.back().size(); iClass++) {//iClass enumerates the sum nodes
            std::cout<<"\n"<<layers.back().at(iClass)->getName()<<" OutValueInt "<<layers.back().at(iClass)->getOutValueInt()<<std::endl;
            for(auto& node : layers.back().at(iClass)->getInputNodes()) { //input nodes of the sum node - so the last layer of the lutNodes, which has int (not binary) out values
                LutBinary* inputLutNode = static_cast<LutBinary*> (node);
                //propagating the gradient back, i.e. setting the inputLutNode->getLastGradientVsOutVal()
                std::cout<<inputLutNode->getName()<<" OutValueInt "<<inputLutNode->getOutValueInt()<<"\t"
                        <<inputLutNode->getFloatValues()[inputLutNode->getLastAddr()] <<std::endl;
            }
        }
    }*/

}

void BinaryLutNetwork::runTrainingNaiveBayes(std::vector<double> expextedResult) {
    //TODO change the input to the number of the expected result, which should be remembered in the input data
    unsigned int iClass = 0;
    for(; iClass < expextedResult.size(); iClass++) { //assuming expextedResults is hot-one
        if(expextedResult[iClass])
            break;
    }

    auto& topNode = layers.back().at(iClass); //this should be SumNode TODO does it work with subclasses?

    NodeVec layerNodes;
    layerNodes.push_back(topNode.get());


    NodeVec prevLayerNodes;
    while(layerNodes.size() ) {
        for(auto& node : layerNodes) {
            for(auto& inputNode : node->getInputNodes() ) {
                inputNode->updateStat(1, 0, 0);
                prevLayerNodes.push_back(inputNode);
            }
        }
        layerNodes = prevLayerNodes;
        prevLayerNodes.clear();
    }
}

void BinaryLutNetwork::updateLutsNaiveBayes(unsigned int iLayer ) {
    //for(unsigned int iLayer = 0; iLayer < layers.size(); iLayer++ )
    {
        auto& layerConf = getLayersConf()[iLayer];
        //double pdfMaxLogVal = (1 << layerConf->outputBits) -1  -2; //-2 to have some room for updates during training
        double pdfMaxLogVal = (1 << layerConf->outputBits) -1;

        for(unsigned int iNode = 0; iNode < layers[iLayer].size(); iNode++) {
            //out<<*(net.layers[iLayer][iNode])<<std::endl;
            LutBinary* lutNode = dynamic_cast<LutBinary*>(layers[iLayer][iNode].get());
            if(!lutNode)
                continue;

            double minLikelihood = 0.2/lutNode->getEntries().size();//0.001;
            double minPlog =  log(minLikelihood);

            double entriesSum = 0;
            for(auto& entry : lutNode->getEntries()) {
                entriesSum += entry;
            }

            if(entriesSum == 0)
                continue;


            for(unsigned int iAddr = 0; iAddr < lutNode->getEntries().size(); iAddr++) {
                double likelihood = (double)(lutNode->getEntries()[iAddr]) / entriesSum;

                double logLikelihood = 0;

                /*if(layerConf->outputBits == 1) {
                    if(likelihood >= 0.2/lutStats.size()) {
                        value = 1;
                    }
                }
                else */
                {
                    if(likelihood > minLikelihood) {
                        logLikelihood = pdfMaxLogVal - log(likelihood) * pdfMaxLogVal / minPlog;
                    }
                }
                lutNode->getFloatValues()[iAddr] = logLikelihood;
                lutNode->getIntValues()[iAddr] = round(logLikelihood);

                //if(logLikelihood)
                //std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" "<<lutNode->getName()<<" iAddr "<<iAddr<<" logLikelihood "<<logLikelihood<<" value "<<value<<std::endl;
            }
        }
    }
}

void BinaryLutNetwork::runTraining(EventInt* event, CostFunction& costFunction) {
    LutNetworkBase::run(event);
    double currentCost = costFunction.get(event->classLabel, getOutputValues() ); //todo pass the event to the costFunction

   /* {//debug
        std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" currentCost "<<std::setw(12)<<currentCost<<" "<<std::endl;
        for(unsigned int iClass = 0; iClass < event->expextedResult.size(); iClass++) {
            //if(event->expextedResult[iClass]) .. event->expextedResult[iClass]
            std::cout<<" iClass "<<iClass<<" expextedResult "<<(event->classLabel == iClass ? 1 : 0)<<" nnResult "<<std::setw(12)<<std::setw(12)<<event->nnResult[iClass]
                    <<" nodeOut float "<<std::setw(5)<<layers.back()[iClass]->getOutValue()
                    <<" int "<<std::setw(5)<<layers.back()[iClass]->getOutValueInt()<<std::endl;
        }
    }*/




    totalCost += currentCost;
    eventCnt++;

    outputNode->calcualteGradient(costFunction, event->classLabel, event->weight);

    double maxOut = -100000;
    unsigned int nnClassLabel = 1000;
    for(unsigned int iOut = 0 ; iOut < event->nnResult.size(); iOut++) {
        if(event->nnResult[iOut] > maxOut) {
            maxOut = event->nnResult[iOut];
            nnClassLabel = iOut;
        }
    }
    if(event->classLabel == nnClassLabel)
        wellClassifiedEvents++;

    //DEBUG
    /*if(currentCost > 0) {
        std::cout<<"event->number " <<event->number<<" classLabel "<<event->classLabel<<" currentCost "<<currentCost
                <<" totalCost "<<totalCost<<" eventCnt "<<eventCnt<<" wellClassifiedEvents "<<wellClassifiedEvents<<std::endl;

        for(unsigned int iOut = 0 ; iOut < event->nnResult.size(); iOut++) {
            std::cout<<" "<<iOut<<" "<<std::setw(10)<<event->nnResult[iOut]
                     //<<" der "<<std::setw(10)<<costFunction.getDerivative()[iOut] fixme clean the getDerivative issue, looks that it should not be used
                     <<(iOut == nnClassLabel ? " <<<<<<<<<< ": "")
                     <<std::endl;
        }
        std::cout<<std::endl;
    }*/
    //DEBUG

    //calculating the lastGradient for the last but one layer (for the nodes that goes to the sumNodes)

    for(auto& node : layers.back() ) {//node is sumNode
        node->updateGradient(); //propagates gradient back from the sumNode
    }

    for(auto& node : layers.back() ) { //node is sumNode
        for(auto& inputNode : node->getInputNodes() ) { //input nodes of the sum node - so the last layer of the lutNodes, which has int or float (not binary) out values
            LutNode* inputLutNode = dynamic_cast<LutNode*> (inputNode); //static_cast<LutNode*> (node);
            if(!inputLutNode)
                throw std::runtime_error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!");

            //LutNode* inputLutNode = static_cast<LutNode*> (inputNode); TODO replace the above when checked

            if(inputLutNode->getDither() == 0) {
                inputLutNode->getGradients()[inputLutNode->getLastAddr()] += inputLutNode->getLastGradient() ;

                /*std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" address "<<inputLutNode->getLastAddr()<<" outValue "<<inputLutNode->getOutValue()
            		<<" lastGradient "<<inputLutNode->getLastGradient()
            		<<" gradient "<<inputLutNode->getGradients()[inputLutNode->getLastAddr()]<<std::endl;*/

                inputLutNode->updateStat(1, 0, 0);

                //inputLutNode->setDither(0); //clearing the dither - cannot be here!!!!!
            }
            else {
                inputLutNode->getLastGradient() = 0;
            }
        }
    }

    //watch out - the below works only for the binary output values of the inputLutNode!!!!!!!!!!!!!!!!!!!
    for(int iLayer = layers.size() - 2; iLayer >= 1; iLayer--) {
        for(auto& node : layers.at(iLayer)) {
            LutNode* lutNode = static_cast<LutNode*> (node.get());
            if(lutNode->getDither()) {
                lutNode->getLastGradient() = 0; //in principle not needed, but just in case...
                continue;
            }

            float ditherNorm =  1. / (1. -  layersConf[iLayer]->ditherRate/100.); //change ditherRate in % to fraction

            //int lutNondeOutValueInt = lutNode->getOutValueInt();
            for(unsigned int iInputNode = 0; iInputNode < lutNode->getInputNodes().size(); iInputNode++) {
                Node* inputNode = lutNode->getInputNodes()[iInputNode];

                if(inputNode->getDither() == 0 ) {
                    unsigned int newAddr = lutNode->getLastAddr() ^ (1UL << iInputNode); //flip the bit corresponding to the inputLutNode

                    //getting the gradient for the inputLutNode from the lutNode->getLastGradientVsOutVal()
                    auto lastGradient = lutNode->getLastGradient();
                    if(iLayer == (int)layers.size() - 2) {
                        //so here we calculate the lastGradient in the classical way,
                        //not bothering that the the difference of the values might not be small
                        float outValueChange = lutNode->getFloatValues()[newAddr] * ditherNorm - lutNode->getOutValue();
                        lastGradient *= outValueChange;
                    }
                    else {
                        LutBinary* lutBinaryNode = static_cast<LutBinary*>(lutNode);
                        if(lutBinaryNode->getIntValues()[newAddr] == lutNode->getOutValue())
                            lastGradient = 0;
                    }

                    /*if(outValueChange == 0) {//avoid gradient vanishing - assuring there is some gradient even in this case,
                    	if(lutNondeOutValueInt == 0)  //TODO use min lut value instead of 0, but watch out - the 0 as min value is assumed in many other places
                    		outValueChange = 1;
                    	else if(lutNondeOutValueInt == layersConf[iLayer]->maxLutVal)
                    		outValueChange = -1;
                    	else { //this case is possible only for the luts with more then two output values, i.e. in the last layer
                    		if(lutNode->getFloatValues()[lutNode->getLastAddr()] > lutNondeOutValueInt ) //using linear interpolation to the OutVal closest to the  lutNode FloatValue
                    			outValueChange = 1;
                    		else
                    			outValueChange = -1;
                    	}
                    	lutNonde_newOutValueInt = lutNondeOutValueInt + outValueChange;
                    }
                    else*/
                    //outValueChange = 1;


                    //if(iLayer > 0)
                    {
                        LutBinary* inputLutNode = static_cast<LutBinary*>(inputNode);
                        if(lastGradient != 0) {
                            //int inputNodeNewValue = inputLutNode->getOutValueInt() ^ 1UL; //flip the bit
                            inputLutNode->getLastGradient() += lastGradient;// * fabs(dVal);
                            if(inputLutNode->getOutValueInt() == 0) //getOutValueInt == 0,  inputNodeNewValue = 1, so the gradient comes with plus
                                inputLutNode->getGradients()[inputLutNode->getLastAddr()] += lastGradient;// * fabs(dVal);
                            else                 //getOutValueInt == 1,  inputNodeNewValue = 0, so the gradient comes with minus
                                inputLutNode->getGradients()[inputLutNode->getLastAddr()] -= lastGradient;// * fabs(dVal);

                            /*
						std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" lutNode: "<<lutNode->getName()<<"\n"
								<<"getLastGradientVsOutVal: ";
						for(unsigned int outVal = 0; outVal < lutNode->getLastGradientVsOutVal().size(); outVal++) {
							std::cout<<"["<<outVal<<"] "<<lutNode->getLastGradientVsOutVal()[outVal]<<" : ";
						}
						std::cout<<"\n"<<" LastAddr "<<lutNode->getLastAddr()
								<<" : newAddr "<<newAddr<<" : lutNondeOutValueInt "<<lutNondeOutValueInt<<" : lutNonde_newOutValueInt "<<lutNonde_newOutValueInt<<"\n"
								<<" : getFloatValues[LastAddr] "<<lutNode->getFloatValues()[lutNode->getLastAddr()]<<" : getFloatValues[newAddr] "<<lutNode->getFloatValues()[newAddr]
								<<" : dVal "<<dVal<<" lastGradient "<<lastGradient<<std::endl;
						std::cout<<"inputLutNode "<<inputLutNode->getName()<<" address "<<std::setw(2)<<inputLutNode->getLastAddr()
								<<" : outValueInt "<<inputLutNode->getOutValueInt()<<" : outFloatValue "<<std::setw(10)<<inputLutNode->getFloatValues()[inputLutNode->getLastAddr()]
								<<" : inputNodeNewValue "<<inputNodeNewValue<<"\n"
								<<" : gradientChange "<<lastGradient * dVal
								<<" : LastGradientVsOutVal[0] "<<std::setw(10)<<(inputLutNode->getLastGradientVsOutVal()[0])
								<<" : LastGradientVsOutVal[1] "<<std::setw(10)<<(inputLutNode->getLastGradientVsOutVal()[1])
								<<" : gradient "<<std::setw(10)<<inputLutNode->getGradients().at(inputLutNode->getLastAddr())<<std::endl<<std::endl;*/
                        }
                        inputLutNode->updateStat(1, 0, 0);
                    }
                    /*else {//Training of the inputNodeBinary->threshold's
                        InputNodeBinary* inputNodeBinary = static_cast<InputNodeBinary*>(inputNode);//watch out, dont use other input nodes!!!!!
                        if(lastGradient != 0 && fabs(inputNodeBinary->getDeltaValThresh()) < 32) {
                            //auto dVal = (lutNode->getFloatValues()[newAddr] -lutNode->getFloatValues()[lutNode->getLastAddr()]);
                            //dVal = dVal - int(dVal); //to handle the case when the abs(dVal) is greater then 1

                            int inputNodeNewValue = inputNodeBinary->getOutValueInt() ^ 1UL; //flip the bit


                            if(inputNodeNewValue) //getOutValueInt == 0,  inputNodeNewValue = 1, so the gradient comes with minus
                                inputNodeBinary->getGradient() -= lastGradient;// * fabs(dVal);
                            else                 //getOutValueInt == 1,  inputNodeNewValue = 0, so the gradient comes with plus
                                inputNodeBinary->getGradient() += lastGradient;// * fabs(dVal);

                            float currentGradient = 0;
                            if(inputNodeNewValue) //getOutValueInt == 0,  inputNodeNewValue = 1, so the gradient comes with minus
                                currentGradient -= lastGradient;// * fabs(dVal);
                            else                 //getOutValueInt == 1,  inputNodeNewValue = 0, so the gradient comes with plus
                                currentGradient += lastGradient;
                            inputNodeBinary->getGradient() += currentGradient;

                            //debug
                            if(inputNodeBinary->getName() == "InputNodeBinary_707_inputNum_411") {
                                std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" lutNode: "<<lutNode->getName()<<"\n"<<"getLastGradientVsOutVal: ";
                                for(unsigned int outVal = 0; outVal < lutNode->getLastGradientVsOutVal().size(); outVal++) {
                                    std::cout<<"["<<outVal<<"] "<<lutNode->getLastGradientVsOutVal()[outVal]<<" : ";
                                }
                                std::cout<<"\n"<<" LastAddr "<<lutNode->getLastAddr()
                                             <<" : newAddr "<<newAddr<<" : lutNondeOutValueInt "<<lutNondeOutValueInt<<" : lutNonde_newOutValueInt "<<lutNonde_newOutValueInt<<"\n"
                                             <<" : getFloatValues[LastAddr] "<<lutNode->getFloatValues()[lutNode->getLastAddr()]<<" : getFloatValues[newAddr] "<<lutNode->getFloatValues()[newAddr]
                                                                                                                                                                                           <<" lastGradient "<<lastGradient<<std::endl;

                                std::cout<<"inputLutNode "<<inputNodeBinary->getName()<<" threshold "<<std::setw(2)<<inputNodeBinary->getThreshold()
                                             <<" deltaValThresh "<<inputNodeBinary->getDeltaValThresh()
                                             <<" : outValueInt "<<inputNodeBinary->getOutValueInt()
                                             <<" : inputNodeNewValue "<<inputNodeNewValue
                                             <<" : currentGradient "<<currentGradient
                                             <<" : getGradient "<<inputNodeBinary->getGradient()<<std::endl<<std::endl;
                            }
                        }
                    }*/
                }

            }

            lutNode->getLastGradient() = 0;
        }
    }
}


void BinaryLutNetwork::updateLuts(std::vector<LearnigParams>& learnigParamsVec) {
    //calcualteAdamBiasCorr();

    int changedLutValues = 0;
    for(unsigned int iLayer = 0; iLayer < layers.size() -1; iLayer++) { //last layer is sumNode, so is not updated
        auto& layer = layers[iLayer];

        double lambda = learnigParamsVec[iLayer].lambda;
        int maxLutVal = layersConf[iLayer]->maxLutVal;

        unsigned int deadAddrCnt = 0; //TODO something is wrong with dead nodes, this is not correct place, but if moved into a good place, works worse
        for(auto& node : layer) {
            //std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" "<<node->getName()<<std::endl;
            LutNode* lutNode = static_cast<LutNode*> (node.get());
            //int maxOutVal = lutNode->getGradients()[0].size() -1;
            for(unsigned int iAddr = 0; iAddr < lutNode->getGradients().size(); iAddr++) {
                //int currentIntVal = lutNode->getIntValues()[iAddr];

                //if(lutNode->getEntries()[iAddr] == 0) //then lambda is not applied,
                //	continue;

                float gradient = lutNode->getGradients().at(iAddr);
                //if for a given value the gradient is positive, then it means that this value is better - has lower cost

                //if(lutNode->getLutStat()[iAddr].entries)
                //gradient /= lutNode->getLutStat()[iAddr].entries; //eventCnt; //TODO maybe rather divide by number of events in a given bin, or by eventCnt in the miniBatch
                //else continue;

                //float newFloatVal = lutNode->getFloatValues()[iAddr] + gradient * learnigParamsVec[iLayer].learnigRate;

                //double change = getLutChangeAdam(lutNode, iAddr, gradient, learnigParamsVec[iLayer].learnigRate);

                //kind of regularization - including the gradient from the addresses that differ by one bit,
                /*if(iLayer == 0) {
                    for(unsigned int iBit = 0; iBit < (layersConf[iLayer]->bitsPerNodeInput * layersConf[iLayer]->nodeInputCnt); iBit++) {
                        unsigned int newAddr = iAddr ^ (1UL<<iBit);
                        gradient += lutNode->getGradients().at(newAddr) * 0.8;
                    }
                }*/


                //L2 normalization has no sens here, TODO remove
                double change = (gradient + (lutNode->getFloatValues()[iAddr] - layersConf.at(iLayer)->middleLutVal ) * lambda ) * learnigParamsVec[iLayer].learnigRate; //with L2 normalization

                float newFloatVal = lutNode->getFloatValues()[iAddr] - change;

                if(newFloatVal < layersConf.at(iLayer)->minLutVal)
                    newFloatVal = layersConf.at(iLayer)->minLutVal;
                else if(newFloatVal > maxLutVal)
                    newFloatVal = maxLutVal;

                if(iLayer < layers.size() - 2) {
                    LutBinary* lutBinary = static_cast<LutBinary*> (lutNode);
                    //not sure if it helps, worth to check, but maybe just fluctuation
                    float histeresisSize = 0.001;
                    if(newFloatVal > (lutBinary->getIntValues()[iAddr] + 0.5 + histeresisSize) ) {
                        //lutNode->getIntValues()[iAddr]++; //change can be > 1, so it cannot be done like that
                        lutBinary->getIntValues()[iAddr] = round(newFloatVal);
                        changedLutValues++;
                    }
                    else if(newFloatVal < (lutBinary->getIntValues()[iAddr] - 0.5 - histeresisSize) )
                    {
                        //lutNode->getIntValues()[iAddr]--; //change can be > 1, so it cannot be done like that
                        lutBinary->getIntValues()[iAddr] = round(newFloatVal);
                        changedLutValues++;
                    }
                }

                /*if(gradient)
                            	std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<lutNode->getName()<<" address "<<std::setw(2)<<iAddr<<" flaotVal "<<lutNode->getFloatValues()[iAddr]<<" intVal "<<lutNode->getIntValues()[iAddr]
            						<<" newFloatVal "<<newFloatVal<<" gradient "<<gradient<<" change "<<change<<std::endl;*/

                lutNode->getFloatValues()[iAddr] = newFloatVal;

                /*if(lutNode->getIntValues()[iAddr] < 0 || lutNode->getIntValues()[iAddr] > maxOutVal) {
                            	std::cout<<"lutNode->getIntValues()[iAddr] "<<lutNode->getIntValues()[iAddr]<<std::endl;
                            }

                            if(lutNode->getIntValues()[iAddr] != round(newFloatVal)) {
                            	std::cout<<"lutNode->getIntValues()[iAddr] "<<lutNode->getIntValues()[iAddr]<<" newFloatVal "<<newFloatVal<<std::endl;
                            }*/

                /*if(lutNode->getIntValues()[iAddr] != round(newFloatVal)) {
                            	lutNode->getIntValues()[iAddr] = round(newFloatVal);
                            	changedLutValues++;
                            }*/

                if( fabs(newFloatVal - maxLutVal/2.) <  0.05) //TODO not good rather
                    deadAddrCnt++;
            }
            lutNode->reset(); //reset gradient and stat, TODO maybe should be earlier?

            /*if(deadAddrCnt == lutNode->getGradients().size() ) {
            	if(lutNode->isDead() ==  false) //just died
            		std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<lutNode->getName()<<" is dead"<<std::endl;
            	lutNode->setDead(true);
                for(auto& val : lutNode->getLastGradientVsOutVal())
                	val = 0; //clear
            }*/

        }
    }
    std::cout<<"updateLuts changedLutValues: "<<std::dec<<changedLutValues<<std::endl;

    /*//trainig the inputNodeBinary->threshold's
    float learingRate = 0.1;
    float maxThreshold =  255;
    for(unsigned int inputNode = 0; inputNode < getInputNodes().size(); inputNode++) {
        InputNodeBinary* inputNodeBinary = static_cast<InputNodeBinary*>(getInputNodes()[inputNode].get());
        auto oldThreshold = inputNodeBinary->getThreshold();

        inputNodeBinary->getThreshold() -=  (inputNodeBinary->getGradient() * learingRate);

        if(inputNodeBinary->getThreshold() < 0)
            inputNodeBinary->getThreshold() = 0;
        else if(inputNodeBinary->getThreshold() > maxThreshold)
            inputNodeBinary->getThreshold() = maxThreshold;

        std::cout<<__FUNCTION__<<":"<<__LINE__<<"inputLutNode "<<inputNodeBinary->getName()
                 <<" oldThreshold "<<oldThreshold
                 <<" threshold "<<std::setw(2)<<inputNodeBinary->getThreshold()
                 <<" : gradient "<<inputNodeBinary->getGradient()<<std::endl<<std::endl;

        inputNodeBinary->getGradient() = 0;
    }*/

    double alpha = 1./100.;
    averageEfficiency = (1 - alpha) * averageEfficiency + alpha * getEfficiency();
}

/*
void BinaryLutNetwork::runTrainingAndUpdate(EventInt* event, CostFunction& costFunction, std::vector<LearnigParams>& learnigParamsVec) {
    LutNetworkBase::run(event);
    double currentCost = costFunction(event->expextedResult, getOutputValues() );

    /*{//debug
        std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" currentCost "<<std::setw(12)<<currentCost<<" "<<std::endl;
        for(unsigned int iClass = 0; iClass < event->expextedResult.size(); iClass++) {
            //if(event->expextedResult[iClass])
            std::cout<<" iClass "<<iClass<<" expextedResult "<<event->expextedResult[iClass]<<" nnResult "<<std::setw(12)<<std::setw(12)<<event->nnResult[iClass]
                     <<" nodeOut "<<std::setw(5)<<layers.back()[iClass]->getOutValueInt()<<" getOutputValues "<<std::setw(5)<<getOutputValues()[iClass]<<std::endl;
        }
    }@/

    totalCost += currentCost;
    eventCnt++;

    //todo why double here?
    std::vector<double> lastLayerOut(layers.back().size(), 0); //last layer is the layer of sumNodes
    //std::vector<double> softMaxValues(layers.back().size(), 0);
    auto out = lastLayerOut.begin();
    for(auto& node : layers.back()) { //copying the values from the last layer to the lastLayerOut
        *out = node->getOutValueInt();
        out++;
    }

    //softMaxFunction(lastLayerOut, softMaxValues);
    //std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" currentCost "<<currentCost<<std::endl;

    //calculating the lastGradient for the last but one layer (for the nodes that goes to the sumNodes)
    //int maxChange = (1 << layersConf.end()[-2]->outputBits) -1;  // = maxOutValue
    for(unsigned int iClass = 0; iClass < event->expextedResult.size(); iClass++) {//iClass enumerates the sum nodes
        int maxChange = layersConf.end()[-2]->maxLutVal;
        //std::vector<double> upadetedOutValues(lastLayerOut);

        //calculating the gradient (cost function change) for every possible change out  value of given sum node (i.e. class)
        //we assuming only one branch of given sum node is changing at one moment, therefore the possible change is from -maxChange to maxChange, where maxChange is the maximum value of the branch output node
        //upadetedOutValues.at(iClass) -= maxChange; // = -maxOutValue

        //std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" iClass "<<iClass<<std::endl;
        double upadetedOutValue = lastLayerOut.at(iClass) - maxChange;

        std::vector<double> gradientVersusChange(2 * maxChange + 1, 0);

        for(int change = -maxChange; change <= maxChange; change++) {
            //softMaxFunction(upadetedOutValues, softMaxValues);
            const std::vector<double>& networkOutValues = outputNode->updateInput(iClass, 0, upadetedOutValue);
            gradientVersusChange.at(change + maxChange) =  costFunction(event->expextedResult, networkOutValues ) - currentCost; // + maxChange because the change goes from -maxChange
            //if for a given value the gradient is positive, then it means that this value is better - has lower cost

            //std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" change "<<change<<" gradientVersusChange "<<gradientVersusChange.at(change + maxChange)<<std::endl;
            upadetedOutValue++;
        }

        auto iLayer = layers.size() -1; //sum nodes
        for(auto& node : layers.back().at(iClass)->getInputNodes()) { //input nodes of the sum node - so the last layer of the lutNodes, which has int (not binary) out values
            LutBinary* inputLutNode = static_cast<LutBinary*> (node);
            if(inputLutNode->getDither() == 0) {
                //propagating the gradient back, i.e. setting the inputLutNode->getLastGradientVsOutVal()
                int outValueInt = inputLutNode->getOutValueInt();
                for(int possibleOutVal = 0; possibleOutVal <= maxChange; possibleOutVal++) { //all possible out values of given node
                    inputLutNode->getLastGradientVsOutVal().at(possibleOutVal) = gradientVersusChange.at(maxChange + possibleOutVal - outValueInt);
                    //here inputLutNode is connected to only one node(which is sumNnode), so here is just assignment, not the +=
                    //std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" "<<inputLutNode->getName()
                    //      <<" possibleOutVal "<<possibleOutVal<<" gradient "<<inputLutNode->getLastGradientVsOutVal().at(possibleOutVal)<<(possibleOutVal == inputLutNode->getOutValueInt() ? " currentVal " : "")<<std::endl;
                }

                const auto& lastAddr = inputLutNode->getLastAddr();
                //calculating the gradient for the inputLutNode
                int outValueChange = 0;
                if(outValueInt == 0)  //TODO use min lut value instead
                    outValueChange = 1;
                else if(outValueInt == maxChange)
                    outValueChange = -1;
                else { //this case is possible only for the luts with more then two output values, i.e. in the last layer
                    if(inputLutNode->getFloatValues()[lastAddr] > outValueInt )
                        outValueChange = 1;
                    else
                        outValueChange = -1;
                }

                double gradient =  inputLutNode->getLastGradientVsOutVal()[outValueInt + outValueChange] * outValueChange ;

                //std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" address "<<inputLutNode->getLastAddr()<<" outValueInt "<<outValueInt<<" outFloatValue "<<std::setw(10)<<inputLutNode->getFloatValues()[inputLutNode->getLastAddr()]
                    //<<" gradientChange "<<(inputLutNode->getLastGradientVsOutVal()[outValueInt + outValueChange] * outValueChange)
                //    <<" gradient "<<gradient<<std::endl;

                auto& learnigParams = learnigParamsVec[iLayer -1];

                double lutValchange = (gradient + (inputLutNode->getFloatValues()[lastAddr] - (maxChange * 0.5) ) * learnigParams.lambda ) * learnigParams.learnigRate; //with L2 normalization

                float newFloatVal = inputLutNode->getFloatValues()[lastAddr] - lutValchange;

                if(newFloatVal < 0)
                    newFloatVal = 0;
                else if(newFloatVal > maxChange)
                    newFloatVal = maxChange;

                //inputLutNode->getIntValues() should not be updated here, because it s updated in the further

                inputLutNode->getFloatValues()[lastAddr] = newFloatVal;

                inputLutNode->updateStat(1, 0, 0);

                //inputLutNode->setDither(0); //clearing the dither - cannot be here!!!!!
            }
            else {
                for(int possibleOutVal = 0; possibleOutVal <= maxChange; possibleOutVal++) { //all possible out values of given node
                    inputLutNode->getLastGradientVsOutVal().at(possibleOutVal) = 0;//do not back propagate the error if the node was dithered
                }
            }
        }
    }
    //std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<std::endl;
    //watch out - the below works only for the binary output values of the inputLutNode!!!!!!!!!!!!!!!!!!!
    for(int iLayer = layers.size() - 2; iLayer >= 0; iLayer--) {
        for(auto& node : layers.at(iLayer)) {
            LutBinary* lutNode = static_cast<LutBinary*> (node.get());
            //unsigned int maxChange = (1 << layersConf[iLayer-1]->outputBits); //input layer max change, we do not subtract 1, and then have possibleVal < maxChange (and not possibleVal <= maxChange)
            if(lutNode->getDither()) {
                continue;
            }
            if(iLayer > 0)  {
                int inputNodemaxLutVal = layersConf[iLayer -1]->maxLutVal; //
                //int lutNondeOutValueInt = lutNode->getOutValueInt();
                for(unsigned int iInputNode = 0; iInputNode < lutNode->getInputNodes().size(); iInputNode++) {
                    unsigned int newAddr = lutNode->getLastAddr() ^ (1UL << iInputNode); //flip the bit corresponding to the inputLutNode
                    //std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" layer "<<iLayer<<" newAddr "<<newAddr<<std::endl;
                    int lutNonde_newOutValueInt = lutNode->getIntValues()[newAddr];

                    //getting the gradient for the inputLutNode from the lutNode->getLastGradientVsOutVal()
                    //int outValueChange = lutNonde_newOutValueInt - lutNondeOutValueInt;

                    auto lastGradient = lutNode->getLastGradientVsOutVal()[lutNonde_newOutValueInt];

                    if(lastGradient != 0) {
                        Node* inputNode = lutNode->getInputNodes()[iInputNode];

                        if(inputNode->getDither() == 0 ) {
                            //std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<std::endl;
                            LutBinary* inputLutNode = static_cast<LutBinary*>(inputNode);
                            //std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" lastGradient "<<lastGradient<<std::endl;

                            //auto dVal = (lutNode->getFloatValues()[newAddr] -lutNode->getFloatValues()[lutNode->getLastAddr()]);
                            //dVal = dVal - int(dVal); //to handle the case when the abs(dVal) is greater then 1
                            //std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<std::endl;
                            int inputNodeNewValue = inputLutNode->getOutValueInt() ^ 1UL; //flip the bit
                            inputLutNode->getLastGradientVsOutVal()[inputNodeNewValue] += lastGradient;// * fabs(dVal);
                            //inputLutNode->getGradients()[inputLutNode->getLastAddr()] += lastGradient * dVal; //in principle this is not correct, but for some reasons works better (more regularization) than the below method
                            //std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<std::endl;
                            auto lastAddr = inputLutNode->getLastAddr();

                            //std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" lastAddr "<<lastAddr<<std::endl;
                            /*std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" address "<<inputLutNode->getLastAddr()<<" outValueInt "<<outValueInt<<" outFloatValue "<<std::setw(10)<<inputLutNode->getFloatValues()[inputLutNode->getLastAddr()]
                                <<" gradientChange "<<(inputLutNode->getLastGradientVsOutVal()[outValueInt + outValueChange] * outValueChange)
                                <<" gradient "<<inputLutNode->getGradients()[inputLutNode->getLastAddr()]<<std::endl;@/

                            if(inputNodeNewValue) { //getOutValueInt == 0,  inputNodeNewValue = 1, so the gradient comes with plus
                                ;//newFloatVal += lutValchange;
                            }
                            else                 //getOutValueInt == 1,  inputNodeNewValue = 0, so the gradient comes with minus
                                lastGradient *= (-1);

                            auto& learnigParams = learnigParamsVec[iLayer -1];
                            double lutValchange = (lastGradient + (inputLutNode->getFloatValues()[lastAddr] - (inputNodemaxLutVal * 0.5) ) * learnigParams.lambda ) * learnigParams.learnigRate; //with L2 normalization
                            float newFloatVal = inputLutNode->getFloatValues()[lastAddr] - lutValchange;
                            //FIXME it is not good, that the L2 regularization is not applied when the gradient is 0
                            if(newFloatVal < 0)
                                newFloatVal = 0;
                            else if(newFloatVal > inputNodemaxLutVal)
                                newFloatVal = inputNodemaxLutVal;

                            inputLutNode->getFloatValues()[lastAddr] = newFloatVal;
                            //std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<std::endl;
                            /*
                        std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" lutNode: "<<lutNode->getName()<<"\n"
                                <<"getLastGradientVsOutVal: ";
                        for(unsigned int outVal = 0; outVal < lutNode->getLastGradientVsOutVal().size(); outVal++) {
                            std::cout<<"["<<outVal<<"] "<<lutNode->getLastGradientVsOutVal()[outVal]<<" : ";
                        }
                        std::cout<<"\n"<<" LastAddr "<<lutNode->getLastAddr()
                                <<" : newAddr "<<newAddr<<" : lutNondeOutValueInt "<<lutNondeOutValueInt<<" : lutNonde_newOutValueInt "<<lutNonde_newOutValueInt<<"\n"
                                <<" : getFloatValues[LastAddr] "<<lutNode->getFloatValues()[lutNode->getLastAddr()]<<" : getFloatValues[newAddr] "<<lutNode->getFloatValues()[newAddr]
                                <<" : dVal "<<dVal<<" lastGradient "<<lastGradient<<std::endl;
                        std::cout<<"inputLutNode "<<inputLutNode->getName()<<" address "<<std::setw(2)<<inputLutNode->getLastAddr()
                                <<" : outValueInt "<<inputLutNode->getOutValueInt()<<" : outFloatValue "<<std::setw(10)<<inputLutNode->getFloatValues()[inputLutNode->getLastAddr()]
                                <<" : inputNodeNewValue "<<inputNodeNewValue<<"\n"
                                <<" : gradientChange "<<lastGradient * dVal
                                <<" : LastGradientVsOutVal[0] "<<std::setw(10)<<(inputLutNode->getLastGradientVsOutVal()[0])
                                <<" : LastGradientVsOutVal[1] "<<std::setw(10)<<(inputLutNode->getLastGradientVsOutVal()[1])
                                <<" : gradient "<<std::setw(10)<<inputLutNode->getGradients().at(inputLutNode->getLastAddr())<<std::endl<<std::endl;@/
                        }
                        //inputLutNode->updateStat(1, 0, 0);
                    }
                }
            }
            float histeresisSize = 0.01; //0.001;
            auto lastAddr = lutNode->getLastAddr();
            auto floatVal = lutNode->getFloatValues()[lastAddr];
            if(floatVal > (lutNode->getIntValues()[lastAddr] + 0.5 + histeresisSize) ) {
                //lutNode->getIntValues()[iAddr]++; //change can be > 1, so it cannot be done like that
                lutNode->getIntValues()[lastAddr] = round(floatVal);
                //changedLutValues++;
            }
            else if(floatVal < (lutNode->getIntValues()[lastAddr] - 0.5 - histeresisSize) )
            {
                //lutNode->getIntValues()[iAddr]--; //change can be > 1, so it cannot be done like that
                lutNode->getIntValues()[lastAddr] = round(floatVal);
                //changedLutValues++;
            }

            for(auto& val : lutNode->getLastGradientVsOutVal())
                val = 0; //clear
        }
    }
}
*/

/*
void BinaryLutNetwork::dither(std::vector<LearnigParams>& learnigParamsVec, std::default_random_engine& rndGenerator) {
    //branches [outNum][branchNum][layer]
    unsigned int ditheredAddrs = 0;
    unsigned int allAddrs = 0;

    for(unsigned int iClass = 0; iClass < branches.size(); iClass++) {
        for(unsigned int branchNum = 0; branchNum < branches[iClass].size(); branchNum++) {
            for(unsigned int iLayer = 0; iLayer < branches[iClass][branchNum].size() -2; iLayer++) {//TODO for the moment we are not doing the last layer
                auto& layer = branches[iClass][branchNum][iLayer];
                //int maxOutVal = (1 << layersConf[iLayer]->outputBits) -1;
                for(auto& node : layer) {
                    //std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" "<<node->getName()<<std::endl;
                    LutBinary* lutNode = static_cast<LutBinary*> (node);
                    //unsigned int maxOutVal = lutNode->getGradients()[0].size() -1;
                    for(unsigned int iAddr = 0; iAddr < lutNode->getGradients().size(); iAddr++) {
                        int currentIntVal = lutNode->getIntValues()[iAddr];
                        float absGradient = fabs(lutNode->getGradients()[iAddr] ); //watch out - works only for binary values, //TODO fix

                        std::uniform_real_distribution<> flatDist(0., 1.);
                        double rnd = flatDist(rndGenerator);
                        if(rnd < (0.00005 * exp( - fabs(absGradient) ) ) ) {
                            if(currentIntVal == 0) {
                                lutNode->getIntValues()[iAddr] = 1;
                                lutNode->getFloatValues()[iAddr] = 1;
                            }
                            else {
                                lutNode->getIntValues()[iAddr] = 0;
                                lutNode->getFloatValues()[iAddr] = 0;
                            }
                            ditheredAddrs++;
                        }
                        allAddrs++;
                    }
                }
            }
        }
    }
    std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" ditheredAddrs "<<ditheredAddrs<<" allAddrs "<<allAddrs<<std::endl;
}*/

//void BinaryLutNetwork::updateLuts(std::vector<LearnigParams> learnigParamsVec) {
//    //branches [outNum][branchNum][layer]
//    for(unsigned int iClass = 0; iClass < branches.size(); iClass++) {
//        for(unsigned int branchNum = 0; branchNum < branches[iClass].size(); branchNum++) {
//            float maxGradient = std::numeric_limits<float>::lowest();
//            unsigned int maxGradientAddr = 0;
//            int maxGradientVal = 0;
//            LutNode* maxGradientLutNode = nullptr;
//            //for(unsigned int iLayer = branches[iClass][branchNum].size() -1; iLayer != 0  ; iLayer--) {
//            //in branches there is not sum layer,
//            for(auto layer = branches[iClass][branchNum].rbegin()+1; layer != branches[iClass][branchNum].rend(); ++layer) { //TODO why revers order iteration here?
//                for(auto& node : *layer) {
//                    std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" "<<node->getName()<<std::endl;
//                    LutNode* lutNode = static_cast<LutNode*> (node);
//                    unsigned int maxOutVal = lutNode->getGradients()[0].size();
//                    for(unsigned int iAddr = 0; iAddr < lutNode->getGradients().size(); iAddr++) {
//                        if(maxOutVal == 2) {
//                            for(unsigned int possibleOutVal = 0; possibleOutVal < maxOutVal; possibleOutVal++) {
//                                /*if(lutNode->getGradients()[iAddr][possibleOutVal] != 0)
//                                std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" "<<lutNode->getName()<<" iAddr "<<iAddr<<" possibleOutVal "<<possibleOutVal<<" "<<lutNode->getGradients()[iAddr][possibleOutVal]<<std::endl; */
//                                if(maxGradient < lutNode->getGradients()[iAddr][possibleOutVal]) {
//                                    maxGradient = lutNode->getGradients()[iAddr][possibleOutVal];
//                                    maxGradientAddr = iAddr;
//                                    maxGradientVal = possibleOutVal;
//                                    maxGradientLutNode = lutNode;
//                                }
//                            }
//                        }
//                        else {
//                            unsigned int currentVal = lutNode->getValues()[iAddr];
//                            unsigned int firstVal = (currentVal == 0 ? 0 : currentVal -1);
//                            unsigned int lastVal = (currentVal == (maxOutVal-1) ? currentVal : currentVal + 1);
//                            for(unsigned int possibleOutVal = firstVal; possibleOutVal <= lastVal; possibleOutVal++) {
//                                std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" "<<lutNode->getName()<<" iAddr "<<iAddr<<" possibleOutVal "<<possibleOutVal<<" "<<lutNode->getGradients()[iAddr][possibleOutVal]<<(possibleOutVal == currentVal ? " currentVal " : "")<<std::endl;
//
//                                if(maxGradient < lutNode->getGradients()[iAddr][possibleOutVal]) {
//                                    maxGradient = lutNode->getGradients()[iAddr][possibleOutVal];
//                                    maxGradientAddr = iAddr;
//                                    maxGradientVal = possibleOutVal;
//                                    maxGradientLutNode = lutNode;
//                                }
//                            }
//                            std::cout<<std::endl;
//                        }
//                    }
//                }
//            }
//
//            if(maxGradient > 0) {
//                maxGradientLutNode->getValues()[maxGradientAddr] = maxGradientVal;
//                std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" maxGradientLutNode "<<maxGradientLutNode->getName()<<" maxGradient "
//                        <<maxGradient<<" maxGradientVal "<<maxGradientVal<<" maxGradientAddr "<<maxGradientAddr<<std::endl;
//            }
//        }
//    }
//}


//void BinaryLutNetwork::runTraining(EventInt* event, CostFunction& costFunction) {
//    run(event);
//    double currentCost = costFunction(event->expextedResult, getOutputValues() );
//
//   /* {//debug
//        std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" currentCost "<<std::setw(12)<<currentCost<<" "<<std::endl;
//        for(unsigned int iClass = 0; iClass < event->expextedResult.size(); iClass++) {
//            //if(event->expextedResult[iClass])
//            std::cout<<" iClass "<<iClass<<" expextedResult "<<event->expextedResult[iClass]<<" nnResult "<<std::setw(12)<<std::setw(12)<<event->nnResult[iClass]<<" nodeOut "<<std::setw(5)<<layers.back()[iClass]->getOutValueInt()<<std::endl;
//        }
//    }*/
//
//    totalCost += currentCost;
//    eventCnt++;
//
//    //todo why double here?
//    std::vector<double> lastLayerOut(layers.back().size(), 0); //last layer is the layer of sumNodes
//    //std::vector<double> softMaxValues(layers.back().size(), 0);
//    auto out = lastLayerOut.begin();
//    for(auto& node : layers.back()) { //copying the values from the last layer to the lastLayerOut
//        *out = node->getOutValueInt();
//        out++;
//    }
//
//    //softMaxFunction(lastLayerOut, softMaxValues);
//    //std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" currentCost "<<costFunction(event->expextedResult, softMaxValues )<<std::endl;
//
//    //calculating the lastGradient for the last but one layer (for the nodes that goes to the sumNodes)
//    int maxChange = (1 << layersConf.end()[-2]->outputBits) -1;
//    for(unsigned int iClass = 0; iClass < event->expextedResult.size(); iClass++) {
//        std::vector<double> upadetedOutValues(lastLayerOut);
//
//        //calculating the gradient (cost function change) for every possible change out the value of given sum node (i.e. class)
//        //we assuming only one branch of given sum node is changing at one moment, therefore the possible change is from -maxChange to maxChange, where maxChange is the maximum value of the branch output node
//        upadetedOutValues.at(iClass) -= maxChange;
//        std::vector<double> gradientVersusChange(2 * maxChange + 1, 0);
//
//        for(int change = -maxChange; change <= maxChange; change++) {
//            //softMaxFunction(upadetedOutValues, softMaxValues);
//            const std::vector<double>& softMaxValues = softMaxNode->updateInput(iClass, 0, upadetedOutValues.at(iClass));
//            gradientVersusChange.at(change + maxChange) = currentCost - costFunction(event->expextedResult, softMaxValues ); // + maxChange because the change goes from -maxChange
//            //if for a given value the gradient is positive, then it means that this value is better - has lower cost
//
//            //std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" change "<<change<<" gradientVersusChange "<<gradientVersusChange.at(change + maxChange)<<std::endl;
//            upadetedOutValues.at(iClass)++;
//        }
//
//        for(auto& node : layers.back().at(iClass)->getInputNodes()) { //input nodes of the sum node - so the last layer of the lutNodes, which has int (not binary) out values
//            LutBinary* inputLutNode = static_cast<LutBinary*> (node);
//            for(int possibleOutVal = 0; possibleOutVal <= maxChange; possibleOutVal++) { //all possible out values of given node
//                inputLutNode->getLastGradientVsOutVal().at(possibleOutVal) = gradientVersusChange.at(maxChange + possibleOutVal - inputLutNode->getOutValueInt());
//
//                inputLutNode->getGradients()[inputLutNode->getLastAddr()][possibleOutVal] += inputLutNode->getLastGradientVsOutVal()[possibleOutVal];
//
//                //std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" possibleOutVal "<<possibleOutVal<<" gradient "<<lutNode->getLastGradient().at(possibleOutVal)<<(possibleOutVal == lutNode->getOutValueInt() ? " currentVal " : "")<<std::endl;
//            }
//            inputLutNode->updateStat(1, 0, 0);
//        }
//    }
//
//    //watch out - the below works only for the binary output values!!!!!!!!!!!!!!!!!!!
//    for(unsigned int iLayer = layers.size() - 2; iLayer >= 1; iLayer--) {
//        for(auto& node : layers.at(iLayer)) {
//            LutBinary* lutNode = static_cast<LutBinary*> (node.get());
//            //unsigned int maxChange = (1 << layersConf[iLayer-1]->outputBits); //input layer max change, we do not subtract 1, and then have possibleVal < maxChange (and not possibleVal <= maxChange)
//            for(unsigned int iInputNode = 0; iInputNode < lutNode->getInputNodes().size(); iInputNode++) {
//                LutBinary* inputLutNode = static_cast<LutBinary*>(lutNode->getInputNodes()[iInputNode]);
//
//                int inputNodeNewValue = inputLutNode->getOutValueInt() ^ 1UL; //flip the bit
//
//                unsigned int newAddr = lutNode->getLastAddr() ^ (1UL << iInputNode); //flip the bit corresponding to the inputLutNode
//
//                int newValue = lutNode->getIntValues()[newAddr];
//                if(newValue != lutNode->getOutValueInt()) {
//                    inputLutNode->getLastGradientVsOutVal()[inputNodeNewValue] = lutNode->getLastGradientVsOutVal()[newValue];
//                    inputLutNode->getGradients()[inputLutNode->getLastAddr()][inputNodeNewValue] += lutNode->getLastGradientVsOutVal()[newValue];
//                }
//                else {
//                    inputLutNode->getLastGradientVsOutVal()[inputNodeNewValue] = 0; //cleaning
//                }
//
//                inputLutNode->getLastGradientVsOutVal()[inputLutNode->getOutValueInt()] = 0; //cleaning
//
//                inputLutNode->updateStat(1, 0, 0);
//            }
//        }
//    }
//}
//
//
//void BinaryLutNetwork::updateLuts(std::vector<LearnigParams>& learnigParamsVec) {
//	int changedLutValues = 0;
//
//    for(unsigned int iLayer = 0; iLayer < layers.size() -1; iLayer++){ //last layer is sumNode, so is not updated
//        auto& layer = layers[iLayer];
//        int maxOutVal = (1 << layersConf[iLayer]->outputBits) -1;
//        for(auto& node : layer) {
//            //std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<" "<<node->getName()<<std::endl;
//            LutBinary* lutNode = static_cast<LutBinary*> (node.get());
//            //int maxOutVal = lutNode->getGradients()[0].size() -1;
//            for(unsigned int iAddr = 0; iAddr < lutNode->getGradients().size(); iAddr++) {
//                int currentIntVal = lutNode->getIntValues()[iAddr];
//                if(currentIntVal == 0)
//                    currentIntVal = 1;  //to make possible gradient calculation as below
//                //TODO in principle, if more then 2 out values are possible, the gradient should be (lutNode->getGradients()[iAddr][currentIntVal+1] - lutNode->getGradients()[iAddr][currentIntVal -1])/2. and but then secial care for out of bounds indexes is needed
//                float gradient = lutNode->getGradients()[iAddr][currentIntVal] - lutNode->getGradients()[iAddr][currentIntVal -1];
//                //if for a given value the gradient is positive, then it means that this value is better - has lower cost
//
///*                if(lutNode->getEntries()[iAddr]) should be get stata
//                	gradient /= lutNode->getEntries()[iAddr]; //eventCnt; //TODO maybe rather divide by number of events in a given bin, or by eventCnt in the miniBatch
//                else
//                	continue;*/
//
//                //should be like that
//                if(currentIntVal == 0)
//                	gradient = lutNode->getGradients()[iAddr][1];
//                else if(currentIntVal == maxOutVal)
//                	gradient = -lutNode->getGradients()[iAddr][maxOutVal - 1];
//                else { //this case is possible only for the luts with more then two output values, i.e. in the last layer
//                	if(lutNode->getFloatValues()[iAddr] > currentIntVal )
//                		gradient = lutNode->getGradients()[iAddr][currentIntVal + 1];
//                	else
//                		gradient = -lutNode->getGradients()[iAddr][currentIntVal - 1];
//                }
//                //end
//
//
//                float newFloatVal = lutNode->getFloatValues()[iAddr] + gradient * learnigParamsVec[iLayer].learnigRate;
//                if(newFloatVal < 0)
//                    newFloatVal = 0;
//                else if(newFloatVal > maxOutVal)
//                    newFloatVal = maxOutVal;
//
//                lutNode->getFloatValues()[iAddr] = newFloatVal;
//
//                if(lutNode->getIntValues()[iAddr] != round(newFloatVal))
//                	changedLutValues++;
//
//                lutNode->getIntValues()[iAddr] = round(newFloatVal);
//
//                /*for(unsigned int i = 0; i < lutNode->getGradients()[iAddr].size(); i++) {
//                	lutNode->getGradients()[iAddr][i] = 0;
//                }*/
//            }
//        }
//    }
//
//    std::cout<<"updateLuts changedLutValues: "<<changedLutValues<<std::endl;
//}


} /* namespace lutNN */
