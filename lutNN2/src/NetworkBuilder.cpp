/*
 * NetworkBuilder.cpp
 *
 *  Created on: Jul 10, 2019
 *      Author: kbunkow
 */

#include "lutNN/lutNN2/interface/LutBinary.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <boost/timer/timer.hpp>
#include "lutNN/lutNN2/interface/SumNode.h"

#include "lutNN/lutNN2/interface/NetworkBuilder.h"
#include "lutNN/lutNN2/interface/LutInter.h"


namespace lutNN {

NetworkBuilder::NetworkBuilder(InputNodeFactory* inputNodeFactory): inputNodeFactory(inputNodeFactory) {
    // TODO Auto-generated constructor stub

}

NetworkBuilder::~NetworkBuilder() {
    // TODO Auto-generated destructor stub
}


//void NetworkBuilder::buildTree(LayersConfigs& layersConf, InputLayer& inputLayer, No+deLayers& layers, BinaryLutNetwork::Branches& branches)
void NetworkBuilder::buildTree1(LayersConfigs& layersConf, BinaryLutNetwork& network) {
    unsigned int inputNodesCnt = layersConf.at(0)->nodesInLayer * layersConf.at(0)->nodeInputCnt;// / layersConf.at(0)->bitsPerNodeInput; //Default is ConfigParameters::fullyConnected
/*    if(layersDef[0]->layerType == ConfigParameters::oneToOne) {
        inputNodesCnt = layersConf.at(0)->neuronInputCnt * layersConf.at(0)->nodesInLayer;
    }*/

    for(unsigned int iNode = 0; iNode < inputNodesCnt; iNode++) {
        network.getInputNodes().emplace_back(std::unique_ptr<InputNode>(inputNodeFactory->get(iNode)));
    }


    network.getLayers().resize(layersConf.size());

    unsigned int outCnt = layersConf.back()->nodesInLayer;
    unsigned int branchesPerOutput = layersConf.back()->nodeInputCnt;

    //[outNum][branchNum][layer]
    network.getBranches().resize(outCnt, std::vector<std::vector<std::vector<NodePtr> > >(branchesPerOutput, std::vector<std::vector<NodePtr> >(layersConf.size())) );

    for(int iLayer = layersConf.size() -1; iLayer >= 0; iLayer--) {
        for(unsigned int iNode = 0; iNode < layersConf.at(iLayer)->nodesInLayer; iNode++) {
            std::unique_ptr<Node> node;
            unsigned int outNum = iNode / (layersConf.at(iLayer)->nodesInLayer / outCnt) ;


            /*if(layersConf[iLayer]->nodeType == LayerConfig::softMax) {
                node = std::make_unique<SumNode>(iNode, layersConf[iLayer]->nodeInputCnt);
                node->setName("sumNode_out_" + std::to_string(outNum));
            }
            else*/
            if(layersConf[iLayer]->nodeType == LayerConfig::sumIntNode) {
                node = std::make_unique<SumIntNode>(iNode, layersConf[iLayer]->nodeInputCnt);
                node->setName("sumNode_out_" + std::to_string(outNum));
            }
            else if(layersConf[iLayer]->nodeType == LayerConfig::lutNode) {//TODO fixme why LutBinary is here?
                node = std::make_unique<LutBinary>(iNode, layersConf[iLayer]->bitsPerNodeInput, layersConf[iLayer]->nodeInputCnt, layersConf[iLayer]->outputBits, layersConf[iLayer]->propagateGradient);

                unsigned int nodesPerLayerPerBranch = layersConf.at(iLayer)->nodesInLayer / (outCnt * branchesPerOutput);
                unsigned int numInBranch = iNode % nodesPerLayerPerBranch;
                unsigned int branchNum = ( iNode % (layersConf.at(iLayer)->nodesInLayer / (outCnt)) ) / (layersConf.at(iLayer)->nodesInLayer / (branchesPerOutput * outCnt) ); //
                node->setName("lutNode_out_" + std::to_string(outNum) + "_branch_" + std::to_string(branchNum) + "_layer_"  + std::to_string(iLayer) + "_num_" + std::to_string(numInBranch));

                network.getBranches().at(outNum).at(branchNum).at(iLayer).push_back(node.get());
            }

            //std::cout<<"iLayer "<<iLayer<<" creating node "<<node->getName()<<std::endl;
            //doing connections
            if( (iLayer +1) != (int)layersConf.size()) {
                unsigned int nextLayerNode = iNode / layersConf[iLayer+1]->nodeInputCnt;
                unsigned int nextLayerNodeInput = iNode % layersConf[iLayer+1]->nodeInputCnt;
                //std::cout<<"connecting iLayer "<<iLayer<<" iNode "<<iNode<<" to nextLayerNode "<<nextLayerNode<<" nextLayerNodeInput "<<nextLayerNodeInput<<std::endl;
                network.getLayers().at(iLayer + 1).at(nextLayerNode)->connectInput(node.get(), nextLayerNodeInput);
            }

            network.getLayers()[iLayer].push_back(std::move(node));
        }
    }

    //connecting input layer
    for(unsigned int iNode = 0; iNode < network.getInputNodes().size(); iNode++) {
        unsigned int nextLayerNode = iNode / layersConf[0]->nodeInputCnt;
        unsigned int nextLayerNodeInput = iNode % layersConf[0]->nodeInputCnt;
        network.getLayers().at(0).at(nextLayerNode)->connectInput(network.getInputNodes()[iNode].get(), nextLayerNodeInput);
    }

    SoftMaxWithSubClasses* softMaxNode = dynamic_cast<SoftMaxWithSubClasses*>(network. getOutputNode());
    if(softMaxNode == 0)
        throw std::invalid_argument("NetworkBuilder::buildTree2 getOutputNode is not SoftMaxWithSubClasses" );

    for(unsigned int iNode = 0; iNode < network.getLayers().back().size(); iNode++) {
        softMaxNode->addSubClass(iNode, network.getLayers().back()[iNode].get());
    }
}

void NetworkBuilder::buildTree2(LayersConfigs& layersConf, BinaryLutNetwork& network) {
    network.getLayers().resize(layersConf.size());

    SoftMaxWithSubClasses* softMaxNode = dynamic_cast<SoftMaxWithSubClasses*>(network. getOutputNode());
    if(softMaxNode == 0)
        throw std::invalid_argument("NetworkBuilder::buildTree2 getOutputNode is not SoftMaxWithSubClasses" );

    for(unsigned int iClass = 0; iClass < softMaxNode->getInputNodes().size(); iClass++) {
        addSubClass(layersConf, network, iClass,  softMaxNode->getInputNodesBySubClasses()[iClass].size() );
    }
}


void NetworkBuilder::connectTree(unsigned int iLayer, LayersConfigs& layersConf, unsigned int inputNodesCnt, BinaryLutNetwork& network, std::default_random_engine* generator) {
	std::cout<<"NetworkBuilder::connectTree "<<__LINE__<<" iLayer "<<iLayer<<std::endl;
	unsigned int iInputNode = 0;
	for(unsigned int iNode = 0; iNode < network.getLayers().at(iLayer).size(); iNode++) {
		for(unsigned int iInput = 0; iInput < layersConf.at(iLayer)->nodeInputCnt; iInput++) {
			if(iLayer == 0) {
				if(generator) {
					std::uniform_int_distribution<> flatDist(0, network.getInputNodes().size() -1);
					iInputNode = flatDist(*generator);
				}
				else {
					iInputNode = iInputNode % network.getInputNodes().size();
				}
				network.getLayers().at(iLayer).at(iNode)->connectInput(network.getInputNodes().at(iInputNode).get(), iInput);
				std::cout<<"iLayer "<<iLayer<<" "<<network.getLayers().at(iLayer).at(iNode)->getName()<<" connecting input node "<<network.getInputNodes().at(iInputNode).get()->getName()<<std::endl;
			}
			else {
				iInputNode = iInputNode % network.getLayers().at(iLayer -1).size();

				//std::cout<<"NetworkBuilder::connectTree "<<__LINE__<<" iLayer "<<iLayer<<" "<<network.getLayers().at(iLayer).at(iNode)->getName()<<" connecting  node "<<iInputNode<<std::endl;
				network.getLayers().at(iLayer).at(iNode)->connectInput(network.getLayers().at(iLayer -1).at(iInputNode).get(), iInput);
			}
			iInputNode++;
		}
	}
}



void NetworkBuilder::buildTree3(LayersConfigs& layersConf, unsigned int inputNodesCnt, BinaryLutNetwork& network, std::default_random_engine* generator) {
	buildLayers(layersConf, inputNodesCnt, network);

	std::cout<<"NetworkBuilder::buildTree3, line "<<__LINE__<<std::endl;
	unsigned int iSumNode = 0;
    for(unsigned int iClass = 0; iClass < layersConf.back()->classesCnt; iClass++) {
        auto softMaxNode = dynamic_cast<SoftMaxWithSubClasses*>(network.getOutputNode());
        if(softMaxNode) {
            for(unsigned int iSubClass = 0; iSubClass < layersConf.back()->subClassesCnt; iSubClass++) {
                network.getLayers().back().at(iSumNode)->setName("sumNode_class_" + std::to_string(iClass) + "_" + std::to_string(iSubClass) ) ;
                softMaxNode->addSubClass(iClass, network.getLayers().back().at(iSumNode).get() );
                std::cout<<"NetworkBuilder::buildTree3 "<<__LINE__<<" softMaxNode->addSubClass iClass "<<iClass<<" "<<" iSubClass "<<softMaxNode->getInputNodesBySubClasses().at(iClass).size()-1
                        <<" "<<" iSumNode "
                        <<iSumNode<<" "<<network.getLayers().back().at(iSumNode)->getName()<<std::endl;
                iSumNode++;
            }
        }
        else {
    	    network.getOutputNode()->connectInput(network.getLayers().back().at(iSumNode).get(), iClass);
    	    std::cout<<"NetworkBuilder::buildTree3 "<<__LINE__<<" getOutputNode()->connectInput iClass "<<iClass<<" "<<network.getOutputNode()->getInputNodes().at(iClass)->getName()<<std::endl;
    	    iSumNode++;
    	}
    }

	std::cout<<"NetworkBuilder::buildTree3, line "<<__LINE__<<std::endl;
    for(unsigned int iLayer = 0; iLayer < network.getLayers().size() ; iLayer++) {
    	connectTree(iLayer, layersConf, inputNodesCnt, network, generator);
    }

    std::cout<<"NetworkBuilder::buildTree3 finished, line "<<__LINE__<<std::endl;
}

void NetworkBuilder::buildTree4(LayersConfigs& layersConf, unsigned int inputNodesCnt, BinaryLutNetwork& network, std::default_random_engine* generator, std::vector<EventInt*>& events) {
	buildLayers(layersConf, inputNodesCnt, network);

	std::cout<<"NetworkBuilder::buildTree4 "<<__LINE__<<std::endl;
	unsigned int iSumNode = 0;
	/*    for(unsigned int iClass = 0; iClass < network.getOutputNode()->getInputNodes().size(); iClass++) {
    	for(unsigned int iSubClass = 0; iSubClass < layersConf.back()->subClassesCnt; iSubClass++) {
    		network.getSoftMaxNode()->addSubClass(iClass, network.getLayers().back().at(iSumNode).get() );
    		iSumNode++;
    	}
    }*/

    for(unsigned int iClass = 0; iClass < layersConf.back()->classesCnt; iClass++) {
        auto softMaxNode = dynamic_cast<SoftMaxWithSubClasses*>(network.getOutputNode());
        if(softMaxNode) {
            for(unsigned int iSubClass = 0; iSubClass < layersConf.back()->subClassesCnt; iSubClass++) {
                network.getLayers().back().at(iSumNode)->setName("sumNode_class_" + std::to_string(iClass) + "_" + std::to_string(iSubClass) ) ;
                softMaxNode->addSubClass(iClass, network.getLayers().back().at(iSumNode).get() );
                std::cout<<"NetworkBuilder::buildTree4 "<<__LINE__<<" softMaxNode->addSubClass iClass "<<iClass<<" "<<" iSubClass "<<softMaxNode->getInputNodesBySubClasses().at(iClass).size()-1
                        <<" "<<" iSumNode "
                        <<iSumNode<<" "<<network.getLayers().back().at(iSumNode)->getName()<<std::endl;
                iSumNode++;
            }
        }
        else {
            network.getOutputNode()->connectInput(network.getLayers().back().at(iSumNode).get(), iClass);
            std::cout<<"NetworkBuilder::buildTree4 "<<__LINE__<<" getOutputNode()->connectInput iClass "<<iClass<<" "<<network.getOutputNode()->getInputNodes().at(iClass)->getName()<<std::endl;
            iSumNode++;
        }
    }

    //makeOptimalfirstLayer(layersConf, events, network.getInputNodes(), network.getLayers().front(), generator);
    makeOptimalfirstLayer2(layersConf, events, network.getInputNodes(), network.getLayers().front(), generator);

	std::cout<<"NetworkBuilder::buildTree4 "<<__LINE__<<std::endl;
    for(unsigned int iLayer = 1; iLayer < network.getLayers().size() ; iLayer++) {
    	connectTree(iLayer, layersConf, inputNodesCnt, network, generator);
    }
}

void NetworkBuilder::addSubClass(LayersConfigs& layersConf, BinaryLutNetwork& network, unsigned int classNum, unsigned int subClassNum) {
    std::cout<<"addSubClass classNum "<<classNum<<std::endl;
    NodePtr sumNode = nullptr;
    if(layersConf.back()->nodeType == LayerConfig::sumIntNode) {
        auto node = std::make_unique<SumIntNode>(network.getLayers().back().size(), layersConf.back()->nodeInputCnt);
        node->setName("sumNode_class_" + std::to_string(classNum) + "_" + std::to_string(subClassNum) );
        sumNode = node.get();
        network.getLayers().back().push_back(std::move(node));
    }
    else {
        throw; //TODO
    }


    NodeVec layerNodes;
    layerNodes.push_back(sumNode);

    NodeVec prevLayerNodes;

    //we should start from the sum node and connect the lutNodes to it
    for(int iLayer = layersConf.size()-1; iLayer >= 0; iLayer--) {
        //unsigned int iNode = 0;
        for(auto& node : layerNodes) {
            for(unsigned int iInputNode = 0; iInputNode < node->getInputNodes().size(); iInputNode++) {
                if(iLayer > 0) {
                    auto& inputLayerConf = layersConf[iLayer-1];
                    if(inputLayerConf->nodeType == LayerConfig::lutNode) {
                        unsigned int numInlayer = network.getLayers()[iLayer -1].size();
                        auto inputNode = std::make_unique<LutBinary>(numInlayer, inputLayerConf->bitsPerNodeInput, inputLayerConf->nodeInputCnt, inputLayerConf->outputBits, inputLayerConf->propagateGradient);

                        inputNode->setName("lutNode_class_" + std::to_string(classNum) + "_" + std::to_string(subClassNum)+ "_layer_"  + std::to_string(iLayer-1) + "_num_" + std::to_string(numInlayer));

                        node->connectInput(inputNode.get(), iInputNode);

                        prevLayerNodes.push_back(inputNode.get());
                        std::cout<<"iLayer "<<iLayer<<" adding node "<<inputNode->getName()<<std::endl;

                        network.getLayers()[iLayer-1].push_back(std::move(inputNode));
                    }
                }
                else {
                    network.getInputNodes().emplace_back(std::unique_ptr<InputNode>(inputNodeFactory->get(network.getInputNodes().size() ) ) );

                    node->connectInput(network.getInputNodes().back().get(), iInputNode);
                    std::cout<<"iLayer "<<iLayer<<" adding input node "<<network.getInputNodes().back()->getName()<<std::endl;
                }
            }
        }
        layerNodes = prevLayerNodes;
        prevLayerNodes.clear();
    }

    SoftMaxWithSubClasses* softMaxNode = dynamic_cast<SoftMaxWithSubClasses*>(network. getOutputNode());

    if(softMaxNode)
        softMaxNode->addSubClass(classNum, sumNode);
    else {
        throw std::invalid_argument("NetworkBuilder::addSubClass getOutputNode is not  SoftMaxWithSubClasses" );
    }
}


Node* NetworkBuilder::getNode(const LayerConfig* layerConfig, unsigned int number, unsigned int subClassNum ) {
/*    softMax,
    sumNode,
    lutNode,
    lutInter,
    neuron,*/

    if(layerConfig->nodeType == LayerConfig::lutNode) {
        return new LutNode(number, layerConfig->nodeInputCnt, 1<<(layerConfig->bitsPerNodeInput * layerConfig->nodeInputCnt), layerConfig->propagateGradient);
    }

    if(layerConfig->nodeType == LayerConfig::lutInter) {
        return new LutInter(number, layerConfig->bitsPerNodeInput, layerConfig->propagateGradient, layerConfig->lutRangesCnt, layerConfig->interpolate);
    }

    if(layerConfig->nodeType == LayerConfig::sumNode) {
        return new SumNode(number, layerConfig->nodeInputCnt, layerConfig->outValOffset, layerConfig->biasShift, layerConfig->shiftLastGradient);
    }

    if(layerConfig->nodeType == LayerConfig::lutBinary) {
        return new LutBinary(number, layerConfig->bitsPerNodeInput, layerConfig->nodeInputCnt, layerConfig->outputBits, layerConfig->propagateGradient);
    }

    if(layerConfig->nodeType == LayerConfig::sumIntNode) {
        auto node = new SumIntNode(number, layerConfig->nodeInputCnt);
        node->setName("sumIntNode_class_" + std::to_string(number) + "_" + std::to_string(subClassNum) );
        return node;
    }

    throw std::invalid_argument("NetworkBuilder::getNode not implemented for Node type " + std::to_string(layerConfig->nodeType) );
    return nullptr;
}


void NetworkBuilder::buildLayers(LayersConfigs& layersConf, unsigned int inputNodesCnt, LutNetworkBase& network) {
    std::cout<<"NetworkBuilder::buildLayers "<<__LINE__<<std::endl;

    //unsigned int inputNodesCnt = layersConf.at(0)->nodesInLayer * layersConf.at(0)->nodeInputCnt;

    std::cout<<"NetworkBuilder::buildLayers "<<__LINE__<<" inputNodesCnt "<<inputNodesCnt<<std::endl;
    for(unsigned int iNode = 0; iNode < inputNodesCnt; iNode++) {
        network.getInputNodes().emplace_back(std::unique_ptr<InputNode>(inputNodeFactory->get(iNode)));
    }

    std::cout<<"NetworkBuilder::buildLayers "<<__LINE__<<std::endl;
    network.getLayers().resize(layersConf.size());

    std::cout<<"NetworkBuilder::buildLayers "<<__LINE__<<" network.getLayers().size() "<<network.getLayers().size()<<" layer 0 size "<<network.getLayers()[0].size()<<std::endl;
    for(unsigned int iLayer = 0; iLayer < layersConf.size(); iLayer++ ) {
        //network.getLayers().emplace_back( layersConf[iLayer]->nodesInLayer ); //
    	//network.getLayers().at(iLayer).resize(layersConf[iLayer]->nodesInLayer);
        std::cout<<"NetworkBuilder::buildLayers "<<__LINE__<<" layer "<<iLayer<<" size "<<network.getLayers()[iLayer].size()<<" nodesInLayer "<<layersConf[iLayer]->nodesInLayer<<std::endl;
        for(unsigned int iNode = 0; iNode < layersConf[iLayer]->nodesInLayer; iNode++) {
            network.getLayers()[iLayer].emplace_back(getNode(layersConf[iLayer].get(), iNode));

            network.getLayers()[iLayer].back()->setName(network.getLayers()[iLayer].back()->getName() + "_layer_"  + std::to_string(iLayer) + "_node_" + std::to_string(iNode));
            	//+ std::to_string(classNum) + "_" + std::to_string(subClassNum)+
        }
    }

    std::cout<<"NetworkBuilder::buildLayers "<<__LINE__<<std::endl;
}

void NetworkBuilder::buildNet(LayersConfigs& layersConf, unsigned int inputNodeCnt, LutNetworkBase& network) {
	//unsigned int inputNodesCnt = layersConf.at(0)->nodesInLayer * layersConf.at(0)->nodeInputCnt; //TODO
	buildLayers(layersConf, inputNodeCnt, network);

    //if(layersConf.at(0)->layerType == ConfigParameters::fullyConnected)
    /*{
        for(auto& node : network.getLayers().at(0)) {
            node->setInputNodes(network.getInputNodes());
        }
    }*/

    for(unsigned int iNode = 0; iNode < network.getLayers().at(0).size(); iNode++) {
        network.getLayers()[0][iNode]->connectInput(network.getInputNodes().at(iNode % network.getInputNodes().size()).get(), 0);
    }

    //must be here after connecting the network.getInputNodes() to the layer 0
    if(network.getNoHitCntNode()) {
        network.getInputNodes().emplace_back(std::unique_ptr<InputNode>(network.getNoHitCntNode()));
    }

    if(layersConf[1]->nodeType == LayerConfig::sumNode) {
        for(unsigned int iNode = 0; iNode < network.getLayers()[1].size(); iNode++) {
            SumNode* sumNode = static_cast<SumNode*>(network.getLayers()[1][iNode].get());
            sumNode->setBiasNode(network.getNoHitCntNode());
        }
    }


    std::cout<<"NetworkBuilder::buildNet "<<__LINE__<<std::endl;
}


void NetworkBuilder::connectNet(LayersConfigs& layersConf, LutNetworkBase& network) {
    for(unsigned int iLayer = 1; iLayer < network.getLayers().size(); iLayer++ ) {
        //if(layersDef[iLayer]->layerType == ConfigParameters::fullyConnected)
        if(layersConf[iLayer]->nodeType == LayerConfig::lutNode || layersConf[iLayer]->nodeType == LayerConfig::lutInter) {
            for(unsigned int iNode = 0; iNode < network.getLayers()[iLayer].size(); iNode++) {
                network.getLayers()[iLayer][iNode]->connectInput(network.getLayers()[iLayer -1].at(iNode % network.getLayers()[iLayer-1].size()).get(), 0);
            }
        }

        if(layersConf[iLayer]->nodeType == LayerConfig::sumNode || layersConf[iLayer]->nodeType == LayerConfig::sumIntNode) {
            int numberInInputLayer = 0;
            for(unsigned int iNode = 0; iNode < network.getLayers()[iLayer].size(); iNode++) {
                unsigned int nodeInputCnt = network.getLayers()[iLayer][iNode]->getInputNodes().size();

                for(unsigned int inputNode = 0; inputNode < nodeInputCnt; inputNode++) {
                    /*if(numberInInputLayer >= network.getLayers()[iLayer -1].size()) {
                        break;
                    }*/
                    network.getLayers()[iLayer][iNode]->connectInput(network.getLayers()[iLayer -1].at(numberInInputLayer).get(), inputNode);
                    numberInInputLayer++;
                }
            }
        }
    }

    network.getOutputNode()->connectInputs(network.getLayers().back());
}
/*
void NetworkBuilder::connectOneNetPerOut() {
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

void NetworkBuilder::connect() {
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

void NetworkBuilder::connect1() {
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
}*/


template<typename T>
T& get2DView(std::vector<T>& vec, size_t sizeX, size_t sizeY, size_t x, size_t y) {
    if(x >= sizeX) throw std::invalid_argument("get2DView: x >= sizeX. x " + std::to_string(x) + " sizeX " + std::to_string(sizeX) );
    if(y >= sizeY) throw std::invalid_argument("get2DView: y >= sizeY. y " + std::to_string(y) + " sizeY " + std::to_string(sizeY) );
    if(sizeX * sizeY > vec.size())
        throw std::invalid_argument("sizeX * sizeY > vec.size(), sizeX " + std::to_string(sizeX) + " sizeY " + std::to_string(sizeY) + " vec.size() " + std::to_string(vec.size()) );

    size_t i = x + y * sizeX;
    return vec.at(i);
}


//layerA is input for the layerB
void connect2D(const LayerConfig* layerConfigA, const LayerConfig* layerConfigB, NodeLayer& layerA, NodeLayer& layerB) {
    size_t strideX = layerConfigB->strideX;
    size_t strideY = layerConfigB->strideY;

    size_t repeatX = layerConfigB->repeatX;
    size_t repeatY = layerConfigB->repeatY;

    size_t sizeXB = layerConfigB->sizeX;
    size_t sizeYB = layerConfigB->sizeY;

    size_t sizeXA = layerConfigA->sizeX;
    size_t sizeYA = layerConfigA->sizeY;

    size_t sizeTileXA = layerConfigB->sizeTileX;
    size_t sizeTileYA = layerConfigB->sizeTileY;

    std::cout<<"\nconnect2D strideX "<<strideX<<" strideY "<<strideY<<" repeatX "<<repeatX<<" repeatY "<<repeatY
            <<" sizeXB "<<sizeXB<<" sizeYB "<<sizeYB<<" sizeXA "<<sizeXA<<" sizeYA "<<sizeYA<<std::endl;

    for(size_t yB = 0; yB < sizeYB; yB++) {
        for(size_t xB = 0; xB < sizeXB; xB++) {
            size_t yAStart = (yB / repeatY * strideY) % sizeYA;
            size_t xAStart = (xB / repeatX * strideX) % sizeXA;
            auto& nodeB = get2DView(layerB, sizeXB, sizeYB, xB, yB);
            std::cout<<nodeB->getName()<<" xB "<<xB<<" yB "<<yB<<std::endl;
            size_t inputNum = 0;
            for(size_t yA = 0; yA < sizeTileYA; yA++) {
                for(size_t xA = 0; xA < sizeTileXA; xA++) {
                    std::cout<<"  input "<<inputNum<<" xA "<<xA<<" yA "<<yA<<" xB "<<xB<<" yB "<<yB;

                    auto& nodeA = get2DView(layerA, sizeXA, sizeYA, xAStart + xA, yAStart + yA);

                    std::cout<<" connecting "<<nodeA->getName()<<std::endl;

                    nodeB->connectInput(nodeA.get(), inputNum);
                    inputNum++;
                }
            }
        }
    }
}

void NetworkBuilder::buildTree2D(LayerConfig& inputLayerConfig, LayersConfigs& layersConf, unsigned int inputNodesCnt, BinaryLutNetwork& network) {
    buildLayers(layersConf, inputNodesCnt, network);

    std::cout<<"NetworkBuilder::buildTree2D, line "<<__LINE__<<std::endl;
    unsigned int iSumNode = 0;
    for(unsigned int iClass = 0; iClass < layersConf.back()->classesCnt; iClass++) {
        network.getOutputNode()->connectInput(network.getLayers().back().at(iSumNode).get(), iClass);
        std::cout<<"NetworkBuilder::buildTree2D "<<__LINE__<<" getOutputNode()->connectInput iClass "<<iClass<<" "<<network.getOutputNode()->getInputNodes().at(iClass)->getName()<<std::endl;
        iSumNode++;
    }

    std::cout<<"NetworkBuilder::buildTree2D, line "<<__LINE__<<std::endl;
    for(unsigned int iLayer = 0; iLayer < network.getLayers().size(); iLayer++) { //last layer is sum node
        if(iLayer == 0) {
            NodeLayer& layerA = network.getInputNodes();
            NodeLayer& layerB = network.getLayers().at(iLayer);

            std::cout<<"NetworkBuilder::buildTree2D connecting to the layer "<<iLayer<<std::endl;
            connect2D(&inputLayerConfig, layersConf[iLayer].get(), layerA, layerB);
        }
        else {
            NodeLayer& layerA = network.getLayers().at(iLayer -1);
            NodeLayer& layerB = network.getLayers().at(iLayer);

            std::cout<<"NetworkBuilder::buildTree2D connecting to the layer "<<iLayer<<std::endl;
            connect2D(layersConf[iLayer-1].get(), layersConf[iLayer].get(), layerA, layerB);;
        }
    }

    std::cout<<"NetworkBuilder::buildTree3 finished, line "<<__LINE__<<std::endl;
}

void NetworkBuilder::makeOptimalfirstLayer(LayersConfigs& layersConf, std::vector<EventInt*>& events, InputLayer& inputNodes, NodeLayer& nodeLayer, std::default_random_engine* generator) {
	LayerConfig* layerConf = layersConf.front().get();

	struct NodeWithStat {
    	LutNode* node = nullptr;
    	std::vector<std::vector<int> > stat; //[lutAddr][class]
    	double rank = 0;

    	NodeWithStat(LayerConfig* layerConf, unsigned int classCnt, unsigned int iNode) {
    		node = static_cast<LutNode*>(NetworkBuilder::getNode(layerConf, iNode));

    		stat.resize(node->getFloatValues().size(), std::vector<int>(classCnt) );
    	}
    };

    unsigned int classCnt = layersConf.back()->classesCnt;
    std::vector<NodeWithStat*> nodes;
	for(unsigned int iNode = 0; iNode < 20000; iNode++) {
		nodes.push_back(new NodeWithStat(layerConf, classCnt, iNode));

        std::uniform_int_distribution<> flatDist(0, inputNodes.size() -1);
		for(unsigned int iInput = 0; iInput < layerConf->nodeInputCnt; iInput++) {
			unsigned int iInputNode = flatDist(*generator);

			nodes.back()->node->connectInput(inputNodes.at(iInputNode).get(), iInput);
		}
    }

	std::cout<<" makeOptimalfirstLayer running events "<<std::endl;
	{
		boost::timer::auto_cpu_timer timer1;
		int iEv = 0;
		for(auto& event : events) {
			for(auto& inputNode : inputNodes) {
				inputNode->setOutValue(event->inputs[inputNode->getNumber()]);
			}

			for(auto& nodeWithStat : nodes) {
				nodeWithStat->node->run();
				nodeWithStat->stat.at(nodeWithStat->node->getLastAddr()).at(event->classLabel )++;
			}

			if(iEv%100 == 0)
				std::cout<<" makeOptimalfirstLayer event "<<iEv<<std::endl;
			iEv++;
		}
	}

	std::cout<<" makeOptimalfirstLayer calculating rank "<<std::endl;
	for(auto& nodeWithStat : nodes) {
	    for(unsigned int iAddr = 0; iAddr < nodeWithStat->stat.size(); iAddr++) {
	    	auto& statForAddr = nodeWithStat->stat.at(iAddr);
	    	auto max = std::max_element(statForAddr.begin(), statForAddr.end());
	    	//std::cout << "max element at: " << std::distance(statForAddr.begin(), max) << '\n';
	    	nodeWithStat->rank += *max;
	    }
	}

	std::cout<<" makeOptimalfirstLayer sorting "<<std::endl;
	std::sort(nodes.begin(), nodes.end(),
			[](NodeWithStat* a, NodeWithStat* b) {
				return a->rank > b->rank;
			}
	);

	//debug printout
	/*for(auto& nodeWithStat : nodes) {
		std::cout<<nodeWithStat->node->getName()<<" rank "<<nodeWithStat->rank<<std::endl;

		std::cout<<"   ";
		for(unsigned int iAddr = 0; iAddr < nodeWithStat->stat.size(); iAddr++) {
			std::cout<<std::setw(4)<<iAddr<<" ";
		}
		std::cout<<std::endl;
		for(unsigned int iClass = 0; iClass < nodeWithStat->stat[0].size(); iClass++) {
			std::cout<<iClass<<"  ";
			for(unsigned int iAddr = 0; iAddr < nodeWithStat->stat.size(); iAddr++) {
				std::cout<<std::setw(4)<<nodeWithStat->stat[iAddr][iClass]<<" ";
			}
			std::cout<<std::endl;
		}
	}*/

	std::cout<<" makeOptimalfirstLayer finalizing "<<std::endl;
	for(unsigned int iNode = 0; iNode < nodes.size(); iNode++) {
		if(iNode < nodeLayer.size() ) {
			nodeLayer.at(iNode).reset(nodes.at(iNode)->node);
			std::cout<<" adding node "<<nodes.at(iNode)->node->getName()<<" rank "<<nodes.at(iNode)->rank<<std::endl;
		}
		else {
			delete nodes.at(iNode)->node;
		}
	}

	//TODO shuffle the the nodeLayer!!!!!!!!!!!!!!!!!!!!!!!!!!!!
}

void NetworkBuilder::makeOptimalfirstLayer2(LayersConfigs& layersConf, std::vector<EventInt*>& events, InputLayer& inputNodes, NodeLayer& firstLayer, std::default_random_engine* generator) {
    LayerConfig* layerConf = layersConf.front().get();

    struct NodeWithStat {
        Node* node = nullptr;
        std::vector<std::vector<int> > stat; //[inputValue][class]
        std::vector<double> rank;

        NodeWithStat(LayerConfig* layerConf, unsigned int classCnt, Node*  inputNode):
            node(inputNode), stat(1<<layerConf->bitsPerNodeInput, std::vector<int>(classCnt) ), rank(classCnt) {
        }
    };

    unsigned int classCnt = layersConf.back()->classesCnt;
    std::vector<NodeWithStat*> nodesWithStat;

    std::cout<<" classCnt "<<classCnt<<std::endl;
    for(auto& inputNode : inputNodes) {
        nodesWithStat.push_back(new NodeWithStat(layerConf, classCnt, inputNode.get() ));
    }


    std::cout<<" rankInputs running events "<<std::endl;
    {
        boost::timer::auto_cpu_timer timer1;
        int iEv = 0;
        for(auto& event : events) {
            for(auto& nodeWithStat : nodesWithStat) {
                nodeWithStat->node->setOutValue(event->inputs[nodeWithStat->node->getNumber()]);
                nodeWithStat->stat.at(nodeWithStat->node->getOutValue()).at(event->classLabel )++;
            }

            if(iEv%100 == 0)
                std::cout<<" rankInputs event "<<iEv<<std::endl;
            iEv++;
        }
    }

    std::cout<<" rankInputs calculating rank "<<std::endl;
    for(auto& nodeWithStat : nodesWithStat) {

        std::vector<std::vector<double> > posterior(nodeWithStat->stat.size(), std::vector<double>(nodeWithStat->stat[0].size()) ); //Bayes posterior
        std::vector<double> prior(classCnt, 0); //Bayes prior i.e. P(iClass)
        std::vector<double> pVal(nodeWithStat->stat.size(), 0); //probability of the val, i.e. norm in the Bayes formula

        for(unsigned int val = 0; val < nodeWithStat->stat.size(); val++) {
            double norm = 0; //events with a ginvn feature value
            for(unsigned int iClass = 0; iClass < nodeWithStat->stat[val].size(); iClass++) {
                norm += nodeWithStat->stat[val][iClass];
            }

            pVal.at(val) = norm / events.size();

            for(unsigned int iClass = 0; iClass < nodeWithStat->stat[val].size(); iClass++) {
                if(norm)
                    posterior[val][iClass] = nodeWithStat->stat[val][iClass]/norm;
                else
                    posterior[val][iClass] = 0;
            }
        }

        for(unsigned int iClass = 0; iClass < classCnt; iClass++) {
            double norm = 0; //events in each class
            for(unsigned int val = 0; val < nodeWithStat->stat.size(); val++) {
                norm += nodeWithStat->stat.at(val).at(iClass);
            }

            prior.at(iClass) = norm / events.size();

            for(unsigned int val = 0; val < nodeWithStat->stat.size(); val++) {
                double likelihood = nodeWithStat->stat[val][iClass] / norm;

                //nodeWithStat->rank[iClass] += posterior[val][iClass] * likelihood;

                //nodeWithStat->rank[iClass] += pVal[val] *  (posterior[val][iClass] - prior[iClass] ) * (posterior[val][iClass] - prior[iClass] );
                nodeWithStat->rank[iClass] += sqrt(pVal[val] *  (posterior[val][iClass] - prior[iClass] ) * (posterior[val][iClass] - prior[iClass] ) ); //sqrt to have the distribution weaker

                //checking:
                double posterior_ = likelihood * prior[iClass] / pVal[val];
                if(pVal[val] == 0)
                    posterior_ = 0;
                if( fabs(posterior_ - posterior[val][iClass]) > 0.00000001) {
                    std::cout<<" rankInputs posterior_ != posterior[val][iClass] "
                            <<" posterior_ "<<posterior_
                            <<" posterior[val][iClass] "<<posterior[val][iClass]
                            <<" !!!!!!!!!!!!!!!!!!!!!!!!!!!! "<<std::endl;
                }

            }
        }

    }



    unsigned int nodesPerClass = firstLayer.size() / classCnt;
    for(unsigned int iClass = 0; iClass < classCnt; iClass++) {
        std::cout<<" rankInputs sorting iClass "<<iClass<<std::endl;
        std::sort(nodesWithStat.begin(), nodesWithStat.end(),
                [&](NodeWithStat* a, NodeWithStat* b) {
            return a->rank[iClass] > b->rank[iClass];
        }
        );

        ////////////////////////// debug printout
        /*unsigned int iNode = 0;
        for(auto& nodeWithStat : nodesWithStat) {
            std::cout<<"\n"<<iNode<<" "<<nodeWithStat->node->getName()<<std::endl;
            std::cout<<"   ";
            for(unsigned int iAddr = 0; iAddr < nodeWithStat->stat.size(); iAddr++) {
                std::cout<<std::setw(4)<<iAddr<<" ";
            }
            std::cout<<std::endl;
            for(unsigned int iClass = 0; iClass < nodeWithStat->stat[0].size(); iClass++) {
                std::cout<<iClass<<"  ";
                for(unsigned int iAddr = 0; iAddr < nodeWithStat->stat.size(); iAddr++) {
                    std::cout<<std::setw(4)<<nodeWithStat->stat[iAddr][iClass]<<" ";
                }
                std::cout<<std::endl;
            }

            for(unsigned int iClass = 0; iClass < nodeWithStat->stat[0].size(); iClass++) {
                std::cout<<" iClass "<<iClass<<" rank "<<nodeWithStat->rank[iClass]<<std::endl;
            }

            iNode++;
        }*/
        /////////////////////////////////////////////////////////////////////////////// end debug printout

        std::vector<double> ranks;
        for(auto& nodeWithStat : nodesWithStat) {
            if(nodeWithStat->rank[iClass] > 1e-6)
                ranks.push_back(nodeWithStat->rank[iClass]);
        }

        std::discrete_distribution<> distribution(ranks.begin(), ranks.end());

        for(unsigned int iNode = nodesPerClass * iClass; iNode <  nodesPerClass * (iClass +1); iNode++) {
            auto& node = firstLayer.at(iNode);
            std::cout<<"iClass "<<iClass<<" "<<node->getName()<<std::endl;
            for(unsigned int iInput = 0; iInput < layersConf.at(0)->nodeInputCnt; iInput++) {

                int iInputNode = distribution(*generator);
                auto& nodeWithStat =  nodesWithStat.at(iInputNode);
                node->connectInput(nodeWithStat->node, iInput);
                std::cout<<" connecting input node "<<nodeWithStat->node->getName()<<" rank "<<nodeWithStat->rank[iClass]<<std::endl;
            }
        }
    }

}

} /* namespace lutNN */
