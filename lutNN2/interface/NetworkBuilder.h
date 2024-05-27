/*
 * NetworkBuilder.h
 *
 *  Created on: Jul 10, 2019
 *      Author: kbunkow
 */

#ifndef INTERFACE_NETWORKBUILDER_H_
#define INTERFACE_NETWORKBUILDER_H_

#include "lutNN/lutNN2/interface/BinaryLutNetwork.h"
#include "lutNN/lutNN2/interface/LutNetworkBase.h"
#include "LutNetworkBase.h"

#include "lutNN/lutNN2/interface/Event.h"

namespace lutNN {

class NetworkBuilder {
public:
    NetworkBuilder(InputNodeFactory* inputNodeFactory);

    virtual ~NetworkBuilder();


    //fills the inputLayer and layers , makes the connections between the nodes

    //creates selected number of branches for each class/output, there are no interconections between branches
    //void buildTree(LayersConfigs& layersConf, InputLayer& inputLayer, NodeLayers& layers, BinaryLutNetwork::Branches& branches);

    void buildTree1(LayersConfigs& layersConf, BinaryLutNetwork& network);

    void buildTree2(LayersConfigs& layersConf, BinaryLutNetwork& network);

    void buildTree3(LayersConfigs& layersConf, unsigned int inputNodesCnt, BinaryLutNetwork& network, std::default_random_engine* generator = nullptr);

    //uses makeOptimalfirstLayer
    void buildTree4(LayersConfigs& layersConf, unsigned int inputNodesCnt, BinaryLutNetwork& network, std::default_random_engine* generator, std::vector<EventInt*>& events);

    void addSubClass(LayersConfigs& layersConf, BinaryLutNetwork& network, unsigned int classNum, unsigned int subClassNum);

    void connectTree(unsigned int iLayer, LayersConfigs& layersConf, unsigned int inputNodesCnt, BinaryLutNetwork& network, std::default_random_engine* generator);

    void buildLayers(LayersConfigs& layersConf, unsigned int inputNodesCnt, LutNetworkBase& network);

    void buildNet(LayersConfigs& layersConf, unsigned int inputNodeCnt, LutNetworkBase& network);
    void connectNet(LayersConfigs& layersConf, LutNetworkBase& network);

    void buildTree2D(LayerConfig& inputLayerConfig, LayersConfigs& layersConf, unsigned int inputNodesCnt, BinaryLutNetwork& network);

    static Node* getNode(const LayerConfig* layerConfig, unsigned int number, unsigned int subClassNum = 0);

    void makeOptimalfirstLayer(LayersConfigs& layersConf, std::vector<EventInt*>& events, InputLayer& inputNodes, NodeLayer& nodeLayer, std::default_random_engine* generator);

    void makeOptimalfirstLayer2(LayersConfigs& layersConf, std::vector<EventInt*>& events, InputLayer& inputNodes, NodeLayer& firstLayer, std::default_random_engine* generator);

private:
    InputNodeFactory* inputNodeFactory = nullptr;
};

} /* namespace lutNN */

#endif /* INTERFACE_NETWORKBUILDER_H_ */
