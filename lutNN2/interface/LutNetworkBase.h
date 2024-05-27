/*
 * LutNetwork.h
 *
 *  Created on: May 1, 2018
 *      Author: Karol Bunkowski
 */

#ifndef LUTNETWORKBASE_H_
#define LUTNETWORKBASE_H_

#include "lutNN/lutNN2/interface/Event.h"
#include "lutNN/lutNN2/interface/LutNode.h"
#include "lutNN/lutNN2/interface/InputNodeFactory.h"
#include "lutNN/lutNN2/interface/NetworkOutNode.h"
#include "lutNN/lutNN2/interface/CostFunction.h"
#include "lutNN/lutNN2/interface/LayerConfig.h"

#include <vector>
#include <memory>
#include <iostream>
#include <random>
#include <exception>


#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>


namespace lutNN {

typedef std::vector<NodeLayer> NodeLayers;

class LutNetworkBase {
public:
	LutNetworkBase(): outputNode(nullptr) {};

	//takes the ownership of the outputNode and inputSetter
    LutNetworkBase(NetworkOutNode* outputNode, bool useNoHitCntNode); //, OutputType outputType
    virtual ~LutNetworkBase();

    virtual void initLuts(std::default_random_engine& generator);
    virtual void initLuts2(std::default_random_engine& generator);

    InputLayer& getInputNodes() {
        return inputNodes;
    }

/*    const NeuronLayer& getOutputLayer() const {
        return neuronLayers.back();
    }*/

    virtual const std::vector<double>& getOutputValues() {
        return outputNode->getOutputValues();
    }

    NodeLayers& getLayers() {
        return layers;
    }

    virtual NetworkOutNode* getOutputNode() const {
        return outputNode.get();
    }

/*    template <typename InputType>
    void setInputs(Event<InputType>* event) {
        unsigned int noHitCnt = 0;
        for(unsigned int inputNode = 0; inputNode < getInputNodes().size(); inputNode++) {
            getInputNodes()[inputNode]->setInput(event->inputs);

            if(getInputNodes()[inputNode]->getOutValueInt() == event->noHitVal)
                noHitCnt++;
        }
        if(getNoHitCntNode() != 0)
	    	getNoHitCntNode()->setInput(noHitCnt);
    }*/

/*
    template <typename InputType>
    void setInputs(Event<InputType>* event, int efficiency, unsigned int minHitCnt,
            std::default_random_engine& rndGenerator); //efficiency in percents
*/

    //noHitCnt should be one of the inputs, the convention is that is the last one,
    //but it depends on the connection of the getNoHitCntNode in the NetworkBuilder to the InputNodes
    template <typename InputType>
    void setInputs(const std::vector<InputType>& inputs) {
        //std::cout<<" "<<inputs.size()<<" getInputNodes().size() "<<getInputNodes().size()<<std::endl;
        for(unsigned int inputNode = 0; inputNode < getInputNodes().size(); inputNode++) {
            auto& inputNodePtr = getInputNodes()[inputNode];
            //std::cout<<" inputNode "<<inputNode<<" inputs.at(inputNode) "<<inputs.at(inputNode)<<std::endl;
            inputNodePtr->setOutValue(inputs.at(inputNode));
            //inputNode->setOutValue(inputs[inputNode]->getNumber());
        }
    }

    InputNode* getNoHitCntNode() {
        return noHitCntNode;
    }

    /**
     * runs calculation for one event, the inputNodes should be set before with desired values
     */
    virtual void run(float eventWeight = 1);

    template <typename InputType>
    const std::vector<double>& run(const std::vector<InputType>& inputs, float eventWeight = 1) {
        setInputs<InputType>(inputs);
        run(eventWeight);
        return getOutputValues();
    }

    template <typename InputType>
    void run(Event<InputType>* event) {
        setInputs<InputType>(event->inputs);
        run(event->weight);
        for(unsigned int iOut = 0;iOut < getOutputValues().size();iOut++) {
            event->nnResult[iOut] = getOutputValues()[iOut];
        }
    }

    /*
     * Calculate mean gradient and update LUTs. learnigRates - learnigRate for every layer
     */

    template <typename InputType>
    void runTraining(const std::vector<InputType>& inputs, unsigned short classLabel, float eventWeight, CostFunction& costFunction);

    template <typename InputType>
    void runTraining(const std::vector<InputType>& inputs, const std::vector<double>& expextedResult, float eventWeight, CostFunction& costFunction);

    template <typename InputType>
    void runTrainingClassLabel(Event<InputType>* event, CostFunction& costFunction) {
        runTraining(event->inputs, event->classLabel, event->weight, costFunction);
    }

    template <typename InputType>
    void runTraining(Event<InputType>* event, CostFunction& costFunction) {
        runTraining(event->inputs, event->expextedResult, event->weight, costFunction);
    }

    /*
     * drops the inputs data with probability of 1-efficiency, efficiency is in %
     */
/*    template <typename InputType>
    void runTraining(Event<InputType>* event, CostFunction& costFunction,
            int efficiency, unsigned int minHitCnt, std::default_random_engine& rndGenerator);*/



    virtual void updateLuts(std::vector<LearnigParams>& learnigParamsVec) = 0;

    void calcualteAdamBiasCorr();

    virtual double getLutChangeAdam(LutNode* lutNode, unsigned int addr, double gradient, double alpha);

    //reset gradients, entries, etc, but no the LUT values
    virtual void reset();

    friend std::ostream & operator << (std::ostream &out, LutNetworkBase & net);

    virtual void print();

    virtual void printLayerStat();

    virtual double getMeanCost() const {
        if(eventCnt == 0)
            return 10;
        return totalCost/eventCnt;
    }

    virtual double getEfficiency() const {
        //std::cout<<"getEfficiency wellClassifiedEvents "<<wellClassifiedEvents<<" eventCnt "<<eventCnt<<std::endl;
        return (double)wellClassifiedEvents/((double)eventCnt);
    }


    virtual double getAverageEfficiency() const {
        return averageEfficiency;
    }
    //LayersConfig should be added before building and using the network
    virtual LayersConfigs& getLayersConf() {
        return layersConf;
    }



private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & (layers);
        ar & (inputNodes);
        ar & (noHitCntNode); //TODO check if it serialized correctly
        ar & (outputNode);
    }

protected:
    NodeLayers layers;

    InputLayer inputNodes;

    //std::unique_ptr<InputNode> noHitCntNode;
    InputNode* noHitCntNode = nullptr; //noHitCntNode is added to the inputNodes, so therefore is not unique_ptr here

    std::unique_ptr<NetworkOutNode> outputNode;

    double totalCost = 0;

    int wellClassifiedEvents = 0;

    double averageEfficiency = 0;

    //double meanCost = 0;

    int eventCnt = 0;

    int notUpdatedNodes = 0;

    LayersConfigs layersConf;

    //OutputType outputType = softMax;

    class LayerStat {
    public:
        double averageAbsDeltaLutValue = 0;
        double averageRelativeDeltaLutValue = 0;
        double meanDeltaAddr = 0;

        double averageAbsDeltaLutInSmooth = 0;
        double averageRelativeDeltaLutInSmooth = 0;
    };

    std::vector<LayerStat> layersStat;

    //std::vector<double> outputValues;

    //virtual void calcualteOutputValues();

    //for Adam optimizer
    double beta1 = 0.8;
    double beta2 = 0.999;

    double beta1ToT = 1.;
    double beta2ToT = 1.;

    double biasCorr0 = 1.;
    double biasCorr1 = 1.;
};


//void softMaxFunction(std::vector<double>& inputValues, std::vector<double>& outputValues);

} /* namespace lutNN */

#endif /* LUTNETWORKBASE_H_ */
