/*
 * LutNetwork.h
 *
 *  Created on: May 1, 2018
 *      Author: Karol Bunkowski
 */

#ifndef LUTNETWORK_H_
#define LUTNETWORK_H_

#include "lutNN/interface/Node.h"
#include "lutNN/interface/InputNodeFactory.h"
#include <vector>
#include <memory>
#include <iostream>
#include <random>

namespace lutNN {

class Event {
public:
    Event(unsigned int inputCnt, unsigned int ouptutCnt): inputs(inputCnt, 0), expextedResult(ouptutCnt, 0), nnResult(ouptutCnt, 0) {

    }
    std::vector<int> inputs;
    std::vector<double> expextedResult;

    std::vector<double> nnResult;

    void print();

    unsigned int number = 0;
};

class CostFunction {
public:
    virtual ~CostFunction() {};

    virtual double get(std::vector<double> expextedResults, std::vector<double> nnResults)  = 0;

    virtual double operator() (std::vector<double> expextedResults, std::vector<double> nnResults) {
        return get(expextedResults, nnResults);
    }

    virtual double derivative(std::vector<double> expextedResults, std::vector<double> nnResults, unsigned int outNum)  = 0;
};

class CostFunctionMeanSquaredError: public  CostFunction {
public:
    virtual ~CostFunctionMeanSquaredError() {};

    virtual double get(std::vector<double> expextedResults, std::vector<double> nnResults);

    virtual double derivative(std::vector<double> expextedResults, std::vector<double> nnResults, unsigned int outNum) {
        //return -2. * (expextedResults[outNum ] - nnResults[outNum] );
        return 2. * (nnResults[outNum] - expextedResults[outNum ]);
    }
};

class CostFunctionCrossEntropy: public  CostFunction {
public:
    virtual ~CostFunctionCrossEntropy() {};

    virtual double get(std::vector<double> expextedResults, std::vector<double> nnResults);

    //if the output layer is softmax!!!!
    virtual double derivative(std::vector<double> expextedResults, std::vector<double> nnResults, unsigned int outNum) {
        return nnResults[outNum] - expextedResults[outNum ];
    }
};

class LutNetwork {
public:
    enum OutputType {
        simple,
        softMax
    };

    LutNetwork(std::vector<ConfigParametersPtr>& layersDef, OutputType outputType, InputNodeFactory* inputNodeFactory = new InputNodeFactoryBase());
    virtual ~LutNetwork();

    InputLayer& getInputNodes() {
        return inputNodes;
    }

/*    const NeuronLayer& getOutputLayer() const {
        return neuronLayers.back();
    }*/

    const std::vector<double>& getOutputValues() {
        return outputValues;
    }

    NeuronLayersVec& getLayers() {
        return layers;
    }

    void connectOneNetPerOut();

    void connect();

    void connect1();

    void initLuts();

    void initLuts(std::default_random_engine& generator);

    virtual void setInputs(Event* event) {
        for(unsigned int inputNode = 0; inputNode < getInputNodes().size(); inputNode++) {
            getInputNodes()[inputNode]->setInput(event->inputs);
        }
    }

    /**
     * runs calculation for one event, the inputNodes should be set before with desired values
     */
    virtual void run();

    virtual void run(Event* event) {
        setInputs(event);
        run();
        for(unsigned int iOut = 0; iOut < outputValues.size(); iOut++) {
            event->nnResult[iOut] = outputValues[iOut];
        }
    }

    virtual void runFromLayer(unsigned int fromLayer);

    void runFromNeuron(NeuronNode* neuronNod);

    /**
     * runs gradient calculation for one event, the inputNodes should be set before with desired values
     * @param mode if mode = 1 only the network final outputs are trained
     */
    virtual void runTraining(std::vector<double> expextedResults, CostFunction& costFunction);

    virtual void runTrainingInter(std::vector<double> expextedResults, CostFunction& costFunction);
    /*
     * Calculate mean gradient and update LUTs. learnigRates - learnigRate for every layer
     */
    struct LearnigParams {
        double learnigRate = 0;
        double beta = 0; //momentum rate
        double lambda = 0; //l2 regularization rate
        double stretchRate = 0;
        //double regularizationRate = 0;
        double smoothWeight = 0;
        float maxLutVal = 1;
    };

    virtual void updateLuts(std::vector<LearnigParams> learnigParamsVec);

    virtual void updateLutsInter(std::vector<LearnigParams> learnigParamsVec);

    virtual void smoothLuts(std::vector<LearnigParams> learnigParamsVec);

    virtual void smoothLuts1(std::vector<LearnigParams> learnigParamsVec);

    void weightByEvents(unsigned int iLayer);

    void shiftAndRescale(unsigned int iLayer, double shiftRatio, double scaleRatio);

    virtual void resetStats();

    //to be used during training, the Lut::outWeight is set to 1 with probability retainProb, otherwise to 0
    virtual void setDropOut(double retainProb, std::default_random_engine& generator);

    //to be used during testing, Lut::outWeight is set to the given lutOutWeight, lutOutWeight should be the same as retainProb set in setDropOut
    virtual void setLutOutWeight(double lutOutWeight);


    friend std::ostream & operator << (std::ostream &out, LutNetwork & net);

    void printLayerStat();

    double getMeanCost() const {
        return totalCost/eventCnt;
    }

private:
    NeuronLayersVec layers;

    InputLayer inputNodes;

    double totalCost = 0;

    //double meanCost = 0;

    int eventCnt = 0;

    int notUpdatedNodes = 0;

    std::vector<ConfigParametersPtr > layersDef;

    OutputType outputType = softMax;

    class LayerStat {
    public:
        double averageAbsDeltaLutValue = 0;
        double averageRelativeDeltaLutValue = 0;
        double meanDeltaAddr = 0;

        double averageAbsDeltaLutInSmooth = 0;
        double averageRelativeDeltaLutInSmooth = 0;
    };

    std::vector<LayerStat> layersStat;

    std::vector<double> outputValues;

    void calcualteOutputValues();

    std::unique_ptr<InputNodeFactory> inputNodeFactory;
};

} /* namespace lutNN */

#endif /* LUTNETWORK_H_ */
