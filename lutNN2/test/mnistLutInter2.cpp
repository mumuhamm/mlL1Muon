//============================================================================
// Name        : mnistLutInter1.cpp
// Author      : Karol Bunkowski
// Version     :
// Copyright   : All right reserved
// Description : Hello World in C, Ansi-style
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <time.h>
#include <random>
#include <boost/timer/timer.hpp>

#include "TCanvas.h"
#include "TFile.h"

#include "lutNN/lutNN2/interface/LutInterNetwork.h"
//#include "lutNN/lutNN2/interface/BinaryLutNetwork.h"
#include "lutNN/lutNN2/interface/NetworkBuilder.h"

#include "lutNN/lutNN2/interface/MNISTParser.h"
#include "lutNN/lutNN2/interface/EventsGeneratorMnist.h"
#include "lutNN/lutNN2/interface/Utlis.h"


using namespace lutNN;
using namespace std;
using namespace boost::timer;

int main(void) {
	puts("Hello World!!!");

	unsigned int batchSize = 300; //TODO
	unsigned int iterations = 20000;
	unsigned int printEveryIteration = 100; //TODO

	unsigned int inputCnt = 28 * 28;
	unsigned int outputsCnt = 10;
	//unsigned int branchesPerOutput = 5;

	//unsigned int maxInputVal = 255;

    const int input_I = 2;
    const int input_F = 6;
    const std::size_t networkInputSize = 18;

    const int layer1_neurons = 16;
    const int layer1_lut_I = 3;
    const int layer1_lut_F = 10;

    const int layer1_output_I = 5;
    const int layer2_input_I = 4; //layer1_output_I;

    //const int layer2_neurons = 8 + 1 + 1;
    const int layer2_lut_I = 5;
    const int layer2_lut_F = 10;

    const int layer3_input_I = 7;
    //const int layer3_neurons = 1;
    const int layer3_lut_I = 6;
    const int layer3_lut_F = 10;

	NetworkOutNode* outputNode = new SoftMax(outputsCnt); //todo use uniq_ptr
    LutInterNetwork network(outputNode, false);
	LayersConfigs& layersConf =  network.getLayersConf();

	LayerConfig lutLayerConfig;
	lutLayerConfig.bitsPerNodeInput = 1;
	lutLayerConfig.nodeInputCnt = 6;
	lutLayerConfig.outputBits = 1;
	lutLayerConfig.nodeType = LayerConfig::lutInter;
	//lutLayerConfig.outValOffset; //TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	lutLayerConfig.lutRangesCnt = 1;

    LayerConfig sumLayerConfig;
    sumLayerConfig.bitsPerNodeInput = 10; //does not matter, its flaot
    sumLayerConfig.nodeInputCnt = 5;
    sumLayerConfig.outputBits = 10; //does not matter, its flaot
    sumLayerConfig.nodeType = LayerConfig::sumNode;
    sumLayerConfig.outValOffset = 0;

    CostFunctionCrossEntropy  costFunction;

    unsigned int lutSize = 0;

    //layer 0
    unsigned int neurons = 16;
    lutLayerConfig.bitsPerNodeInput = input_I;
    lutLayerConfig.nodesInLayer = inputCnt * neurons;
    lutLayerConfig.nodeInputCnt = 1;
    lutLayerConfig.lutRangesCnt = 1;
    lutLayerConfig.interpolate = true;
    //lutLayerConfig.lutBinsPerRange = (1<<lutLayerConfig.bitsPerNodeInput) / lutLayerConfig.lutRangesCnt;
    lutLayerConfig.maxLutVal =  0.02;
    lutLayerConfig.minLutVal = -lutLayerConfig.maxLutVal;
    lutLayerConfig.middleLutVal = (lutLayerConfig.maxLutVal + lutLayerConfig.minLutVal)/2;
    lutLayerConfig.propagateGradient = false;   //TODO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    lutLayerConfig.maxLutValChange = lutLayerConfig.maxLutVal/5.;

    lutSize = 1<<lutLayerConfig.bitsPerNodeInput;
    lutLayerConfig.initSlopeMax = (lutLayerConfig.maxLutVal - lutLayerConfig.minLutVal) / (lutSize/lutLayerConfig.lutRangesCnt) / 1.;
    lutLayerConfig.initSlopeMin = lutLayerConfig.initSlopeMax * 0.1;
    layersConf.push_back(make_unique<LayerConfig>(lutLayerConfig) );
    //layer 1
    sumLayerConfig.nodesInLayer = neurons;
    sumLayerConfig.nodeInputCnt = inputCnt;
    sumLayerConfig.outValOffset = 1 << (layer2_input_I-1);
    sumLayerConfig.biasShift    = 0; //
    sumLayerConfig.shiftLastGradient = false;
    layersConf.push_back(make_unique<LayerConfig>(sumLayerConfig) );


    //layer 2
    unsigned int prevLayerNeurons = neurons;
    neurons = outputsCnt;
    lutLayerConfig.bitsPerNodeInput = layer2_input_I;
    lutLayerConfig.nodesInLayer = prevLayerNeurons * neurons;
    lutLayerConfig.nodeInputCnt = 1;
    lutLayerConfig.lutRangesCnt = 1;
    lutLayerConfig.interpolate = true;
    //lutLayerConfig.lutBinsPerRange = (1<<lutLayerConfig.bitsPerNodeInput) / lutLayerConfig.lutRangesCnt;
    lutLayerConfig.maxLutVal =  1<<(layer2_lut_I-1);
    lutLayerConfig.minLutVal = -lutLayerConfig.maxLutVal;
    lutLayerConfig.middleLutVal = (lutLayerConfig.maxLutVal + lutLayerConfig.minLutVal)/2;
    lutLayerConfig.propagateGradient = true;   //TODO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    lutLayerConfig.maxLutValChange = lutLayerConfig.maxLutVal/5.;

    lutSize = 1<<lutLayerConfig.bitsPerNodeInput;
    lutLayerConfig.initSlopeMax = (lutLayerConfig.maxLutVal - lutLayerConfig.minLutVal) / (lutSize/lutLayerConfig.lutRangesCnt) / 2.;
    lutLayerConfig.initSlopeMin = lutLayerConfig.initSlopeMax * 0.1;
    layersConf.push_back(make_unique<LayerConfig>(lutLayerConfig) );
    //layer 3
    sumLayerConfig.nodesInLayer = neurons;
    sumLayerConfig.nodeInputCnt = prevLayerNeurons;
    sumLayerConfig.outValOffset = 1 << (layer3_input_I-1); //should be next layer bitsPerNodeInput
    sumLayerConfig.biasShift    = 0;
    sumLayerConfig.shiftLastGradient = false;

    layersConf.push_back(make_unique<LayerConfig>(sumLayerConfig) );

    //layer 4
/*    prevLayerNeurons = neurons;
    neurons = outputsCnt;
    lutLayerConfig.bitsPerNodeInput = layer3_input_I;
    lutLayerConfig.nodesInLayer = neurons;
    lutLayerConfig.nodeInputCnt = 1;
    lutLayerConfig.lutRangesCnt = 1;
    lutLayerConfig.interpolate = true;
    lutLayerConfig.maxLutVal = 2;
    lutLayerConfig.minLutVal = -lutLayerConfig.maxLutVal;
    lutLayerConfig.middleLutVal = (lutLayerConfig.maxLutVal + lutLayerConfig.minLutVal)/2;
    lutLayerConfig.initSlopeMax = (lutLayerConfig.maxLutVal - lutLayerConfig.minLutVal) / (lutSize/lutLayerConfig.lutRangesCnt) / 4.;
    lutLayerConfig.initSlopeMin = lutLayerConfig.initSlopeMax * 0.1;
lutLayerConfig.propagateGradient = true;   //TODO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    layersConf.push_back(make_unique<LayerConfig>(lutLayerConfig) );*/


    unique_ptr<InputNodeFactory> inputNodeFactory = make_unique<InputNodeFactoryBase>();

    cout<<"building network "<<endl;
    NetworkBuilder networkBuilder(inputNodeFactory.get());
    networkBuilder.buildNet(layersConf, inputCnt, network);

    networkBuilder.connectNet(layersConf, network);

    cout<<"\nlayersConf: "<<endl;
    for(unsigned int iLayer = 0; iLayer < network.getLayers().size(); iLayer++ ) {
        cout<<"iLayer "<<iLayer<<" nodeCnt "<<network.getLayers()[iLayer].size()<<" nodeType "<<LayerConfig::nodeTypeToStr(layersConf[iLayer]->nodeType)
        <<" outValOffset "<<layersConf[iLayer]->outValOffset
        <<" lutRangesCnt "<<layersConf[iLayer]->lutRangesCnt
        <<" lutBinsPerRange "<<layersConf[iLayer]->lutBinsPerRange
        <<" maxLutVal "<<layersConf[iLayer]->maxLutVal<<endl;
    }

    cout<<network<<endl;
    int* a;
    *a = 0;

    std::default_random_engine rndGenerator(11);

    network.initLuts2(rndGenerator);

    string mnistDir = "/afs/cern.ch/work/k/kbunkow/private/lutNN/mnist/";

    EventsGeneratorMnist<EventFloat> eventsGeneratorTrain(mnistDir + "train-images-idx3-ubyte", mnistDir + "train-labels-idx1-ubyte", rndGenerator);
    eventsGeneratorTrain.generateEvents(eventsGeneratorTrain.gelPixelCnt(), outputsCnt, 0, 0, 1./ (1<<input_F));

    EventsGeneratorMnist<EventFloat> eventsGeneratorTest(mnistDir + "t10k-images-idx3-ubyte", mnistDir + "t10k-labels-idx1-ubyte", rndGenerator);
    eventsGeneratorTest.generateEvents(eventsGeneratorTrain.gelPixelCnt(), outputsCnt, 0, 0, 1./ (1<<input_F));

    if( (inputCnt) != eventsGeneratorTrain.getEvents().front()->inputs.size()) {
        throw std::runtime_error("wrong inputCnt");
    }

    for(unsigned int iLayer = 0; iLayer < network.getLayers().size(); iLayer++ ) {
        cout<<"iLayer "<<iLayer<<" nodes "<<network.getLayers()[iLayer].size()<<endl;
    }

    unsigned int maxNodesPerLayer = 5;
    LutNetworkPrint lutNetworkPrint;
    TCanvas* canvasLutNN = lutNetworkPrint.createCanvasLutsAndOutMap(layersConf.size(), maxNodesPerLayer, 2);


    //cout<<network;


    vector<EventFloat*>& testEvents = eventsGeneratorTest.getEvents(); //TODO look for the number
    vector<EventFloat*>& validationEvents = eventsGeneratorTest.getEvents(); //TODO take real set of validation events

    for(unsigned int i = 0; i < 3; i++)
        eventsGeneratorTrain.printImage(i);

    for(unsigned int i = 0; i < 3; i++) {
        cout<<" event "<<i<<endl;
        EventFloat* event = eventsGeneratorTrain.getEvents()[i];
        eventsGeneratorTrain.printEvent(event, 28, 28);
    }

    //outfile.Write();

    vector<EventFloat*> trainingEventsBatch(batchSize);

    lutNetworkPrint.createCostHistory(iterations, printEveryIteration);

    network.reset();

    //eventsGeneratorTrain.getNextMiniBatch(trainingEventsBatch); //todo remove
    std::string canvasCostPng = "../pictures/canvasLutNN_canvasCostHist.png";

    LearnigParams learnigParams;
    learnigParams.learnigRate = 0.002;//0.1/16. * 40; //200 ;
    learnigParams.lambda = 0;//0.005;
    learnigParams.beta = 0; //0.8;
    learnigParams.smoothWeight = 0;//0.01;

    std::vector<LearnigParams> learnigParamsVec(layersConf.size(), learnigParams);

    learnigParamsVec.at(2).learnigRate *= 2.;

    eventsGeneratorTrain.shuffle();
    eventsGeneratorTrain.shuffle();
    eventsGeneratorTrain.shuffle();

    for(unsigned int i = 0; i < iterations; i++) {
        //cout<<"line "<<dec<<__LINE__<<endl;
        //auto_cpu_timer timer1;
        eventsGeneratorTrain.getNextMiniBatch(trainingEventsBatch); //TODO !!!!!!!!!!!!!!!!!!!
        for(unsigned int iEv = 0; iEv < trainingEventsBatch.size(); iEv++) {
            //network.runTraining(trainingEventsBatch[iEv], costFunction);
            network.LutNetworkBase::runTrainingClassLabel(trainingEventsBatch[iEv], costFunction);
        }

        double trainSampleCost = network.getMeanCost();
        cout<<"trainSampleCost "<<trainSampleCost<<std::endl;

        network.updateLuts(learnigParamsVec);


        double validSampleCost = 0;

        if( (i)%printEveryIteration == 0) {
            for(auto& event : validationEvents) {
                network.run(event);
            }

            lutNetworkPrint.printLuts2(network, maxNodesPerLayer);

            validSampleCost = lutNetworkPrint.printExpectedVersusNNOutHotOne(validationEvents, costFunction);

            lutNetworkPrint.updateCostHistory(i, trainSampleCost, validSampleCost, canvasCostPng);

            canvasLutNN->SaveAs( ("../pictures/canvasLutNN_GradientTrain_" + std::to_string(i)+ ".png").c_str());

            cout<<"validSampleCost "<<validSampleCost<<std::endl<<std::endl;

            learnigParams.learnigRate *= 0.9;
        }

        network.reset();
        //outfile.Write();
        //cout<<"line "<<dec<<__LINE__<<endl;
    }

    TFile outfile("../pictures/lutNN2.root", "RECREATE");
    outfile.cd();
    canvasLutNN->Write();
    outfile.Write();
    cout<<"line "<<dec<<__LINE__<<endl;
    return EXIT_SUCCESS;
}
