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

	unsigned int batchSize = 200; //TODO
	unsigned int iterations = 50;
	unsigned int printEveryIteration = 1; //TODO

	unsigned int outputsCnt = 10;
	unsigned int branchesPerOutput = 5;

    InputSetterBase* inputSetter = new InputSetterBase();
    LutInterNetwork network(new SoftMax(outputsCnt), inputSetter, false);
	LayersConfigs& layersConf =  network.getLayersConf();

	LayerConfig layerConfig;
	layerConfig.bitsPerNodeInput = 1;
	layerConfig.nodeInputCnt = 6;
	layerConfig.outputBits = 1;
	layerConfig.nodeType = LayerConfig::lutNode;
	//layerConfig.outValOffset; //TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    CostFunctionCrossEntropy  costFunction;


	//layer 0
	//layerConfig.nodesInLayer = outputsCnt * branchesPerOutput * 6 * 6 * 6;
	//layersConf.push_back(make_unique<LayerConfig>(layerConfig) );

	//layer 1
    layerConfig.nodesInLayer = outputsCnt * branchesPerOutput * 6 * 6;
    layerConfig.nodeType = LayerConfig::lutNode; //TODO
    layersConf.push_back(make_unique<LayerConfig>(layerConfig) );

    //layer 2
    layerConfig.nodesInLayer = outputsCnt * branchesPerOutput * 6;
    layerConfig.outValOffset = 1;
    layersConf.push_back(make_unique<LayerConfig>(layerConfig) );

    //layer 3
    layerConfig.nodesInLayer = outputsCnt * branchesPerOutput;
    layerConfig.outputBits = 4;
    layersConf.push_back(make_unique<LayerConfig>(layerConfig) );

    //layer 4
    layerConfig.nodeType = LayerConfig::sumIntNode;
    layerConfig.nodeInputCnt = branchesPerOutput; //layersConf.back()->nodesInLayer;
    layerConfig.nodesInLayer = outputsCnt;
    layersConf.push_back(make_unique<LayerConfig>(layerConfig) );


    std::default_random_engine rndGenerator(11);

    string mnistDir = "/afs/cern.ch/work/k/kbunkow/private/lutNN/mnist/";

    EventsGeneratorMnist eventsGeneratorTrain(mnistDir + "train-images-idx3-ubyte", mnistDir + "train-labels-idx1-ubyte", rndGenerator);
    eventsGeneratorTrain.generateEvents(eventsGeneratorTrain.gelPixelCnt(), outputsCnt, 0, 0);

    EventsGeneratorMnist eventsGeneratorTest(mnistDir + "t10k-images-idx3-ubyte", mnistDir + "t10k-labels-idx1-ubyte", rndGenerator);
    eventsGeneratorTest.generateEvents(eventsGeneratorTrain.gelPixelCnt(), outputsCnt);


    unique_ptr<InputNodeFactory> inputNodeFactory = make_unique<InputNodeBinaryFactory>(eventsGeneratorTrain.gelPixelCnt(), &rndGenerator);

    NetworkBuilder networkBuilder(inputNodeFactory.get());
    //networkBuilder.buildTree2(layersConf, network); <<<!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!111

    for(unsigned int iLayer = 0; iLayer < network.getLayers().size(); iLayer++ ) {
        cout<<"iLayer "<<iLayer<<" nodes "<<network.getLayers()[iLayer].size()<<endl;
    }

    unsigned int maxNodesPerLayer = 5;
    LutNetworkPrint lutNetworkPrint;
    TCanvas* canvasLutNN = lutNetworkPrint.createCanvasLutsAndOutMap(layersConf.size(), maxNodesPerLayer, 2);


    //cout<<network;


    vector<EventInt*>& testEvents = eventsGeneratorTest.getEvents(); //TODO look for the number
    vector<EventInt*>& validationEvents = eventsGeneratorTest.getEvents(); //TODO take real set of validation events

    for(unsigned int i = 0; i < 3; i++)
        eventsGeneratorTrain.printImage(i);

    for(unsigned int i = 0; i < 3; i++) {
        cout<<" event "<<i<<endl;
        EventInt* event = eventsGeneratorTrain.getEvents()[i];
        eventsGeneratorTrain.printEvent(event, 28, 28);
    }

    //runTrainingNaiveBayes
    for(unsigned int iLayer = 0; iLayer < network.getLayers().size() -1; iLayer++ ) { //network.getLayers().size()
        {
            //cout<<"line "<<dec<<__LINE__<<endl;
            //auto_cpu_timer timer1;
            for(unsigned int i = 0; i < 10000; i++) {
                //cout<<" event "<<i<<endl;
                EventInt* event = eventsGeneratorTrain.getEvents()[i];

                network.run(event);
                //network.runTrainingNaiveBayes(event->expextedResult);
            }
            //cout<<"line "<<dec<<__LINE__;
        }
        //cout<<"line "<<__LINE__ << timer.format() << '\n'; timer.stop(); timer.start();

        {
            //auto_cpu_timer timer1;
            //network.updateLutsNaiveBayes(iLayer);

            //cout<<"line "<<dec<<__LINE__;
        }
        //cout<<"line "<<__LINE__ << timer.format() << '\n'; timer.stop(); timer.start();

        lutNetworkPrint.printLuts2(network, maxNodesPerLayer);

        //cout<<"line "<<__LINE__ << timer.format() << '\n'; timer.stop(); timer.start();

        network.reset();

        //cout<<"line "<<__LINE__ << timer.format() << '\n'; timer.stop(); timer.start();

        {
            //auto_cpu_timer timer1;
            for(auto& event : validationEvents) {
                network.run(event);
            }
            //cout<<"line "<<dec<<__LINE__;
        }

        //cout<<"line "<<__LINE__ << timer.format() << '\n'; timer.stop(); timer.start();
        {
            //auto_cpu_timer timer1;
            lutNetworkPrint.printExpectedVersusNNOutHotOne(validationEvents, costFunction);
            //cout<<"line "<<dec<<__LINE__;
        }
        //cout<<"line "<<__LINE__ << timer.format() << '\n'; timer.stop(); timer.start();

        canvasLutNN->SaveAs( ("../pictures/canvasLutNN_BayesTrain_" + std::to_string(iLayer)+ ".png").c_str());
        //cout<<"line "<<dec<<__LINE__<<endl;
    }

    //outfile.Write();

    vector<EventInt*> trainingEventsBatch(batchSize);

    lutNetworkPrint.createCostHistory(iterations, printEveryIteration);

    network.reset();

    //eventsGeneratorTrain.getNextMiniBatch(trainingEventsBatch); //todo remove
    std::string canvasCostPng = "../pictures/canvasLutNN_canvasCostHist.png";

    LearnigParams learnigParams;
    learnigParams.learnigRate = 0.002 ;
    std::vector<LearnigParams> learnigParamsVec(layersConf.size(), learnigParams);

    for(unsigned int i = 0; i < iterations; i++) {
        //cout<<"line "<<dec<<__LINE__<<endl;
        //auto_cpu_timer timer1;
        eventsGeneratorTrain.getNextMiniBatch(trainingEventsBatch); //TODO !!!!!!!!!!!!!!!!!!!
        for(unsigned int iEv = 0; iEv < trainingEventsBatch.size(); iEv++) {
            network.runTraining(trainingEventsBatch[iEv], costFunction);
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
