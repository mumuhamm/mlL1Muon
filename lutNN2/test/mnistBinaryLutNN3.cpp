//============================================================================
// Name        : lutNN2.cpp
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

//#include "lutNN/lutNN2/interface/LutNetworkBase.h"
#include "lutNN/lutNN2/interface/BinaryLutNetwork.h"
#include "lutNN/lutNN2/interface/NetworkBuilder.h"

#include "lutNN/lutNN2/interface/MNISTParser.h"
#include "lutNN/lutNN2/interface/EventsGeneratorMnist.h"
#include "lutNN/lutNN2/interface/Utlis.h"


using namespace lutNN;
using namespace std;
using namespace boost::timer;

int main(void) {
	puts("Hello World!!!");
    std::default_random_engine rndGenerator(11);
    //std::mt19937_64 rndGenerator(11);

	unsigned int batchSize = 500; //TODO
	unsigned int iterations = 30000;
	unsigned int printEveryIteration = 100; //TODO

	iterations++;

	unsigned int outputsCnt = 10;
	unsigned int branchesPerOutput = 35;

    //softMaxNode = new SoftMaxWithSubClasses(outputsCnt);

    //outputNode.reset(softMaxNode);
    //outputNode.reset(new NetworkOutNode(outputsCnt));
    //outputNode.reset(new SoftMaxWithSubClasses(outputsCnt));
    //outputNode.reset(new SoftMax(outputsCnt));
    NetworkOutNode* outputNode = new SoftMax(outputsCnt); //todo use uniq_ptr
    //NetworkOutNode* outputNode = new NetworkOutNode(outputsCnt); //todo use uniq_ptr

    //remember that in the constructor softMaxNode = new SoftMaxWithSubClasses(layersConf.back()->nodesInLayer)
    BinaryLutNetwork network(outputNode, &rndGenerator);
	LayersConfigs& layersConf =  network.getLayersConf();

	LayerConfig layerConfig;
	layerConfig.bitsPerNodeInput = 1;
	layerConfig.nodeInputCnt = 6;
	layerConfig.outputBits = 1;
	layerConfig.maxLutVal = 1.;
	layerConfig.nodeType = LayerConfig::lutBinary;

    CostFunctionCrossEntropy  costFunction;


/*	//layer 0
	layerConfig.nodesInLayer = outputsCnt * branchesPerOutput * 6 * 6;
	layerConfig.bitsPerNodeInput = 1;
	layerConfig.nodeInputCnt = 6;
	layersConf.push_back(make_unique<LayerConfig>(layerConfig) );*/

	//layer 1
    layerConfig.nodesInLayer = outputsCnt * branchesPerOutput * 7 * 7 / 2;
	layerConfig.bitsPerNodeInput = 1;
	layerConfig.nodeInputCnt = 7;
    layersConf.push_back(make_unique<LayerConfig>(layerConfig) );

    //layer 2
	layerConfig.bitsPerNodeInput = 1;
	layerConfig.nodeInputCnt = 7;
    layerConfig.nodesInLayer = outputsCnt * branchesPerOutput * 7 / 2;
    layersConf.push_back(make_unique<LayerConfig>(layerConfig) );

    //layer 3
    layerConfig.nodesInLayer = outputsCnt * branchesPerOutput;
    layerConfig.nodeInputCnt = 7;
    layerConfig.outputBits = 2;
    layerConfig.maxLutVal = (1 << layerConfig.outputBits) -1 -1;
    layersConf.push_back(make_unique<LayerConfig>(layerConfig) );

    //layer 4
    layerConfig.nodeType = LayerConfig::sumIntNode;
    layerConfig.nodeInputCnt = branchesPerOutput; //layersConf.back()->nodesInLayer;
    layerConfig.nodesInLayer = outputsCnt;
    layersConf.push_back(make_unique<LayerConfig>(layerConfig) );

    string mnistDir = "/afs/cern.ch/work/k/kbunkow/private/lutNN/mnist/";

    {
        EventsGeneratorMnist eventsGeneratorTrain(mnistDir + "train-images-idx3-ubyte", mnistDir + "train-labels-idx1-ubyte", rndGenerator);
        eventsGeneratorTrain.generateEvents(eventsGeneratorTrain.gelPixelCnt(), outputsCnt, 0, 0); //here we don't want stride, therefore we have this in the sceptered block

        unique_ptr<InputNodeFactory> inputNodeFactory = make_unique<InputNodeBinaryFactory>(eventsGeneratorTrain.gelPixelCnt(), nullptr); //, rndGenerator
        //unique_ptr<InputNodeFactory> inputNodeFactory = make_unique<InputNodeSelectedBitsFactory>(eventsGeneratorTrain.gelPixelCnt(), rndGenerator);

    	NetworkBuilder networkBuilder(inputNodeFactory.get());

    	//unsigned int inputNodesCnt = layersConf.at(0)->nodesInLayer * layersConf.at(0)->nodeInputCnt;
    	 unsigned int inputNodesCnt = eventsGeneratorTrain.gelPixelCnt();

    	networkBuilder.buildTree4(layersConf, inputNodesCnt, network, &rndGenerator, eventsGeneratorTrain.getEvents()); //TODO
    }


    SoftMaxWithSubClasses* softMaxNode = dynamic_cast<SoftMaxWithSubClasses*>(network. getOutputNode());
    if(softMaxNode == 0)
        throw std::invalid_argument("NetworkBuilder::buildTree2 getOutputNode is not SoftMaxWithSubClasses" );
    softMaxNode->setMaxInVal(layersConf.at(layersConf.size() -2)->maxLutVal/2 * branchesPerOutput);
    cout<<"network.getSoftMaxNode()->getMaxInVal() "<<softMaxNode->getMaxInVal()<<std::endl;

    network.print();

    int totalNodes = 0;
    for(unsigned int iLayer = 0; iLayer < network.getLayers().size(); iLayer++ ) {
        cout<<"iLayer "<<iLayer<<" nodes "<<network.getLayers()[iLayer].size()<<endl;
        totalNodes += network.getLayers()[iLayer].size();
    }
    cout<<"totalNodes "<<totalNodes<<endl;
	
    unsigned int maxNodesPerLayer = 20;
    LutNetworkPrint lutNetworkPrint;
    TCanvas* canvasLutNN = lutNetworkPrint.createCanvasLutsAndOutMap(layersConf.size(), maxNodesPerLayer, 2);

    EventsGeneratorMnist eventsGeneratorTrain(mnistDir + "train-images-idx3-ubyte", mnistDir + "train-labels-idx1-ubyte", rndGenerator);
    eventsGeneratorTrain.generateEvents(eventsGeneratorTrain.gelPixelCnt(), outputsCnt, 0, 0);

    EventsGeneratorMnist eventsGeneratorTest(mnistDir + "t10k-images-idx3-ubyte", mnistDir + "t10k-labels-idx1-ubyte", rndGenerator);
    eventsGeneratorTest.generateEvents(eventsGeneratorTest.gelPixelCnt(), outputsCnt);

    //cout<<network;
    eventsGeneratorTrain.shuffle();
    eventsGeneratorTrain.shuffle();
    eventsGeneratorTrain.shuffle();

    vector<EventInt*>& testEvents = eventsGeneratorTest.getEvents(); //TODO look for the number
    vector<EventInt*>& validationEvents = eventsGeneratorTest.getEvents(); //TODO take real set of validation events
    //vector<EventInt*> validationEvents(10000);  //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //eventsGeneratorTrain.getNextMiniBatch(validationEvents);

    for(unsigned int i = 0; i < 3; i++)
        eventsGeneratorTrain.printImage(i);

    for(unsigned int i = 0; i < 3; i++) {
        cout<<" event "<<i<<endl;
        EventInt* event = eventsGeneratorTrain.getEvents()[i];
        eventsGeneratorTrain.printEvent(event, 28, 28);
    }

    //runTrainingNaiveBayes
    network.initLutsRnd(rndGenerator);

    //outfile.Write();

    vector<EventInt*> trainingEventsBatch(batchSize);

    lutNetworkPrint.createCostHistory(iterations, printEveryIteration);

    network.reset();

    //eventsGeneratorTrain.getNextMiniBatch(trainingEventsBatch); //todo remove
    std::string canvasCostPng = "../pictures/canvasLutNN_canvasCostHist.png";

    LearnigParams learnigParams;
    learnigParams.learnigRate = 0.05 ;
    learnigParams.lambda = 0.01;
    
    std::vector<LearnigParams> learnigParamsVec(layersConf.size(), learnigParams);

/*	learnigParamsVec.at(0).learnigRate = 0.01 ;
	learnigParamsVec.at(1).learnigRate = 0.01 ;
	learnigParamsVec.at(2).learnigRate = 0.05 ;*/

    for(unsigned int iLayer = 0; iLayer < layersConf.size(); iLayer++)
    	layersConf[iLayer]->ditherRate = 0;

    std::normal_distribution<> normalDist(0, 35);
    for(unsigned int i = 0; i < iterations; i++) {
        //cout<<"line "<<dec<<__LINE__<<endl;
        //auto_cpu_timer timer1;
        eventsGeneratorTrain.getNextMiniBatch(trainingEventsBatch); //TODO !!!!!!!!!!!!!!!!!!!

        {
        	auto_cpu_timer timer1;
        	for(unsigned int iEv = 0; iEv < trainingEventsBatch.size(); iEv++) {
        		EventInt eventWithNoise = *(trainingEventsBatch[iEv]);
        		for(auto& input : eventWithNoise.inputs) {
        			input += normalDist(rndGenerator);
        		}
        		network.runTraining(&eventWithNoise, costFunction);
        	}
        	//cout<<"line "<<__LINE__ << timer1.format() << '\n';
        }

        double trainSampleCost = network.getMeanCost();
        cout<<setw(6)<<i<<" trainSampleCost "<<trainSampleCost<<std::endl;

        network.updateLuts(learnigParamsVec);

        if(i == 1000) {
        	learnigParamsVec.at(0).learnigRate = 0.05 ;
        	learnigParamsVec.at(1).learnigRate = 0.05 ;
        	learnigParamsVec.at(2).learnigRate = 0.05 ;
        }
        else if(i == 2000) {
        	learnigParamsVec.at(0).learnigRate = 0.04 ;
        	learnigParamsVec.at(1).learnigRate = 0.04 ;
        	learnigParamsVec.at(2).learnigRate = 0.04 ;
        }
        else if(i == 5000) {
        	learnigParamsVec.at(0).learnigRate = 0.03 ;
        	learnigParamsVec.at(1).learnigRate = 0.03 ;
        	learnigParamsVec.at(2).learnigRate = 0.03 ;
        }
        else  if(i == 7000) {
        	learnigParamsVec.at(0).learnigRate = 0.02;
        	learnigParamsVec.at(1).learnigRate = 0.02;
        	learnigParamsVec.at(2).learnigRate = 0.02;
        }
        else  if(i == 10000) {
        	learnigParamsVec.at(0).learnigRate = 0.01;
        	learnigParamsVec.at(1).learnigRate = 0.01;
        	learnigParamsVec.at(2).learnigRate = 0.01;
        }
        else  if(i == 15000) {
        	learnigParamsVec.at(0).learnigRate = 0.007;
        	learnigParamsVec.at(1).learnigRate = 0.007;
        	learnigParamsVec.at(2).learnigRate = 0.007;
        }

        /*if( i%20 == 1) //todo
            network.dither(learnigParamsVec, rndGenerator);*/

        double validSampleCost = 0;

        if( (i)%printEveryIteration == 0) {
        	network.reset();
            for(unsigned int iLayer = 0; iLayer < layersConf.size(); iLayer++) {
            	layersConf[iLayer]->ditherRate = 0;
            }

        	{
        		auto_cpu_timer timer1;
        		for(auto& event : validationEvents) {
        			network.LutNetworkBase::run(event);
        		}
        	}

        	for(unsigned int iLayer = 0; iLayer < layersConf.size(); iLayer++)
        		layersConf[iLayer]->ditherRate = 0.0;
            //cout<<"line "<<__LINE__ << timer1.format() << '\n';

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
