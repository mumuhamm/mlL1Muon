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

	unsigned int batchSize = 200; //TODO
	unsigned int iterations = 30000;
	unsigned int printEveryIteration = 200; //TODO

	iterations++;

	unsigned int outputsCnt = 10;
	unsigned int branchesPerOutput = 10;

    //NetworkOutNode* outputNode = new SoftMax(outputsCnt); //todo use uniq_ptr
    NetworkOutNode* outputNode = new NetworkOutNode(outputsCnt);
    BinaryLutNetwork network(outputNode, &rndGenerator);

	LayersConfigs& layersConf =  network.getLayersConf();

	LayerConfig layerConfig;
	layerConfig.bitsPerNodeInput = 1;
	layerConfig.nodeInputCnt = 6;
	layerConfig.outputBits = 1;
	layerConfig.maxLutVal = 1.;
	layerConfig.nodeType = LayerConfig::lutBinary;

    //CostFunctionCrossEntropy  costFunction;
    CostFunctionHingeLost  costFunction;
    //CostFunctionAbsoluteError  costFunction;
    //CostFunctionMeanSquaredError costFunction;

/*	//layer 0
	layerConfig.nodesInLayer = outputsCnt * branchesPerOutput * 5 * 5 * 5;
	layerConfig.bitsPerNodeInput = 1;
	layerConfig.nodeInputCnt = 5;
	layerConfig.propagateGradient = false;
	layersConf.push_back(make_unique<LayerConfig>(layerConfig) );*/

	//layer 1
    layerConfig.nodesInLayer = outputsCnt * branchesPerOutput * 6 * 6;// /10;
	layerConfig.bitsPerNodeInput = 1;
	layerConfig.nodeInputCnt = 6;
	layerConfig.propagateGradient = false;   //TODO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    layersConf.push_back(make_unique<LayerConfig>(layerConfig) );

    //layer 2
	layerConfig.bitsPerNodeInput = 1;
	layerConfig.nodeInputCnt = 6;
    layerConfig.nodesInLayer = outputsCnt * branchesPerOutput * 6;
    layerConfig.propagateGradient = true;
    layersConf.push_back(make_unique<LayerConfig>(layerConfig) );

    //layer 3
    layerConfig.nodeType = LayerConfig::lutNode;
    layerConfig.nodesInLayer = outputsCnt * branchesPerOutput;
    layerConfig.nodeInputCnt = 6;
    layerConfig.outputBits = 3;
    layerConfig.maxLutVal = (1 << layerConfig.outputBits) -1 -1;
    layerConfig.propagateGradient = true;
    layersConf.push_back(make_unique<LayerConfig>(layerConfig) );

    //layer 4

    layerConfig.nodeType = LayerConfig::sumNode;
    layerConfig.subClassesCnt = 1;
    layerConfig.nodeInputCnt = branchesPerOutput / layerConfig.subClassesCnt; //layersConf.back()->nodesInLayer;
    layerConfig.classesCnt = outputsCnt;
    layerConfig.nodesInLayer = outputsCnt * layerConfig.subClassesCnt;
    layerConfig.propagateGradient = true;
    layersConf.push_back(make_unique<LayerConfig>(layerConfig) );

    for(unsigned int iLayer = 0; iLayer < layersConf.size(); iLayer++) {
        cout<<"alyer "<<iLayer<<" layersConf maxLutVal "<<layersConf[iLayer]->maxLutVal<<std::endl;
    }

    string mnistDir = "/afs/cern.ch/work/k/kbunkow/private/lutNN/mnist/";

    EventsGeneratorMnist<EventInt> eventsGeneratorTrain(mnistDir + "train-images-idx3-ubyte", mnistDir + "train-labels-idx1-ubyte", rndGenerator);
    eventsGeneratorTrain.generateEvents(eventsGeneratorTrain.gelPixelCnt(), outputsCnt, 0, 0);

    EventsGeneratorMnist<EventInt> eventsGeneratorTest(mnistDir + "t10k-images-idx3-ubyte", mnistDir + "t10k-labels-idx1-ubyte", rndGenerator);
    eventsGeneratorTest.generateEvents(eventsGeneratorTrain.gelPixelCnt(), outputsCnt);

    //TODO here the input nodes are randomly connected to the inputs, there is one, unique input node for each input of the first lutNode layer
    //unsigned int inputNodesCnt = layersConf.at(0)->nodesInLayer * layersConf.at(0)->nodeInputCnt;
    //unique_ptr<InputNodeFactory> inputNodeFactory = make_unique<InputNodeBinaryFactory>(eventsGeneratorTrain.gelPixelCnt(), &rndGenerator);
    //unique_ptr<InputNodeFactory> inputNodeFactory = make_unique<InputNodeSelectedBitsFactory>(eventsGeneratorTrain.gelPixelCnt(), rndGenerator);

    unsigned int inputNodesCnt = eventsGeneratorTrain.gelPixelCnt();
    unique_ptr<InputNodeFactory> inputNodeFactory = make_unique<InputNodeBinaryFactory>(eventsGeneratorTrain.gelPixelCnt(), nullptr); //, rndGenerator


    NetworkBuilder networkBuilder(inputNodeFactory.get());
    networkBuilder.buildTree3(layersConf, inputNodesCnt, network, nullptr); // then should be the &rndGenerator in the inputNodeFactory
    //networkBuilder.buildTree3(layersConf, inputNodesCnt, network, &rndGenerator); //// then should be the nullptr in the inputNodeFactory
    //networkBuilder.buildTree4(layersConf, inputNodesCnt, network, &rndGenerator, eventsGeneratorTrain.getEvents()); //TODO

    //static_cast<SoftMaxWithSubClasses*>(network.getOutputNode())->setMaxInVal(layersConf.at(layersConf.size() -2)->maxLutVal/2 * branchesPerOutput); //TODO !!!!!!!!!!!!!!!!!!!

    //network.getSoftMaxNode()->setMaxInVal(layersConf.at(layersConf.size() -2)->maxLutVal/2 * branchesPerOutput);
    //cout<<"network.getSoftMaxNode()->getMaxInVal() "<<network.getSoftMaxNode()->getMaxInVal()<<std::endl;

    network.print();

    int totalNodes = 0;
    for(unsigned int iLayer = 0; iLayer < network.getLayers().size(); iLayer++ ) {
        cout<<"iLayer "<<iLayer<<" nodes "<<network.getLayers()[iLayer].size()<<endl;
        totalNodes += network.getLayers()[iLayer].size();
    }
    cout<<"totalNodes "<<totalNodes<<endl;

    //return 0; //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,,

    unsigned int maxNodesPerLayer = 20;
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

    //outfile.Write();

    network.initLutsRnd(rndGenerator); //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

    vector<EventInt*> trainingEventsBatch(batchSize);

    lutNetworkPrint.createCostHistory(iterations, printEveryIteration);

    network.reset();

    //eventsGeneratorTrain.getNextMiniBatch(trainingEventsBatch); //todo remove
    std::string canvasCostPng = "../pictures/canvasLutNN_canvasCostHist.png";

    LearnigParams learnigParams;
    learnigParams.learnigRate = 0.05 ;
    learnigParams.lambda = 0;//0.005;

    std::vector<LearnigParams> learnigParamsVec(layersConf.size(), learnigParams);

	learnigParamsVec.at(0).learnigRate = 0.005 ; //*2
	learnigParamsVec.at(1).learnigRate = 0.005 ;
	learnigParamsVec.at(2).learnigRate = 0.005 ;

    //learnigParamsVec.at(0).lambda = 0.003 ;
    //learnigParamsVec.at(1).lambda = 0.003 ;
    //learnigParamsVec.at(2).lambda = 0.003 ;

    for(unsigned int iLayer = 0; iLayer < layersConf.size(); iLayer++)
    	layersConf[iLayer]->ditherRate = 0;

    eventsGeneratorTrain.shuffle();
    eventsGeneratorTrain.shuffle();
    eventsGeneratorTrain.shuffle();

    std::normal_distribution<> normalDist(0, 50);
    for(unsigned int i = 0; i < iterations; i++) {
        //cout<<"line "<<dec<<__LINE__<<endl;
        //auto_cpu_timer timer1;
        eventsGeneratorTrain.getNextMiniBatch(trainingEventsBatch); //TODO !!!!!!!!!!!!!!!!!!!

        {
        	auto_cpu_timer timer1;
        	for(unsigned int iEv = 0; iEv < trainingEventsBatch.size(); iEv++) {
        		/*EventInt eventWithNoise = *(trainingEventsBatch[iEv]);
        		for(auto& input : eventWithNoise.inputs) {
        			input += normalDist(rndGenerator);
        		}*/
        		//network.runTraining(&eventWithNoise, costFunction);  //TODO!!!!!!!!!!!!!!!!!!!!
        		//network.LutNetworkBase::runTrainingClassLabel(&eventWithNoise, costFunction);
                network.LutNetworkBase::runTrainingClassLabel(trainingEventsBatch[iEv], costFunction);
        		//network.runTraining(trainingEventsBatch[iEv], costFunction);
        		//cout<<"line "<<dec<<__LINE__<<endl;
        		//network.runTrainingAndUpdate(&eventWithNoise, costFunction, learnigParamsVec);
        		//cout<<"line "<<dec<<__LINE__<<endl;
        	}
        	//cout<<"line "<<__LINE__ << timer1.format() << '\n';
        }

        double trainSampleCost = network.getMeanCost();
        cout<<setw(6)<<dec<<i<<" trainSampleCost "<<trainSampleCost<<std::endl;

        network.updateLuts(learnigParamsVec);

        if(i > 0 && i%500 == 0) {
            learnigParamsVec.at(0).learnigRate *= 0.9;
            learnigParamsVec.at(1).learnigRate *= 0.9;
            learnigParamsVec.at(2).learnigRate *= 0.9;

            cout<<"learnigParamsVec.at(2).learnigRate  "<<learnigParamsVec.at(2).learnigRate <<endl;
        }


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

        	/*for(unsigned int iLayer = 0; iLayer < layersConf.size()-2; iLayer++)
        		layersConf[iLayer]->ditherRate = 0;*/

        	layersConf[layersConf.size()-2]->ditherRate = 0;//50; //this is in percent!!!!!
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
