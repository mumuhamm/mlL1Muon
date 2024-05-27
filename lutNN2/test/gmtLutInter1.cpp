//============================================================================
// Name        : omtfLutNN2.cpp
// Author      : Karol Bunkowski
// Version     :
// Copyright   : All right reserved
// Description : lutNN trainer for the omtf
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
#include "lutNN/lutNN2/interface/LutNetworkBase.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "TCanvas.h"
#include "TFile.h"
#include "TH2F.h"

#include "lutNN/lutNN2/interface/LutInterNetwork.h"
#include "lutNN/lutNN2/interface/NetworkBuilder.h"
#include "lutNN/lutNN2/interface/LutInter.h"
#include "lutNN/lutNN2/interface/NetworkOutNode.h"

#include "lutNN/lutNN2/interface/EventsGeneratorGmt.h"
#include "lutNN/lutNN2/interface/Utlis.h"
#include "lutNN/lutNN2/interface/NetworkSerialization.h"
#include "lutNN/lutNN2/interface/GmtAnalyzer.h"

using namespace lutNN;
using namespace std;
using namespace boost::timer;

int main(void) {
	puts("Hello World!!!");

    std::default_random_engine rndGenerator(11);
    
    unsigned int batchSize = 1000; //TODO
	unsigned int iterations = 20000;
	unsigned int printEveryIteration = 1000; //TODO
	
	bool useHitsValid = false;
	bool addNoHitCntToInputs = true;

	iterations++;
	
	unsigned int outputsCnt = 3;

    CostFunctionCrossEntropy  costFunction;

    NetworkOutNode* outputNode = new SoftMax(outputsCnt); //todo use uniq_ptr
    LutInterNetwork network(outputNode, addNoHitCntToInputs);

	LayersConfigs& layersConf = network.getLayersConf();

	unsigned int inputCnt = 28;

	unsigned int lutLayerCnt = 3; //TODO           for printing

	LayerConfig lutLayerConfig;
	lutLayerConfig.bitsPerNodeInput = 10;
	lutLayerConfig.nodeInputCnt = 1;
	lutLayerConfig.outputBits = 10; //does not matter, its float
	lutLayerConfig.nodeType = LayerConfig::lutInter;
	lutLayerConfig.lutRangesCnt = 8;

    LayerConfig neuronLayerConfig;
    neuronLayerConfig.bitsPerNodeInput = 10; //does not matter, its float
    neuronLayerConfig.nodeInputCnt = 5;
    neuronLayerConfig.outputBits = 10; //does not matter, its float
    neuronLayerConfig.nodeType = LayerConfig::sumNode;
    neuronLayerConfig.outValOffset = 0;

	//layer 0
    unsigned int neurons = 7;
    lutLayerConfig.nodeType = LayerConfig::lutInter; //TODO  make it  working with the lutNode - now LutInterNetwork::updateLuts works only with the lutInter;
    lutLayerConfig.nodesInLayer = inputCnt * neurons;
    lutLayerConfig.nodeInputCnt = 1;
    lutLayerConfig.lutRangesCnt = 1;
    lutLayerConfig.bitsPerNodeInput = 6;
    lutLayerConfig.maxLutVal =  (1<<lutLayerConfig.bitsPerNodeInput) / (float)inputCnt/2. * 60./8;  //
    lutLayerConfig.minLutVal = -lutLayerConfig.maxLutVal;
    lutLayerConfig.middleLutVal = (lutLayerConfig.maxLutVal + lutLayerConfig.minLutVal)/2;
	layersConf.push_back(make_unique<LayerConfig>(lutLayerConfig) );
	//layer 1
	neuronLayerConfig.nodesInLayer = neurons;
	neuronLayerConfig.nodeInputCnt = inputCnt;
	neuronLayerConfig.outValOffset = 0; //is set later!!!!!
	neuronLayerConfig.biasShift    = 5; //change together with lutLayerConfig.bitsPerNodeInput of the layer 2, should be 4 bits less
	layersConf.push_back(make_unique<LayerConfig>(neuronLayerConfig) );


    //layer 2
	unsigned int prevLayerNeurons = neurons;
    neurons = outputsCnt;
    lutLayerConfig.nodeType = LayerConfig::lutInter;
    lutLayerConfig.nodesInLayer = prevLayerNeurons * neurons;
    lutLayerConfig.nodeInputCnt = 1;
    lutLayerConfig.lutRangesCnt = 16;
    lutLayerConfig.bitsPerNodeInput = 9;
    lutLayerConfig.maxLutVal =  (1<<lutLayerConfig.bitsPerNodeInput) / (float)lutLayerConfig.nodesInLayer;
    lutLayerConfig.minLutVal = -lutLayerConfig.maxLutVal;
    lutLayerConfig.middleLutVal = (lutLayerConfig.maxLutVal + lutLayerConfig.minLutVal)/2;
    layersConf.push_back(make_unique<LayerConfig>(lutLayerConfig) );
    //layer 3
    neuronLayerConfig.nodesInLayer = neurons;
    neuronLayerConfig.nodeInputCnt = prevLayerNeurons;
    neuronLayerConfig.outValOffset = 0; //is set later!!!!!
    neuronLayerConfig.biasShift    = 0;
    layersConf.push_back(make_unique<LayerConfig>(neuronLayerConfig) );

    //layer 4
    prevLayerNeurons = neurons;
    neurons = outputsCnt;
    lutLayerConfig.nodesInLayer = neurons;
    lutLayerConfig.nodeInputCnt = 1;
    lutLayerConfig.lutRangesCnt = 1;
    lutLayerConfig.bitsPerNodeInput = 10;
    lutLayerConfig.maxLutVal = 15;
    lutLayerConfig.minLutVal = -lutLayerConfig.maxLutVal;
    lutLayerConfig.middleLutVal = (lutLayerConfig.maxLutVal + lutLayerConfig.minLutVal)/2;
    layersConf.push_back(make_unique<LayerConfig>(lutLayerConfig) );

    layersConf.at(1)->outValOffset = (1<<layersConf.at(2)->bitsPerNodeInput) / layersConf.at(2)->lutRangesCnt / 2; //TODO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,,

    layersConf.at(3)->outValOffset =  (1<<layersConf.at(4)->bitsPerNodeInput -1);


    //string gmtDir = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_11_x_x_l1tOfflinePhase2/CMSSW_11_1_7/src/L1Trigger/Phase2L1GMT/test/";
    string gmtDirDYToLL = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_11_x_x_l1tOfflinePhase2/CMSSW_11_1_7/src/usercode/MuCorrelatorAnalyzer/crab/crab_Phase2L1GMT_org_MC_analysis_DYToLL_M-50_Summer20_PU200_t207/results/";

    EventsGeneratorGmt eventsGeneratorTrain(rndGenerator, outputsCnt, useHitsValid, addNoHitCntToInputs);
/*    eventsGeneratorTrain.readEvents(gmtDirDYToLL + "muCorrelatorTTAnalysis1_2.root");
    eventsGeneratorTrain.readEvents(gmtDirDYToLL + "muCorrelatorTTAnalysis1_3.root");
    eventsGeneratorTrain.readEvents(gmtDirDYToLL + "muCorrelatorTTAnalysis1_4.root");
    eventsGeneratorTrain.readEvents(gmtDirDYToLL + "muCorrelatorTTAnalysis1_5.root");*/
    eventsGeneratorTrain.readEvents(gmtDirDYToLL + "muCorrelatorTTAnalysis1_6.root");
    eventsGeneratorTrain.readEvents(gmtDirDYToLL + "muCorrelatorTTAnalysis1_7.root");
    //eventsGeneratorTrain.readEvents(gmtDirDYToLL + "muCorrelatorTTAnalysis1_8.root");
    eventsGeneratorTrain.readEvents(gmtDirDYToLL + "muCorrelatorTTAnalysis1_9.root");
    eventsGeneratorTrain.readEvents(gmtDirDYToLL + "muCorrelatorTTAnalysis1_13.root");
    eventsGeneratorTrain.readEvents(gmtDirDYToLL + "muCorrelatorTTAnalysis1_14.root");
    eventsGeneratorTrain.readEvents(gmtDirDYToLL + "muCorrelatorTTAnalysis1_15.root");
    eventsGeneratorTrain.readEvents(gmtDirDYToLL + "muCorrelatorTTAnalysis1_16.root");

    eventsGeneratorTrain.readEvents(gmtDirDYToLL + "muCorrelatorTTAnalysis1_9.root");

    string gmtDirJPsi = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_11_x_x_l1tOfflinePhase2/CMSSW_11_1_7/src/usercode/MuCorrelatorAnalyzer/crab/crab_Phase2L1GMT_org__MC_analysis_JPsiToMuMu_Summer20_PU200_t207/results/";
    eventsGeneratorTrain.readEvents(gmtDirJPsi + "muCorrelatorTTAnalysis1_1.root");
    eventsGeneratorTrain.readEvents(gmtDirJPsi + "muCorrelatorTTAnalysis1_2.root");
    eventsGeneratorTrain.readEvents(gmtDirJPsi + "muCorrelatorTTAnalysis1_3.root");
    eventsGeneratorTrain.readEvents(gmtDirJPsi + "muCorrelatorTTAnalysis1_4.root");
    eventsGeneratorTrain.readEvents(gmtDirJPsi + "muCorrelatorTTAnalysis1_5.root");
    eventsGeneratorTrain.readEvents(gmtDirJPsi + "muCorrelatorTTAnalysis1_6.root");
    eventsGeneratorTrain.readEvents(gmtDirJPsi + "muCorrelatorTTAnalysis1_7.root");

    string gmtDirTauTo3Mu = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_11_x_x_l1tOfflinePhase2/CMSSW_11_1_7/src/usercode/MuCorrelatorAnalyzer/crab/crab_Phase2L1GMT_MC_analysis_TauTo3Mu_Summer20_PU200_withNewMB_t207/results/";
    eventsGeneratorTrain.readEvents(gmtDirTauTo3Mu + "muCorrelatorTTAnalysis1_1.root");
    eventsGeneratorTrain.readEvents(gmtDirTauTo3Mu + "muCorrelatorTTAnalysis1_2.root");
    eventsGeneratorTrain.readEvents(gmtDirTauTo3Mu + "muCorrelatorTTAnalysis1_3.root");
    eventsGeneratorTrain.readEvents(gmtDirTauTo3Mu + "muCorrelatorTTAnalysis1_4.root");
    eventsGeneratorTrain.readEvents(gmtDirTauTo3Mu + "muCorrelatorTTAnalysis1_5.root");
    eventsGeneratorTrain.readEvents(gmtDirTauTo3Mu + "muCorrelatorTTAnalysis1_6.root");
    eventsGeneratorTrain.readEvents(gmtDirTauTo3Mu + "muCorrelatorTTAnalysis1_7.root");

    EventsGeneratorGmt eventsGeneratorTest(rndGenerator, outputsCnt, useHitsValid, addNoHitCntToInputs);
    eventsGeneratorTest.readEvents(gmtDirDYToLL  + "muCorrelatorTTAnalysis1_20.root");
    eventsGeneratorTest.readEvents(gmtDirJPsi  + "muCorrelatorTTAnalysis1_10.root");
    eventsGeneratorTest.readEvents(gmtDirTauTo3Mu  + "muCorrelatorTTAnalysis1_10.root");

    //TODO here the input nodes are randomly connected to the inputs, there is one, unique input node for each input of the first lutNode layer
    //unsigned int inputNodesCnt = layersConf.at(0)->nodesInLayer * layersConf.at(0)->nodeInputCnt;
    //unique_ptr<InputNodeFactory> inputNodeFactory = make_unique<InputNodeBinaryFactory>(eventsGeneratorTrain.gelPixelCnt(), &rndGenerator);
    //unique_ptr<InputNodeFactory> inputNodeFactory = make_unique<InputNodeSelectedBitsFactory>(eventsGeneratorTrain.gelPixelCnt(), rndGenerator);

    //unsigned int inputNodesCnt = eventsGeneratorTrain.getEvents().front()->inputs.size();

    if( (inputCnt + addNoHitCntToInputs) != eventsGeneratorTrain.getEvents().front()->inputs.size()) {
        throw std::runtime_error("wrong inputCnt");
    }


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


    cout<<"network:\n\n";
    cout<<network<<endl;

    network.initLuts(rndGenerator);

    //return 0;

    unsigned int maxNodesPerLayer = 18;
    LutNetworkPrint lutNetworkPrint;
    TCanvas* canvasLutNN = lutNetworkPrint.createCanvasLutsAndOutMap(lutLayerCnt, maxNodesPerLayer, 2);

    //vector<EventInt*>& testEvents = eventsGeneratorTest.getEvents(); //TODO look for the number
    vector<EventInt*>& validationEvents = eventsGeneratorTest.getEvents(); //TODO take real set of validation events

    vector<EventInt*> trainingEventsBatch(batchSize);

    lutNetworkPrint.createCostHistory(iterations, printEveryIteration);

    network.reset();

    //eventsGeneratorTrain.getNextMiniBatch(trainingEventsBatch); //todo remove
    std::string canvasCostPng = "../pictures/canvasLutNN_canvasCostHist.png";

    LearnigParams learnigParams;
    learnigParams.learnigRate = 0.1; //0.1/16. * 40; //200 ;
    learnigParams.lambda = 0.0005;
    learnigParams.beta = 0; //0.8;
    learnigParams.smoothWeight = 0;//0.01;
    std::vector<LearnigParams> learnigParamsVec(layersConf.size(), learnigParams);

    learnigParamsVec[0].learnigRate *= 1.;
    learnigParamsVec[2].learnigRate *= 10.;
    learnigParamsVec[4].learnigRate *= 0.5;

/*
    learnigParamsVec[0].lambda *= 1.;
    learnigParamsVec[2].lambda *= 0.2;
    learnigParamsVec[4].lambda *= 1.;
*/

    learnigParamsVec[4].lambda = 0.;

    cout<<"learnigParams "<<endl;
    for(unsigned int iLayer = 0; iLayer < learnigParamsVec.size(); iLayer++ ) {
        cout<<" iLayer "<<iLayer<<" learnigRate "<<learnigParamsVec[iLayer].learnigRate<<" lambda "<<learnigParamsVec[iLayer].lambda
                <<" beta "<<learnigParamsVec[iLayer].beta<<" smoothWeight "<<learnigParamsVec[iLayer].smoothWeight<<endl;
    }

    string outDir = "../pictures/omtfAnalyzer";

    //lambda
    auto printValidation = [&](vector<EventInt*> events, unsigned int iteration, double trainSampleCost) {
        network.reset();
        for(auto& event : events) {
            network.run(event);
            //network.runTrainingClassLabel(event, costFunction);
        }
        cout<<"line "<<dec<<__LINE__<<endl;
        lutNetworkPrint.printLuts2(network, maxNodesPerLayer);
        cout<<"line "<<dec<<__LINE__<<endl;
        double validSampleCost = lutNetworkPrint.printExpectedVersusNNOutHotOne(events, costFunction);
        cout<<"line "<<dec<<__LINE__<<endl;
        lutNetworkPrint.updateCostHistory(iteration, trainSampleCost, validSampleCost, canvasCostPng);

        canvasLutNN->SaveAs( ("../pictures/canvasLutNN_GradientTrain_" + std::to_string(iteration)+ ".png").c_str());

        cout<<"validSampleCost "<<validSampleCost<<std::endl<<std::endl;
        return validSampleCost;
    };

    double trainSampleCost = 0;
    double minCost = 1000000.;
    double minCostIteration = 0;
    for(unsigned int i = 0; i < iterations; i++) {
        //cout<<"line "<<dec<<__LINE__<<endl;
        //auto_cpu_timer timer1;
        eventsGeneratorTrain.getNextMiniBatch(trainingEventsBatch); //TODO !!!!!!!!!!!!!!!!!!!
        for(unsigned int iEv = 0; iEv < trainingEventsBatch.size(); iEv++) {
            network.runTrainingClassLabel(trainingEventsBatch[iEv], costFunction);
            //network.runTraining(trainingEventsBatch[iEv], costFunction, 90, 3, rndGenerator);
        }

        trainSampleCost = network.getMeanCost();
        //cout<<setw(6)<<dec<<i<<" trainSampleCost "<<setw(10)<<trainSampleCost<<" efficiency "<<setw(10)<<network.getEfficiency()
        //        <<" AverageEfficiency "<<network.getAverageEfficiency()<<std::endl;

        if(i > 0) {//just to see the initial LUTs
            network.updateLuts(learnigParamsVec);
            if(i < 1000)
            	network.smoothLuts(learnigParamsVec);
        }

        //cout<<"line "<<dec<<__LINE__<<endl;
        if( (i)%1000 == 0) {
            cout<<"iteration "<<i<<" trainSampleCost "<<trainSampleCost<<std::endl;
            printValidation(eventsGeneratorTest.getEvents(), i, trainSampleCost);
        }
        else if( (i)%printEveryIteration == 0) {
            cout<<"iteration "<<i<<" trainSampleCost "<<trainSampleCost<<std::endl;
            double validSampleCost = printValidation(validationEvents, i, trainSampleCost);

            if(minCost > validSampleCost) {
                minCost = validSampleCost;
                minCostIteration = i;
            }
        }

        network.reset();
        //outfile.Write();
        //cout<<"line "<<dec<<__LINE__<<endl;

        if(i == 1000) {
            learnigParamsVec[0].lambda *= 0.;
            learnigParamsVec[2].lambda *= 0.;
            learnigParamsVec[4].lambda *= 0.;

            learnigParamsVec[0].learnigRate *= 0.5;
            learnigParamsVec[2].learnigRate *= 0.5;
            learnigParamsVec[4].learnigRate *= 0.5;
        }

        /*if(i == 2000) {
            learnigParamsVec[0].lambda *= 0.;
            learnigParamsVec[2].lambda *= 0.;
            learnigParamsVec[4].lambda *= 0.;
        }*/

        if(i == 3000 || i == 10000 || i == 20000) {
            trainingEventsBatch.resize(trainingEventsBatch.size() * 2, nullptr);
        }

        if(i == 5000) {
            /*std::ofstream ofs("lutNN_omtfClassifier_38050.txt");
            {
                boost::archive::text_oarchive oa(ofs);
                registerClasses(oa);
                oa << network;
            }*/

            learnigParamsVec[0].learnigRate *= 0.5;
            learnigParamsVec[2].learnigRate *= 0.5;
            learnigParamsVec[4].learnigRate *= 0.5;
/*
            learnigParamsVec[0].learnigRate *= 0.1;
            learnigParamsVec[2].learnigRate *= 0.1;
            learnigParamsVec[4].learnigRate *= 0.1;*/

            learnigParamsVec[0].lambda *= 0.;
            learnigParamsVec[2].lambda *= 0.;
            learnigParamsVec[4].lambda *= 0.;
        }

        if(i == 8000) {
            learnigParamsVec[0].learnigRate *= 0.5;
            learnigParamsVec[2].learnigRate *= 0.5;
            learnigParamsVec[4].learnigRate *= 0.5;
        }
    }

    printValidation(eventsGeneratorTest.getEvents(), iterations, trainSampleCost);

    // create and open a character archive for output
    std::ofstream ofs("gmtLutInterNN1.txt");
    // save data to archive
    {
        boost::archive::text_oarchive oa(ofs);
        // write class instance to archive
        registerClasses(oa);
        oa << network;
        // archive and stream closed when destructors are called
    }

    cout<<"eventsGeneratorTest.getEvents().size() "<<eventsGeneratorTest.getEvents().size()<<endl;


    TFile outfile("../pictures/lutNN2.root", "RECREATE");
    outfile.cd();
    canvasLutNN->Write();
    outfile.Write();


    cout<<" minCost "<<minCost<<" minCostIteration "<<minCostIteration<<endl;

    cout<<"line "<<dec<<__LINE__<<endl;

    {
        bool useMargin = false;

        if(dynamic_cast<SoftMax*>(network.getOutputNode()) != 0)
            useMargin = false;
        else
            useMargin = true;

        eventsGeneratorTest.clear();
        std::string signalSample = "TauTo3Mu";
        eventsGeneratorTest.readEvents(gmtDirTauTo3Mu  + "muCorrelatorTTAnalysis1_10.root");
        eventsGeneratorTest.readEvents(gmtDirTauTo3Mu  + "muCorrelatorTTAnalysis1_11.root");
        eventsGeneratorTest.readEvents(gmtDirTauTo3Mu  + "muCorrelatorTTAnalysis1_12.root");
        eventsGeneratorTest.readEvents(gmtDirTauTo3Mu  + "muCorrelatorTTAnalysis1_13.root");
        eventsGeneratorTest.readEvents(gmtDirTauTo3Mu  + "muCorrelatorTTAnalysis1_14.root");
        eventsGeneratorTest.readEvents(gmtDirTauTo3Mu  + "muCorrelatorTTAnalysis1_15.root");

        auto& events = eventsGeneratorTest.getEvents();


        cout<<"events.size() "<<events.size()<<endl;

        for(auto& event : events) {
            //network.run(event);
            network.LutNetworkBase::run(event);
        }

        std::string outFilePath = "../pictures/";

        GmtAnalyzer gmtAnalyzer(outFilePath + signalSample + "_results.root");

        gmtAnalyzer.analyze(events, signalSample, false, useMargin);

        eventsGeneratorTest.clear();

        string gmtDir = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_11_x_x_l1tOfflinePhase2/CMSSW_11_1_7/src/usercode/MuCorrelatorAnalyzer/crab/crab_Phase2L1GMT_MC_analysis_MinBias_Summer20_PU200_t207/results/";
        std::string falseSampl = "MinBias";

        eventsGeneratorTest.readEvents(gmtDir  + "muCorrelatorTTAnalysis1_11.root");
        events = eventsGeneratorTest.getEvents();

        for(auto& event : events) {
            network.LutNetworkBase::run(event);
        }
        gmtAnalyzer.analyze(events, falseSampl, true, useMargin);

        gmtAnalyzer.makeRocCurve(signalSample, falseSampl);
    }
    return EXIT_SUCCESS;
}
