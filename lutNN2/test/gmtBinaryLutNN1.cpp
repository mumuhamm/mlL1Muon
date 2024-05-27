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

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "TCanvas.h"
#include "TFile.h"

//#include "lutNN/lutNN2/interface/LutNetworkBase.h"
#include "lutNN/lutNN2/interface/BinaryLutNetwork.h"
#include "lutNN/lutNN2/interface/NetworkBuilder.h"


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
    //std::mt19937_64 rndGenerator(11);

	unsigned int batchSize = 500; //TODO
	unsigned int iterations = 20000;
	unsigned int printEveryIteration = 1000; //TODO

	iterations++;

	unsigned int outputsCnt = 3;
	unsigned int branchesPerOutput = 20;

	unsigned int branchesShare = 3;

	bool useHitsValid = false;

    //CostFunctionCrossEntropy  costFunction;
    CostFunctionHingeLost  costFunction;
	//CostFunctionMeanSquaredError costFunction;

    //softMaxNode = new SoftMaxWithSubClasses(outputsCnt);

    //outputNode.reset(softMaxNode);
    //outputNode.reset(new NetworkOutNode(outputsCnt));
    //outputNode.reset(new SoftMaxWithSubClasses(outputsCnt));
    //outputNode.reset(new SoftMax(outputsCnt));
    //NetworkOutNode* outputNode = new SoftMax(outputsCnt); //todo use uniq_ptr
    NetworkOutNode* outputNode = new NetworkOutNode(outputsCnt);

    BinaryLutNetwork network(outputNode, &rndGenerator);
	LayersConfigs& layersConf =  network.getLayersConf();

	LayerConfig layerConfig;
	layerConfig.bitsPerNodeInput = 1;
	layerConfig.nodeInputCnt = 6;
	layerConfig.outputBits = 1;
	layerConfig.maxLutVal = 1.;
	layerConfig.minLutVal = 0;
	layerConfig.middleLutVal = (layerConfig.maxLutVal - layerConfig.minLutVal)/ 2.;
	layerConfig.nodeType = LayerConfig::lutBinary;

/*	//layer 0
	layerConfig.nodesInLayer = outputsCnt * branchesPerOutput * 5 * 5 * 5;
	layerConfig.bitsPerNodeInput = 1;
	layerConfig.nodeInputCnt = 5;
	layersConf.push_back(make_unique<LayerConfig>(layerConfig) );*/

	//layer 1
    layerConfig.nodesInLayer = outputsCnt / branchesShare * branchesPerOutput * 5 * 6;// /10;
	layerConfig.bitsPerNodeInput = 6;//TODO!!!!!!!!!!!!!!!!!!!!!!!!!!change to 7 for the curvature!!!!!!!!!!!!!!!!!!!
	layerConfig.nodeInputCnt = 1;
    layersConf.push_back(make_unique<LayerConfig>(layerConfig) );

    //layer 2
	layerConfig.bitsPerNodeInput = 1;
	layerConfig.nodeInputCnt = 5;
    layerConfig.nodesInLayer = outputsCnt / branchesShare  * branchesPerOutput * 6;
    layersConf.push_back(make_unique<LayerConfig>(layerConfig) );

    //layer 3
    layerConfig.nodeType = LayerConfig::lutNode;
    layerConfig.nodesInLayer = outputsCnt * branchesPerOutput;
    layerConfig.nodeInputCnt = 6;
    layerConfig.outputBits = 3;
    layerConfig.maxLutVal = 10;
    layerConfig.minLutVal = -layerConfig.maxLutVal;
    layerConfig.middleLutVal = (layerConfig.maxLutVal + layerConfig.minLutVal)/ 2.;
    layersConf.push_back(make_unique<LayerConfig>(layerConfig) );

    //layer 4

    layerConfig.nodeType = LayerConfig::sumNode;
    layerConfig.subClassesCnt = 1;
    layerConfig.nodeInputCnt = branchesPerOutput / layerConfig.subClassesCnt; //layersConf.back()->nodesInLayer;
    layerConfig.classesCnt = outputsCnt;
    layerConfig.nodesInLayer = outputsCnt * layerConfig.subClassesCnt;
    layersConf.push_back(make_unique<LayerConfig>(layerConfig) );

    for(unsigned int iLayer = 0; iLayer < layersConf.size(); iLayer++) {
        cout<<"alyer "<<iLayer<<" layersConf minLutVal "<<layersConf[iLayer]->minLutVal<<" maxLutVal "<<layersConf[iLayer]->maxLutVal
                <<" middleLutVal "<<layersConf[iLayer]->middleLutVal <<std::endl;
    }

    //string gmtDir = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_11_x_x_l1tOfflinePhase2/CMSSW_11_1_7/src/L1Trigger/Phase2L1GMT/test/";
    string gmtDirDYToLL = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_11_x_x_l1tOfflinePhase2/CMSSW_11_1_7/src/usercode/MuCorrelatorAnalyzer/crab/crab_Phase2L1GMT_org_MC_analysis_DYToLL_M-50_Summer20_PU200_t207/results/";

    EventsGeneratorGmt eventsGeneratorTrain(rndGenerator, outputsCnt, useHitsValid, false);
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

    EventsGeneratorGmt eventsGeneratorTest(rndGenerator, outputsCnt, useHitsValid, false);
    eventsGeneratorTest.readEvents(gmtDirDYToLL  + "muCorrelatorTTAnalysis1_20.root");
    eventsGeneratorTest.readEvents(gmtDirJPsi  + "muCorrelatorTTAnalysis1_10.root");
    eventsGeneratorTest.readEvents(gmtDirTauTo3Mu  + "muCorrelatorTTAnalysis1_10.root");

    //TODO here the input nodes are randomly connected to the inputs, there is one, unique input node for each input of the first lutNode layer
    //unsigned int inputNodesCnt = layersConf.at(0)->nodesInLayer * layersConf.at(0)->nodeInputCnt;
    //unique_ptr<InputNodeFactory> inputNodeFactory = make_unique<InputNodeBinaryFactory>(eventsGeneratorTrain.gelPixelCnt(), &rndGenerator);
    //unique_ptr<InputNodeFactory> inputNodeFactory = make_unique<InputNodeSelectedBitsFactory>(eventsGeneratorTrain.gelPixelCnt(), rndGenerator);

    unsigned int inputNodesCnt = eventsGeneratorTrain.getEvents().front()->inputs.size();
    unique_ptr<InputNodeFactory> inputNodeFactory = make_unique<InputNodeFactoryBase>(); //, rndGenerator


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


    //vector<EventInt*>& testEvents = eventsGeneratorTest.getEvents(); //TODO look for the number
    vector<EventInt*>& validationEvents = eventsGeneratorTest.getEvents(); //TODO take real set of validation events

    //outfile.Write();

    network.initLutsRnd(rndGenerator); //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
    //network.initLutsAnd(rndGenerator);

    vector<EventInt*> trainingEventsBatch(batchSize);

    lutNetworkPrint.createCostHistory(iterations, printEveryIteration);

    network.reset();

    //eventsGeneratorTrain.getNextMiniBatch(trainingEventsBatch); //todo remove
    std::string canvasCostPng = "../pictures/canvasLutNN_canvasCostHist.png";

    LearnigParams learnigParams;
    learnigParams.learnigRate = 0.05 ;
    learnigParams.lambda = 0;//0.005;

    std::vector<LearnigParams> learnigParamsVec(layersConf.size(), learnigParams);

	learnigParamsVec.at(0).learnigRate = 0.002 ; //*2
	learnigParamsVec.at(1).learnigRate = 0.002 ;
	learnigParamsVec.at(2).learnigRate = 0.002 ;

    //learnigParamsVec.at(0).lambda = 0.003 ;
    //learnigParamsVec.at(1).lambda = 0.003 ;
    //learnigParamsVec.at(2).lambda = 0.01 ;

    for(unsigned int iLayer = 0; iLayer < layersConf.size(); iLayer++)
    	layersConf[iLayer]->ditherRate = 0;

    layersConf[layersConf.size()-2]->ditherRate = 0;

    eventsGeneratorTrain.shuffle();
    eventsGeneratorTrain.shuffle();
    eventsGeneratorTrain.shuffle();

    cout<<"starting training"<<endl;
    std::normal_distribution<> normalDist(0, 64);
    for(unsigned int i = 0; i < iterations; i++) {
        //cout<<"line "<<dec<<__LINE__<<endl;
        //auto_cpu_timer timer1;
        //if(i > 0)
        {
            eventsGeneratorTrain.getNextMiniBatch(trainingEventsBatch); //TODO !!!!!!!!!!!!!!!!!!!
            //cout<<"line "<<__LINE__<<" trainingEventsBatch.size() "<<trainingEventsBatch.size()<<endl;
            {
                auto_cpu_timer timer1;
                for(unsigned int iEv = 0; iEv < trainingEventsBatch.size(); iEv++) {
                    /*EventInt eventWithNoise = *(trainingEventsBatch[iEv]);
                    for(auto& input : eventWithNoise.inputs) {
                        input += normalDist(rndGenerator);
                    }*/

                    //network.runTraining(&eventWithNoise, costFunction);  //TODO!!!!!!!!!!!!!!!!!!!!

                    network.LutNetworkBase::runTrainingClassLabel(trainingEventsBatch[iEv], costFunction);
                    //cout<<"line "<<dec<<__LINE__<<endl;
                    //network.runTrainingAndUpdate(&eventWithNoise, costFunction, learnigParamsVec);
                    //cout<<"line "<<dec<<__LINE__<<endl;
                }
                //cout<<"line "<<__LINE__ << timer1.format() << '\n';
            }

            network.updateLuts(learnigParamsVec);
        }
        double trainSampleCost = network.getMeanCost();
        cout<<setw(6)<<dec<<i<<" trainSampleCost "<<setw(10)<<trainSampleCost<<" efficiency "<<setw(10)<<network.getEfficiency()
                <<" AverageEfficiency "<<network.getAverageEfficiency()<<std::endl;


        if(i > 0 && i%200 == 0) {
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
/*

        	if(i < 600) {
        	    layersConf[layersConf.size()-2]->ditherRate = 90. - i/20.;//50; //this is in percent!!!!!
        	}
        	else
        	    layersConf[layersConf.size()-2]->ditherRate = 60;//50; //this is in percent!!!!!
*/

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


    // create and open a character archive for output
    std::ofstream ofs("gmtBinaryLutNN1.txt");
    // save data to archive
    {
        boost::archive::text_oarchive oa(ofs);
        // write class instance to archive
        registerClasses(oa);
        oa << network;
        // archive and stream closed when destructors are called
    }



    TFile outfile("../pictures/lutNN2.root", "RECREATE");
    outfile.cd();
    canvasLutNN->Write();
    outfile.Write();
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
