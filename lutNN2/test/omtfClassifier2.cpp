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
#include "lutNN/lutNN2/interface/EventsGeneratorOmtf.h"
#include "lutNN/lutNN2/interface/Utlis.h"
#include "lutNN/lutNN2/interface/NetworkSerialization.h"
#include "lutNN/lutNN2/interface/OmtfAnalyzer.h"

using namespace lutNN;
using namespace std;
using namespace boost::timer;

int main(void) {
	puts("Hello World!!!");

    std::default_random_engine rndGenerator(11);



    std::vector<float> ptBins = {
/*    		4,
			6,
			8,
			10,
			12,
			15,
			18,
			21,
			26,
			36,
			60,
			100,
			1000000 */
    		
    		4,
			7,
			9,
    		12,
			15,
			18,
			21,
			26,
			36,
			60,
			1000000

/*    		4,
			7,
			9,
    		12,
			16,
			20,
			26,
			36,
			60,
			1000000*/

	   	/*	    5,
	    		10,
				15,
				20,
				30,
				1000000*/
    };


    unsigned int outputCnt = ptBins.size();// * 2;

   EventsGeneratorOmtf eventsGeneratorTrain(rndGenerator, outputCnt, ptBins);


    string dataPath = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_12_x_x_official/CMSSW_12_3_0_pre4/src/L1Trigger/L1TMuonOverlapPhase1/test/expert/omtf/";
    //eventsGeneratorTrain.readEvents(dataPath + "OMTFHits_pats0x00012_newerSample_files_1_80.root");

    eventsGeneratorTrain.readEvents(dataPath + "OMTFHits_pats0x00012_oldSample_files_16_20.root", 2);

    //eventsGeneratorTrain.readEvents(dataPath + "OMTFHits_pats0x00012_oldSample_files_1_15_PtUpTo13GeV.root", 2);


/*    dataPath = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_12_x_x_official/CMSSW_12_3_0_pre4/src/L1Trigger/L1TMuonOverlapPhase1/test/crab/";

    string datdDir = dataPath + "crab_OmtfDataDump_analysis_DoubleMuon_gun_Summer20_PU200_t130/results/";
    for(int iFile = 1; iFile <= 50; iFile++)  //TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        eventsGeneratorTrain.readEvents(datdDir + "omtfHitsDump_" + to_string(iFile) + ".root", 2);

    datdDir = dataPath +  "crab_OmtfDataDump_analysis_DoubleMuon_gun_Summer20_NoPU_t130/results/";
    for(int iFile = 1; iFile <= 13; iFile++)  //TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        eventsGeneratorTrain.readEvents(datdDir + "omtfHitsDump_" + to_string(iFile) + ".root", 2);

    datdDir = dataPath +  "crab_OmtfDataDump_analysis_MuFlatPt_PhaseIITDRSpring19DR_PU200_t130/results/";
    for(int iFile = 1; iFile <= 65; iFile++)  //TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        eventsGeneratorTrain.readEvents(datdDir + "omtfHitsDump_" + to_string(iFile) + ".root", 2);*/
/*
    datdDir = dataPath + "crab_OmtfDataDump_analysis_JPsiToMuMu_Summer20_PU140_t130/results/";
    for(int iFile = 1; iFile <= 11; iFile++)  //TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        eventsGeneratorTrain.readEvents(datdDir + "omtfHitsDump_" + to_string(iFile) + ".root", 2);*/

/*    datdDir = dataPath +  "crab_OmtfDataDump_analysis_SingleNeutrino_PhaseIITDRSpring19_PU250_t130/results/";
    for(int iFile = 1; iFile <= 64; iFile++)  //TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        eventsGeneratorTrain.readEvents(datdDir + "omtfHitsDump_" + to_string(iFile) + ".root", 2);

    datdDir = dataPath +  "crab_OmtfDataDump_analysis_SingleNeutrino_PhaseIITDRSpring19_PU200_t130/results/";
    for(int iFile = 1; iFile <= 57; iFile++)  //TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        eventsGeneratorTrain.readEvents(datdDir + "omtfHitsDump_" + to_string(iFile) + ".root", 2);

    datdDir = dataPath +  "crab_OmtfDataDump_analysis_JPsiToMuMu_Summer20_PU200_t130/results/";
    for(int iFile = 1; iFile <= 10; iFile++)  //TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        eventsGeneratorTrain.readEvents(datdDir + "omtfHitsDump_" + to_string(iFile) + ".root", 2);*/


    cout<<"eventsGeneratorTrain.getEvents().size() "<<eventsGeneratorTrain.getEvents().size()<<endl;

    eventsGeneratorTrain.setEventWeight(nullptr, nullptr);

    EventsGeneratorOmtf eventsGeneratorTest(rndGenerator, outputCnt, ptBins);

    dataPath = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_12_x_x_official/CMSSW_12_3_0_pre4/src/L1Trigger/L1TMuonOverlapPhase1/test/expert/omtf/";
    eventsGeneratorTrain.readEvents(dataPath + "OMTFHits_pats0x00012_newerSample_files_1_80.root", 2);
    eventsGeneratorTest.readEvents(dataPath + "OMTFHits_pats0x00012_newerSample_files_81_100.root", 2);

/*
    datdDir = dataPath + "crab_OmtfDataDump_analysis_DoubleMuon_gun_Summer20_PU200_t130/results/";
    for(int iFile = 51; iFile <= 75; iFile++) {
        eventsGeneratorTest.readEvents(datdDir + "omtfHitsDump_" + to_string(iFile) + ".root", 2);
    }

    datdDir = dataPath + "crab_OmtfDataDump_analysis_JPsiToMuMu_Summer20_PU140_t130/results/";
    for(int iFile = 1; iFile <= 11; iFile++)  //TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        eventsGeneratorTest.readEvents(datdDir + "omtfHitsDump_" + to_string(iFile) + ".root", 2);*/

    cout<<"eventsGeneratorTest.getEvents().size() "<<eventsGeneratorTest.getEvents().size()<<endl;

    eventsGeneratorTest.setEventWeight(nullptr, nullptr);


	unsigned int batchSize = 1000; //TODO

	unsigned int iterations = 15000;
	unsigned int printEveryIteration = 200; //TODO

    unsigned int inputCnt = 18; //eventsGeneratorTrain.getInputCnt();

	unsigned int lutLayerCnt = 3; //TODO           for printing

	//unsigned int branchesPerOutput = 5;


    NetworkOutNode* outputNode = new SoftMax(outputCnt); //todo use uniq_ptr
    LutInterNetwork network(outputNode, true);

	LayersConfigs& layersConf = network.getLayersConf();

    const int input_I = 10;
    const int input_F = 4;
    const std::size_t networkInputSize = 18;

    const int layer1_neurons = 16;
    const int layer1_lut_I = 3;
    const int layer1_lut_F = 10;

    const int layer1_output_I = 6;
    //4 bits are for the count of the noHit layers which goes to the input of the layer2, other 4 for layer1_output_I
    const int layer2_input_I = layer1_output_I + 4;

    const int layer2_neurons = 8 + 1 + 1;
    const int layer2_lut_I = 5;
    const int layer2_lut_F = 10;

    const int layer3_input_I = 10;
    const int layer3_neurons = 1;
    const int layer3_lut_I = 6;
    const int layer3_lut_F = 10;


	LayerConfig lutLayerConfig;
	lutLayerConfig.bitsPerNodeInput = 10;
	lutLayerConfig.nodeInputCnt = 1;
	lutLayerConfig.outputBits = 10; //does not matter, its flaot
	lutLayerConfig.nodeType = LayerConfig::lutInter;
	lutLayerConfig.lutRangesCnt = 8;

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
    lutLayerConfig.lutRangesCnt = 8;
    lutLayerConfig.interpolate = true;
    //lutLayerConfig.lutBinsPerRange = (1<<lutLayerConfig.bitsPerNodeInput) / lutLayerConfig.lutRangesCnt;
    lutLayerConfig.maxLutVal =  1<<(layer1_lut_I-1);
    lutLayerConfig.minLutVal = -lutLayerConfig.maxLutVal;
    lutLayerConfig.middleLutVal = (lutLayerConfig.maxLutVal + lutLayerConfig.minLutVal)/2;

    lutSize = 1<<lutLayerConfig.bitsPerNodeInput;
    lutLayerConfig.initSlopeMax = (lutLayerConfig.maxLutVal - lutLayerConfig.minLutVal) / (lutSize/lutLayerConfig.lutRangesCnt) / 1.;
    lutLayerConfig.initSlopeMin = lutLayerConfig.initSlopeMax * 0.1;
	layersConf.push_back(make_unique<LayerConfig>(lutLayerConfig) );
	//layer 1
	sumLayerConfig.nodesInLayer = neurons;
	sumLayerConfig.nodeInputCnt = inputCnt;
	sumLayerConfig.outValOffset = 1 << (layer1_output_I-1);
	sumLayerConfig.biasShift    = layer1_output_I; //
	sumLayerConfig.shiftLastGradient = false;
	layersConf.push_back(make_unique<LayerConfig>(sumLayerConfig) );


    //layer 2
	unsigned int prevLayerNeurons = neurons;
    neurons = outputCnt;
    lutLayerConfig.bitsPerNodeInput = layer2_input_I;
    lutLayerConfig.nodesInLayer = prevLayerNeurons * neurons;
    lutLayerConfig.nodeInputCnt = 1;
    lutLayerConfig.lutRangesCnt = 16;
    lutLayerConfig.interpolate = true;
    //lutLayerConfig.lutBinsPerRange = (1<<lutLayerConfig.bitsPerNodeInput) / lutLayerConfig.lutRangesCnt;
    lutLayerConfig.maxLutVal =  1<<(layer2_lut_I-1);
    lutLayerConfig.minLutVal = -lutLayerConfig.maxLutVal;
    lutLayerConfig.middleLutVal = (lutLayerConfig.maxLutVal + lutLayerConfig.minLutVal)/2;

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
    prevLayerNeurons = neurons;
    neurons = outputCnt;
    lutLayerConfig.bitsPerNodeInput = layer3_input_I;
    lutLayerConfig.nodesInLayer = neurons;
    lutLayerConfig.nodeInputCnt = 1;
    lutLayerConfig.lutRangesCnt = 1;
    lutLayerConfig.interpolate = true;
    lutLayerConfig.maxLutVal = 15;
    lutLayerConfig.minLutVal = -lutLayerConfig.maxLutVal;
    lutLayerConfig.middleLutVal = (lutLayerConfig.maxLutVal + lutLayerConfig.minLutVal)/2;
    layersConf.push_back(make_unique<LayerConfig>(lutLayerConfig) );


    unique_ptr<InputNodeFactory> inputNodeFactory = make_unique<InputNodeFactoryBase>();

    if( (inputCnt + 1) != eventsGeneratorTrain.getEvents().front()->inputs.size()) {
        throw std::runtime_error("wrong inputCnt");
    }

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

    network.initLuts2(rndGenerator);

    //return 0;

    unsigned int maxNodesPerLayer = 18;
    LutNetworkPrint lutNetworkPrint;
    TCanvas* canvasLutNN = lutNetworkPrint.createCanvasLutsAndOutMap(lutLayerCnt, maxNodesPerLayer, 2);


    //vector<EventFloat*>& testEvents = eventsGeneratorTest.getEvents(); //TODO look for the number


    //vector<EventFloat*>& validationEvents = eventsGeneratorTest.getEvents(); //TODO take real set of validation events

    eventsGeneratorTest.shuffle(); //todo change to eventsGeneratorTest

    vector<EventFloat*> validationEvents(10000, nullptr);

    if(validationEvents.size() > eventsGeneratorTest.getEvents().size() )
        validationEvents.resize(eventsGeneratorTest.getEvents().size(), nullptr); //todo it better
    std::copy(eventsGeneratorTest.getEvents().begin(), eventsGeneratorTest.getEvents().begin() + validationEvents.size(), validationEvents.begin());

    vector<EventFloat*> trainingEventsBatch(batchSize);

    lutNetworkPrint.createCostHistory(iterations, printEveryIteration);

    network.reset();

    //eventsGeneratorTrain.getNextMiniBatch(trainingEventsBatch); //todo remove
    std::string canvasCostPng = "../pictures/canvasLutNN_canvasCostHist.png";

    LearnigParams learnigParams;
    learnigParams.learnigRate = 0.1;//0.1/16. * 40; //200 ;
    learnigParams.lambda = 0.005;
    learnigParams.beta = 0; //0.8;
    learnigParams.smoothWeight = 0;//0.01;
    std::vector<LearnigParams> learnigParamsVec(layersConf.size(), learnigParams);

/*    learnigParamsVec[0].learnigRate *= 1.;
    learnigParamsVec[2].learnigRate *= 20.;
    learnigParamsVec[4].learnigRate *= 0.5;*/

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
    auto printValidation = [&](vector<EventFloat*> events, unsigned int iteration, double trainSampleCost) {
        network.reset();
        for(auto& event : events) {
            network.run(event);
            //network.runTraining(event, costFunction); //todo runTraining fill the entries, find how to do this with run
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

    eventsGeneratorTrain.shuffle();
    eventsGeneratorTrain.shuffle();

    double trainSampleCost = 0;
    double minCost = 1000000.;
    double minCostIteration = 0;
    for(unsigned int i = 0; i < iterations; i++) {
        //cout<<"line "<<dec<<__LINE__<<endl;
        //auto_cpu_timer timer1;
        eventsGeneratorTrain.getNextMiniBatch(trainingEventsBatch); //TODO !!!!!!!!!!!!!!!!!!!
        for(unsigned int iEv = 0; iEv < trainingEventsBatch.size(); iEv++) {
            network.runTrainingClassLabel(trainingEventsBatch[iEv], costFunction);
            //network.runTraining(trainingEventsBatch[iEv], costFunction);
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

            learnigParamsVec[0].learnigRate *= 0.05;
            learnigParamsVec[2].learnigRate *= 0.05;
            learnigParamsVec[4].learnigRate *= 0.05;
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
    std::ofstream ofs("lutNN_omtfClassifier.txt");
    // save data to archive
    {
        boost::archive::text_oarchive oa(ofs);
        // write class instance to archive
        registerClasses(oa);
        oa << ptBins;
        oa << network;
        // archive and stream closed when destructors are called
    }

    cout<<"eventsGeneratorTest.getEvents().size() "<<eventsGeneratorTest.getEvents().size()<<endl;

    OmtfAnalyzer omtfAnalyzer;

    omtfAnalyzer.muonAlgorithms.emplace_back(new OmtfAlgorithm(18, 12, "omtf_HighQ", kBlue));
    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPAlgorithm(15, 12, "lutNN_maxP_HighQ", kMagenta, ptBins));
    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierPSumAlgorithm(15, 12, "lutNN_PSum_04_HighQ", kGreen, ptBins, 0.4));
    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierPSumAlgorithm(15, 12, "lutNN_PSum_05_HighQ", kGreen + 1, ptBins, 0.5));
    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierPSumAlgorithm(15, 12, "lutNN_PSum_06_HighQ", kGreen + 2, ptBins, 0.6));
    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierPSumAlgorithm(15, 12, "lutNN_PSum_07_HighQ", kGreen + 3, ptBins, 0.7));
    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPInterAlgorithm(15, 12, "lutNN_MaxPInter_HighQ", kGreen + 1, ptBins, 1));

    MuonAlgorithm* omtfAlgo = omtfAnalyzer.muonAlgorithms[0].get();

    for(unsigned int iAlgo = 1; iAlgo < omtfAnalyzer.muonAlgorithms.size(); iAlgo++) {
    	omtfAnalyzer.muonAlgorithms[iAlgo]->algosToCompare.push_back(omtfAlgo);
    }

    //omtfAnalyzer.analyse(eventsGeneratorTest.getEvents(), outDir); //OmtfAnalysisType::regression TODO select rootFileTest1     !!!!!!!!!!!!!!!!!!!
    omtfAnalyzer.analyse(eventsGeneratorTest.getEvents(), "", outDir);

    TFile outfile("../pictures/lutNN2.root", "RECREATE");
    outfile.cd();
    canvasLutNN->Write();
    outfile.Write();



    cout<<" minCost "<<minCost<<" minCostIteration "<<minCostIteration<<endl;

    cout<<"line "<<dec<<__LINE__<<endl;
    return EXIT_SUCCESS;
}
