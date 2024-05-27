//============================================================================
// Name        : omtfClassifierTestFixedPoint.cpp
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
#include "lutNN/lutNN2/interface/ClassifierToRegression.h"
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


#include "lutNN/lutNN2/interface/LutNetworkFixedPoint.h"

using namespace lutNN;
using namespace std;
using namespace boost::timer;

int main(void) {
	puts("Hello World!!!");

    const int input_I = 10;
    const int input_F = 4;
    const std::size_t networkInputSize = 18;

    const int layer1_neurons = 16;
    const int layer1_lut_I = 3;
    const int layer1_lut_F = 10;

    const int layer1_output_I = 6;
    //4 bits are for the count of the noHit layers which goes to the input of the layer2
    const int layer2_input_I = 10;

    const int layer2_neurons = 2 * 11;
    const int layer2_lut_I = 5;
    const int layer2_lut_F = 10;

    const int layer3_input_I = 10;
    const int layer3_neurons = 1;
    const int layer3_lut_I = 6;
    const int layer3_lut_F = 10;

    LutNetworkFixedPoint<input_I,  input_F,  networkInputSize,
                         layer1_lut_I, layer1_lut_F, layer1_neurons,        //layer1_lutSize = 2 ^ input_I
                         layer1_output_I,
                         layer2_input_I,
                         layer2_lut_I, layer2_lut_F, layer2_neurons,
                         layer3_input_I,
                         layer3_lut_I, layer3_lut_F, layer3_neurons> lutNetworkFP; //, network_output_W, network_output_I, network_outputSize>



	//OmtfAnalysisType omtfAnalysisType = OmtfAnalysisType::classifier; //OmtfAnalysisType::regression;

    std::default_random_engine rndGenerator(11);

   /* string rootFile1 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/OMTFHits_pats0x0005_oldSample_files10_14.root";
    string rootFile2 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/OMTFHits_pats0x0005_oldSample_files20_24.root";
    string rootFile3 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/OMTFHits_pats0x0005_oldSample_files30_39.root";
*/
    //string rootFileTest1 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/OMTFHits_oldSample_files15.root";
    //string rootFileTest1 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/OMTFHits_pats0x0005_newerSample_files1_100.root";
    //string rootFile3 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/OMTFHits_lowPt_3_9.root";

    string rootFile1 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/omtfHits_omtfAlgo0x0006_v1/OMTFHits_pats0x0003_oldSample_files_30_40.root";

    //string rootFileTest1 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/omtfHits/OMTFHits_pats0x0006_2_newerSample_files_1_100.root";
    string rootFileTest1 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/omtfHits_omtfAlgo0x0006_v1/OMTFHits_pats0x0003_newerSample_files_1_100.root";
    //string rootFileTest1 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/omtfHits/OMTFHits_pats0x00031_newerSample_files_1_100.root";
    //string rootFileTest1 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/omtfHits/OMTFHits_pats0x00031_newerSample_files_1_100_omtfPtCont1.root";

    //string rootFileTest1 ="/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/omtfHits/OMTFHits_pats0x0006_2_oldSample_files_40_45.root";

    const bool enableMaxPLut = false;

    //rootFileTest1 = rootFile1;

    InputSetterBase* inputSetter = new InputSetterWithNoHitCnt();
    LutInterNetwork network(nullptr, inputSetter, true);

    string dir = "../results/omtfClassifier1_v38/";

    std::vector<float> ptBins = {
    		//omtfClassifier1_v27
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
			1000000*/

    		//omtfClassifier1_v21

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

    		/*5,
    		10,
			15,
			20,
			30,
			1000000*/
    };


    //string outDir = dir + "omtfAnalyzer_pats0x0006_2_oldSample";
    string outDir = dir + "omtfAnalyzer_pats0x0003_algo_0x0006_newerSample_";
    //string outDir = dir + "omtfAnalyzer_pats0x0006_";

    // create and open a character archive for output
    //std::ifstream ifs("lutNN_omtfRegression.txt");
    //std::ifstream ifs("../results/omtfRegression_v17/lutNN_omtfRegression.txt");
    //std::ifstream ifs("lutNN_omtfClassifier_38050.txt");
    //std::ifstream ifs("lutNN_omtfClassifier.txt");
    std::ifstream ifs(dir + "lutNN_omtfClassifier.txt");
    //std::ifstream ifs(dir + "lutNN_omtfClassifier_withPtBins.txt");
    {
        boost::archive::text_iarchive ia(ifs);
        registerClasses(ia);
        ptBins.clear();
        ia >> ptBins;
        cout<<"LINE:"<<__LINE__<<endl;
        ia >> network;
        // archive and stream closed when destructors are called
    }

    cout<<network<<endl;

    lutNetworkFP.initLuts(network);

    //exit(1);//<<<<<<<<<<<<<<<<<<<<!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    unsigned int outputCnt = network.getOutputNode()->getOutputValues().size();

    /*if(outputCnt == 1) {
        omtfAnalysisType = OmtfAnalysisType::regression; //TODO
    }
    else {
        omtfAnalysisType = OmtfAnalysisType::classifier;
    }*/

    //OMTFHits_4_p.root OMTFHits_allPt_1
    //EventsGeneratorOmtf eventsGeneratorTrain(rootFile1, rndGenerator);
    //eventsGeneratorTrain.readEvents(rootFile2);
    //eventsGeneratorTrain.readEvents(rootFile3);
    //eventsGeneratorTrain.setEventWeight(nullptr, nullptr);



    // save data to archive
    if(false){
    	std::ofstream ofs(dir + "lutNN_omtfClassifier_withPtBins.txt");
        boost::archive::text_oarchive oa(ofs);
        // write class instance to archive
        registerClasses(oa);
        oa << ptBins;
        oa << network;
        // archive and stream closed when destructors are called
    }


    //unsigned int outputCnt = ptBins.size() * 2;

    ClassifierToRegressionMaxPLut classifierToRegressionMaxPLut(ptBins, 1024);
    if(enableMaxPLut) {
        EventsGeneratorOmtf eventsGeneratorTrain(rndGenerator, outputCnt, ptBins);
        eventsGeneratorTrain.readEvents(rootFileTest1, 1); //TODO

        //eventsGeneratorTrain.setEventWeight(nullptr, nullptr);

        eventsGeneratorTrain.shuffle();
        std::vector<EventFloat*>& events = eventsGeneratorTrain.getEvents();

        for(auto& event : events) {
            //network.run(event);
            //cout<<(*event)<<endl;
            network.run(event); //todo runTraining fill the entries, find how to do this with run
        }

        classifierToRegressionMaxPLut.train(events);

    }


    EventsGeneratorOmtf eventsGeneratorTest(rndGenerator, outputCnt, ptBins);
    eventsGeneratorTest.readEvents(rootFileTest1, 1); //TODO

    eventsGeneratorTest.setEventWeight(nullptr, nullptr);

    std::vector<EventFloat*>& events = eventsGeneratorTest.getEvents();

/*    std::vector<EventFloat*> events;

    EventFloatOmtf* event = new EventFloatOmtf(18, 1, 1023.);

    event->inputs.at(0) = 64;
    event->inputs.at(1) = 37.375;
    event->inputs.at(10) = 67.5;
    event->inputs.at(11) = 56.5;

    events.push_back(event);*/

	//unsigned int branchesPerOutput = 5;

	//CostFunctionCrossEntropy  costFunction;


    cout<<"events.size() "<<events.size()<<endl;

    for(auto& event : events) {
        //network.run(event);
        //network.run(event);

        lutNetworkFP.run(event);
    }

    //exit(1);

	TFile outfile((dir + "PtCalibration.root").c_str(), "RECREATE");
	outfile.cd();

    PtCalibration ptCalibrationMaxPLut("ptCalibrationMaxPLut");
    ptCalibrationMaxPLut.train(events, classifierToRegressionMaxPLut, 0);

    ClassifierToRegressionPSum classifierToRegressionPSum04(ptBins, 0.4);

    PtCalibration ptCalibrationPSumInter04("ptCalibrationPSumInter04");
    ptCalibrationPSumInter04.train(events, classifierToRegressionPSum04, 0);


    ClassifierToRegressionPSum classifierToRegressionPSum05(ptBins, 0.5);

    PtCalibration ptCalibrationPSumInter05("ptCalibrationPSumInter05");
    ptCalibrationPSumInter05.train(events, classifierToRegressionPSum05, 0);



    ClassifierToRegressionPSum classifierToRegressionPSum035(ptBins, 0.35);
    PtCalibration ptCalibrationPSumInter035("ptCalibrationPSumInter035");
    ptCalibrationPSumInter035.train(events, classifierToRegressionPSum035, 0);


    ClassifierToRegressionPSum classifierToRegressionPSum045(ptBins, 0.45);
    PtCalibration ptCalibrationPSumInter045("ptCalibrationPSumInter045");
    ptCalibrationPSumInter045.train(events, classifierToRegressionPSum045, 0);

    ClassifierToRegressionPSum classifierToRegressionPSum055(ptBins, 0.55);
    PtCalibration ptCalibrationPSumInter055("ptCalibrationPSumInter055");
    ptCalibrationPSumInter055.train(events, classifierToRegressionPSum055, 0);


    ClassifierToRegressionMeanP classifierToRegressionMeanP(ptBins);
    PtCalibration ptCalibrationMeanP("ptCalibrationMeanP");
    ptCalibrationMeanP.train(events, classifierToRegressionMeanP, 0);

    OmtfAnalyzer omtfAnalyzer;

    omtfAnalyzer.muonAlgorithms.emplace_back(new OmtfAlgorithm(18, 0,  "omtf", kBlue));
    omtfAnalyzer.muonAlgorithms.emplace_back(new OmtfAlgorithm(20, 0,  "omtf", kBlue));
    omtfAnalyzer.muonAlgorithms.emplace_back(new OmtfAlgorithm(18, 12, "omtf_HighQ", kBlack));
    omtfAnalyzer.muonAlgorithms.emplace_back(new OmtfAlgorithm(20, 12, "omtf_HighQ", kBlack));
    omtfAnalyzer.muonAlgorithms.emplace_back(new OmtfAlgorithm(25, 12, "omtf_HighQ", kBlack));

    omtfAnalyzer.muonAlgorithms.emplace_back(new OmtfAlgorithm(10, 0, "omtf", kBlue));

    omtfAnalyzer.muonAlgorithms.emplace_back(new OmtfAlgorithm(0, 12, "omtfHighQual", kBlack));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAnalyzer.muonAlgorithms[2].get());


    omtfAnalyzer.muonAlgorithms.emplace_back(new OmtfAlgorithm(18.5, 12, "omtf_HighQ_", kGreen));
    omtfAnalyzer.muonAlgorithms.emplace_back(new OmtfAlgorithm(19, 12, "omtf_HighQ", kRed));
    omtfAnalyzer.muonAlgorithms.emplace_back(new OmtfAlgorithm(21, 12, "omtf_HighQ", kMagenta));

    omtfAnalyzer.muonAlgorithms.emplace_back(new OmtfAlgorithm(60, 12, "omtf_HighQ", kBlack));

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPAlgorithm(16, 0, "nn_maxP", kMagenta, ptBins));
/*    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierPSumAlgorithm(16, 0, "nn_PSum_04", kGreen, ptBins, 0.4));
    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierPSumAlgorithm(16, 0, "nn_PSum_05", kGreen + 1, ptBins, 0.5));

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPInterAlgorithm(14, 0, "nn_MaxPInter", kGreen + 1, ptBins, 1));
    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPInterAlgorithm(15, 0, "nn_MaxPInter", kGreen + 1, ptBins, 1));*/
    //omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPInterAlgorithm(16, 0, "nn_MaxPInter", kGreen + 1, ptBins, 1));

    MuonAlgorithm* omtfAlgo = omtfAnalyzer.muonAlgorithms[3].get();

    for(unsigned int iAlgo = 1; iAlgo < omtfAnalyzer.muonAlgorithms.size(); iAlgo++) {
    	omtfAnalyzer.muonAlgorithms[iAlgo]->algosToCompare.push_back(omtfAlgo);
    }


    omtfAlgo = omtfAnalyzer.muonAlgorithms[3].get();

/*    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierPSumAlgorithm(16, 12, "nn_PSum_04_HighQ", kGreen, ptBins, 0.4));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierPSumAlgorithm(16, 12, "nn_PSum_05_HighQ", kGreen + 1, ptBins, 0.5));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPAlgorithm(16, 12, "nn_maxP_HighQ", kMagenta, ptBins));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);


    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPInterAlgorithm(13, 12, "nn_MaxPInter_HighQ", kGreen + 1, ptBins, 1));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPInterAlgorithm(14, 12, "nn_MaxPInter_HighQ", kGreen + 1, ptBins, 1));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPInterAlgorithm(15, 12, "nn_MaxPInter_HighQ", kGreen + 1, ptBins, 1));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPInterAlgorithm(16, 12, "nn_MaxPInter_HighQ", kGreen + 1, ptBins, 1));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPInterAlgorithm(17, 12, "nn_MaxPInter_HighQ", kGreen + 1, ptBins, 1));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);*/


    omtfAlgo = omtfAnalyzer.muonAlgorithms[3].get(); //20

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPInterAlgorithm(17, 0, "nn_MaxPInter", kGreen + 1, ptBins, 1));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPInterAlgorithm(17, 12, "nn_MaxPInter_HighQ", kGreen + 1, ptBins, 1));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

/*    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPInterAlgorithm(17, 0, "nn_MaxPInter_PThres008", kGreen + 1, ptBins, 0.08));
        omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPInterAlgorithm(17, 0, "nn_MaxPInter_PThres01", kGreen + 1, ptBins, 0.1));
        omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPInterAlgorithm(17, 0, "nn_MaxPInter_PThres015", kGreen + 1, ptBins, 0.15));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPInterAlgorithm(17, 0, "nn_MaxPInter_PThres02", kGreen + 1, ptBins, 0.2));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);*/


/*    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPInterAlgorithm(18, 0, "nn_MaxPInter", kGreen + 1, ptBins, 1));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierPSumAlgorithm(18, 12, "nn_PSum_04_HighQ", kGreen, ptBins, 0.4));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierPSumAlgorithm(18, 12, "nn_PSum_05_HighQ", kGreen + 1, ptBins, 0.5));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPAlgorithm(18, 12, "nn_maxP_HighQ", kMagenta, ptBins));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);


    omtfAlgo = omtfAnalyzer.muonAlgorithms[2].get();
    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPInterAlgorithm(18, 12, "nn_MaxPInter_HighQ", kGreen + 1, ptBins, 1));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPInterAlgorithm(19, 12, "nn_MaxPInter_HighQ", kGreen + 1, ptBins, 1));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPInterAlgorithm(20, 12, "nn_MaxPInter_HighQ", kGreen + 1, ptBins, 1));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);*/


    omtfAlgo = omtfAnalyzer.muonAlgorithms[4].get();

/*
    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPInterAlgorithm(21, 0, "nn_MaxPInter", kGreen + 1, ptBins, 1));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);22

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPInterAlgorithm(22, 0, "nn_MaxPInter", kGreen + 1, ptBins, 1));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPInterAlgorithm(23, 0, "nn_MaxPInter", kGreen + 1, ptBins, 1));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);
*/

/*    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPInterAlgorithm(21, 12, "nn_MaxPInter_HighQ", kGreen + 1, ptBins, 1));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPInterAlgorithm(22, 12, "nn_MaxPInter_HighQ", kGreen + 1, ptBins, 1));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPInterAlgorithm(23, 12, "nn_MaxPInter_HighQ", kGreen + 1, ptBins, 1));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);*/



/*
    omtfAlgo = omtfAnalyzer.muonAlgorithms[2].get(); //18

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPInterLutAlgorithm(14, 0, "nn_MaxPInterLut", kGreen + 1, ptBins, classifierToRegression));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPInterLutAlgorithm(15, 0, "nn_MaxPInterLut", kGreen + 1, ptBins, classifierToRegression));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPInterLutAlgorithm(15.5, 0, "nn_MaxPInterLut", kGreen + 1, ptBins, classifierToRegression));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    //omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPInterLutAlgorithm(16, 0, "nn_MaxPInterLut", kGreen + 1, ptBins, classifierToRegression));
    //omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);
*/

    if(enableMaxPLut) {
    omtfAlgo = omtfAnalyzer.muonAlgorithms[3].get(); //20
    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(16, 0, "nn_MaxPInterLut", kGreen + 1, ptBins, classifierToRegressionMaxPLut));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(16.5, 0, "nn_MaxPInterLut_", kGreen + 1, ptBins, classifierToRegressionMaxPLut));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(17, 0, "nn_MaxPInterLut", kGreen + 1, ptBins, classifierToRegressionMaxPLut));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

/*    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPInterLutAlgorithm(18, 0, "nn_MaxPInterLut", kGreen + 1, ptBins, classifierToRegressionMaxPLut));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierMaxPInterLutAlgorithm(19, 0, "nn_MaxPInterLut", kGreen + 1, ptBins, classifierToRegressionMaxPLut));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);*/

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(20, 0, "nn_MaxPInterLut_calib", kGreen + 1, ptBins, classifierToRegressionMaxPLut, &ptCalibrationMaxPLut));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(21, 0, "nn_MaxPInterLut_calib", kGreen + 1, ptBins, classifierToRegressionMaxPLut, &ptCalibrationMaxPLut));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAlgo = omtfAnalyzer.muonAlgorithms[4].get(); //25

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(20, 0, "nn_MaxPInterLut", kGreen + 1, ptBins, classifierToRegressionMaxPLut));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(21, 0, "nn_MaxPInterLut", kGreen + 1, ptBins, classifierToRegressionMaxPLut));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(22, 0, "nn_MaxPInterLut", kGreen + 1, ptBins, classifierToRegressionMaxPLut));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(24, 0, "nn_MaxPInterLut_calib", kGreen + 1, ptBins, classifierToRegressionMaxPLut, &ptCalibrationMaxPLut));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(25, 0, "nn_MaxPInterLut_calib", kGreen + 1, ptBins, classifierToRegressionMaxPLut, &ptCalibrationMaxPLut));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAlgo = omtfAnalyzer.muonAlgorithms[5].get(); //10

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(8, 0, "nn_MaxPInterLut", kGreen + 1, ptBins, classifierToRegressionMaxPLut));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(9, 0, "nn_MaxPInterLut", kGreen + 1, ptBins, classifierToRegressionMaxPLut));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(10, 0, "nn_MaxPInterLut", kGreen + 1, ptBins, classifierToRegressionMaxPLut));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);
    }


/*
    omtfAlgo = omtfAnalyzer.muonAlgorithms[3].get(); //20
    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierPSumAlgorithm(20, 0, "nn_PSum_04", kGreen, ptBins, 0.4));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierPSumAlgorithm(20, 0, "nn_PSum_045", kGreen, ptBins, 0.45));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierPSumAlgorithm(20, 0, "nn_PSum_05", kGreen + 1, ptBins, 0.5));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAlgo = omtfAnalyzer.muonAlgorithms[4].get(); //20
    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierPSumAlgorithm(25, 0, "nn_PSum_04", kGreen, ptBins, 0.4));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierPSumAlgorithm(25, 0, "nn_PSum_045", kGreen, ptBins, 0.45));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierPSumAlgorithm(25, 0, "nn_PSum_05", kGreen + 1, ptBins, 0.5));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);
*/




    omtfAlgo = omtfAnalyzer.muonAlgorithms[3].get(); //20
    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(17, 0, "nn_PSumInter_04", kRed, ptBins, classifierToRegressionPSum04));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(18, 0, "nn_PSumInter_04", kRed, ptBins, classifierToRegressionPSum04));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(19, 0, "nn_PSumInter_04", kRed, ptBins, classifierToRegressionPSum04));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(18, 0, "nn_PSumInter_04_calib", kRed, ptBins, classifierToRegressionPSum04, &ptCalibrationPSumInter04) );
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(20, 0, "nn_PSumInter_04_calib", kRed, ptBins, classifierToRegressionPSum04, &ptCalibrationPSumInter04) );
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(21, 0, "nn_PSumInter_04_calib", kRed, ptBins, classifierToRegressionPSum04, &ptCalibrationPSumInter04) );
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(20, 12, "nn_PSumInter_04_HighQ_calib", kRed, ptBins, classifierToRegressionPSum04, &ptCalibrationPSumInter04) );
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(21, 12, "nn_PSumInter_04_HighQ_calib", kRed, ptBins, classifierToRegressionPSum04, &ptCalibrationPSumInter04) );
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(22, 0, "nn_PSumInter_04_calib", kRed, ptBins, classifierToRegressionPSum04, &ptCalibrationPSumInter04) );
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAlgo = omtfAnalyzer.muonAlgorithms[4].get(); //25
    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(20, 0, "nn_PSumInter_04", kRed, ptBins, classifierToRegressionPSum04));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(21, 0, "nn_PSumInter_04", kRed, ptBins, classifierToRegressionPSum04));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(22, 0, "nn_PSumInter_04", kRed, ptBins, classifierToRegressionPSum04));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);





    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(25, 0, "nn_PSumInter_04_calib", kRed, ptBins, classifierToRegressionPSum04, &ptCalibrationPSumInter04) );
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(24, 0, "nn_PSumInter_04_calib", kRed, ptBins, classifierToRegressionPSum04, &ptCalibrationPSumInter04) );
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAlgo = omtfAnalyzer.muonAlgorithms[5].get(); //10
    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(8, 0, "nn_PSumInter_04", kRed, ptBins, classifierToRegressionPSum04));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(8.5, 0, "nn_PSumInter_04_", kRed, ptBins, classifierToRegressionPSum04));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(9, 0, "nn_PSumInter_04", kRed, ptBins, classifierToRegressionPSum04));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(9, 0, "nn_PSumInter_04_calib", kRed, ptBins, classifierToRegressionPSum04, &ptCalibrationPSumInter04) );
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(10, 0, "nn_PSumInter_04_calib", kRed, ptBins, classifierToRegressionPSum04, &ptCalibrationPSumInter04) );
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);


    omtfAlgo = omtfAnalyzer.muonAlgorithms[7].get(); //60
    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(60, 0, "nn_PSumInter_04", kRed, ptBins, classifierToRegressionPSum04));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);


    omtfAlgo = omtfAnalyzer.muonAlgorithms[3].get(); //20
    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(17, 0, "nn_PSumInter_05", kRed, ptBins, classifierToRegressionPSum05));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(18, 0, "nn_PSumInter_05", kRed, ptBins, classifierToRegressionPSum05));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(19, 0, "nn_PSumInter_05", kRed, ptBins, classifierToRegressionPSum05));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(19, 0, "nn_PSumInter_05_calib", kRed, ptBins, classifierToRegressionPSum05, &ptCalibrationPSumInter04) );
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(20, 0, "nn_PSumInter_05_calib", kRed, ptBins, classifierToRegressionPSum05, &ptCalibrationPSumInter04) );
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);


    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(21, 0, "nn_MeanP_calib", kRed, ptBins, classifierToRegressionMeanP, &ptCalibrationMeanP) );
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(22, 0, "nn_MeanP_calib", kRed, ptBins, classifierToRegressionMeanP, &ptCalibrationMeanP) );
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);



    omtfAlgo = omtfAnalyzer.muonAlgorithms[4].get(); //25
    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(20, 0, "nn_PSumInter_05", kRed, ptBins, classifierToRegressionPSum05));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(21, 0, "nn_PSumInter_05", kRed, ptBins, classifierToRegressionPSum05));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(22, 0, "nn_PSumInter_05", kRed, ptBins, classifierToRegressionPSum05));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(23, 0, "nn_PSumInter_05_calib", kRed, ptBins, classifierToRegressionPSum05, &ptCalibrationPSumInter04) );
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(25, 0, "nn_PSumInter_05_calib", kRed, ptBins, classifierToRegressionPSum05, &ptCalibrationPSumInter04) );
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAlgo = omtfAnalyzer.muonAlgorithms[5].get(); //10
    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(8, 0, "nn_PSumInter_05", kRed, ptBins, classifierToRegressionPSum05));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(8.5, 0, "nn_PSumInter_05_", kRed, ptBins, classifierToRegressionPSum05));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(9, 0, "nn_PSumInter_05", kRed, ptBins, classifierToRegressionPSum05));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(8, 0, "nn_PSumInter_05_calib", kRed, ptBins, classifierToRegressionPSum05, &ptCalibrationPSumInter04) );
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNClassifierCalibrated(10, 0, "nn_PSumInter_05_calib", kRed, ptBins, classifierToRegressionPSum05, &ptCalibrationPSumInter04) );
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);


    omtfAnalyzer.analyse(events, rootFileTest1, outDir); //OmtfAnalysisType::regression TODO select rootFileTest1     !!!!!!!!!!!!!!!!!!!
/*    TFile outfile( (dir + "lutNN2_luts3.root").c_str(), "RECREATE");
    outfile.cd();
    printLuts3(network, 2);*/

    cout<<"line "<<dec<<__LINE__<<endl;
    return EXIT_SUCCESS;
}
