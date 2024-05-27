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

using namespace lutNN;
using namespace std;
using namespace boost::timer;

int main(void) {
	puts("Hello World!!!");

	//OmtfAnalysisType omtfAnalysisType = OmtfAnalysisType::classifier; //OmtfAnalysisType::regression;

    std::default_random_engine rndGenerator(11);

   /* string rootFile1 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/OMTFHits_pats0x0005_oldSample_files10_14.root";
    string rootFile2 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/OMTFHits_pats0x0005_oldSample_files20_24.root";
    string rootFile3 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/OMTFHits_pats0x0005_oldSample_files30_39.root";
*/
    //string rootFileTest1 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/OMTFHits_oldSample_files15.root";
    //string rootFileTest1 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/OMTFHits_pats0x0005_newerSample_files1_100.root";
    //string rootFile3 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/OMTFHits_lowPt_3_9.root";

    //string rootFile1 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/omtfHits_omtfAlgo0x0006_v1/OMTFHits_pats0x0003_oldSample_files_30_40.root";

    //string rootFileTest1 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/omtfHits/OMTFHits_pats0x0006_2_newerSample_files_1_100.root";
	string rootFileTest1 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/omtfHits_omtfAlgo0x0006_v1/OMTFHits_pats0x0003_newerSample_files_1_100.root";
    //string rootFileTest1 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/omtfHits/OMTFHits_pats0x00031_newerSample_files_1_100.root";
    //string rootFileTest1 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/omtfHits/OMTFHits_pats0x00031_newerSample_files_1_100_omtfPtCont1.root";

    //string rootFileTest1 ="/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/omtfHits/OMTFHits_pats0x0006_2_oldSample_files_40_45.root";

    //string rootFileTest1 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/omtfHits_omtfAlgo0x0006_v1/OMTFHits_pats0x0003_oldSample_files_1_10.root";

    //rootFileTest1 = rootFile1;

    LutInterNetwork network(nullptr, true);

    string dir = "../results/omtfRegression_v116_10neuronsInLayer0_8layer2_EvWeight6/";

    //string outDir = dir + "omtfAnalyzer_pats0x0006_2_oldSample";
    string outDir = dir + "omtfAnalyzer_pats0x0003_algo_0x0006_newerSample_";
    //string outDir = dir + "omtfAnalyzer_pats0x0006_";

    // create and open a character archive for output
    //std::ifstream ifs("lutNN_omtfRegression.txt");
    //std::ifstream ifs("../results/omtfRegression_v17/lutNN_omtfRegression.txt");
    //std::ifstream ifs("lutNN_omtfClassifier_38050.txt");
    //std::ifstream ifs("lutNN_omtfClassifier.txt");
    
    string fileName = dir + "lutNN_omtfRegression.txt";
    
    std::ifstream ifs(fileName);
    
    if(ifs.fail()) {
        cout<<"something wrong with the file "<<fileName<<endl;
        return 0;
    }
    
    //std::ifstream ifs(dir + "lutNN_omtfClassifier_withPtBins.txt");
    {
        cout<<"LINE:"<<__LINE__<<endl;
        cout<<fileName<<endl;
        
        boost::archive::text_iarchive ia(ifs);
        registerClasses(ia);
        ia >> network;
        // archive and stream closed when destructors are called
    }

    cout<<network<<endl;

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
/*    if(false){
    	std::ofstream ofs(dir + "lutNN_omtfClassifier_withPtBins.txt");
        boost::archive::text_oarchive oa(ofs);
        // write class instance to archive
        registerClasses(oa);
        oa << network;
        // archive and stream closed when destructors are called
    }*/


    //unsigned int outputCnt = ptBins.size() * 2;


    EventsGeneratorOmtf eventsGeneratorTest(rndGenerator, outputCnt);
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
        network.run(event); //todo runTraining fill the entries, find how to do this with run
    }

	TFile outfile((dir + "PtCalibration.root").c_str(), "RECREATE");
	outfile.cd();



    OmtfAnalyzer omtfAnalyzer;

    omtfAnalyzer.muonAlgorithms.emplace_back(new OmtfAlgorithm(18, 0,  "omtf", kBlue));
    omtfAnalyzer.muonAlgorithms.emplace_back(new OmtfAlgorithm(20, 0,  "omtf", kBlue));
    omtfAnalyzer.muonAlgorithms.emplace_back(new OmtfAlgorithm(18, 12, "omtf_HighQ", kBlack));
    omtfAnalyzer.muonAlgorithms.emplace_back(new OmtfAlgorithm(20, 12, "omtf_HighQ", kBlack));
    omtfAnalyzer.muonAlgorithms.emplace_back(new OmtfAlgorithm(25, 12, "omtf_HighQ", kBlack));

    omtfAnalyzer.muonAlgorithms.emplace_back(new OmtfAlgorithm(10, 0, "omtf", kBlue));

    omtfAnalyzer.muonAlgorithms.emplace_back(new OmtfAlgorithm(0, 12, "omtfHighQual", kBlack));
    //omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAnalyzer.muonAlgorithms[2].get());

/*
    omtfAnalyzer.muonAlgorithms.emplace_back(new OmtfAlgorithm(18.5, 12, "omtf_HighQ_", kGreen));
    omtfAnalyzer.muonAlgorithms.emplace_back(new OmtfAlgorithm(19, 12, "omtf_HighQ", kRed));
    omtfAnalyzer.muonAlgorithms.emplace_back(new OmtfAlgorithm(21, 12, "omtf_HighQ", kMagenta));*/

    omtfAnalyzer.muonAlgorithms.emplace_back(new OmtfAlgorithm(60, 12, "omtf_HighQ", kBlack));

    MuonAlgorithm* omtfAlgo = omtfAnalyzer.muonAlgorithms[3].get();

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNRegressionAlgorithm(13.5 , 0, "NN_Regression", kMagenta));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNRegressionAlgorithm(14 , 0, "NN_Regression", kMagenta));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNRegressionAlgorithm(14.5, 0, "NN_Regression", kMagenta));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNRegressionAlgorithm(15 , 0, "NN_Regression", kMagenta));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNRegressionAlgorithm(15.5, 0, "NN_Regression", kMagenta));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNRegressionAlgorithm(17.5, 0, "NN_Regression", kMagenta));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNRegressionAlgorithm(18, 0, "NN_Regression", kMagenta));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNRegressionAlgorithm(18.5, 0, "NN_Regression", kMagenta));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNRegressionAlgorithm(19, 0, "NN_Regression", kMagenta));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNRegressionAlgorithm(20, 0, "NN_Regression", kMagenta));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);


    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNRegressionAlgorithm(18 , 12, "NN_Regression_HighQ", kMagenta));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNRegressionAlgorithm(18.5, 12, "NN_Regression_HighQ", kMagenta));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNRegressionAlgorithm(19, 12, "NN_Regression_HighQ", kMagenta));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNRegressionAlgorithm(20, 12, "NN_Regression_HighQ", kMagenta));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);


    omtfAlgo = omtfAnalyzer.muonAlgorithms[5].get();
    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNRegressionAlgorithm(6 , 0, "NN_Regression", kMagenta));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNRegressionAlgorithm(7 , 0, "NN_Regression", kMagenta));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNRegressionAlgorithm(8 , 0, "NN_Regression", kMagenta));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNRegressionAlgorithm(9 , 0, "NN_Regression", kMagenta));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);

    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNRegressionAlgorithm(10, 0, "NN_Regression", kMagenta));
    omtfAnalyzer.muonAlgorithms.back()->algosToCompare.push_back(omtfAlgo);



    omtfAnalyzer.analyse(events, rootFileTest1, outDir);

/*    TFile outfile( (dir + "lutNN2_luts3.root").c_str(), "RECREATE");
    outfile.cd();
    printLuts3(network, 2);*/

    cout<<"line "<<dec<<__LINE__<<endl;
    return EXIT_SUCCESS;
}
