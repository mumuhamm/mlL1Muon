/*
 * gmtBinaryLutNNTest.cpp
 *
 *  Created on: Dec 28, 2021
 *      Author: kbunkow
 */


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

#include "lutNN/lutNN2/interface/BinaryLutNetwork.h"
#include "lutNN/lutNN2/interface/NetworkBuilder.h"
#include "lutNN/lutNN2/interface/LutInter.h"
#include "lutNN/lutNN2/interface/NetworkOutNode.h"
#include "lutNN/lutNN2/interface/EventsGeneratorOmtf.h"
#include "lutNN/lutNN2/interface/Utlis.h"
#include "lutNN/lutNN2/interface/NetworkSerialization.h"
#include "lutNN/lutNN2/interface/EventsGeneratorGmt.h"

#include "lutNN/lutNN2/interface/GmtAnalyzer.h"

using namespace lutNN;
using namespace std;
using namespace boost::timer;

int main(void) {
    puts("Hello World!!!");


    std::default_random_engine rndGenerator(11);


    //BinaryLutNetwork network;
    LutInterNetwork network;

    bool useHitsValid = false; //TODO!!!!!!!!!!!!!!!!!!!!!!! must be the same as during the trainig!!!!!!!!!!!!!!!!!!!!!!!!
    //bool addNoHitCntToInputs = true;
    bool addNoHitCntToInputs = false;

    bool useMargin = false;

    //string dir = "../results/gmtBinaryLutNN_v100/";
    //string dir = "../results/gmtBinaryLutNN_v103_withHitsValid/";
    //string dir = "../results/gmtBinaryLutNN_v103/";
    //string dir = "../results/gmtBinaryLutNN_v104_withHitsValidInPhiAndEta/";
    //string dir = "../results/gmtBinaryLutNN_v104_10branches/";
    //string dir = "../results/gmtBinaryLutNN_v105_10branches_withMoreCurvBits/";
    //string dir = "../results/gmtBinaryLutNN_v105_10branches_with_tttBendChi2/";
    //string dir = "../results/gmtBinaryLutNN_v105_10branches_with_tttBendChi2_intSumNode/";
    //string dir = "../results/gmtBinaryLutNN_v106_10branches/";

    //string dir = "../results/gmtBinaryLutNN_v107_10branches_hinge/";

    //string dir = "../results/gmtBinaryLutNN_v107_10branches_minSquare/";
    //string dir = "../results/gmtBinaryLutNN_v109_5branches_minSquare/";

    //string dir = "../results/gmtLutInterNN_v108_7Neurons_1/";


    string dir = "../results/gmtBinaryLutNN_v123_hinge/";


    std::string outFilePath = dir;

    // create and open a character archive for output

    string fileName = dir + "gmtBinaryLutNN1.txt";
    //string fileName = dir + "gmtBinaryLutNN1_test.txt";

    if(dir.find("LutInter") != string::npos) {
        fileName = dir + "gmtLutInterNN1.txt";
        addNoHitCntToInputs = true;
    }

    std::ifstream ifs(fileName);
    if(ifs.fail()) {
        cout<<"something wrong with the file "<<fileName<<endl;
        return 0;
    }

    /*std::array<char, 20> a;
    ifs.getline(&a[0], 20);
    cout<<&a[0]<<endl;;*/
    {
        cout<<fileName<<endl;
        boost::archive::text_iarchive ia(ifs);
        registerClasses(ia);
        cout<<"LINE:"<<__LINE__<<endl;
        ia >> network;
        // archive and stream closed when destructors are called
    }

    cout<<network<<endl;

    cout<<"addNoHitCntToInputs "<<addNoHitCntToInputs<<std::endl;
    unsigned int outputCnt = network.getOutputNode()->getOutputValues().size();
    EventsGeneratorGmt eventsGeneratorTest(rndGenerator, outputCnt, useHitsValid, addNoHitCntToInputs);

    //string gmtDir = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_11_x_x_l1tOfflinePhase2/CMSSW_11_1_7/src/L1Trigger/Phase2L1GMT/test/";
    //std::string outFileName = "test.root";

    //string gmtDir = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_11_x_x_l1tOfflinePhase2/CMSSW_11_1_7/src/usercode/MuCorrelatorAnalyzer/crab/crab_Phase2L1GMT_org_MC_analysis_DYToLL_M-50_Summer20_PU200_t204/results/";
    //std::string outFileName = "out_DYToLL.root";
/*
    string gmtDir = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_11_x_x_l1tOfflinePhase2/CMSSW_11_1_7/src/usercode/MuCorrelatorAnalyzer/crab/crab_Phase2L1GMT_org__MC_analysis_JPsiToMuMu_Summer20_PU200_t207/results/";
    std::string signalSample = "JPsiToMuMu";

    eventsGeneratorTest.readEvents(gmtDir  + "muCorrelatorTTAnalysis1_8.root");
    eventsGeneratorTest.readEvents(gmtDir  + "muCorrelatorTTAnalysis1_9.root");
    eventsGeneratorTest.readEvents(gmtDir  + "muCorrelatorTTAnalysis1_10.root");*/


    string gmtDir = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_11_x_x_l1tOfflinePhase2/CMSSW_11_1_7/src/usercode/MuCorrelatorAnalyzer/crab/crab_Phase2L1GMT_MC_analysis_TauTo3Mu_Summer20_PU200_withNewMB_t207/results/";
    std::string signalSample = "TauTo3Mu";
    eventsGeneratorTest.readEvents(gmtDir  + "muCorrelatorTTAnalysis1_10.root");
    eventsGeneratorTest.readEvents(gmtDir  + "muCorrelatorTTAnalysis1_11.root");
    eventsGeneratorTest.readEvents(gmtDir  + "muCorrelatorTTAnalysis1_12.root");
    eventsGeneratorTest.readEvents(gmtDir  + "muCorrelatorTTAnalysis1_13.root");
    eventsGeneratorTest.readEvents(gmtDir  + "muCorrelatorTTAnalysis1_14.root");
    eventsGeneratorTest.readEvents(gmtDir  + "muCorrelatorTTAnalysis1_15.root");


    //eventsGeneratorTest.readEvents(gmtDir  + "muCorrelatorTTAnalysis1.root"); //test

//    eventsGeneratorTest.readEvents(gmtDir  + "muCorrelatorTTAnalysis1_1.root");
//    eventsGeneratorTest.readEvents(gmtDir  + "muCorrelatorTTAnalysis1_2.root");
//    eventsGeneratorTest.readEvents(gmtDir  + "muCorrelatorTTAnalysis1_3.root");
//    eventsGeneratorTest.readEvents(gmtDir  + "muCorrelatorTTAnalysis1_4.root");

    //eventsGeneratorTest.readEvents(gmtDir  + "muCorrelatorTTAnalysis1_8.root");
    //eventsGeneratorTest.readEvents(gmtDir  + "muCorrelatorTTAnalysis1_9.root");

    auto& events = eventsGeneratorTest.getEvents();


    cout<<"events.size() "<<events.size()<<endl;

    for(auto& event : events) {
        //network.run(event);
        network.LutNetworkBase::run(event);
    }

    if(dynamic_cast<SoftMax*>(network.getOutputNode()) != 0)
        useMargin = false;
    else
        useMargin = true;

    GmtAnalyzer gmtAnalyzer(outFilePath + signalSample + "_results.root");

    gmtAnalyzer.analyze(events, signalSample, false, useMargin);

    eventsGeneratorTest.clear();

    gmtDir = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_11_x_x_l1tOfflinePhase2/CMSSW_11_1_7/src/usercode/MuCorrelatorAnalyzer/crab/crab_Phase2L1GMT_MC_analysis_MinBias_Summer20_PU200_t207/results/";
    std::string falseSampl = "MinBias";

    eventsGeneratorTest.readEvents(gmtDir  + "muCorrelatorTTAnalysis1_11.root");
    events = eventsGeneratorTest.getEvents();

    for(auto& event : events) {
        network.LutNetworkBase::run(event);
    }
    gmtAnalyzer.analyze(events, falseSampl, true, useMargin);

    gmtAnalyzer.makeRocCurve(signalSample, falseSampl);
}

