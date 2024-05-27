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

void initLastLutsWithAverageExpected(vector<EventFloat*>& events, LutInterNetwork& network) {

}

int main(void) {
	puts("Hello World!!!");

    std::default_random_engine rndGenerator(12);

    //string rootFile1 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/OMTFHits_pats0x0005_oldSample_files10_14.root";
    //string rootFile2 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/OMTFHits_pats0x0005_oldSample_files20_24.root";
    //string rootFile3 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/OMTFHits_pats0x0005_oldSample_files30_39.root";

    //string rootFileTest1 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/OMTFHits_oldSample_files15.root";
    //string rootFileTest1 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/OMTFHits_pats0x0005_newerSample_files1_100.root";
    //string rootFile3 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/OMTFHits_lowPt_3_9.root";

    string rootFile1 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/omtfHits_omtfAlgo0x0006_v1/OMTFHits_pats0x0003_oldSample_files_1_10.root";
    string rootFile2 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/omtfHits_omtfAlgo0x0006_v1/OMTFHits_pats0x0003_oldSample_files_15_25.root";
    string rootFile3 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/omtfHits_omtfAlgo0x0006_v1/OMTFHits_pats0x0003_oldSample_files_30_40.root";
    //string rootFile4 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/omtfHits_omtfAlgo0x0006/omtfHits/OMTFHits_pats0x0003_oldSample_files_65_95_ptUpTo13.root";


    string rootFileTest1 = "/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/test/expert/omtf/omtfHits_omtfAlgo0x0006_v1/OMTFHits_pats0x0003_newerSample_files_1_100.root";
    //rootFileTest1 = rootFile2;

    unsigned int outputCnt = 2;

    //OMTFHits_4_p.root OMTFHits_allPt_1
    EventsGeneratorOmtf eventsGeneratorTrain(rndGenerator, outputCnt);
    eventsGeneratorTrain.readEvents(rootFile1, 1);
    //eventsGeneratorTrain.readEvents(rootFile2); <<!!!!!!! TODO
    //eventsGeneratorTrain.readEvents(rootFile3);

    eventsGeneratorTrain.setEventWeight(nullptr, nullptr);

    EventsGeneratorOmtf eventsGeneratorTest(rndGenerator, outputCnt);
    eventsGeneratorTest.readEvents(rootFileTest1, 1);
    eventsGeneratorTest.setEventWeight(nullptr, nullptr);

	unsigned int batchSize = 2000; //TODO

	unsigned int iterations = 10000; //15000;
	unsigned int printEveryIteration = 100; //TODO

    unsigned int inputCnt = 18; //eventsGeneratorTrain.getInputCnt();

	unsigned int lutLayerCnt = 3; //TODO           for printing

	//unsigned int branchesPerOutput = 5;

	//double expectValMax = pow(1./log(2.5), 1.);
    double expectValMax = 200;
	double expectValMin = 0;// - expectValMax;
	double expectValRange = expectValMax - expectValMin;

	cout<<"expectValMin "<<expectValMin<<" expectValMax "<<expectValMax<<endl;

    NetworkOutNode* outputNode = new NetworkOutNode(outputCnt); //todo use uniq_ptr
    LutInterNetwork network(outputNode, true); //fixme should it be true hre?
	LayersConfigs& layersConf = network.getLayersConf();


    const int input_I = 10;
    const int input_F = 4;
    const std::size_t networkInputSize = 18;

    const int layer1_neurons = 16;
    const int layer1_lut_I = 3;
    const int layer1_lut_F = 10;

    const int layer1_output_I = 4;
    //4 bits are for the count of the noHit layers which goes to the input of the layer2, other 4 for layer1_output_I
    const int layer2_input_I = layer1_output_I + 4;

    const int layer2_neurons = 9;
    const int layer2_lut_I = 5;
    const int layer2_lut_F = 10;

    const int layer3_input_I = 5;
    const int layer3_neurons = 1;
    const int layer3_lut_I = 6;
    const int layer3_lut_F = 10;



	LayerConfig lutLayerConfig;
	lutLayerConfig.maxLutValChange = 0.2;
    lutLayerConfig.nodeType = LayerConfig::lutInter;
    lutLayerConfig.nodeInputCnt = 1;

	lutLayerConfig.bitsPerNodeInput = 10;
	lutLayerConfig.outputBits = 10; //does not matter, its flaot
	lutLayerConfig.lutRangesCnt = 8;

    LayerConfig sumLayerConfig;
    sumLayerConfig.bitsPerNodeInput = 10; //does not matter, its flaot
    sumLayerConfig.nodeInputCnt = 5;
    sumLayerConfig.outputBits = 10; //does not matter, its flaot
    sumLayerConfig.nodeType = LayerConfig::sumNode;
    sumLayerConfig.outValOffset = 0;

    //CostFunctionMeanSquaredError  costFunction;
    CostFunctionAbsoluteError costFunction;

    unsigned int lutSize = 0;

	//layer 0
    lutLayerConfig.bitsPerNodeInput = input_I;
    lutLayerConfig.nodesInLayer = inputCnt * layer1_neurons;
    lutLayerConfig.nodeInputCnt = 1;
    lutLayerConfig.lutRangesCnt = 8;
    lutLayerConfig.interpolate = true;
    //lutLayerConfig.lutBinsPerRange = (1<<lutLayerConfig.bitsPerNodeInput) / lutLayerConfig.lutRangesCnt;
    lutLayerConfig.maxLutVal =  1<<(layer1_lut_I-1);
    lutLayerConfig.minLutVal = -lutLayerConfig.maxLutVal;
    lutLayerConfig.middleLutVal = (lutLayerConfig.maxLutVal + lutLayerConfig.minLutVal)/2;

    lutSize = 1<<lutLayerConfig.bitsPerNodeInput;
    lutLayerConfig.initSlopeMax = (lutLayerConfig.maxLutVal - lutLayerConfig.minLutVal) / (lutSize/lutLayerConfig.lutRangesCnt) / 10.;
    lutLayerConfig.initSlopeMin = lutLayerConfig.initSlopeMax * 0.1;
	layersConf.push_back(make_unique<LayerConfig>(lutLayerConfig) );
	//layer 1
	sumLayerConfig.nodesInLayer = layer1_neurons;
	sumLayerConfig.nodeInputCnt = inputCnt;
	sumLayerConfig.outValOffset = 1 << (layer1_output_I-1);
	sumLayerConfig.biasShift    = layer1_output_I; //
	sumLayerConfig.shiftLastGradient = false;
	layersConf.push_back(make_unique<LayerConfig>(sumLayerConfig) );


    //layer 2
    lutLayerConfig.bitsPerNodeInput = layer2_input_I;
    lutLayerConfig.nodesInLayer = layer1_neurons * layer2_neurons;
    lutLayerConfig.nodeInputCnt = 1;
    lutLayerConfig.lutRangesCnt = 16;
    lutLayerConfig.interpolate = true;
    //lutLayerConfig.lutBinsPerRange = (1<<lutLayerConfig.bitsPerNodeInput) / lutLayerConfig.lutRangesCnt;
    lutLayerConfig.maxLutVal =  1<<(layer2_lut_I-1);
    lutLayerConfig.minLutVal = -lutLayerConfig.maxLutVal;
    lutLayerConfig.middleLutVal = (lutLayerConfig.maxLutVal + lutLayerConfig.minLutVal)/2;

    lutSize = 1<<lutLayerConfig.bitsPerNodeInput;
    lutLayerConfig.initSlopeMax = (lutLayerConfig.maxLutVal - lutLayerConfig.minLutVal) / (lutSize/lutLayerConfig.lutRangesCnt) / 20.;
    lutLayerConfig.initSlopeMin = lutLayerConfig.initSlopeMax * 0.1;
    layersConf.push_back(make_unique<LayerConfig>(lutLayerConfig) );
    //layer 3
    sumLayerConfig.nodesInLayer = layer2_neurons;
    sumLayerConfig.nodeInputCnt = layer1_neurons;
    sumLayerConfig.outValOffset = 1 << (layer3_input_I-1); //should be next layer bitsPerNodeInput
    sumLayerConfig.biasShift    = 0;
    sumLayerConfig.shiftLastGradient = false;

    layersConf.push_back(make_unique<LayerConfig>(sumLayerConfig) );

    //layer 4
    lutLayerConfig.bitsPerNodeInput = layer3_input_I;
    lutLayerConfig.nodesInLayer =  layer2_neurons; //TODO is it good?
    lutLayerConfig.nodeInputCnt = 1;
    lutLayerConfig.lutRangesCnt = 1;
    lutLayerConfig.interpolate = true;
    //lutLayerConfig.lutBinsPerRange = (1<<lutLayerConfig.bitsPerNodeInput) / lutLayerConfig.lutRangesCnt;
    lutLayerConfig.maxLutVal = expectValMax;
    lutLayerConfig.minLutVal = -lutLayerConfig.maxLutVal;
    lutLayerConfig.middleLutVal = (lutLayerConfig.maxLutVal + lutLayerConfig.minLutVal)/2;

    lutSize = 1<<lutLayerConfig.bitsPerNodeInput;
    lutLayerConfig.initSlopeMax = (lutLayerConfig.maxLutVal - lutLayerConfig.minLutVal) / (lutSize/lutLayerConfig.lutRangesCnt)/20. ;
    lutLayerConfig.initSlopeMin = lutLayerConfig.initSlopeMax * 0.1;
    layersConf.push_back(make_unique<LayerConfig>(lutLayerConfig) );

    //layer 5
    sumLayerConfig.nodesInLayer = 1; //outputCnt;
    sumLayerConfig.nodeInputCnt = 8;//layer2_neurons/outputCnt;
    sumLayerConfig.outValOffset = 0; //1<<(lutLayerConfig.bitsPerNodeInput -1);
    sumLayerConfig.biasShift    = 0;
    sumLayerConfig.shiftLastGradient = false;

    layersConf.push_back(make_unique<LayerConfig>(sumLayerConfig) );

    unique_ptr<InputNodeFactory> inputNodeFactory = make_unique<InputNodeFactoryBase>();

    if( (inputCnt + 1) != eventsGeneratorTrain.getEvents().front()->inputs.size()) {
        throw std::runtime_error("wrong inputCnt");
    }

    cout<<"building network "<<endl;
    NetworkBuilder networkBuilder(inputNodeFactory.get());
    networkBuilder.buildNet(layersConf, inputCnt, network);

    //second output for the sign
    sumLayerConfig.nodeInputCnt = 1; //it is sum node, but just with one input
    auto& lastLayer = network.getLayers().back();
    lastLayer.emplace_back(networkBuilder.getNode(&sumLayerConfig, 1));
    lastLayer.back()->setName(lastLayer.back()->getName() + "layer_"  + std::to_string(5) + "_node_" + std::to_string(1));

    //network.getOutputNode()->connectInputs(lastLayer); FIXME should be not needed, it is done in connectNet

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
    //network.initLuts(rndGenerator);


    //return 0;

    unsigned int maxNodesPerLayer = 18;
    LutNetworkPrint lutNetworkPrint;
    TCanvas* canvasLutNN = lutNetworkPrint.createCanvasLutsAndOutMap(lutLayerCnt, maxNodesPerLayer, 2);


    //vector<EventFloat*>& testEvents = eventsGeneratorTest.getEvents(); //TODO look for the number


    //vector<EventFloat*>& validationEvents = eventsGeneratorTest.getEvents(); //TODO take real set of validation events

    eventsGeneratorTest.shuffle();
    eventsGeneratorTest.shuffle();
    eventsGeneratorTest.shuffle();

    vector<EventFloat*> validationEvents(10000, nullptr);

    if(validationEvents.size() > eventsGeneratorTest.getEvents().size() )
        validationEvents.resize(eventsGeneratorTest.getEvents().size(), nullptr); //todo it better

    std::copy(eventsGeneratorTest.getEvents().begin(), eventsGeneratorTest.getEvents().begin() + validationEvents.size(), validationEvents.begin());

    vector<EventFloat*> trainingEventsBatch(batchSize);

    lutNetworkPrint.createCostHistory(iterations, printEveryIteration, 0.1, 100);

    //reset gradients, entries, etc, but no the LUT values
    network.reset();

    //eventsGeneratorTrain.getNextMiniBatch(trainingEventsBatch); //todo remove
    std::string canvasCostPng = "../pictures/canvasLutNN_canvasCostHist.png";

    LearnigParams learnigParams;
    learnigParams.learnigRate = 0.01; //200 ;
    learnigParams.lambda = 0;//0.0001;
    learnigParams.beta = 0;//0.8;
    learnigParams.smoothWeight = 0.;
    std::vector<LearnigParams> learnigParamsVec(layersConf.size(), learnigParams);

    //learnigParamsVec[0].learnigRate /= 4.;
    //learnigParamsVec[2].learnigRate *= 16.;
    //learnigParamsVec[4].learnigRate *= 0.7;

    //learnigParamsVec[0].learnigRate *= 10; //layersConf[2]->lutBinsPerRange / expectValRange ;
    //learnigParamsVec[2].learnigRate *= 100; //layersConf[4]->lutBinsPerRange / expectValRange ;
    //learnigParamsVec[4].learnigRate *= expectValRange/40;


    //learnigParamsVec[0].learnigRate = 0.1;
    //learnigParamsVec[2].learnigRate = 0.;
    learnigParamsVec[4].learnigRate = 0.02;

    learnigParamsVec[0].lambda *= 1.5;
    learnigParamsVec[2].lambda /= 6.;
    learnigParamsVec[4].lambda /= 6.;

    //learnigParamsVec[4].lambda *= 0.;

    learnigParamsVec[0].smoothWeight *= 1;
    learnigParamsVec[2].smoothWeight *= 0.4;
    learnigParamsVec[4].smoothWeight *= 1.;



    cout<<"learnigParams "<<endl;
    for(unsigned int iLayer = 0; iLayer < learnigParamsVec.size(); iLayer++ ) {
        cout<<" iLayer "<<iLayer<<" learnigRate "<<learnigParamsVec[iLayer].learnigRate<<" lambda "<<learnigParamsVec[iLayer].lambda
                <<" beta "<<learnigParamsVec[iLayer].beta<<" smoothWeight "<<learnigParamsVec[iLayer].smoothWeight<<endl;
    }

    std::vector<std::pair<double, double> > ranges;
    ranges.emplace_back(expectValMin, expectValMax);

    //lambda
    auto printValidation = [&](vector<EventFloat*> events, unsigned int iteration, double trainSampleCost) {
        network.reset();
        for(auto& event : events) {
            network.run(event);
            //network.runTraining(event, costFunction);
        }
        cout<<"line "<<dec<<__LINE__<<endl;
        lutNetworkPrint.printLuts2(network, maxNodesPerLayer);
        cout<<"line "<<dec<<__LINE__<<endl;
        double validSampleCost = lutNetworkPrint.printExpectedVersusNNOut(ranges, events, costFunction);//TODO add cost function if needed
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
            trainingEventsBatch[iEv];
            network.runTraining(trainingEventsBatch[iEv], costFunction);
            //network.runTraining(trainingEventsBatch[iEv], costFunction, 90, 3, rndGenerator);
            //network.runTraining(trainingEventsBatch[iEv], costFunction);
        }

        trainSampleCost = network.getMeanCost();

        if(i > 0) {//just to see the initial LUTs
            network.updateLuts(learnigParamsVec);
            //network.smoothLuts(learnigParamsVec);
        }

        /*if(i > 1) {
             network.smoothLuts(learnigParamsVec);
         }*/

        //cout<<"line "<<dec<<__LINE__<<endl;
        if( (i)%10000 == 0 && i > 0) {
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

        /*if(i == 200) {
            learnigParamsVec[0].learnigRate = 0.01;
            learnigParamsVec[2].learnigRate = 0.01;
            learnigParamsVec[4].learnigRate = 0.01;
        }*/
        if(i%200 == 0 && i > 0) {
            learnigParamsVec[0].lambda /= 6.;
            learnigParamsVec[2].lambda /= 4.;
            learnigParamsVec[4].lambda /= 6.;

            learnigParamsVec[0].learnigRate *= 0.8;
            learnigParamsVec[2].learnigRate *= 0.8;
            learnigParamsVec[4].learnigRate *= 0.8;
        }

       /* if(i == 500) {
            learnigParamsVec[0].lambda /= 6.;
            learnigParamsVec[2].lambda /= 4.;
            learnigParamsVec[4].lambda /= 6.;

            learnigParamsVec[0].learnigRate *= 0.5;
            learnigParamsVec[2].learnigRate *= 0.5;
            learnigParamsVec[4].learnigRate *= 0.5;
        }*/
        /*if(i == 2000) {
            learnigParamsVec[0].lambda /= 6.;
            learnigParamsVec[2].lambda /= 4.;
            learnigParamsVec[4].lambda /= 6.;

            learnigParamsVec[0].learnigRate *= 0.01;
            learnigParamsVec[2].learnigRate *= 0.01;
            learnigParamsVec[4].learnigRate *= 0.01;
        }*/
       /* if(i == 1000) {
            learnigParamsVec[0].lambda /= 6.;
            learnigParamsVec[2].lambda /= 4.;
            learnigParamsVec[4].lambda /= 6.;

            learnigParamsVec[0].learnigRate *= 0.01;
            learnigParamsVec[2].learnigRate *= 0.01;
            learnigParamsVec[4].learnigRate *= 0.01;
        }*/

        /*if(i == 10000) {
            learnigParamsVec[0].lambda /= 6.;
            learnigParamsVec[2].lambda /= 4.;
            learnigParamsVec[4].lambda /= 6.;

            learnigParamsVec[0].learnigRate *= 0.6;
            learnigParamsVec[2].learnigRate *= 0.6;
            learnigParamsVec[4].learnigRate *= 0.6;
        }

        if( i == 20000) {//i == 3000 || i == 10000 ||
            trainingEventsBatch.resize(trainingEventsBatch.size() * 2, nullptr);
        }*/
       /* if(i == 100) {
            learnigParamsVec[0].learnigRate = 0.04;
            learnigParamsVec[2].learnigRate = 0.04;
            learnigParamsVec[4].learnigRate = 0.04;


            learnigParamsVec[0].beta = 0.8;
            learnigParamsVec[2].beta = 0.8;
            learnigParamsVec[4].beta = 0.8;

            //learnigParamsVec[4].smoothWeight *= 0.1;

            //learnigParamsVec[0].lambda = 0.0001;
            //learnigParamsVec[2].lambda = 0.0001;
        }*/


        /*if(i == 20000) {
            learnigParamsVec[0].learnigRate *= 0.5;
            learnigParamsVec[2].learnigRate *= 0.5;
            learnigParamsVec[4].learnigRate *= 0.5;
        }*/
    }

    printValidation(eventsGeneratorTest.getEvents(), iterations, trainSampleCost);

    cout<<"eventsGeneratorTest.getEvents().size() "<<eventsGeneratorTest.getEvents().size()<<endl;

    string outDir = "../pictures/omtfAnalyzer";

    OmtfAnalyzer omtfAnalyzer;

    omtfAnalyzer.muonAlgorithms.emplace_back(new OmtfAlgorithm(20, 12, "omtf_HighQ", kBlue));
    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNRegressionAlgorithm(15.5, 12, "lutNN_Regression_HighQ", kMagenta));
    omtfAnalyzer.muonAlgorithms.emplace_back(new LutNNRegressionAlgorithm(15.5, 1, "lutNN_Regression", kMagenta));

    MuonAlgorithm* omtfAlgo = omtfAnalyzer.muonAlgorithms[0].get();

    for(unsigned int iAlgo = 1; iAlgo < omtfAnalyzer.muonAlgorithms.size(); iAlgo++) {
        omtfAnalyzer.muonAlgorithms[iAlgo]->algosToCompare.push_back(omtfAlgo);
    }

    omtfAnalyzer.analyse(eventsGeneratorTest.getEvents(), rootFileTest1, outDir); //OmtfAnalysisType::regression TODO select rootFileTest1     !!!!!!!!!!!!!!!!!!!

    TFile outfile("../pictures/lutNN2.root", "RECREATE");
    outfile.cd();
    canvasLutNN->Write();
    //outfile.Write();

    // create and open a character archive for output
    std::ofstream ofs("lutNN_omtfRegression.txt");
    // save data to archive
    {
        boost::archive::text_oarchive oa(ofs);
        // write class instance to archive
        registerClasses(oa);
        oa << network;
        // archive and stream closed when destructors are called
    }

    cout<<" minCost "<<minCost<<" minCostIteration "<<minCostIteration<<endl;

    cout<<"line "<<dec<<__LINE__<<endl;
    return EXIT_SUCCESS;
}
