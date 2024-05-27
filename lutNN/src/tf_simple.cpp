//============================================================================
// Name        : tf.cpp
// Author      : Karol Bunkowski
// Version     :
// Copyright   : All rights reserved
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <time.h>
#include <random>

#include "TH2I.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TStyle.h"
#include "THStack.h"
#include "TText.h"
#include "TGraph.h"
#include "TFitResultPtr.h"
#include "TFitResult.h"
#include "TPaveLabel.h"

#include "lutNN/interface/Node.h"
#include "lutNN/interface/LutNetwork.h"
#include "lutNN/interface/Utlis.h"

#include "boost/dynamic_bitset.hpp"

using namespace lutNN;
using namespace std;

vector<double> planeY = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
};

class EventsGeneratorTracks: public EventsGeneratorBase {
private:
    unsigned int inputLutSize;
    std::vector<std::pair<double, double> >& ranges;
    std::default_random_engine& generator;
public:
    EventsGeneratorTracks(unsigned int inputLutSize, std::vector<std::pair<double, double> >& ranges,  std::default_random_engine& generator):
        EventsGeneratorBase(), inputLutSize(inputLutSize), ranges(ranges), generator(generator) {}

    virtual void generateEvents(vector<Event>& events);
};

void EventsGeneratorTracks::generateEvents(vector<Event>& events) {
    //std::uniform_int_distribution<int> a0dist(ranges[0].first, ranges[0].second);
    //std::uniform_int_distribution<int> a0dist(60, inputLutSize-60);
    std::uniform_real_distribution<> a0dist(ranges[0].first, ranges[0].second); //TODO!!!!!!!!!!!!!!!!!!!!!
    std::uniform_real_distribution<> a1dist(ranges[1].first, ranges[1].second);
    std::uniform_real_distribution<> a2dist(ranges[2].first, ranges[2].second);
    std::uniform_int_distribution<int> hitProbDist(0, 100);
    double a0 = 0;
    double a1 = 0;
    double a2 = 0;

    boost::dynamic_bitset<> firedPlanes(events[0].inputs.size() - 1);
    for(auto& event : events) {
        a0 = a0dist(generator);
        a1 = a1dist(generator);
        a2 = a2dist(generator);
        //a0 = inputLutSize/2; //TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
        event.expextedResult[0] = a2;//a1;//a0;

        //event.expextedResult[1] = a1;
        //event.expextedResult[2] = a2;
        //cout<<"Event "<<"a0 "<<a0<<" a1 "<<a1;
        //the a0, a1, a3 should have the same range for proper training in the lut
        //so here we recalculate such thay have proper value for polynomial
        /*a0 = ( 0.5 * a0 - ranges[0].first) * inputLutSize / (ranges[0].second - ranges[0].first);
        a1 = a1 * 10. / (ranges[1].second - ranges[1].first);
        a2 = a2 * 0.4/ (ranges[2].second - ranges[2].first);*/

        a0 = inputLutSize/4. + 0.5 * a0 * inputLutSize;
        a1 = (a1 - 0.5) * 10.;
        a2 = (a2 - 0.5) * 0.3;

        int hitCnt = 0;
        firedPlanes.reset();
        for(unsigned int i = 0; i < event.inputs.size() -1; i++) {
            int hitProb = hitProbDist(generator);
            int x = round(a0 + a1 * planeY[i] + a2 * planeY[i] * planeY[i]); //
            if(x > 0 && x < inputLutSize && hitProb < 80) {//  && hitProb < 80//TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                event.inputs[i] = x;
                hitCnt++;
                firedPlanes[i] = true;
            }
            else
                event.inputs[i] = 0;
            //cout<<" in"<<i<<" "<<x;
        }
        event.inputs[event.inputs.size() - 1] = firedPlanes.to_ulong();
        //string str;
        //boost::to_string(firedPlanes, str);
        //cout<<" str "<<str<<" firedPlanes.to_ulong() "<<firedPlanes.to_ulong()<<" input "<<event.inputs[event.inputs.size() - 1]<<std::endl;;
        //event.inputs[event.inputs.size() -1] = hitCnt;
        //cout<<endl;

    }
}


void makeFit(TPad& padOutMap,  std::vector<std::pair<double, double> >& ranges, std::vector<Event>& events, unsigned int inputLutSize) {
    vector<TH2F*> hists;
    unsigned int inCnt =  events[0].inputs.size() -1;//TODO!!!!!!!!!!!!!!!!!!
    unsigned int outCnt =  events[0].expextedResult.size();
    for(unsigned int iOut = 0; iOut < outCnt; iOut++) {
        ostringstream ostr;
        ostr<<"hist_RootFit_iOut_"<<iOut;
        string histName = ostr.str();;

        TH2F* hist = new TH2F(histName.c_str(), histName.c_str(), 50, ranges[iOut].first, ranges[iOut].second, 50, ranges[iOut].first, ranges[iOut].second);
        cout<<__FUNCTION__<<":"<<__LINE__<<" creating hist "<<histName<<endl;
        hists.push_back(hist);
    }

    double* x = new double[inCnt];
    double* y = new double[inCnt];


    for(auto& event : events) {
        unsigned int iPoint = 0;
        for(unsigned int iIn = 0; iIn < inCnt; iIn++) {
            if(event.inputs[iIn]) {
                y[iPoint] = event.inputs[iIn];
                x[iPoint] = planeY[iIn];
                iPoint++;
            }
        }

        TGraph* gr = new TGraph(iPoint, x, y);

        TFitResultPtr fitResult = gr->Fit("pol2","S");
        Int_t fitStatus = fitResult;
        if(fitStatus == 0) {
            /*for(unsigned int iOut = 0; iOut < outCnt; iOut++) {
                hists[iOut]->Fill(event.expextedResult[iOut], fitResult->Value(iOut +1) ); //TODO remove +1!!!!!!!!!!!!!!!!!!
            }*/

            //inputLutSize/4. + 0.5 * a0 * inputLutSize;
            //hists[0]->Fill(event.expextedResult[0], (fitResult->Value(0) -inputLutSize/4.) / (0.5 * inputLutSize) ); //TODO a1 watch aout for the indexes!!!!
            //hists[0]->Fill(event.expextedResult[0], fitResult->Value(1)  / 10. + 0.5); //TODO a1 watch aout for the indexes!!!!
            hists[0]->Fill(event.expextedResult[0], fitResult->Value(2)  / 0.3 + 0.5); //TODO a2 watch aout for the indexes!!!!
            //hists[0]->Fill(event.expextedResult[0], fitResult->Value(2) * (ranges[2].second - ranges[2].first) /0.4); //TODO watch aout for the indexes!!!!

            fitResult->Clear();
        }

        //cout<<" neuron.getOutAddr(): "<<neuron.getOutAddr()<<endl;

        gr->Delete();

    }

    for(unsigned int iOut = 0; iOut < outCnt; iOut++) {
        padOutMap.cd(iOut + 1)->SetGrid();
        hists[iOut]->SetStats(0);
        hists[iOut]->Draw("colz");
    }
}

TH1F* createCostHistory(int iterations, int printEveryIteration) {
    int bins = iterations / printEveryIteration;
    return  new TH1F("costHistory", "costHistory", bins, -0.5, iterations - 0.5);
}

void updateCostHistory(TPad& padRateHistory, TH1F* costHist, int iteration, double cost) {
    padRateHistory.cd()->SetGrid();
    costHist->Fill(iteration, cost);
    costHist->SetStats(0);
    costHist->Draw("hist");
}

int main(void) {
    //puts("Hello World!!!");

    /*	std::shared_ptr<ConfigParameters> config(new ConfigParameters() );
	config->lutAddrOffset = 0;
	cout<<"config: lutSize "<<config->lutSize<<" lutAddrOffset "<<config->lutAddrOffset<<endl;*/

    unsigned int inputCnt = 11;
    unsigned int outputCnt = 1;
    unsigned int lutSize = (1<<8) -1;
    cout<<"lutSize "<<lutSize<<endl;
    std::vector<std::shared_ptr<ConfigParameters> > layersDef;
    layersDef.emplace_back(ConfigParametersPtr(new ConfigParameters(50, lutSize, lutSize, inputCnt, ConfigParameters::fullyConnected) ) );
    //layersDef.emplace_back(ConfigParametersPtr(new ConfigParameters(3, lutSize, lutSize, 7, ConfigParameters::fullyConnected) ) );
    //layersDef.emplace_back(ConfigParametersPtr(new ConfigParameters(2, lutSize, lutSize/2, 2) ) );
    //layersDef.emplace_back(ConfigParametersPtr(new ConfigParameters(20, lutSize, lutSize, 4, ConfigParameters::fullyConnected) ) );
    layersDef.emplace_back(ConfigParametersPtr(new ConfigParameters(outputCnt, lutSize, 0, 4, ConfigParameters::fullyConnected) ) );
    //layersDef.emplace_back(ConfigParametersPtr(new ConfigParameters(outputCnt, 2*lutSize, lutSize/2, 1, ConfigParameters::singleLut) ) );

    //TODO there is mistake in stretching the neuron output values, it does not work well when the lutSize is different in each layer - check if it is true

    //making sure the values of neuronInputCnt are correct
    for(unsigned int iLayer = 0; iLayer < layersDef.size(); iLayer++ ) {
        if(iLayer == 0)
            layersDef[iLayer]->neuronInputCnt = inputCnt;
        else {
            if(layersDef[iLayer]->layerType == ConfigParameters::fullyConnected)
                layersDef[iLayer]->neuronInputCnt = layersDef[iLayer - 1]->neuronsInLayer;
            else if(layersDef[iLayer]->layerType == ConfigParameters::singleLut)
                layersDef[iLayer]->neuronInputCnt = 1;
        }

        if(iLayer < layersDef.size() -1) {
            layersDef[iLayer]->nextLayerLutSize = layersDef[iLayer + 1]->lutSize;
        }
    }

    LutNetwork lutNetwork(inputCnt, layersDef);

    std::cout<<lutNetwork<<std::endl;
    std::default_random_engine generator(11);
    lutNetwork.initLuts(generator);
    //lutNetwork.initLuts();

    unsigned int maxNodesPerLayer = 0;
    for(unsigned int iLayer = 0; iLayer < lutNetwork.getLayers().size(); iLayer++ ) {
        if(maxNodesPerLayer < lutNetwork.getLayers()[iLayer].size() )
            maxNodesPerLayer = lutNetwork.getLayers()[iLayer].size();
    }
    maxNodesPerLayer = 3;
    cout<<"maxNodesPerLayer "<<maxNodesPerLayer<<endl;

    //--------------------------------------------------
    gStyle->SetOptStat(0);
    TCanvas canvasLutNN("canvasLutNN", "canvasLutNN", 1.5* 1900, 1.5* 900);
    canvasLutNN.cd();
    TPad padLuts("padLuts", "padLuts", 0.0, 0.0, 0.7, 1.);
    padLuts.SetGrid();
    padLuts.Divide(lutNetwork.getLayers().size(), maxNodesPerLayer * 2);
    padLuts.Draw();

    TPad padOutMap("padOutMap", "padOutMap", 0.7, 0., 1., 1.);
    padOutMap.SetRightMargin(0.2);
    padOutMap.SetGrid();
    padOutMap.Divide(2, 2); //todo increase if more nn outputs then 4
    padOutMap.Draw();

    TCanvas canvasGradients("canvasGradients", "canvasGradients", 1.5* 1900, 1.5* 900);
    canvasGradients.cd();
    TPad padGradients("padGradients", "padGradients", 0.0, 0.0, 1., 1.); //0.0, 0.0, 0.7, 1.
    padGradients.SetGrid();
    padGradients.Divide(lutNetwork.getLayers().size(), maxNodesPerLayer * 2);
    padGradients.Draw();

    TFile outfile("../pictures/lutNN.root", "RECREATE");
    outfile.cd();

    printLuts2(padLuts, padGradients, lutNetwork, maxNodesPerLayer);

    canvasLutNN.SaveAs("../pictures/canvasLutNN_0.png");

    LutNetwork::LearnigParams learnigParams;
    learnigParams.learnigRate = 0;//0.01 ;
    learnigParams.stretchRate = 0.2 *  0.05;
    learnigParams.regularizationRate = 0;
    std::vector<LutNetwork::LearnigParams> learnigParamsVec(layersDef.size(), learnigParams);

    //learnigRates[0] = 0.02;
    //learnigRates[1] = 0.02;

    TPaveLabel textCosts(0.7, .01, 1., 0.05, "aaaa");
    canvasLutNN.cd();
    ostringstream ostr;
    ostr<<"cost";
    textCosts.Draw();

    unsigned int batchSize = 10000; //TODO
    vector<Event> trainingEvents(batchSize, {inputCnt, outputCnt} );
    vector<Event> testEvents(10000, {inputCnt, outputCnt} );


    std::vector<std::pair<double, double> > ranges =
    {
            {0. , 1 },
            {0. , 1 },
            {0. , 1 }
    };

    std::uniform_int_distribution<int> distribution(0,9);
    int mode = 0;
    int iterations = 1000; //TODO
    int printEveryIteration = 100; //TODO
    bool stochastic = true; //TODO
    TH1F* costHist = createCostHistory(iterations, printEveryIteration);
    TCanvas canvasCostHist("canvasCostHist", "canvasCostHist", 1400, 800);
    canvasCostHist.SetLogy();

    EventsGeneratorTracks eventsGenerator(lutSize, ranges, generator);

    eventsGenerator.generateEvents(testEvents);

    //########################### printNN lambda ###########################
    auto printNN = [&](int iteration, vector<Event>& events) {
        //eventsGenerator.generateEvents(events);

        printLuts2(padLuts, padGradients, lutNetwork, maxNodesPerLayer);
        printGradinets(padGradients, lutNetwork, maxNodesPerLayer * 2);

        for(auto& event : events) {
            lutNetwork.run(event);
            //std::cout<<" "<<std::endl;
        }
        /*cout<<"printing event 0."<<endl;
        events[0].print();*/


        printExpectedVersusNNOut(padOutMap, ranges, events);
        ostr.str("");
        ostr<<"iteration "<<setw(7)<<iteration<<" mean cost "<<setw(5)<<lutNetwork.getMeanCost();
        textCosts.SetLabel(ostr.str().c_str() );

        padLuts.Update();
        padOutMap.Update();
        canvasLutNN.Update();
        canvasLutNN.SaveAs( ("../pictures/luts/canvasLutNN_" + to_string(iteration+1) + ".png").c_str() );

        padGradients.Update();
        canvasGradients.Update();
        canvasGradients.SaveAs( ("../pictures/gradients/canvasGradients_" + to_string(iteration+1) + ".png").c_str() );

        updateCostHistory(canvasCostHist, costHist, iteration, lutNetwork.getMeanCost());
        canvasCostHist.Update();
        canvasCostHist.SaveAs( "../pictures/canvasLutNN_canvasCostHist.png" );
    };
    //######################################################

    int iteration = -20;

    //scaling and shifting the initial LUT contents so that the layer outputs fill the next layer input address
    for(unsigned int iLayer = 0; iLayer < lutNetwork.getLayers().size(); iLayer++) {
        double shiftRatio = 1;
        double scaleRatio = 1;

        eventsGenerator.generateEvents(trainingEvents);
        for(auto& event : trainingEvents) {
            lutNetwork.setInputs(event);
            //lutNetwork.runTrainingInter(event.expextedResult); //fills the statistics
            lutNetwork.runTraining(event.expextedResult, 0); //fills the statistics
            //std::cout<<" "<<std::endl;
        }
        printNN(iteration, trainingEvents);

        if(iLayer < lutNetwork.getLayers().size() -1)
            lutNetwork.shiftAndRescale(iLayer, shiftRatio, scaleRatio);

        lutNetwork.resetStats();
        iteration++;
    }

    for(int i = 0; i < iterations; i++) {
       /* if(i%10 == 0)
            std::cout<<"runTraining iteration "<<i<<std::endl;*/
        //these iterations will just capture the range of the expected values (i.e. the network outputs)

        /*if(i == -1) { //not sure if needed
            for(unsigned int iLayer = 0; iLayer < learnigParamsVec.size(); iLayer++ ) {
                auto& learnigParams = learnigParamsVec[iLayer];
                learnigParams.learnigRate = 0.1 ;
                learnigParams.stretchRate = 0.;// * learnigParams.learnigRate;
                learnigParams.regularizationRate = 0;
                if(iLayer == 0)
                    learnigParams.regularizationRate = 0.05 * learnigParams.learnigRate;
                else
                    learnigParams.regularizationRate = 1.5 * learnigParams.learnigRate;
            }
            mode = 1; //in mode 1 only the LUTs in the last layer are updated
        }*/

        //here normal trainig starts
       /* if(i == 0) { //not sure if needed
            for(unsigned int iLayer = 0; iLayer < learnigParamsVec.size(); iLayer++ ) {
                auto& learnigParams = learnigParamsVec[iLayer];
                learnigParams.learnigRate =0.01 ;
                if(iLayer == learnigParamsVec.size() -1) {
                    learnigParams.regularizationRate = 0.2;//0.1;// * learnigParams.learnigRate;
                }
                else {
                    learnigParams.regularizationRate = 0;// * learnigParams.learnigRate;
                    learnigParams.learnigRate = 0.;;
                }
            }
            mode = 0;
        }
        if(i == 80) { //not sure if needed
            for(unsigned int iLayer = 0; iLayer < learnigParamsVec.size(); iLayer++ ) {
                auto& learnigParams = learnigParamsVec[iLayer];
                learnigParams.learnigRate =0. ;
                if(iLayer == learnigParamsVec.size() -1) {
                    learnigParams.regularizationRate = 0.2;//0.1;// * learnigParams.learnigRate;
                }
                else {
                    learnigParams.regularizationRate = 0;// * learnigParams.learnigRate;
                    learnigParams.learnigRate = 0.;;
                }
            }
            mode = 0;
        }
        else*/
        //for 2 layers network
        if(i == 0) { //not sure if needed
            for(unsigned int iLayer = 0; iLayer < learnigParamsVec.size(); iLayer++ ) {
                auto& learnigParams = learnigParamsVec[iLayer];
                if(iLayer == learnigParamsVec.size() -1) {
                    learnigParams.learnigRate = 0.0005 ;
                    learnigParams.regularizationRate = 0.1;//0.1;// * learnigParams.learnigRate;
                }
                else {
                    learnigParams.learnigRate = 0.5;//0.5;//2.;
                    learnigParams.regularizationRate = 0;// * learnigParams.learnigRate;
                }
            }
            mode = 0;
        }

        else if(i == 2000) { //not sure if needed
            for(unsigned int iLayer = 0; iLayer < learnigParamsVec.size(); iLayer++ ) {
                auto& learnigParams = learnigParamsVec[iLayer];
                if(iLayer == learnigParamsVec.size() -1) {
                    learnigParams.learnigRate = 0.001 ;
                    learnigParams.regularizationRate = 0.1;// * learnigParams.learnigRate;
                }
                else {
                    learnigParams.learnigRate = 2.;
                    learnigParams.regularizationRate = 0;// * learnigParams.learnigRate;
                }
            }
            mode = 0;
        }
        else if(i == 10000) { //not sure if needed
            for(unsigned int iLayer = 0; iLayer < learnigParamsVec.size(); iLayer++ ) {
                auto& learnigParams = learnigParamsVec[iLayer];
                if(iLayer == learnigParamsVec.size() -1) {
                    learnigParams.learnigRate = 0.005 ;
                    learnigParams.regularizationRate = 0.01;// * learnigParams.learnigRate;
                }
                else {
                    learnigParams.learnigRate = 4.;
                    learnigParams.regularizationRate = 0;// * learnigParams.learnigRate;
                }
            }
            mode = 0;
        }
        else if(i == 15000) { //not sure if needed
            for(unsigned int iLayer = 0; iLayer < learnigParamsVec.size(); iLayer++ ) {
                auto& learnigParams = learnigParamsVec[iLayer];
                if(iLayer == learnigParamsVec.size() -1) {
                    learnigParams.learnigRate = 0.01 ;
                    learnigParams.regularizationRate = 0.001;// * learnigParams.learnigRate;
                }
                else {
                    learnigParams.learnigRate = 4.;
                    learnigParams.regularizationRate = 0;// * learnigParams.learnigRate;
                }
            }
            mode = 0;
        }
        else if(i == 20000) { //not sure if needed
            for(unsigned int iLayer = 0; iLayer < learnigParamsVec.size(); iLayer++ ) {
                auto& learnigParams = learnigParamsVec[iLayer];
                if(iLayer == learnigParamsVec.size() -1) {
                    learnigParams.learnigRate = 0.01 ;
                    learnigParams.regularizationRate = 0;//0.001;// * learnigParams.learnigRate;
                }
                else {
                    learnigParams.learnigRate = 4.;
                    learnigParams.regularizationRate = 0;// * learnigParams.learnigRate;
                }
            }
            mode = 0;
        }
 //for 3 layer network
/*        if(i == 0) { //not sure if needed
            for(unsigned int iLayer = 0; iLayer < learnigParamsVec.size(); iLayer++ ) {
                auto& learnigParams = learnigParamsVec[iLayer];
                if(iLayer == 2) {
                    learnigParams.learnigRate = 0.0001 ;
                    learnigParams.regularizationRate = 0;//0.1;// * learnigParams.learnigRate;
                }
                else if(iLayer == 1) {
                    learnigParams.learnigRate = 1;
                    learnigParams.regularizationRate = 0;//0.01;// * learnigParams.learnigRate;
                }
                else if(iLayer == 0) {
                    learnigParams.learnigRate = 1.;
                    learnigParams.regularizationRate = 0;// * learnigParams.learnigRate;
                }
            }
            mode = 0;
        }
        else if(i == 5000) { //not sure if needed
            for(unsigned int iLayer = 0; iLayer < learnigParamsVec.size(); iLayer++ ) {
                auto& learnigParams = learnigParamsVec[iLayer];
                if(iLayer == 2) {
                    learnigParams.learnigRate = 0.001 ;
                    learnigParams.regularizationRate = 0;//0.1;// * learnigParams.learnigRate;
                }
                else if(iLayer == 1) {
                    learnigParams.learnigRate = 2.;
                    learnigParams.regularizationRate = 0;//0.01;// * learnigParams.learnigRate;
                }
                else if(iLayer == 0) {
                    learnigParams.learnigRate = 2.;
                    learnigParams.regularizationRate = 0;// * learnigParams.learnigRate;
                }
            }
            mode = 0;
        }
        else if(i == 10000) { //not sure if needed
            for(unsigned int iLayer = 0; iLayer < learnigParamsVec.size(); iLayer++ ) {
                auto& learnigParams = learnigParamsVec[iLayer];
                if(iLayer == 2) {
                    learnigParams.learnigRate = 0.02 ;
                    learnigParams.regularizationRate = 0;//0.1;// * learnigParams.learnigRate;
                }
                else if(iLayer == 1) {
                    learnigParams.learnigRate = 4.;
                    learnigParams.regularizationRate = 0;//0.01;// * learnigParams.learnigRate;
                }
                else if(iLayer == 0) {
                    learnigParams.learnigRate = 4.;
                    learnigParams.regularizationRate = 0;// * learnigParams.learnigRate;
                }
            }
            mode = 0;
        }
        else if(i == 20000) { //not sure if needed
            for(unsigned int iLayer = 0; iLayer < learnigParamsVec.size(); iLayer++ ) {
                auto& learnigParams = learnigParamsVec[iLayer];
                if(iLayer == 2) {
                    learnigParams.learnigRate = 0.04 ;
                    learnigParams.regularizationRate = 0;//0.1;// * learnigParams.learnigRate;
                }
                else if(iLayer == 1) {
                    learnigParams.learnigRate = 8.;
                    learnigParams.regularizationRate = 0;//0.01;// * learnigParams.learnigRate;
                }
                else if(iLayer == 0) {
                    learnigParams.learnigRate = 8.;
                    learnigParams.regularizationRate = 0;// * learnigParams.learnigRate;
                }
            }
            mode = 0;
        }*/
        /*
        else if(i == 300) { //not sure if needed
            for(unsigned int iLayer = 0; iLayer < learnigParamsVec.size(); iLayer++ ) {
                auto& learnigParams = learnigParamsVec[iLayer];
                if(iLayer == 2) {
                    learnigParams.learnigRate = 0.01 ;
                    learnigParams.regularizationRate = 0.1;// * learnigParams.learnigRate;
                }
                else if(iLayer == 1) {
                    learnigParams.regularizationRate = 0.01;// * learnigParams.learnigRate;
                    learnigParams.learnigRate = 4.;
                }
                else if(iLayer == 0) {
                    learnigParams.regularizationRate = 0;// * learnigParams.learnigRate;
                    learnigParams.learnigRate = 1.;
                }
            }
            mode = 0;
        }
        else if(i == 1000) { //not sure if needed
            for(unsigned int iLayer = 0; iLayer < learnigParamsVec.size(); iLayer++ ) {
                auto& learnigParams = learnigParamsVec[iLayer];
                if(iLayer == 2) {
                    learnigParams.learnigRate = 0.01 ;
                    learnigParams.regularizationRate = 0.001;// * learnigParams.learnigRate;
                }
                else if(iLayer == 1) {
                    learnigParams.regularizationRate = 0.001;// * learnigParams.learnigRate;
                    learnigParams.learnigRate = 4.;
                }
                else if(iLayer == 0) {
                    learnigParams.regularizationRate = 0;// * learnigParams.learnigRate;
                    learnigParams.learnigRate = 1.;
                }
            }
            mode = 0;
        }*/

        /*else if(i == 1000) { //not sure if needed
            for(unsigned int iLayer = 0; iLayer < learnigParamsVec.size(); iLayer++ ) {
                auto& learnigParams = learnigParamsVec[iLayer];
                learnigParams.learnigRate = 0.02 ;
                if(iLayer == learnigParamsVec.size() -1) {
                    learnigParams.regularizationRate = 0.1;// * learnigParams.learnigRate;
                }
                else {
                    learnigParams.regularizationRate = 0;// * learnigParams.learnigRate;
                    learnigParams.learnigRate = 4.;
                }
            }
            mode = 0;
        }*/
        /*else if(i == 2000) { //not sure if needed
            for(unsigned int iLayer = 0; iLayer < learnigParamsVec.size(); iLayer++ ) {
                auto& learnigParams = learnigParamsVec[iLayer];
                learnigParams.learnigRate = 0.01 ;
                if(iLayer == learnigParamsVec.size() -1) {
                    learnigParams.regularizationRate = 0.001;// * learnigParams.learnigRate;//0.001 * learnigParams.learnigRate;
                }
                else {
                    learnigParams.regularizationRate = 0.001;// * learnigParams.learnigRate;//0.03 * learnigParams.learnigRate;
                    learnigParams.learnigRate *= 1000;
                }
            }
        }

        if(i == 5000) { //not sure if needed
            for(unsigned int iLayer = 0; iLayer < learnigParamsVec.size(); iLayer++ ) {
                auto& learnigParams = learnigParamsVec[iLayer];
                learnigParams.learnigRate = 0.01 ;
                if(iLayer == learnigParamsVec.size() -1) {
                    learnigParams.regularizationRate = 0;//0.0001;//0.001 * learnigParams.learnigRate;
                }
                else {
                    learnigParams.regularizationRate = 0;//0.03 * learnigParams.learnigRate;
                    learnigParams.learnigRate *= 500;
                }
            }
        }*/

      /*  if(i > 6000 && i < 6100) {
            printEveryIteration = 1;
        }
        else
            printEveryIteration = 100;*/

        /*else if(i == 5000) { //not sure if needed
            for(unsigned int iLayer = 0; iLayer < learnigParamsVec.size(); iLayer++ ) {
                auto& learnigParams = learnigParamsVec[iLayer];
                learnigParams.learnigRate = 0.0005 ;
                learnigParams.stretchRate = 0. * learnigParams.learnigRate;
                if(iLayer == 0)
                    learnigParams.regularizationRate = 0;//0.001 * learnigParams.learnigRate;
                else
                    learnigParams.regularizationRate = 0;//0.03 * learnigParams.learnigRate;
            }
        }*/

        /*else if(i == 10000) { //not sure if needed
            for(unsigned int iLayer = 0; iLayer < learnigParamsVec.size(); iLayer++ ) {
                auto& learnigParams = learnigParamsVec[iLayer];
                learnigParams.learnigRate = 0.0 ;
                learnigParams.stretchRate = 0. * learnigParams.learnigRate;
                if(iLayer == 0)
                    learnigParams.regularizationRate = 0;//0.001 * learnigParams.learnigRate;
                else
                    learnigParams.regularizationRate = 0;//0.03 * learnigParams.learnigRate;
            }
        }*/

        if(stochastic)
            eventsGenerator.generateEvents(trainingEvents); //TODO!!!!!!!!!!!!!!!!!!!!!!!!!!

        for(auto& event : trainingEvents) {
            lutNetwork.setInputs(event);
            //lutNetwork.runTrainingInter(event.expextedResult);
            lutNetwork.runTraining(event.expextedResult, 0);
            //std::cout<<" "<<std::endl;
        }

        lutNetwork.updateLuts(learnigParamsVec);
        if( i < 0) {
            //scaling and shifting the initial LUT contents so that the layer outputs fill the next layer input address
            for(unsigned int iLayer = 0; iLayer < lutNetwork.getLayers().size() -1; iLayer++) {
                lutNetwork.shiftAndRescale(iLayer, 1, 1);
            }
        }

        if( (i)%printEveryIteration == 0 ) { //|| (i > 900 && i < 1000)
            if(stochastic) {
            //eventsGenerator.generateEvents(testEvents);
                printNN(i, testEvents);
            }
            else
                printNN(i, trainingEvents);
            lutNetwork.printLayerStat();
        }

        lutNetwork.resetStats();

        if(i ==  0) {
            TCanvas canvasRootFit("canvasRootFit", "canvasRootFit", 1400, 800);
            canvasRootFit.Divide(2, 2);
            makeFit(canvasRootFit,  ranges, testEvents, lutSize);
            canvasRootFit.Update();
            canvasRootFit.SaveAs( "../pictures/canvasRootFit.png" );
        }

    }


    canvasLutNN.Write();
    outfile.Write();

    std::cout<<"done - OK"<<std::endl;
    return EXIT_SUCCESS;
}
