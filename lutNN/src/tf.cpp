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

    vector<Event*> events;
public:
    EventsGeneratorTracks(unsigned int inputLutSize, std::vector<std::pair<double, double> >& ranges,  std::default_random_engine& generator):
        EventsGeneratorBase(), inputLutSize(inputLutSize), ranges(ranges), generator(generator) {}

    virtual ~EventsGeneratorTracks() {
        for(auto& event : events)
            delete event;
    }

    virtual void generateEvents(unsigned int eventCount, unsigned int inputCnt, unsigned int outputCnt);

    virtual void getRandomEvents(vector<Event*>& events);

    virtual vector<Event*>& getEvents() {
        return events;
    }
};

bool useFiredPlanes = false;
void EventsGeneratorTracks::generateEvents(unsigned int eventCount, unsigned int inputCnt, unsigned int outputCnt) {
    events.resize(eventCount);

    //std::uniform_int_distribution<int> a0dist(ranges[0].first, ranges[0].second);
    //std::uniform_int_distribution<int> a0dist(60, inputLutSize-60);
    std::uniform_real_distribution<> a0dist(ranges[0].first, ranges[0].second); //TODO!!!!!!!!!!!!!!!!!!!!!
    std::uniform_real_distribution<> a1dist(ranges[1].first, ranges[1].second);
    std::uniform_real_distribution<> a2dist(ranges[2].first, ranges[2].second);
    std::uniform_int_distribution<int> hitProbDist(0, 100);
    std::uniform_int_distribution<unsigned int> firedPlaneDist(0, (1 << events[0]->inputs.size()) -1);
    double a0 = 0;
    double a1 = 0;
    double a2 = 0;

    //boost::dynamic_bitset<> firedPlanes(events[0].inputs.size() - useFiredPlanes);
    uint64_t firedPlanes = 0;

    for(auto& event : events) {
        event = new Event(inputCnt, outputCnt);
        //a0 = a0dist(generator);
        a1 = a1dist(generator);
        a2 = a2dist(generator);
        event->expextedResult[0] = a2;//a1;//a0;

        //event.expextedResult[1] = a1;
        //event.expextedResult[2] = a2;
        //cout<<"Event "<<"a0 "<<a0<<" a1 "<<a1;
        //the a0, a1, a3 should have the same range for proper training in the lut
        //so here we recalculate such thay have proper value for polynomial
        /*a0 = ( 0.5 * a0 - ranges[0].first) * inputLutSize / (ranges[0].second - ranges[0].first);
        a1 = a1 * 10. / (ranges[1].second - ranges[1].first);
        a2 = a2 * 0.4/ (ranges[2].second - ranges[2].first);*/

        //a0 = inputLutSize/4. + 0.5 * a0 * inputLutSize;
        a0 = inputLutSize/2; //TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
        a1 = (a1 - 0.5) * 5.;
        a2 = (a2 - 0.5) * 0.5;

        int hitCnt = 0;
        while(true) {
            firedPlanes = firedPlaneDist(generator);
            unsigned int firedPlanesCnt = 0;
            for(unsigned int i = 0; i < (event->inputs.size() -useFiredPlanes); i++) {
                if(firedPlanes & (1<<i) )
                    firedPlanesCnt++;
            }
            if(firedPlanesCnt >= 4)
                break;
        }

        for(unsigned int i = 0; i < (event->inputs.size() -useFiredPlanes); i++) {
            //int hitProb = hitProbDist(generator);
            int x = round(a0 + a1 * planeY[i] + a2 * planeY[i] * planeY[i]); //
            if(x > 0 && x < inputLutSize && ( firedPlanes & (1<<i) ) ) {//  && hitProb < 80//TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                event->inputs[i] = x;
                hitCnt++;
                /*if(useFiredPlanes)
                    firedPlanes[i] = true;*/
            }
            else
                event->inputs[i] = 0;
            //cout<<" in"<<i<<" "<<x;
        }
        if(useFiredPlanes)
            event->inputs[event->inputs.size() - 1] = firedPlanes;//.to_ulong();
        //string str;
        //boost::to_string(firedPlanes, str);
        //cout<<" str "<<str<<" firedPlanes.to_ulong() "<<firedPlanes.to_ulong()<<" input "<<event.inputs[event.inputs.size() - 1]<<std::endl;;
        //event.inputs[event.inputs.size() -1] = hitCnt;
        //cout<<endl;

    }
}


void makeFit(TPad& padOutMap,  std::vector<std::pair<double, double> >& ranges, std::vector<Event*>& events, unsigned int inputLutSize) {
    vector<TH2F*> hists;
    unsigned int inCnt =  events[0]->inputs.size() - useFiredPlanes;//TODO!!!!!!!!!!!!!!!!!!
    unsigned int outCnt =  events[0]->expextedResult.size();
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

    int iEvent = 0;
    for(auto& event : events) {
        unsigned int iPoint = 0;
        for(unsigned int iIn = 0; iIn < inCnt; iIn++) {
            if(event->inputs[iIn]) {
                y[iPoint] = event->inputs[iIn];
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
            hists[0]->Fill(event->expextedResult[0], fitResult->Value(2)  / 0.5 + 0.5); //TODO a2 watch aout for the indexes!!!!
            //hists[0]->Fill(event.expextedResult[0], fitResult->Value(2) * (ranges[2].second - ranges[2].first) /0.4); //TODO watch aout for the indexes!!!!

            fitResult->Clear();
        }

        //cout<<" neuron.getOutAddr(): "<<neuron.getOutAddr()<<endl;

        if(iEvent < 100) {
            TCanvas canvasGraph("canvasGraph", "canvasGraph");
            canvasGraph.cd();
            gr->Draw("A*");
            //gr->GetXaxis()->SetRangeUser(0, 10);
            gr->GetXaxis()->SetLimits(0, planeY.size() - 1);
            gr->GetYaxis()->SetRangeUser(0, inputLutSize);
            gr->Draw("A*");
            canvasGraph.Update();
            canvasGraph.SaveAs( ("../pictures/graphs/canvasGraph" + to_string(iEvent) + ".png").c_str() );
        }
        iEvent++;
        gr->Delete();

    }

    for(unsigned int iOut = 0; iOut < outCnt; iOut++) {
        padOutMap.cd(iOut + 1)->SetGrid();
        hists[iOut]->SetStats(0);
        hists[iOut]->Draw("colz");
    }
}

int main(void) {
    //puts("Hello World!!!");

    /*	std::shared_ptr<ConfigParameters> config(new ConfigParameters() );
	config->lutAddrOffset = 0;
	cout<<"config: lutSize "<<config->lutSize<<" lutAddrOffset "<<config->lutAddrOffset<<endl;*/

    unsigned int batchSize = 3000; //TODO

    unsigned int inputCnt = 10;// +1;
    planeY = vector<double>(inputCnt, 0);
    for(unsigned int i = 0; i < planeY.size(); i++) {
        planeY[i] = i;
    }

    CostFunctionMeanSquaredError  costFunction;
    LutNetwork::OutputType outputType = LutNetwork::simple;

    unsigned int outputCnt = 1;
    unsigned int lutSize = (1<<7) -1;
    cout<<"lutSize "<<lutSize<<endl;
    std::vector<std::shared_ptr<ConfigParameters> > layersDef;
    unsigned int neuronCntL1 = 20;
    layersDef.emplace_back(ConfigParametersPtr(new ConfigParameters(neuronCntL1,               lutSize, lutSize, inputCnt, ConfigParameters::fullyConnected) ) );
    layersDef.emplace_back(ConfigParametersPtr(new ConfigParameters(outputCnt,                 lutSize, 0      , neuronCntL1, ConfigParameters::fullyConnected) ) );
    //layersDef.emplace_back(ConfigParametersPtr(new ConfigParameters(outputCnt, 2*lutSize, lutSize/2, 1, ConfigParameters::singleLut) ) );

    //TODO there is mistake in stretching the neuron output values, it does not work well when the lutSize is different in each layer - check if it is true

    //making sure the values of neuronInputCnt are correct
 /*   for(unsigned int iLayer = 0; iLayer < layersDef.size(); iLayer++ ) {
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
    }*/

    LutNetwork lutNetwork(inputCnt, layersDef, outputType);

    std::cout<<lutNetwork<<std::endl;
    std::default_random_engine generator(11);
    lutNetwork.initLuts(generator);
    //lutNetwork.initLuts();

    unsigned int maxNodesPerLayer = 0;
    /*for(unsigned int iLayer = 0; iLayer < lutNetwork.getNeuronLayers().size(); iLayer++ ) {
        if(maxNodesPerLayer < lutNetwork.getNeuronLayers()[iLayer].size() )
            maxNodesPerLayer = lutNetwork.getNeuronLayers()[iLayer].size();
    }*/
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
    learnigParams.smoothWeight = 0;
    std::vector<LutNetwork::LearnigParams> learnigParamsVec(layersDef.size(), learnigParams);

    //learnigRates[0] = 0.02;
    //learnigRates[1] = 0.02;

    TPaveLabel textCosts(0.7, .01, 1., 0.05, "aaaa");
    canvasLutNN.cd();
    ostringstream ostr;
    ostr<<"cost";
    textCosts.Draw();



    std::vector<std::pair<double, double> > ranges =
    {
            {0. , 1 },
            {0. , 1 },
            {0. , 1 }
    };

    std::uniform_int_distribution<int> distribution(0,9);
    int iterations = 10000; //TODO
    int printEveryIteration = 100; //TODO
    bool stochastic = true; //TODO
    TH1F* costHist = createCostHistory(iterations, printEveryIteration);
    TCanvas canvasCostHist("canvasCostHist", "canvasCostHist", 1400, 800);
    canvasCostHist.SetLogy();

    EventsGeneratorTracks eventsGeneratorTrain(lutSize, ranges, generator);
    eventsGeneratorTrain.generateEvents(100000, inputCnt, outputCnt);
    vector<Event*> trainingEvents(batchSize);

    EventsGeneratorTracks eventsGeneratorTest(lutSize, ranges, generator);
    eventsGeneratorTrain.generateEvents(10000, inputCnt, outputCnt);

    vector<Event*>& testEvents = eventsGeneratorTest.getEvents();

    //########################### printNN lambda ###########################
    auto printNN = [&](int iteration, vector<Event*>& events) {
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

        eventsGeneratorTrain.getRandomEvents(trainingEvents);
        for(auto& event : trainingEvents) {
            lutNetwork.setInputs(event);
            //lutNetwork.runTrainingInter(event.expextedResult); //fills the statistics
            lutNetwork.runTraining(event->expextedResult, costFunction); //fills the statistics
            //std::cout<<" "<<std::endl;
        }
        printNN(iteration, trainingEvents);

        if(iLayer < lutNetwork.getLayers().size() -1)
            lutNetwork.shiftAndRescale(iLayer, shiftRatio, scaleRatio);

        lutNetwork.resetStats();
        iteration++;
    }

    for(int i = 0; i < iterations; i++) {
        //for 2 layers network
        double speed = 0.1;
        if(i == 0) { //not sure if needed
            learnigParamsVec[0].learnigRate = speed * 700;//0.5;//0.5;//2.;
            learnigParamsVec[0].beta = 0.9;
            learnigParamsVec[0].smoothWeight = 0;// * learnigParams.learnigRate;

            learnigParamsVec[1].learnigRate = speed * 0.0005 ;
            learnigParamsVec[1].beta = 0.9;
            learnigParamsVec[1].smoothWeight = 1;//0.1;// * learnigParams.learnigRate;
        }
        if(i == 2000) { //not sure if needed
            speed = 0.1;
            learnigParamsVec[0].learnigRate = speed * 700;//0.5;//0.5;//2.;
            learnigParamsVec[0].beta = 0.9;
            learnigParamsVec[0].smoothWeight = 0;// * learnigParams.learnigRate;

            learnigParamsVec[1].learnigRate = speed * 0.0005 ;
            learnigParamsVec[1].beta = 0.9;
            learnigParamsVec[1].smoothWeight = 1;//0.1;// * learnigParams.learnigRate;
        }
        else if(i == 4000) { //not sure if needed
            learnigParamsVec[0].learnigRate = speed * 700;//0.5;//0.5;//2.;
            learnigParamsVec[0].beta = 0.9;
            learnigParamsVec[0].smoothWeight = 0;// * learnigParams.learnigRate;

            learnigParamsVec[1].learnigRate = speed * 0.0005 ;
            learnigParamsVec[1].beta = 0.9;
            learnigParamsVec[1].smoothWeight = 0;//0.1;// * learnigParams.learnigRate;
        }
        else if(i == 6000) { //not sure if needed
            speed = 0.3;
            learnigParamsVec[0].learnigRate = speed * 700;//0.5;//0.5;//2.;
            learnigParamsVec[0].beta = 0.9;
            learnigParamsVec[0].smoothWeight = 0;// * learnigParams.learnigRate;

            learnigParamsVec[1].learnigRate = speed * 0.0005 ;
            learnigParamsVec[1].beta = 0.9;
            learnigParamsVec[1].smoothWeight = 0;//0.1;// * learnigParams.learnigRate;
        }


        if(stochastic)
            eventsGeneratorTrain.getRandomEvents(trainingEvents); //TODO!!!!!!!!!!!!!!!!!!!!!!!!!!

        for(auto& event : trainingEvents) {
            lutNetwork.setInputs(event);
            //lutNetwork.runTrainingInter(event.expextedResult);
            lutNetwork.runTraining(event->expextedResult, costFunction);
            //std::cout<<" "<<std::endl;
        }

        lutNetwork.updateLuts(learnigParamsVec);
        if(i%10 == 0)
            lutNetwork.smoothLuts(learnigParamsVec); //TODO!!!!!!!!!!!!!!!
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
            //printNN(i+1, testEvents);
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
