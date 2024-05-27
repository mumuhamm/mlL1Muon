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

#include "lutNN/interface/MNISTParser.h"

#include "boost/dynamic_bitset.hpp"

using namespace lutNN;
using namespace std;

class EventsGeneratorMnist: public EventsGeneratorBase {
private:
    unsigned int inputLutSize;

    std::string imageFile;
    std::string labelsFile;

    std::default_random_engine& generator;

    std::string filename;
    std::vector<std::vector<uint8_t> > images;
    std::vector<uint8_t> labels;

    unsigned int rowCnt = 0;
    unsigned int columnCnt = 0;
public:
    EventsGeneratorMnist(unsigned int inputLutSize, std::string imageFile, std::string labelsFile, std::default_random_engine& generator):
        EventsGeneratorBase(), inputLutSize(inputLutSize), imageFile(imageFile), labelsFile(labelsFile) , generator(generator) {

        readMnistImages(imageFile, images, rowCnt, columnCnt);

        readMnistLabel(labelsFile, labels);

        cout<<"read imageFile "<<imageFile<<" size "<<images.size()<<" rowCnt "<<rowCnt<<" columnCnt "<<columnCnt<<endl;
    }

    void printImage(unsigned int num);

    virtual void generateEvents(vector<Event>& events, int mode);
};

void EventsGeneratorMnist::printImage(unsigned int num) {
    std::vector<uint8_t> image = images.at(num);
    for(unsigned int row = 0; row < rowCnt; row++) {
        for(unsigned int column = 0; column < columnCnt; column++) {
            if(image[row * columnCnt + column] > 64)
                cout<<("*");
            else
                cout<<" ";
        }
        cout<<endl;
    }
    cout<<"label "<<(unsigned int)labels[num]<<endl;
}

void EventsGeneratorMnist::generateEvents(vector<Event>& events, int mode) {
    std::uniform_int_distribution<int> randomEventDist(0, images.size() -1);

    int num = 0;
    for(auto& event : events) {
        if(mode == 0)
            num = randomEventDist(generator);

        std::vector<uint8_t> image = images.at(num);
        event.expextedResult[0] = (unsigned int)labels[num];

        //hot one
        /*for(unsigned int iRes = 0; iRes < event.expextedResult.size(); iRes++) {
            if(iRes == (unsigned int)labels[num])
                event.expextedResult[iRes] = 1;
            else
                event.expextedResult[iRes] = 0;
        }*/

        num++; //todo do not use the num after, cause it will be screwed!!!!!!

        unsigned int a = 3;
        /*for(unsigned int iInput = 0; iInput < event.inputs.size(); iInput++) {
            //int hitProb = hitProbDist(generator);
            unsigned int inBits = 0;

            for(unsigned int iBit = 0; iBit < a*a; iBit++) {
                //column = iBit%a
                //row = iBit/a
                if(image[iInput * a + iBit%a +  iBit/a * columnCnt] > 0) {
                    inBits |= (1<<iBit);
                }
            }

            event.inputs[iInput] = inBits;
            //cout<<" in"<<iInput<<" "<<x;
        }
*/
        for(unsigned int iInput = 0; iInput < event.inputs.size(); iInput++) {
            event.inputs[iInput] = 0;
        }
        for(unsigned int row = 0; row < rowCnt; row++) {
            for(unsigned int column = 0; column < columnCnt; column++) {
                if(image[row * columnCnt + column] > 64) {
                    unsigned int iInput =  column / a + row / a * (columnCnt +2)/ a; //+2 is if columnCnt%a != 0
                    unsigned int iBit = column % a + row%a * a;
                    event.inputs[iInput] |= (1<<iBit);
                    //cout<<column<<" "<<row<<" "<<iInput<<" "<<" "<<iBit<<" "<<event.inputs[iInput]<<endl;
                }
            }
        }



/*        for(unsigned int row = 0; row < 10; row++) {
            for(unsigned int column = 0; column < 10; column++) {
                cout<<setw(3)<<event.inputs[row * 10 + column];
            }
            cout<<endl;
        }
        cout<<"label "<<event.expextedResult[0]<<endl;
        printImage(num);*/

    }
}

int main(void) {
    int iterations = 100000; //TODO
    int printEveryIteration = 100; //TODO
    bool stochastic = true; //TODO

    unsigned int batchSize = 3000; //TODO


    unsigned int inputCnt = 100;// +1;

    unsigned int outputCnt = 1;
    unsigned int lutSize = (1<<9) -1;
    cout<<"lutSize "<<lutSize<<endl;
    std::vector<std::shared_ptr<ConfigParameters> > layersDef;
    unsigned int neuronCntL1 = 3;
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

    LutNetwork lutNetwork(inputCnt, layersDef);

    //std::cout<<lutNetwork<<std::endl;
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
    padOutMap.Divide(1, 1); //todo increase if more nn outputs then 4
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

    vector<Event> trainingEvents(batchSize, {inputCnt, outputCnt} );
    //vector<Event>& testEvents = trainingEvents;
    vector<Event> testEvents(10000, {inputCnt, outputCnt} ); //TODO look for the number


    std::vector<std::pair<double, double> > ranges =
    {
            {0. , 10 }
    };

    std::uniform_int_distribution<int> distribution(0,9);

    TH1F* costHist = createCostHistory(iterations, printEveryIteration);
    TCanvas canvasCostHist("canvasCostHist", "canvasCostHist", 1400, 800);
    canvasCostHist.SetLogy();

    string mnistDir = "/afs/cern.ch/work/k/kbunkow/private/lutNN/mnist/";

    EventsGeneratorMnist eventsGeneratorTrain(lutSize, mnistDir + "train-images-idx3-ubyte", mnistDir + "train-labels-idx1-ubyte", generator);
    EventsGeneratorMnist eventsGeneratorTest(lutSize, mnistDir + "t10k-images-idx3-ubyte", mnistDir + "t10k-labels-idx1-ubyte", generator);

    //eventsGenerator.generateEvents(testEvents);

    eventsGeneratorTrain.printImage(0);
    eventsGeneratorTrain.printImage(1);

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

        eventsGeneratorTest.generateEvents(testEvents, 1);
        for(auto& event : testEvents) {
            lutNetwork.setInputs(event);
            //lutNetwork.runTrainingInter(event.expextedResult); //fills the statistics
            lutNetwork.runTraining(event.expextedResult, 0); //fills the statistics
            //std::cout<<" "<<std::endl;
        }
        printNN(iteration, testEvents);

        if(iLayer < lutNetwork.getLayers().size() -1)
            lutNetwork.shiftAndRescale(iLayer, shiftRatio, scaleRatio);

        lutNetwork.resetStats();
        iteration++;
    }

    for(int i = 0; i < iterations; i++) {
        //for 2 layers network
        double speed = 5;
        if(i == 0) { //not sure if needed
            learnigParamsVec[0].learnigRate = speed * 2;//0.5;//0.5;//2.;
            learnigParamsVec[0].beta = 0.95;
            learnigParamsVec[0].regularizationRate = 0;// * learnigParams.learnigRate;

            learnigParamsVec[1].learnigRate = speed * 0.002 ;
            learnigParamsVec[1].beta = 0.95;
            learnigParamsVec[1].regularizationRate = 1;//0.1;// * learnigParams.learnigRate;
        }
        if(i == 500) { //not sure if needed
            learnigParamsVec[0].learnigRate = speed * 3;//0.5;//0.5;//2.;
            learnigParamsVec[0].beta = 0.95;
            learnigParamsVec[0].regularizationRate = 0;// * learnigParams.learnigRate;

            learnigParamsVec[1].learnigRate = speed * 0.002 ;
            learnigParamsVec[1].beta = 0.95;
            learnigParamsVec[1].regularizationRate = 1;//0.1;// * learnigParams.learnigRate;
        }
/*        if(i == 2000) { //not sure if needed
            speed = 0.2;
            learnigParamsVec[0].learnigRate = speed * 7;//0.5;//0.5;//2.;
            learnigParamsVec[0].beta = 0.95;
            learnigParamsVec[0].regularizationRate = 0;// * learnigParams.learnigRate;

            learnigParamsVec[1].learnigRate = speed * 0.0005 ;
            learnigParamsVec[1].beta = 0.95;
            learnigParamsVec[1].regularizationRate = 1;//0.1;// * learnigParams.learnigRate;
        }*/
        if(i == 80000) { //not sure if needed
            speed = 5;
            learnigParamsVec[0].learnigRate = speed * 3;//0.5;//0.5;//2.;
            learnigParamsVec[0].beta = 0.95;
            learnigParamsVec[0].regularizationRate = 0;// * learnigParams.learnigRate;

            learnigParamsVec[1].learnigRate = speed * 0.002 ;
            learnigParamsVec[1].beta = 0.95;
            learnigParamsVec[1].regularizationRate = 0;//0.1;// * learnigParams.learnigRate;
        }

        if(stochastic)
            eventsGeneratorTrain.generateEvents(trainingEvents, 0); //TODO!!!!!!!!!!!!!!!!!!!!!!!!!!

        for(auto& event : trainingEvents) {
            lutNetwork.setInputs(event);
            //lutNetwork.runTrainingInter(event.expextedResult);
            lutNetwork.runTraining(event.expextedResult, 0);
            //std::cout<<" "<<std::endl;
        }

        lutNetwork.updateLuts(learnigParamsVec);
        //if(i%10 == 0)
            lutNetwork.smoothLuts(learnigParamsVec); //TODO!!!!!!!!!!!!!!!

        if( i < 0) {
            //scaling and shifting the initial LUT contents so that the layer outputs fill the next layer input address
            for(unsigned int iLayer = 0; iLayer < lutNetwork.getLayers().size() -1; iLayer++) {
                lutNetwork.shiftAndRescale(iLayer, 1, 1);
            }
        }

        if( (i)%printEveryIteration == 0 ) { //|| (i > 900 && i < 1000)
            printNN(i, testEvents);

            lutNetwork.printLayerStat();
            //printNN(i+1, testEvents);
        }

        lutNetwork.resetStats();

    }


    canvasLutNN.Write();
    outfile.Write();

    std::cout<<"done - OK"<<std::endl;
    return EXIT_SUCCESS;
}
