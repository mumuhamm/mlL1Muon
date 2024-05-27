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
#include <fstream>
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

    std::string filename;
    std::vector<std::vector<uint8_t> > images;
    std::vector<uint8_t> labels;

    unsigned int rowCnt = 0;
    unsigned int columnCnt = 0;

public:
    EventsGeneratorMnist(unsigned int inputLutSize, std::string imageFile, std::string labelsFile, std::default_random_engine& generator):
        EventsGeneratorBase(generator), inputLutSize(inputLutSize), imageFile(imageFile), labelsFile(labelsFile) {

        readMnistImages(imageFile, images, rowCnt, columnCnt);

        readMnistLabel(labelsFile, labels);

        cout<<"read imageFile "<<imageFile<<" size "<<images.size()<<" rowCnt "<<rowCnt<<" columnCnt "<<columnCnt<<endl;
    }

    virtual ~EventsGeneratorMnist() {
        for(auto& event : events)
            delete event;
    }

    void printImage(unsigned int num, ostream& ostr = cout);

    virtual void generateEvents(unsigned int inputCnt, unsigned int outputCnt, int xShift = 0, int yShift = 0);

    virtual void getRandomEvents(vector<Event*>& events);

    virtual vector<Event*>& getEvents() {
        return events;
    }

    int gelPixelCnt() {
        return rowCnt * columnCnt;
    }
};

void EventsGeneratorMnist::printImage(unsigned int num, ostream& ostr) {
    std::vector<uint8_t> image = images.at(num);
    for(unsigned int column = 0; column < columnCnt; column++)
        ostr<<"#";
    ostr<<"#"<<endl;
    for(unsigned int row = 0; row < rowCnt; row++) {
        for(unsigned int column = 0; column < columnCnt; column++) {
            int indx= row * columnCnt + column;
            if(image.at(indx) == 0)
                ostr<<(" ");
            else if(image[indx] < 64)
                ostr<<(".");
            else if(image[indx] < 128)
                ostr<<("o");
            else
                ostr<<("@");
        }
        ostr<<"#"<<endl;
    }
    for(unsigned int column = 0; column < columnCnt; column++)
        ostr<<"#";
    ostr<<"#"<<endl;
    ostr<<"label "<<(unsigned int)labels[num]<<endl;
}

void EventsGeneratorMnist::generateEvents(unsigned int inputCnt, unsigned int outputCnt, int xShift, int yShift) {
    events.resize( (2*xShift + 1) * (2*yShift + 1) * images.size());
    unsigned int iEvent = 0;
    for(unsigned int iImage = 0; iImage < images.size(); iImage++) {
        std::vector<uint8_t> image = images.at(iImage);

        for(int xS = -xShift; xS <= xShift; xS++) {
            for(int yS = -yShift; yS <= yShift; yS++) {
                Event* event = new Event(inputCnt, outputCnt);

                //hot one
                for(unsigned int iRes = 0; iRes < event->expextedResult.size(); iRes++) {
                    if(iRes == (unsigned int)labels[iImage])
                        event->expextedResult[iRes] = 1;
                    else
                        event->expextedResult[iRes] = 0;
                }

                for(unsigned int row = 0; row < rowCnt; row++) {
                    for(unsigned int column = 0; column < columnCnt; column++) {
                        if(row + yS < 0 || row + yS >= rowCnt || column + xS < 0 || column + xS >= columnCnt)
                            continue;
                        event->inputs[(row) * columnCnt + (column)] = image[ (row + yS) * columnCnt + (column + xS) ];
                    }
                }

                event->number = iEvent;
                events.at(iEvent++) = event;
            }
        }
    }

    miniBatchBegin = events.begin();
}

void EventsGeneratorMnist::getRandomEvents(vector<Event*>& rndEvents) {
    std::uniform_int_distribution<int> randomEventDist(0, this->events.size() -1);
    for(auto& event : rndEvents) {
        int num = randomEventDist(generator);
        event = this->events[num];
    }
}

void printWrongClasified( LutNetwork& lutNetwork, EventsGeneratorMnist& eventsGeneratorTest, vector<Event*>& events) {
    ofstream ofstr("WrongClasified.txt", std::ofstream::out);
    for(auto& event : events) {
        lutNetwork.run(event);
    }

    unsigned int iE = 0;
    for(auto& event : events) {
        double maxOutVal = -10000;
        unsigned int maxOutNum = 0;
        unsigned int expectedOutNum = 0;

        for(unsigned int iOut = 0; iOut < event->expextedResult.size(); iOut++) {
            if(event->nnResult[iOut] >= maxOutVal) {
                maxOutVal = event->nnResult[iOut];
                maxOutNum = iOut;
            }
            if(event->expextedResult[iOut] == 1)
                expectedOutNum = iOut;

        }

        if(expectedOutNum != maxOutNum) {
            eventsGeneratorTest.printImage(iE, ofstr);
            ofstr<<"classified as "<<maxOutNum<<endl;
        }
        iE++;
    }

}

int main(void) {
    int iterations = 20000; //TODO
    int printEveryIteration = 100; //TODO
    bool stochastic = true; //TODO

    unsigned int batchSize = 3000; //TODO

    double retainProb = 1; //dropout regularization parameter
    unsigned int inputNodeCnt = 20;// +1;

    unsigned int outputCnt = 10;
    unsigned int lutAddrBitCnt = 10;
    unsigned int lutSize = (1<<lutAddrBitCnt) -1;
    cout<<"lutSize "<<lutSize<<endl;
    std::vector<std::shared_ptr<ConfigParameters> > layersDef;
    unsigned int neuronCntL1 = 50;
    unsigned int  disabledNeuronCnt = 30;

    CostFunctionCrossEntropy  costFunction;
    LutNetwork::OutputType outputType = LutNetwork::softMax;
    ConfigParameters::LutType lutType = ConfigParameters::interpolated;

    layersDef.emplace_back(ConfigParametersPtr(new ConfigParameters(neuronCntL1,               lutSize, lutSize/4, inputNodeCnt, ConfigParameters::oneToOne, ConfigParameters::discrete) ) );
    layersDef.emplace_back(ConfigParametersPtr(new ConfigParameters(outputCnt,                 lutSize/4, 0      , neuronCntL1, ConfigParameters::fullyConnected, ConfigParameters::interpolated) ) );
    //layersDef.emplace_back(ConfigParametersPtr(new ConfigParameters(outputCnt,                 lutSize, 0      , neuronCntL1/outputCnt, ConfigParameters::singleNetForOut) ) ); //TODO!!!!!!!!!!!
    //layersDef.emplace_back(ConfigParametersPtr(new ConfigParameters(outputCnt, 64, 0, 1, ConfigParameters::singleLut) ) );

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

    string mnistDir = "/afs/cern.ch/work/k/kbunkow/private/lutNN/mnist/";

    std::default_random_engine generator(11);
    EventsGeneratorMnist eventsGeneratorTrain(lutSize, mnistDir + "train-images-idx3-ubyte", mnistDir + "train-labels-idx1-ubyte", generator);
    eventsGeneratorTrain.generateEvents(eventsGeneratorTrain.gelPixelCnt(), outputCnt, 0, 0);

    EventsGeneratorMnist eventsGeneratorTest(lutSize, mnistDir + "t10k-images-idx3-ubyte", mnistDir + "t10k-labels-idx1-ubyte", generator);
    eventsGeneratorTest.generateEventthiss(eventsGeneratorTrain.gelPixelCnt(), outputCnt);

    InputNodeFactory* inputNodeFactory = new InputNodeSelBinaryInputsFactoryFlatDist(eventsGeneratorTrain.gelPixelCnt(), lutAddrBitCnt, generator);

    LutNetwork lutNetwork(layersDef, outputType, inputNodeFactory);

    //std::cout<<lutNetwork<<std::endl;

    lutNetwork.initLuts(generator);

    //lutNetwork.initLuts();

    cout<<lutNetwork<<endl;

    unsigned int maxNodesPerLayer = 0;
    /*for(unsigned int iLayer = 0; iLayer < lutNetwork.getNeuronLayers().size(); iLayer++ ) {
        if(maxNodesPerLayer < lutNetwork.getNeuronLayers()[iLayer].size() )
            maxNodesPerLayer = lutNetwork.getNeuronLayers()[iLayer].size();
    }*/
    maxNodesPerLayer = 4;
    cout<<"maxNodesPerLayer "<<maxNodesPerLayer<<endl;

    //--------------------------------------------------
    gStyle->SetOptStat(0);
    TCanvas canvasLutNN("canvasLutNN", "canvasLutNN", 2 * 1900, 2 * 900);
    canvasLutNN.cd();
    TPad padLuts("padLuts", "padLuts", 0.0, 0.0, 0.7, 1.);
    padLuts.SetGrid();
    padLuts.Divide(lutNetwork.getLayers().size(), maxNodesPerLayer * 2);
    padLuts.Draw();

    TPad padOutMap("padOutMap", "padOutMap", 0.7, 0., 1., 1.);
    padOutMap.SetRightMargin(0.2);
    padOutMap.SetGrid();
    padOutMap.Divide(1, 2); //todo increase if more nn outputs then 4
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

    vector<Event*> trainingEvents(batchSize);

    std::vector<std::pair<double, double> > ranges =
    {
            {0. , 10 }
    };

    std::uniform_int_distribution<int> distribution(0,9);

    createCostHistory(iterations, printEveryIteration);
    TCanvas canvasCostHist("canvasCostHist", "canvasCostHist", 1400, 800);
    canvasCostHist.SetLogy();

    vector<Event*>& testEvents = eventsGeneratorTest.getEvents(); //TODO look for the number

    for(unsigned int i = 0; i < 3; i++)
        eventsGeneratorTrain.printImage(i);

    for(unsigned int i = 0; i < 20; i++) {
        cout<<" event "<<i<<endl;
        Event* event = eventsGeneratorTrain.getEvents()[i];

        for(unsigned int row = 0; row < 10; row++) {
            for(unsigned int column = 0; column < 10; column++) {
                cout<<setw(3)<<hex<<event->inputs.at(row * 10 + column);
            }
            cout<<endl;
        }
        cout<<"expected out: "<<endl;
        for(unsigned int iOut = 0; iOut < event->expextedResult.size(); iOut++) {
            cout<<iOut<<": "<<event->expextedResult[iOut]<<"  ";
        }
        cout<<endl<<endl;;
    }

    //########################### printNN lambda ###########################
    auto printNN = [&](int iteration, vector<Event*>& events) {
        //eventsGenerator.generateEvents(events);

        printLuts2(padLuts, padGradients, lutNetwork, maxNodesPerLayer);
        //printGradinets(padGradients, lutNetwork, maxNodesPerLayer * 2);

        for(auto& event : events) {
            lutNetwork.run(event);
            //std::cout<<" "<<std::endl;
        }

        cout<<"iteration "<<dec<<setw(7)<<iteration<<endl;
        double validSampleCost = printExpectedVersusNNOutHotOne(padOutMap, ranges, events, costFunction);
        ostr.str("");
        ostr<<"iteration "<<setw(7)<<iteration<<" cost: train: "<<setw(5)<<lutNetwork.getMeanCost()<<" valid "<<setw(5)<<validSampleCost;

        cout<<" validation sample cost:    "<<setw(5)<<validSampleCost<<endl;
        cout<<" training sample mean cost: "<<setw(5)<<lutNetwork.getMeanCost()<<endl;

        textCosts.SetLabel(ostr.str().c_str() );

        padLuts.Update();
        padOutMap.Update();
        canvasLutNN.Update();
        canvasLutNN.SaveAs( ("../pictures/luts/canvasLutNN_" + to_string(iteration+1) + ".png").c_str() );

        padGradients.Update();
        canvasGradients.Update();
        canvasGradients.SaveAs( ("../pictures/gradients/canvasGradients_" + to_string(iteration+1) + ".png").c_str() );

        updateCostHistory(canvasCostHist, iteration, lutNetwork.getMeanCost(), validSampleCost);
        canvasCostHist.Update();
        canvasCostHist.SaveAs( "../pictures/canvasLutNN_canvasCostHist.png" );
    };
    //######################################################

    int iteration = -20;

    //scaling and shifting the initial LUT contents so that the layer outputs fill the next layer input address
/*    for(unsigned int iLayer = 0; iLayer < lutNetwork.getNeuronLayers().size(); iLayer++) {
        double shiftRatio = 1;
        double scaleRatio = 1;

        //eventsGeneratorTest.generateEvents(testEvents, 1);
        for(auto& event : testEvents) {
            lutNetwork.setInputs(event);
            //lutNetwork.runTrainingInter(event->expextedResult); //fills the statistics
            lutNetwork.runTraining(event->expextedResult, 1); //fills the statistics
            //std::cout<<" "<<std::endl;
        }

        if(iLayer == 0) {//TODO!!!!!!!!!!!!
            lutNetwork.weightByEvents(iLayer);
            lutNetwork.resetStats();
            for(auto& event : testEvents) {
                lutNetwork.setInputs(event);
                //lutNetwork.runTrainingInter(event->expextedResult); //fills the statistics
                lutNetwork.runTraining(event->expextedResult, 1); //fills the statistics
                //std::cout<<" "<<std::endl;
            }
        }

        printNN(iteration, testEvents);

        if(iLayer < lutNetwork.getNeuronLayers().size() -1)
            lutNetwork.shiftAndRescale(iLayer, shiftRatio, scaleRatio);

        lutNetwork.resetStats();
        iteration++;
    }*/


    {//TODO, must be done after the shiftAndRescale
        /*for(unsigned int iNeuron = 0; iNeuron < disabledNeuronCnt; iNeuron++) {//lutNetwork.getNeuronLayers()[0].size()
            lutNetwork.getNeuronLayers()[0].at(iNeuron)->enable(false);
            for(auto& childLut : lutNetwork.getNeuronLayers()[0].at(iNeuron)->getChildLuts()) {
                for(unsigned int iAddr = 0; iAddr < childLut->getValues().size(); iAddr++) {
                    childLut->getValues()[iAddr] = 0;
                }
            }
        }*/
    }

    int wrongClassEventWeight = 1;
    unsigned int neuronToEnable = 0;
    unsigned int enablingEpoch = 0;
    for(int i = 0; i < iterations; i++) {
        //for 2 layers network
        double speed = 3;
        if(i == 0) {
            //wrongClassEventWeight = 3;
            learnigParamsVec[0].learnigRate = speed * 60.;//0.5;//0.5;//2.;
            learnigParamsVec[0].beta = 0.0;
            learnigParamsVec[0].lambda = 0;//0.0000001  * 1.;
            learnigParamsVec[0].smoothWeight = 10000;// * learnigParams.learnigRate;
            learnigParamsVec[0].maxLutVal = 2;

            learnigParamsVec[1].learnigRate = 0;//speed * 0.1;
            learnigParamsVec[1].beta = 0.0;
            learnigParamsVec[1].lambda = 0;// 0.001;//0.000005;
            learnigParamsVec[1].smoothWeight = 0;//0.1;// * learnigParams.learnigRate;
            learnigParamsVec[1].maxLutVal = 5;

            for(unsigned int iLP = 0; iLP < learnigParamsVec.size(); iLP++) {
                cout<<"iter "<<i<<" layer "<<iLP<<" learnigRate "<<learnigParamsVec[iLP].learnigRate<<" beta "<<learnigParamsVec[iLP].beta<<" lambda "<< learnigParamsVec[iLP].lambda<<endl;
            }
        }
        if(i == 400) {
            speed = 10;
            //wrongClassEventWeight = 3;
            learnigParamsVec[0].learnigRate = speed * 4;//0.5;//0.5;//2.;
            learnigParamsVec[0].beta = 0.0;
            learnigParamsVec[0].lambda = 0.0000005;
            learnigParamsVec[0].smoothWeight = 8000;// * learnigParams.learnigRate;
            learnigParamsVec[0].maxLutVal = 2;

            learnigParamsVec[1].learnigRate = speed * 2.;
            learnigParamsVec[1].beta = 0.0;
            learnigParamsVec[1].lambda = 0.0000001;
            learnigParamsVec[1].smoothWeight = 30;//0.1;// * learnigParams.learnigRate;
            learnigParamsVec[1].maxLutVal = 5;

            for(unsigned int iLP = 0; iLP < learnigParamsVec.size(); iLP++) {
                cout<<"iter "<<i<<" layer "<<iLP<<" learnigRate "<<learnigParamsVec[iLP].learnigRate<<" beta "<<learnigParamsVec[iLP].beta<<" lambda "<< learnigParamsVec[iLP].lambda<<endl;
            }
        }
        if(i == 8000) {
            trainingEvents.resize(batchSize * 2);
        }

        /*        if(i > 500) {
            double betaMod = 1;//(1. - 1. / (i - 400) );
            learnigParamsVec[0].beta = 0.90 * betaMod;
            learnigParamsVec[1].beta = 0.90 * betaMod;
            //cout<<"i "<<i<<" beta 0 "<<learnigParamsVec[0].beta<<" beta 1 "<<learnigParamsVec[1].beta<<endl;
        }
        if(i == 1000) {
            //learnigParamsVec[0].learnigRate = speed * 0.5;
            learnigParamsVec[0].smoothWeight = 300;
            learnigParamsVec[0].beta = 0.90;
            learnigParamsVec[1].beta = 0.90;
        }
        if(i == 4000) {
            trainingEvents.resize(batchSize * 2);
            learnigParamsVec[0].beta = 0.90;
            learnigParamsVec[1].beta = 0.90;
        }
        if(i == 6000) {
            trainingEvents.resize(batchSize * 3);
            learnigParamsVec[0].beta = 0.95;
            learnigParamsVec[1].beta = 0.95;
        }
        if(i == 8000) {
            trainingEvents.resize(batchSize * 4);
        }*/

      /*  if(i == 8000) {
            learnigParamsVec[0].learnigRate = speed * 5.;
            learnigParamsVec[1].learnigRate = speed * 5.;
        }
        else if(i == 8001) {
            learnigParamsVec[0].learnigRate = speed * 1.;
            learnigParamsVec[1].learnigRate = speed * 1.;
        }*/
        /*if(i > 8000) {
            double learnigRate = speed * (0.5 + 2. * (1 - cos( (i-8500.)/628.) ) );
            learnigParamsVec[0].learnigRate = learnigRate;
            learnigParamsVec[1].learnigRate = learnigRate;
        }*/

        /*if(i > 1000 && i % 100 == 50) { //not sure if needed
            for(unsigned int iLP = 0; iLP < learnigParamsVec.size(); iLP++) {
                learnigParamsVec[iLP].learnigRate *= 0.99;//0.5;//0.5;//2.;
                learnigParamsVec[iLP].beta = 0.95;
                //if(iLP == 0)
                learnigParamsVec[iLP].lambda *= 0.96;
                //learnigParamsVec[iLP].regularizationRate = 0;// * learnigParams.learnigRate;
                cout<<"iter "<<i<<" layer "<<iLP<<" learnigRate "<<learnigParamsVec[iLP].learnigRate<<" beta "<<learnigParamsVec[iLP].beta<<" lambda "<< learnigParamsVec[iLP].lambda<<endl;            }

            // learnigParamsVec[1].learnigRate += (learnigParamsVec[1].learnigRate * 0.1);
        }*/
        /*if(i > 1 && i %2000 == 0) { //not sure if needed
            for(unsigned int iNeuron = 0; (iNeuron < enablingEpoch * 10) && iNeuron < lutNetwork.getNeuronLayers()[0].size(); iNeuron++) {//lutNetwork.getNeuronLayers()[0].size()
               // if(iNeuron == neuronToEnable)
                    lutNetwork.getNeuronLayers()[0].at(iNeuron)->enable(true);
                else
                    lutNetwork.getNeuronLayers()[0].at(iNeuron)->enable(false);
            }
            enablingEpoch++;
            neuronToEnable++;
            //wrongClassEventWeight = 3;
        }*/
        /*if(i == 3000) { //not sure if needed
            //wrongClassEventWeight = 3;
            learnigParamsVec[0].learnigRate /=2;//0.5;//0.5;//2.;
            learnigParamsVec[0].beta = 0.95;
            learnigParamsVec[0].lambda /=2;
            learnigParamsVec[0].regularizationRate = 0;// * learnigParams.learnigRate;

            learnigParamsVec[1].learnigRate /=2 ;
            learnigParamsVec[1].beta = 0.95;
            learnigParamsVec[1].lambda /=2;
            learnigParamsVec[1].regularizationRate = 1;//0.1;// * learnigParams.learnigRate;
        }*/


        if(stochastic) {
            //eventsGeneratorTrain.getRandomEvents(trainingEvents); //TODO!!!!!!!!!!!!!!!!!!!!!!!!!!
            eventsGeneratorTrain.getNextMiniBatch(trainingEvents);
        }
        //lutNetwork.setDropOut(retainProb, generator);

        for(auto& event : trainingEvents) {
            //cout<<" iteration "<<dec<<i<<" processing event "<<event->number<<endl;

            lutNetwork.setInputs(event);

            if(lutType == ConfigParameters::interpolated)
                lutNetwork.runTrainingInter(event->expextedResult, costFunction);
            else if(lutType == ConfigParameters::discrete)
                lutNetwork.runTraining(event->expextedResult, costFunction);
            //std::cout<<" "<<std::endl;
        }

        if(lutType == ConfigParameters::interpolated)
            lutNetwork.updateLutsInter(learnigParamsVec);
        else if(lutType == ConfigParameters::discrete)
            lutNetwork.updateLuts(learnigParamsVec);

        //if(i%5 == 0)
            lutNetwork.smoothLuts(learnigParamsVec); //TODO!!!!!!!!!!!!!!!
            lutNetwork.smoothLuts1(learnigParamsVec); //TODO!!!!!!!!!!!!!!!

        /*if(i == 1000)
            lutNetwork.shiftAndRescale(0, 1, 1);
*/
/*        if( i < 0) { //TODO!!!!
            //scaling and shifting the initial LUT contents so that the layer outputs fill the next layer input address
            for(unsigned int iLayer = 0; iLayer < lutNetwork.getNeuronLayers().size() -1; iLayer++) {
                lutNetwork.shiftAndRescale(iLayer, 1, 1);
            }
        }*/

        if( (i)%printEveryIteration == 0) { //|| (i > 900 && i < 1000)
            //lutNetwork.setLutOutWeight(retainProb);

            printNN(i, testEvents);

            lutNetwork.printLayerStat();
            //printNN(i+1, testEvents);
            cout<<endl;
        }

        lutNetwork.resetStats();

    }


    canvasLutNN.Write();
    outfile.Write();

    printWrongClasified(lutNetwork, eventsGeneratorTest, testEvents);
    std::cout<<"done - OK"<<std::endl;


    /*for(unsigned char i = 0; i != 255; i++) {
        cout<<(unsigned int)i<<" "<<i<<endl;
    }*/

    return EXIT_SUCCESS;
}
