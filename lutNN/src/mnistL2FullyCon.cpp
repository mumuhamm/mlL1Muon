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

    std::default_random_engine& generator;

    std::string filename;
    std::vector<std::vector<uint8_t> > images;
    std::vector<uint8_t> labels;

    vector<Event*> events;

    unsigned int rowCnt = 0;
    unsigned int columnCnt = 0;
public:
    EventsGeneratorMnist(unsigned int inputLutSize, std::string imageFile, std::string labelsFile, std::default_random_engine& generator):
        EventsGeneratorBase(), inputLutSize(inputLutSize), imageFile(imageFile), labelsFile(labelsFile) , generator(generator) {

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
};

void EventsGeneratorMnist::printImage(unsigned int num, ostream& ostr) {
    std::vector<uint8_t> image = images.at(num);
    for(unsigned int column = 0; column < columnCnt; column++)
        ostr<<"#";
    ostr<<"#"<<endl;
    for(unsigned int row = 0; row < rowCnt; row++) {
        for(unsigned int column = 0; column < columnCnt; column++) {
            int indx= row * columnCnt + column;
            if(image[indx] == 0)
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


                unsigned int a = 3;

                for(unsigned int iInput = 0; iInput < event->inputs.size(); iInput++) {
                    event->inputs[iInput] = 0;
                }
                for(unsigned int row = 0; row < rowCnt; row++) {
                    for(unsigned int column = 0; column < columnCnt; column++) {
                        if(row + yS < 0 || row + yS >= rowCnt ||
                           column + xS < 0 || column + xS >= columnCnt)
                            continue;
                        if(image[ (row + yS) * columnCnt + (column + xS) ] > 150) {
                            unsigned int iInput =  column / a + row / a * (columnCnt +2)/ a; //TODO +2 is if columnCnt%a != 0
                            unsigned int iBit = column % a + row%a * a;
                            event->inputs[iInput] |= (1<<iBit);
                            //cout<<column<<" "<<row<<" "<<iInput<<" "<<" "<<iBit<<" "<<event->inputs[iInput]<<endl;
                        }
                    }
                }

                events.at(iEvent++) = event;
            }
        }
    }
}

void EventsGeneratorMnist::getRandomEvents(vector<Event*>& rndEvents) {
    std::uniform_int_distribution<unsigned int> randomEventDist(0, this->events.size() -1);
    for(auto& event : rndEvents) {
        unsigned int num = randomEventDist(generator);
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
    unsigned int inputCnt = 100;// +1;

    unsigned int outputCnt = 10;
    unsigned int lutSize = (1<<9) -1;
    cout<<"lutSize "<<lutSize<<endl;
    std::vector<std::shared_ptr<ConfigParameters> > layersDef;
    unsigned int neuronCntL1 = 30;
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
    learnigParams.regularizationRate = 0;
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

    TH1F* costHist = createCostHistory(iterations, printEveryIteration);
    TCanvas canvasCostHist("canvasCostHist", "canvasCostHist", 1400, 800);
    canvasCostHist.SetLogy();

    string mnistDir = "/afs/cern.ch/work/k/kbunkow/private/lutNN/mnist/";

    EventsGeneratorMnist eventsGeneratorTrain(lutSize, mnistDir + "train-images-idx3-ubyte", mnistDir + "train-labels-idx1-ubyte", generator);
    eventsGeneratorTrain.generateEvents(inputCnt, outputCnt, 2, 2);

    EventsGeneratorMnist eventsGeneratorTest(lutSize, mnistDir + "t10k-images-idx3-ubyte", mnistDir + "t10k-labels-idx1-ubyte", generator);
    eventsGeneratorTest.generateEvents(inputCnt, outputCnt);

    vector<Event*>& testEvents = eventsGeneratorTest.getEvents(); //TODO look for the number

    for(unsigned int i = 0; i < 3; i++)
        eventsGeneratorTrain.printImage(i);

    for(unsigned int i = 0; i < 20; i++) {
        cout<<" event "<<i<<endl;
        Event* event = eventsGeneratorTrain.getEvents()[i];

        for(unsigned int row = 0; row < 10; row++) {
            for(unsigned int column = 0; column < 10; column++) {
                cout<<setw(3)<<hex<<event->inputs[row * 10 + column];
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
        printGradinets(padGradients, lutNetwork, maxNodesPerLayer * 2);

        for(auto& event : events) {
            lutNetwork.run(event);
            //std::cout<<" "<<std::endl;
        }


        printExpectedVersusNNOutHotOne(padOutMap, ranges, events);
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

        //eventsGeneratorTest.generateEvents(testEvents, 1);
        for(auto& event : testEvents) {
            lutNetwork.setInputs(event);
            //lutNetwork.runTrainingInter(event->expextedResult); //fills the statistics
            lutNetwork.runTraining(event->expextedResult, 0); //fills the statistics
            //std::cout<<" "<<std::endl;
        }

        if(iLayer == 0) {//TODO!!!!!!!!!!!!
            lutNetwork.weightByEvents(iLayer);
            lutNetwork.resetStats();
            for(auto& event : testEvents) {
                lutNetwork.setInputs(event);
                //lutNetwork.runTrainingInter(event->expextedResult); //fills the statistics
                lutNetwork.runTraining(event->expextedResult, 0); //fills the statistics
                //std::cout<<" "<<std::endl;
            }
        }

        printNN(iteration, testEvents);

        if(iLayer < lutNetwork.getLayers().size() -1)
            lutNetwork.shiftAndRescale(iLayer, shiftRatio, scaleRatio);

        lutNetwork.resetStats();
        iteration++;
    }

    for(int i = 0; i < iterations; i++) {
        //for 2 layers network
        double speed = 0.5;
        if(i == 0) { //not sure if needed
            learnigParamsVec[0].learnigRate = speed * 5;//0.5;//0.5;//2.;
            learnigParamsVec[0].beta = 0.95;
            learnigParamsVec[0].lambda = 0.001;
            learnigParamsVec[0].regularizationRate = 0;// * learnigParams.learnigRate;

            learnigParamsVec[1].learnigRate = speed * 0.3 ;
            learnigParamsVec[1].beta = 0.95;
            learnigParamsVec[1].lambda = 0.0005;
            learnigParamsVec[1].regularizationRate = 1;//0.1;// * learnigParams.learnigRate;
        }
        if(i == 1500) { //not sure if needed
            learnigParamsVec[0].learnigRate = speed * 4;//0.5;//0.5;//2.;
            learnigParamsVec[0].beta = 0.95;
            learnigParamsVec[0].lambda = 0.00003;
            learnigParamsVec[0].regularizationRate = 0;// * learnigParams.learnigRate;

            learnigParamsVec[1].learnigRate = speed * 0.3 ;
            learnigParamsVec[1].beta = 0.95;
            learnigParamsVec[1].lambda = 0.0001;
            learnigParamsVec[1].regularizationRate = 1;//0.1;// * learnigParams.learnigRate;
        }
        if(i == 10000) { //not sure if needed
            learnigParamsVec[0].learnigRate = speed * 8;//0.5;//0.5;//2.;
            learnigParamsVec[0].beta = 0.95;
            learnigParamsVec[0].lambda = 0.00001;
            learnigParamsVec[0].regularizationRate = 0;// * learnigParams.learnigRate;

            learnigParamsVec[1].learnigRate = speed * 0.3 ;
            learnigParamsVec[1].beta = 0.95;
            learnigParamsVec[1].lambda = 0.0001;
            learnigParamsVec[1].regularizationRate = 1;//0.1;// * learnigParams.learnigRate;
        }
       /* if(i == 500) { //not sure if needed
            speed = 0.1;
            learnigParamsVec[0].learnigRate = speed * 8;//0.5;//0.5;//2.;
            learnigParamsVec[0].beta = 0.95;
            learnigParamsVec[0].lambda = 0.006;
            learnigParamsVec[0].regularizationRate = 0;// * learnigParams.learnigRate;

            learnigParamsVec[1].learnigRate = speed * 0.3 ;
            learnigParamsVec[1].beta = 0.95;
            learnigParamsVec[1].lambda = 0.0007;
            learnigParamsVec[1].regularizationRate = 1;//0.1;// * learnigParams.learnigRate;
        }*/
        /*if(i == 1500) { //not sure if needed
            speed = 0.1;
            learnigParamsVec[0].learnigRate = speed * 12;//0.5;//0.5;//2.;
            learnigParamsVec[0].beta = 0.95;
            learnigParamsVec[0].lambda = 0.001;
            learnigParamsVec[0].regularizationRate = 0;// * learnigParams.learnigRate;

            learnigParamsVec[1].learnigRate = speed * 0.1 ;
            learnigParamsVec[1].beta = 0.95;
            learnigParamsVec[1].lambda = 0.001;
            learnigParamsVec[1].regularizationRate = 1;//0.1;// * learnigParams.learnigRate;
        }
        if(i == 2600) { //not sure if needed
            speed = 0.1;
            learnigParamsVec[0].learnigRate = speed * 25;//0.5;//0.5;//2.;
            learnigParamsVec[0].beta = 0.95;
            learnigParamsVec[0].lambda = 0.001;
            learnigParamsVec[0].regularizationRate = 0;// * learnigParams.learnigRate;

            learnigParamsVec[1].learnigRate = speed * 0.1 ;
            learnigParamsVec[1].beta = 0.95;
            learnigParamsVec[1].lambda = 0.001;
            learnigParamsVec[1].regularizationRate = 1;//0.1;// * learnigParams.learnigRate;
        }*/

        if(stochastic)
            eventsGeneratorTrain.getRandomEvents(trainingEvents); //TODO!!!!!!!!!!!!!!!!!!!!!!!!!!

        lutNetwork.setDropOut(retainProb, generator);

        for(auto& event : trainingEvents) {
            lutNetwork.setInputs(event);
            //lutNetwork.runTrainingInter(event->expextedResult);
            lutNetwork.runTraining(event->expextedResult, 0);
            //std::cout<<" "<<std::endl;
        }

        lutNetwork.updateLuts(learnigParamsVec);
        if(i%5 == 0)
            lutNetwork.smoothLuts(learnigParamsVec); //TODO!!!!!!!!!!!!!!!


        /*if(i == 1000)
            lutNetwork.shiftAndRescale(0, 1, 1);
*/
/*        if( i < 0) { //TODO!!!!
            //scaling and shifting the initial LUT contents so that the layer outputs fill the next layer input address
            for(unsigned int iLayer = 0; iLayer < lutNetwork.getNeuronLayers().size() -1; iLayer++) {
                lutNetwork.shiftAndRescale(iLayer, 1, 1);
            }
        }*/

        if( (i)%printEveryIteration == 0 ) { //|| (i > 900 && i < 1000)
            lutNetwork.setLutOutWeight(retainProb);

            printNN(i, testEvents);

            lutNetwork.printLayerStat();
            //printNN(i+1, testEvents);
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
