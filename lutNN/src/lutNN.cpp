//============================================================================
// Name        : lutNN.cpp
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
#include "TPaveLabel.h"

#include "lutNN/interface/Node.h"
#include "lutNN/interface/LutNetwork.h"
#include "lutNN/interface/Utlis.h"


using namespace lutNN;
using namespace std;

void generateEventsElipse(vector<Event>& events, unsigned int inputLutSize, std::default_random_engine& generator) {
    std::uniform_int_distribution<int> distribution(0, inputLutSize);
    unsigned int b = inputLutSize/2;
    for(auto& event : events) {
        double res = 0;
        for(auto in : event.inputs) {
            in = distribution(generator);
            res = pow( (in - b) / 10., 2);
        }
        event.expextedResult[0] = res;
    }
}

int main(void) {
    //puts("Hello World!!!");

    /*	std::shared_ptr<ConfigParameters> config(new ConfigParameters() );
	config->lutAddrOffset = 0;
	cout<<"config: lutSize "<<config->lutSize<<" lutAddrOffset "<<config->lutAddrOffset<<endl;*/

    unsigned int inputCnt = 2;
    unsigned int outputCnt = 1;
    unsigned int lutSize = 2<<6;
    std::vector<std::shared_ptr<ConfigParameters> > layersDef;
    layersDef.emplace_back(ConfigParametersPtr(new ConfigParameters(7, lutSize, lutSize, 2) ) );
    //layersDef.emplace_back(ConfigParametersPtr(new ConfigParameters(2, lutSize, lutSize, 2) ) );
    //layersDef.emplace_back(ConfigParametersPtr(new ConfigParameters(2, lutSize, lutSize/2, 2) ) );
    layersDef.emplace_back(ConfigParametersPtr(new ConfigParameters(outputCnt, lutSize, lutSize, 2) ) );


    //making sure the values of neuronInputCnt are correct
    for(unsigned int iLayer = 0; iLayer < layersDef.size(); iLayer++ ) {
        if(iLayer == 0)
            layersDef[iLayer]->neuronInputCnt = inputCnt;
        else {
            layersDef[iLayer]->neuronInputCnt = layersDef[iLayer - 1]->neuronsInLayer;
        }
    }

    LutNetwork lutNetwork(inputCnt, layersDef);

    std::cout<<lutNetwork<<std::endl;

    unsigned int maxNodesPerLayer = 0;
    for(unsigned int iLayer = 0; iLayer < lutNetwork.getLayers().size(); iLayer++ ) {
        if(maxNodesPerLayer < lutNetwork.getLayers()[iLayer].size() )
            maxNodesPerLayer = lutNetwork.getLayers()[iLayer].size();
    }
    cout<<"maxNodesPerLayer "<<maxNodesPerLayer<<endl;

    //--------------------------------------------------
    gStyle->SetOptStat(0);
    TCanvas canvasLutNN("canvasLutNN", "canvasLutNN", 1600, 900);
    canvasLutNN.cd();
    TPad padLuts("canvasLuts", "canvasLuts", 0.0, 0.0, 0.7, 1.);
    padLuts.SetGrid();
    padLuts.Divide(lutNetwork.getLayers().size(), maxNodesPerLayer * 2);
    padLuts.Draw();

    TPad padOutMap("padOutMap", "padOutMap", 0.7, 0., 1., 0.5);
    padOutMap.SetRightMargin(0.2);
    padOutMap.Draw();


    printLuts(padLuts, lutNetwork, maxNodesPerLayer);
    printOutMap(padOutMap,  lutNetwork, lutSize);

    canvasLutNN.SaveAs("../pictures/canvasLutNN_0.png");

    /*padLuts.Write();
	padOutMap.Write();*/

    canvasLutNN.cd();
    TPad padTargetMap("padTargetMap", "padTargetMap", 0.7, 0.5, 1., 1.);
    padTargetMap.SetRightMargin(0.2);
    padTargetMap.Draw();
    TH2F* histTargetMap =  new TH2F("histTargetMap", "target", lutSize, -0.5, lutSize - 0.5, lutSize, -0.5, lutSize - 0.5);


    vector<Event> events(lutSize * lutSize, {inputCnt, outputCnt} );
    for(int iX = 0; iX < lutSize; iX++) {
        for(int iY = 0; iY < lutSize; iY++) {
            events[iX * lutSize + iY].inputs[0] = iX;
            events[iX * lutSize + iY].inputs[1] = iY;
            events[iX * lutSize + iY].expextedResult[0] = 0;
            /*if(iX == 80 && iY == 80)
                events[iX * lutSize + iY].expextedResult[0] = 1;*/
            //events[iX * lutSize + iY].expextedResult[0] = 1 * pow( (iX - lutSize/2.)/10., 2) + 1 * pow( (iY - lutSize/2.)/10., 2) + 0.005* (iX - lutSize/2.) * (iY - lutSize/2.);

            double val = 0; //1 * pow( (iX - lutSize/5.)/10., 2) + 1 * pow( (iY - lutSize/5.)/10., 2) + 0.007* (iX - lutSize/5.) * (iY - lutSize/4.);
            //val += 30 * exp(- ( pow( (iX - 1*lutSize/7.)/40., 2) + pow( (iY - 1.*lutSize/7.)/30., 2)  + 0.0007* (iX - 1.*lutSize/7.) * (-iY - 1.*lutSize/7.) ));
            //val += 30 * exp(- ( pow( (iX - 4*lutSize/5.)/30., 2) + pow( (iY - 4.*lutSize/5.)/40., 2)  - 0.0007* (iX - 4.*lutSize/5.) * (iY - 4.*lutSize/5.) ));

            val +=  30 * exp(- ( pow( (iX - 1.*lutSize/2.)/70., 2) + pow( (iY - 1.*lutSize/2. )/70., 2) ));//  + 0.0007* (iX - 1.*lutSize/2.) * (-iY - 1.*lutSize/2.) ));
            val += -30 * exp(- ( pow( (iX - 1.*lutSize/2.)/20., 2) + pow( (iY - 1.*lutSize/2. )/20., 2) )); // - 0.0007* (iX - 1.*lutSize/3.) * (iY - 1.*lutSize/3.) ));

            events[iX * lutSize + iY].expextedResult[0] = val;
            //1 * pow( (iX - 3.*lutSize/4.)/10., 2) + 1 * pow( (iY - 3.*lutSize/4.)/10., 2);

            histTargetMap->Fill(iX, iY, events[iX * lutSize + iY].expextedResult[0]);
        }
    }

    /*	vector<Event> events(1, {2, 1} );
	events[0].inputs[0] = 32;
	events[0].inputs[1] = 32;
	events[0].expextedResult[0] = 2;*/

    padTargetMap.cd();
    histTargetMap->Draw("colz");

    LutNetwork::LearnigParams learnigParams;
    learnigParams.learnigRate = 0;//0.01 ;
    learnigParams.stretchRate = 0.2 *  0.05;
    learnigParams.regularizationRate = 0;
    std::vector<LutNetwork::LearnigParams> learnigParamsVec(layersDef.size(), learnigParams);

    //learnigRates[0] = 0.02;
    //learnigRates[1] = 0.02;

    TPaveLabel textCosts(0.4, .01, 0.7, 0.1, "aaaa");
    canvasLutNN.cd();
    ostringstream ostr;
    ostr<<"cost";
    textCosts.Draw();

    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0,9);
    int mode = 0;
    for(int i = 0; i < 30000; i ++) {
        std::cout<<"runTraining iteration "<<i<<std::endl;

        //these iterations will just stretch the output values such that they fill the LUT inputs of the next layer evenly
        if(i == 0) { //not sure if needed
            //unsigned int layer = 0;
            for(auto& learnigParams :learnigParamsVec ) {
                learnigParams.learnigRate = 0 ;
                learnigParams.stretchRate = 0.2 *  0.5;
                learnigParams.regularizationRate = 0;
            }
            mode = 0;
        }

        //these iterations will just capture the range of the expected values (i.e. the network outputs)
        if(i == 100) { //not sure if needed
            //unsigned int layer = 0;
            for(auto& learnigParams :learnigParamsVec ) {
                learnigParams.learnigRate = 0.1 ;
                learnigParams.stretchRate = 0. * learnigParams.learnigRate;
                learnigParams.regularizationRate = 0.5 * learnigParams.learnigRate;
            }
            mode = 1; //in mode 1 only the LUTs in the last layer are updated
        }

        //here normal trainig starts
        if(i == 200) { //not sure if needed
            //unsigned int layer = 0;
            for(auto& learnigParams :learnigParamsVec ) {
                learnigParams.learnigRate = 0.01 ;
                learnigParams.stretchRate = 0.1 * learnigParams.learnigRate;
                learnigParams.regularizationRate = 0.8 * learnigParams.learnigRate;
            }
            mode = 0;
        }

        else if(i == 30000) { //not sure if needed
            for(auto& learnigParams :learnigParamsVec ) {
                learnigParams.learnigRate = 0.003 ;
                learnigParams.stretchRate = 0. * learnigParams.learnigRate;
                learnigParams.regularizationRate = 0.3 * learnigParams.learnigRate;
            }
        }

        for(auto& event : events) {
            if(distribution(generator) < 8) //stochastic learning
                continue;

            lutNetwork.getInputNodes()[0]->setInput(event.inputs[0]);
            lutNetwork.getInputNodes()[1]->setInput(event.inputs[1]);

            lutNetwork.runTraining(event.expextedResult, mode);

            //std::cout<<" "<<std::endl;
        }

        lutNetwork.updateLuts(learnigParamsVec);

        if( (i)%100 == 1)
        {
            printLuts(padLuts, lutNetwork, maxNodesPerLayer);
            printOutMap(padOutMap,  lutNetwork, lutSize);

            ostr.str("");
            ostr<<"iteration "<<setw(7)<<i<<" mean cost "<<setw(5)<<lutNetwork.getMeanCost();
            textCosts.SetLabel(ostr.str().c_str() );

            padLuts.Update();
            padOutMap.Update();
            canvasLutNN.Update();
            canvasLutNN.SaveAs( ("../pictures/canvasLutNN_" + to_string(i+1) + ".png").c_str() );
        }
        lutNetwork.resetStats();
    }

    TFile outfile("../pictures/lutNN.root", "RECREATE");
    outfile.cd();
    canvasLutNN.Write();

    std::cout<<"done - OK"<<std::endl;
    return EXIT_SUCCESS;
}
