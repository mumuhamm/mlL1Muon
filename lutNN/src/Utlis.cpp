/*
 * Utlis.cpp
 *
 *  Created on: Jul 3, 2018
 *      Author: kbunkow
 */

#include "lutNN/interface/Utlis.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <iomanip>

#include "TH2I.h"
#include "TH1F.h"
#include "TFile.h"
#include "TStyle.h"
#include "THStack.h"
#include "TText.h"
#include "TPaveLabel.h"

namespace lutNN {
using namespace std;

Utlis::Utlis() {
    // TODO Auto-generated constructor stub

}

Utlis::~Utlis() {
    // TODO Auto-generated destructor stub
}



void printLuts(TPad& padLuts, LutNetwork& lutNetwork, unsigned int maxNodesPerLayer) {
    for(unsigned int iLayer = 0; iLayer < lutNetwork.getLayers().size(); iLayer++ ) {
        for(unsigned int iNode = 0; iNode < lutNetwork.getLayers()[iLayer].size(); iNode++) {
            ostringstream ostrLut;
            ostrLut<<"layer_"<<iLayer<<"_neuron_"<<iNode;
            string strEntries = ostrLut.str() + "Entries";
            TObject* obj =  gDirectory->Get(ostrLut.str().c_str());
            THStack* histLutStack = 0;
            THStack* histLutEntriesStack = 0;
            if(obj == 0) {
                histLutStack = new THStack(ostrLut.str().c_str(), ostrLut.str().c_str());
                gDirectory->Add(histLutStack);
                histLutEntriesStack =  new THStack(strEntries.c_str(), strEntries.c_str());
                gDirectory->Add(histLutEntriesStack);
            }
            else {
                histLutStack = (THStack*)obj;
                histLutEntriesStack = (THStack*)gDirectory->Get(strEntries.c_str());
            }
            for(unsigned int iLut = 0; iLut < lutNetwork.getLayers()[iLayer][iNode]->getLuts().size(); iLut++) {
                Lut* lut = lutNetwork.getLayers()[iLayer][iNode]->getLuts()[iLut].get();
                ostringstream ostr;
                ostr<<"layer_"<<iLayer<<"_neuron_"<<iNode<<"_lut_"<<iLut;
                TObject* histObj =  gDirectory->Get(ostr.str().c_str());
                TH1F* histLut = 0;
                TH1F* histEntries = 0;
                strEntries = ostr.str() + "Entries";
                if(histObj == 0) {
                    histLut =  new TH1F(ostr.str().c_str(), ostr.str().c_str(), lut->getValues().size(), -0.5, lut->getValues().size() - 0.5 );
                    if(iLayer != lutNetwork.getLayers().size() -1) {
                        //histLut->GetYaxis()->SetRangeUser(lut->getValues().size()/(-2.), lut->getValues().size()/2);
                        //double range = lutNetwork.getNeuronLayers()[iLayer][iNode]->getLuts()[iLut]->getValues().size()/2;
                        //histLut->GetYaxis()->SetRangeUser(-range, range);
                    }
                    cout<<__FUNCTION__<<":"<<__LINE__<<" creating hist "<<ostr.str()<<endl;
                    histLutStack->Add(histLut);

                    histEntries = new TH1F(strEntries.c_str(), strEntries.c_str(), lut->getValues().size(), -0.5, lut->getValues().size() - 0.5 );
                    histLutEntriesStack->Add(histEntries);
                    int color = iLut + 2;
                    if (color == kWhite)
                        color = kMagenta + 3;
                    histLut->SetLineColor(color);
                    histEntries->SetLineColor(color);
                }
                else {
                    histLut = (TH1F*)histObj;
                    histEntries = (TH1F*)gDirectory->Get(strEntries.c_str());
                    //cout<<__FUNCTION__<<":"<<__LINE__<<" got hist "<<ostr.str()<<" from gDirectory"<<endl;
                }

                for(unsigned int iAddr = 0; iAddr < lut->getValues().size(); iAddr++) {
                    histLut->SetBinContent(iAddr +1, lut->getValues()[iAddr]);
                    histEntries->SetBinContent(iAddr +1, lut->getLutStat()[iAddr].entries);
                    //cout<<__FUNCTION__<<":"<<__LINE__<<" "<<lut->getLutStat()[iAddr].entries<<endl;
                }
                /*if(iLut == 0) {
                    histLut->Draw("hist");
                }
                else
                    histLut->Draw("histsame");*/
            }
            if(obj == 0) {
                padLuts.cd(iLayer + 2 * iNode * lutNetwork.getLayers().size()  +1)->SetGrid();
                histLutStack->Draw("nostackhist");

                padLuts.cd(iLayer + (2*iNode+1) * lutNetwork.getLayers().size()  +1)->SetGrid();
                padLuts.cd(iLayer + (2*iNode+1) * lutNetwork.getLayers().size()  +1)->SetLogy();
                histLutEntriesStack->Draw("nostackhist");
            }
        }
    }
}

void printLuts2(TPad& padLuts, TPad& padGradients, LutNetwork& lutNetwork, unsigned int maxNodesPerLayer) {
    //cout<<" gDirectory->GetName() "<<gDirectory->GetName()<<endl;
    //gDirectory->pwd();
    for(unsigned int iLayer = 0; iLayer < lutNetwork.getLayers().size(); iLayer++ ) {
        for(unsigned int iNode = 0; iNode < lutNetwork.getLayers()[iLayer].size(); iNode++) {

            auto getHistStac = [&](string name, TPad& pad, int padOffset, bool logY) {
                ostringstream ostr;
                ostr<<name<<"_layer_"<<iLayer<<"_neuron_"<<iNode;

                TObject* obj =  gDirectory->Get(ostr.str().c_str());
                THStack* histStack = 0;
                if(obj == 0) {
                    histStack = new THStack(ostr.str().c_str(), ostr.str().c_str());
                    gDirectory->Add(histStack);

                    for(unsigned int iLut = 0; iLut < lutNetwork.getLayers()[iLayer][iNode]->getLuts().size(); iLut++) {
                        Lut* lut = lutNetwork.getLayers()[iLayer][iNode]->getLuts()[iLut].get();
                        ostringstream ostrHistName;
                        ostrHistName<<ostr.str()<<"_lut_"<<iLut;
                        TH1F* hist  =  new TH1F(ostrHistName.str().c_str(), ostrHistName.str().c_str(), lut->getValues().size(), -0.5, lut->getValues().size() - 0.5 );
                        if(iLayer != lutNetwork.getLayers().size() -1) {
                            //histLut->GetYaxis()->SetRangeUser(lut->getValues().size()/(-2.), lut->getValues().size()/2);
                            //double range = lutNetwork.getNeuronLayers()[iLayer][iNode]->getLuts()[iLut]->getValues().size()/2;
                            //histLut->GetYaxis()->SetRangeUser(-range, range);
                        }
                        //cout<<__FUNCTION__<<":"<<__LINE__<<" creating hist "<<ostrHistName.str()<<endl;
                        histStack->Add(hist);
                        int color = iLut + 2;
                        if (color == 10)
                            color =  46;
                        hist->SetLineColor(color);
                    }

                    if(iNode < maxNodesPerLayer) {
                        pad.cd(iLayer + (2*iNode+padOffset) * lutNetwork.getLayers().size()  +1)->SetGrid();
                        if(logY)
                            pad.cd(iLayer + (2*iNode+padOffset) * lutNetwork.getLayers().size()  +1)->SetLogy();

                        histStack->Draw("nostackhist");
                        /*if(name == "GradientSum") {
                        histStack->SetMinimum(-0.02);
                        histStack->SetMaximum( 0.02);
                    }
                    if(name == "AbsGradientSum") {
                        histStack->SetMinimum(0);
                        histStack->SetMaximum(0.2);
                    }*/
                        histStack->Draw("nostackhist");
                        pad.cd(iLayer + (2*iNode+padOffset) * lutNetwork.getLayers().size()  +1)->Update();
                    }
                }
                else {
                    histStack = (THStack*)obj;
                }
                return histStack;
            };

            getHistStac("Values", padLuts, 0, false);
            getHistStac("Entries", padLuts, 1, true);
            //getHistStac("GradientSum", padGradients, 0, false);
            //getHistStac("AbsGradientSum", padGradients, 1, false);

            for(unsigned int iLut = 0; iLut < lutNetwork.getLayers()[iLayer][iNode]->getLuts().size(); iLut++) {
                Lut* lut = lutNetwork.getLayers()[iLayer][iNode]->getLuts()[iLut].get();
                ostringstream ostr;
                ostr<<"_layer_"<<iLayer<<"_neuron_"<<iNode<<"_lut_"<<iLut;

                TH1F* histLut = (TH1F*)gDirectory->Get( ("Values" + ostr.str()).c_str());
                TH1F* histEntries = (TH1F*)gDirectory->Get( ("Entries" + ostr.str()).c_str());
                //TH1F* histGradientSum = (TH1F*)gDirectory->Get( ("GradientSum" + ostr.str()).c_str());
                //TH1F* histAbsGradientSum = (TH1F*)gDirectory->Get( ("AbsGradientSum" + ostr.str()).c_str());

                for(unsigned int iAddr = 0; iAddr < lut->getValues().size(); iAddr++) {
                    histLut->SetBinContent(iAddr +1, lut->getValues()[iAddr]);
                    histEntries->SetBinContent(iAddr +1, lut->getLutStat()[iAddr].entries);

                    //histGradientSum->SetBinContent(iAddr +1, lut->getLutStat()[iAddr].gradientSumPos);
                    //histAbsGradientSum->SetBinContent(iAddr +1, lut->getLutStat()[iAddr].absGradientSum);

                    //cout<<__FUNCTION__<<":"<<__LINE__<<" "<<lut->getLutStat()[iAddr].entries<<endl;
                }
            }
        }
    }
}


void printGradinets(TPad& padGradients, LutNetwork& lutNetwork, unsigned int maxHistsPerLayer) {
    //cout<<" gDirectory->GetName() "<<gDirectory->GetName()<<endl;
    //gDirectory->pwd();
    for(unsigned int iLayer = 0; iLayer < lutNetwork.getLayers().size(); iLayer++ ) {
        unsigned int iPad = 0;
        for(unsigned int iNode = 0; iNode < lutNetwork.getLayers()[iLayer].size(); iNode++) {
            auto getHists = [&](string name, TPad& pad) {
                ostringstream ostr;
                ostr<<name<<"_layer_"<<iLayer<<"_neuron_"<<iNode;

                for(unsigned int iLut = 0; iLut < lutNetwork.getLayers()[iLayer][iNode]->getLuts().size(); iLut++) {
                    Lut* lut = lutNetwork.getLayers()[iLayer][iNode]->getLuts()[iLut].get();
                    ostringstream ostrHistName;
                    ostrHistName<<ostr.str()<<"_lut_"<<iLut;

                    TObject* obj =  gDirectory->Get(ostrHistName.str().c_str());
                    if(obj == 0) {
                        TH1F* hist  =  new TH1F(ostrHistName.str().c_str(), ostrHistName.str().c_str(), lut->getValues().size(), -0.5, lut->getValues().size() - 0.5 );

                        //cout<<__FUNCTION__<<":"<<__LINE__<<" creating hist "<<ostrHistName.str()<<endl;
                        int color = iLut + 2;
                        if (color == 10)
                            color =  46;
                        hist->SetLineColor(color);

                        if(iPad < maxHistsPerLayer) {
                            pad.cd(iLayer + iPad * lutNetwork.getLayers().size() +1 )->SetGrid();
                            if(iLayer == lutNetwork.getLayers().size() -1)
                                hist->GetYaxis()->SetRangeUser(-0.005, 0.005);
                            else
                                hist->GetYaxis()->SetRangeUser(-0.05, 0.05);
                            hist->Draw("hist");
                        }
                        iPad++;
                    }
                }
                /*
                if(logY)
                    pad.cd(iLayer + (iNode+padOffset) * lutNetwork.getNeuronLayers().size()  +1)->SetLogy();

                pad.cd(iLayer + (iNode+padOffset) * lutNetwork.getNeuronLayers().size()  +1)->Update();*/
            };

            getHists("momentum", padGradients);

            for(unsigned int iLut = 0; iLut < lutNetwork.getLayers()[iLayer][iNode]->getLuts().size(); iLut++) {
                Lut* lut = lutNetwork.getLayers()[iLayer][iNode]->getLuts()[iLut].get();
                ostringstream ostr;
                ostr<<"_layer_"<<iLayer<<"_neuron_"<<iNode<<"_lut_"<<iLut;

                TH1F* histGradientSum = (TH1F*)gDirectory->Get( ("momentum" + ostr.str()).c_str());

                for(unsigned int iAddr = 0; iAddr < lut->getValues().size(); iAddr++) {
                    histGradientSum->SetBinContent(iAddr +1, lut->getLutStat()[iAddr].momentum);
                    //cout<<__FUNCTION__<<":"<<__LINE__<<" "<<lut->getLutStat()[iAddr].entries<<endl;
                }
            }
        }
    }
}

void printOutMap(TPad& padOutMap,  LutNetwork& lutNetwork, unsigned int& lutSize) {
    string histName = "lutNNOutMap";

    TObject* histObj =  gDirectory->Get(histName.c_str());
    TH2F* hist = 0;
    if(histObj == 0) {
        hist =  new TH2F(histName.c_str(), histName.c_str(), lutSize, -0.5, lutSize - 0.5, lutSize, -0.5, lutSize - 0.5);
        cout<<__FUNCTION__<<":"<<__LINE__<<" creating hist "<<histName<<endl;
    }
    else {
        hist = (TH2F*)histObj;
        //cout<<__FUNCTION__<<":"<<__LINE__<<" got hist "<<histName<<" from gDirectory"<<endl;
    }

    for(int iX = 0; iX < lutSize; iX++) {
        for(int iY = 0; iY < lutSize; iY++) {
            lutNetwork.getInputNodes()[0]->setInput(iX);
            lutNetwork.getInputNodes()[1]->setInput(iY);
            lutNetwork.run();
            hist->SetBinContent(iX +1, iY +1, lutNetwork.getOutputValues()[0]);
            //cout<<" neuron.getOutAddr(): "<<neuron.getOutAddr()<<endl;
        }
    }

    padOutMap.cd();
    hist->SetStats(0);
    hist->Draw("colz");
}

double printExpectedVersusNNOut(TPad& padOutMap,  std::vector<std::pair<double, double> >& ranges, std::vector<Event*>& events) {
    double cost = 0;
    for(unsigned int iOut = 0; iOut < ranges.size(); iOut++) {
        ostringstream ostr;
        ostr<<"hist_iOut_"<<iOut;
        string histName = ostr.str();;

        TObject* histObj =  gDirectory->Get(histName.c_str());
        TH2F* hist = 0;
        if(histObj == 0) {
            hist =  new TH2F(histName.c_str(), histName.c_str(), 50, ranges[iOut].first, ranges[iOut].second, 50, ranges[iOut].first, ranges[iOut].second);
            cout<<__FUNCTION__<<":"<<__LINE__<<" creating hist "<<histName<<endl;
        }
        else {
            hist = (TH2F*)histObj;
            hist->Reset();
            //cout<<__FUNCTION__<<":"<<__LINE__<<" got hist "<<histName<<" from gDirectory"<<endl;
        }

        unsigned int resultsCnt = 10;//TODO!!!!!!!!!!!!
        vector<unsigned int> wellClassifiedCnt(resultsCnt, 0);
        vector<unsigned int> allCnt(resultsCnt, 0);
        unsigned int totalEfficiency = 0;
        for(auto& event : events) {
            hist->Fill(event->expextedResult[iOut], event->nnResult[iOut]);
            cost += pow(event->expextedResult[iOut] - event->nnResult[iOut], 2);
            //cout<<" neuron.getOutAddr(): "<<neuron.getOutAddr()<<endl;

            if(round(event->nnResult[iOut]) == event->expextedResult[iOut]) {
                wellClassifiedCnt[event->expextedResult[iOut] ]++;
                totalEfficiency++;
            }
            allCnt[event->expextedResult[iOut] ]++;

        }
        padOutMap.cd(iOut + 1)->SetGrid();
        hist->SetStats(0);
        hist->Draw("colz");

        for(unsigned int iOut = 0; iOut < wellClassifiedCnt.size(); iOut++) {
            cout<<iOut<<" eff: "<<setw(10)<<wellClassifiedCnt[iOut]/(double)allCnt[iOut]<<endl;
        }

        double totalEff = totalEfficiency/(double)events.size();
        cout<<" total eff: "<<setw(10)<<totalEff<<" mismatch "<<(1-totalEff)<<endl;
    }
    cout<<" training sample cost: "<<(cost / events.size() )<<endl;
    return cost;

}

double printExpectedVersusNNOutHotOne(TPad& padOutMap,  std::vector<std::pair<double, double> >& ranges, std::vector<Event*>& events, CostFunction& costFunction) {
    string histName = "histOutValues";

    TObject* histObj =  gDirectory->Get(histName.c_str());
    TH2F* histOutValues = 0;
    unsigned int resultsCnt = events.at(0)->expextedResult.size();
    if(histObj == 0) {
        histOutValues =  new TH2F(histName.c_str(), histName.c_str(), resultsCnt, -.5, resultsCnt -0.5, 50, -1, 2);
        padOutMap.cd(1)->SetGrid();
        histOutValues->SetStats(0);
        histOutValues->Draw("box");

        cout<<__FUNCTION__<<":"<<__LINE__<<" creating "<<histName<<endl;
    }
    else {
        histOutValues = (TH2F*)histObj;
        histOutValues->Reset();
        //cout<<__FUNCTION__<<":"<<__LINE__<<" got histOutValues "<<histName<<" from gDirectory"<<endl;
    }

    TH2F* histOutVsExpected = 0;
    histName = "histOutVsExpected";
    histObj =  gDirectory->Get(histName.c_str());
    if(histObj == 0) {
        histOutVsExpected =  new TH2F(histName.c_str(), histName.c_str(), resultsCnt, -.5, resultsCnt -0.5, resultsCnt, -.5, resultsCnt -0.5);
        padOutMap.cd(2)->SetGrid();
        histOutVsExpected->SetStats(0);
        histOutVsExpected->Draw("colz");
        cout<<__FUNCTION__<<":"<<__LINE__<<" creating "<<histName<<endl;
    }
    else {
        histOutVsExpected = (TH2F*)histObj;
        histOutVsExpected->Reset();
        //cout<<__FUNCTION__<<":"<<__LINE__<<" got histOutValues "<<histName<<" from gDirectory"<<endl;
    }

    double cost = 0;

    vector<unsigned int> wellClassifiedCnt(resultsCnt, 0);
    vector<unsigned int> allCnt(resultsCnt, 0);
    unsigned int totalEfficiency = 0;
    for(auto& event : events) {
        double maxOutVal = -10000;
        unsigned int maxOutNum = 0;
        unsigned int expectedOutNum = 0;

        cost += costFunction(event->expextedResult, event->nnResult);

        for(unsigned int iOut = 0; iOut < event->expextedResult.size(); iOut++) {
            histOutValues->Fill(iOut, event->nnResult[iOut]);
            if(event->nnResult[iOut] >= maxOutVal) {
                maxOutVal = event->nnResult[iOut];
                maxOutNum = iOut;
            }
            if(event->expextedResult[iOut] == 1)
                expectedOutNum = iOut;
        }

        histOutVsExpected->Fill(expectedOutNum, maxOutNum);

        if(expectedOutNum == maxOutNum) {
            wellClassifiedCnt[expectedOutNum]++;
            totalEfficiency++;
        }
        allCnt[expectedOutNum]++;
    }
    cost = cost / events.size();
    //cout<<" validation sample cost: "<<(cost )<<endl;
    for(unsigned int iOut = 0; iOut < wellClassifiedCnt.size(); iOut++) {
        cout<<iOut<<" eff: "<<setw(10)<<wellClassifiedCnt[iOut]/(double)allCnt[iOut]<<endl;
    }

    double totalEff = totalEfficiency/(double)events.size();
    cout<<" total eff: "<<setw(10)<<totalEff<<" mismatch "<<(1-totalEff)<<endl;

    return cost;
}

TH1F* trainSampleCostHist = 0;
TH1F* validSampleCostHist = 0;

void createCostHistory(int iterations, int printEveryIteration) {
    int bins = iterations / printEveryIteration;
    trainSampleCostHist = new TH1F("trainSampleCostHist", "trainSampleCostHist", bins, -0.5, iterations - 0.5);
    trainSampleCostHist->SetStats(0);
    validSampleCostHist = new TH1F("validSampleCostHist", "validSampleCostHist", bins, -0.5, iterations - 0.5);
    validSampleCostHist->SetStats(0);
    validSampleCostHist->SetLineColor(kRed);
}

void updateCostHistory(TPad& padRateHistory, int iteration, double trainSampleCost, double validSampleCost) {
    padRateHistory.cd()->SetGrid();
    trainSampleCostHist->Fill(iteration, trainSampleCost);
    validSampleCostHist->Fill(iteration, validSampleCost);
    trainSampleCostHist->Draw("hist");
    validSampleCostHist->Draw("samehist");
}


void EventsGeneratorBase::shuffle() {
    std::uniform_int_distribution<int> randomEventDist(0, this->events.size() -1);
    for(unsigned int i = 0; i < events.size()/2; i++) {
        auto first = events.begin() + randomEventDist(generator);
        auto second = events.begin() + randomEventDist(generator);
        std::swap((*first), (*second));
    }
}

void EventsGeneratorBase::getNextMiniBatch(vector<Event*>& miniBatchEvents) {
    if(miniBatchBegin + miniBatchEvents.size() >= events.end() ) {
        shuffle();
        miniBatchBegin = events.begin();
    }
    std::copy(miniBatchBegin, miniBatchBegin + miniBatchEvents.size(), miniBatchEvents.begin());
    miniBatchBegin = miniBatchBegin + miniBatchEvents.size();
}

} /* namespace lutNN */
