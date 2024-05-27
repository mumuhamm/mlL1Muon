/*
 * Utlis.cpp
 *
 *  Created on: Jul 3, 2018
 *      Author: kbunkow
 */

#include "lutNN/lutNN2/interface/LutInter.h"
#include "lutNN/lutNN2/interface/Utlis.h"
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
#include "TPad.h"

namespace lutNN {
using namespace std;

Utlis::Utlis() {
    // TODO Auto-generated constructor stub

}

Utlis::~Utlis() {
    // TODO Auto-generated destructor stub
}

LutNetworkPrint::LutNetworkPrint() {

}

LutNetworkPrint::~LutNetworkPrint() {
    std::cout<<__FUNCTION__<<":"<<std::dec<<__LINE__<<std::endl;
}

TCanvas* LutNetworkPrint::createCanvasLutsAndOutMap(unsigned int lutLayerCnt, unsigned int maxNodesPerLayer, int padOutMapSubPpadsCnt) {
    this->lutLayerCnt = lutLayerCnt;
    gStyle->SetOptStat(0);
    canvasLutNN = new TCanvas("canvasLutNN", "canvasLutNN", 3 * 1900, 8 * 900);
    canvasLutNN->Divide(2, 1);
    padLuts = canvasLutNN->cd(1);
    padLuts->SetPad("padLuts", "padLuts", 0.0, 0.0, 0.7, 1., kWhite);
    //TPad padLuts("padLuts", "padLuts", 0.0, 0.0, 0.7, 1.);
    padLuts->SetGrid();
    padLuts->Divide(lutLayerCnt, maxNodesPerLayer * 2, 0.001, 0.001);
    padLuts->Draw();

    padOutMap = canvasLutNN->cd(2);
    padOutMap->SetPad("padOutMap", "padOutMap", 0.7, 0., 1., 1., kWhite);
    //TPad padOutMap("padOutMap", "padOutMap", 0.7, 0., 1., 1.);
    padOutMap->SetRightMargin(0.2);
    padOutMap->SetGrid();
    padOutMap->Divide(1, padOutMapSubPpadsCnt); //todo increase if more nn outputs then 4
    padOutMap->Draw();

    textCosts = new TPaveLabel(0.7, .01, 1., 0.05, "aaaa");
    canvasLutNN->cd();
    ostringstream ostr;
    ostr<<"cost";
    textCosts->Draw();

    return canvasLutNN;
}

/*
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
                if(iLut == 0) {
                    histLut->Draw("hist");
                }
                else
                    histLut->Draw("histsame");
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
}*/

void printLut(string name, TVirtualPad* pad, LutNode* lutNode, unsigned int iLayer, bool logY) {
    if(lutNode == nullptr)
        return;

    ostringstream ostrHistName;
    ostrHistName<<name<<"_layer_"<<iLayer<<"_"<<lutNode->getName();

    TObject* obj =  gDirectory->Get(ostrHistName.str().c_str());
    TH1F* hist = nullptr;
    auto& lutValues = lutNode->getFloatValues();
    if(obj == 0) {
        hist  =  new TH1F(ostrHistName.str().c_str(), ostrHistName.str().c_str(), lutValues.size(), -0.5, lutValues.size() - 0.5 );
        if(name == "Values")
            hist->SetLineColor(kBlue);
        else
            hist->SetLineColor(kRed);

        //gDirectory->Add(hist);
        hist->SetStats(0);
    }
    else {
        hist = dynamic_cast<TH1F*>(obj);
        if(obj == nullptr) {
            cout<<__FUNCTION__<<":"<<__LINE__<<" obj == nullptr "<<endl;
        }
    }

    if(name == "Values") {
        for(unsigned int iAddr = 0; iAddr < lutValues.size() -1; iAddr++) { //TODO last bin is not trained, so it is not displayed
            hist->SetBinContent(iAddr +1, lutValues[iAddr]);
        }
        hist->SetBinContent(lutValues.size() -1, 0); //TODO last bin is not trained, so it is not displayed
    }
    else
        for(unsigned int iAddr = 0; iAddr < lutValues.size(); iAddr++) {
            hist->SetBinContent(iAddr +1, lutNode->getEntries()[iAddr]);
        }

    //pad->cd();
    pad->SetGrid();
    pad->SetTopMargin(0.01);
    pad->SetBottomMargin(0.05);
    if(logY)
        pad->SetLogy();

    hist->GetYaxis()->SetLabelSize(0.1);
    if(name == "Values") {
        hist->Draw("L");
    }
    else
        hist->Draw("hist");
    pad->Update();
    return; // hist;
};

void printLuts3(LutNetworkBase& lutNetwork, unsigned int lutsPerCanvas) {
    //unsigned int lutLayer = 0;
   // TCanvas canvas;
    //canvas.cd();
    for(unsigned int iLayer = 0; iLayer < lutNetwork.getLayers().size(); iLayer++ ) {

        TVirtualPad* pad = nullptr;
        unsigned int padNum = 1;
        for(unsigned int iNode = 0; iNode < lutNetwork.getLayers()[iLayer].size(); iNode++) {
            LutNode* lutNode = dynamic_cast<LutNode*>(lutNetwork.getLayers()[iLayer][iNode].get());
            if(lutNode == nullptr)
                continue;

            if( (iNode % lutsPerCanvas) == 0 || (iNode % 18) == 0) {
                ostringstream name;
                name<<"Layer_"<<iLayer<<"_luts_"<<iNode<<"_"<<(iNode+lutsPerCanvas-1);
                pad = new TCanvas(name.str().c_str(), name.str().c_str(), 800, 1000);
                //pad = new TPad(name.str().c_str(), name.str().c_str(), 0, 0, 1, 1); not allowd
                pad->cd();
                pad->Divide(1, lutsPerCanvas * 2);
                cout<<"printLuts3: creating canvas "<<pad->GetName()<<endl;
            }

            printLut("Values", pad->cd(padNum), lutNode, iLayer,  false);
            padNum++;
            printLut("Entries", pad->cd(padNum), lutNode, iLayer,  true);
            padNum++;

            if( ( (iNode % lutsPerCanvas) == (lutsPerCanvas-1) ) || (iNode == (lutNetwork.getLayers()[iLayer].size() -1) ) ) {
                pad->Write();
                pad->Close();
                padNum = 1;
            }


        }

    }
}

void LutNetworkPrint::printLuts2(LutNetworkBase& lutNetwork, unsigned int maxNodesPerLayer) {
    //cout<<" gDirectory->GetName() "<<gDirectory->GetName()<<endl;
    //gDirectory->pwd();

    unsigned int lutLayer = 0;
    for(unsigned int iLayer = 0; iLayer < lutNetwork.getLayers().size(); iLayer++ ) {
        if(lutNetwork.getLayersConf()[iLayer]->nodeType >= LayerConfig::lutNode) {
            for(unsigned int iNode = 0; iNode < lutNetwork.getLayers()[iLayer].size(); iNode++) {
                //lambda
                auto getHist = [&](string name, TVirtualPad* pad, int padOffset, bool logY) {
                    LutNode* lutNode = dynamic_cast<LutNode*>(lutNetwork.getLayers()[iLayer][iNode].get());
                    if(lutNode == nullptr)
                        return;

                    ostringstream ostrHistName;
                    ostrHistName<<name<<"_layer_"<<iLayer<<"_"<<lutNode->getName();

                    TObject* obj =  gDirectory->Get(ostrHistName.str().c_str());
                    TH1F* hist = nullptr;
                    auto& lutValues = lutNode->getFloatValues();
                    if(obj == 0) {
                        hist  =  new TH1F(ostrHistName.str().c_str(), ostrHistName.str().c_str(), lutValues.size(), -0.5, lutValues.size() - 0.5 );
                        if(iLayer != lutNetwork.getLayers().size() -1) {
                            //histLut->GetYaxis()->SetRangeUser(lut->getValues().size()/(-2.), lut->getValues().size()/2);
                            //double range = lutNetwork.getNeuronLayers()[iLayer][iNode]->getLuts()[iLut]->getValues().size()/2;
                            //histLut->GetYaxis()->SetRangeUser(-range, range);
                        }
                        //cout<<__FUNCTION__<<":"<<__LINE__<<" creating hist "<<ostrHistName.str()<<endl;

                        if(name == "Values") {
                            hist->SetLineColor(kBlue);
                            //hist->GetYaxis()->SetRangeUser(lutNetwork.getLayersConf().at(iLayer)->minLutVal, lutNetwork.getLayersConf().at(iLayer)->maxLutVal);
                        }
                        else
                            hist->SetLineColor(kRed);

                        gDirectory->Add(hist);

                        if(iNode < maxNodesPerLayer) {
                            pad->cd(lutLayer + (2*iNode+padOffset) * lutLayerCnt  +1)->SetGrid();
                            pad->cd(lutLayer + (2*iNode+padOffset) * lutLayerCnt  +1)->SetTopMargin(0);
                            pad->cd(lutLayer + (2*iNode+padOffset) * lutLayerCnt  +1)->SetBottomMargin(0);
                            if(logY)
                                pad->cd(lutLayer + (2*iNode+padOffset) * lutLayerCnt  +1)->SetLogy();

                            hist->GetYaxis()->SetLabelSize(0.1);
                            if(name == "Values")
                                hist->Draw("l");
                            else
                                hist->Draw("hist");

                            pad->cd(lutLayer + (2*iNode+padOffset) * lutLayerCnt  +1)->Update();
                        }
                    }
                    else {
                        hist = dynamic_cast<TH1F*>(obj);
                        if(obj == nullptr) {
                            cout<<__FUNCTION__<<":"<<__LINE__<<" obj == nullptr "<<endl;
                        }
                    }

                    if(name == "Values") {
                        for(unsigned int iAddr = 0; iAddr < lutValues.size() ; iAddr++) { //TODO -1 last bin is not trained, so it is not displayed
                            hist->SetBinContent(iAddr +1, lutValues[iAddr]);
                        }

                        //if(dynamic_cast<LutInter*>(lutNode) ) //only makes confusion
                        //	hist->SetBinContent(lutValues.size(), 0); //TODO last bin is not trained, so it is not displayed
                    }
                    else
                        for(unsigned int iAddr = 0; iAddr < lutValues.size(); iAddr++) {
                            hist->SetBinContent(iAddr +1, lutNode->getEntries()[iAddr]);
                        }

                    return; // hist;
                };

                getHist("Values", padLuts, 0, false);
                getHist("Entries", padLuts, 1, true);
                //getHistStac("GradientSum", padGradients, 0, false);
                //getHistStac("AbsGradientSum", padGradients, 1, false);
            }
            lutLayer++;
        }
    }
}

/*
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

                if(logY)
                    pad.cd(iLayer + (iNode+padOffset) * lutNetwork.getNeuronLayers().size()  +1)->SetLogy();

                pad.cd(iLayer + (iNode+padOffset) * lutNetwork.getNeuronLayers().size()  +1)->Update();
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
}*/

void LutNetworkPrint::printOutMap(LutNetworkBase& lutNetwork, unsigned int& lutSize) {
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

    for(unsigned int iX = 0; iX < lutSize; iX++) {
        for(unsigned int iY = 0; iY < lutSize; iY++) {
            lutNetwork.getInputNodes()[0]->setOutValue(iX);
            lutNetwork.getInputNodes()[1]->setOutValue(iY);
            lutNetwork.run();
            hist->SetBinContent(iX +1, iY +1, lutNetwork.getOutputValues()[0]);
            //cout<<" neuron.getOutAddr(): "<<neuron.getOutAddr()<<endl;
        }
    }

    padOutMap->cd();
    hist->SetStats(0);
    hist->Draw("colz");
}

template <typename EventType>
double LutNetworkPrint::printExpectedVersusNNOut(std::vector<std::pair<double, double> >& ranges, std::vector<EventType*>& events, CostFunction& costFunction) {
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

        for(auto& event : events) {
            hist->Fill(event->expextedResult[iOut], event->nnResult[iOut]);
            //cost += pow(event->expextedResult[iOut] - event->nnResult[iOut], 2);
            if(iOut == 0) //to avoid double counting in case more then one output exists
              cost += costFunction(event->expextedResult, event->nnResult);
        }
        padOutMap->cd(iOut + 1)->SetGrid();
        padOutMap->cd(iOut + 1)->SetLogz(); //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,
        hist->SetStats(0);
        hist->Draw("colz");
    }
    cost = cost / events.size();
    //cout<<" training sample cost: "<<(cost)<<endl;
    return cost;

}


template <typename EventType>
double LutNetworkPrint::printExpectedVersusNNOutHotOne(std::vector<EventType*>& events, CostFunction& costFunction) {
    string histName = "histOutValues";

    TObject* histObj =  gDirectory->Get(histName.c_str());
    TH2F* histOutValues = 0;
    unsigned int resultsCnt = events.at(0)->nnResult.size();
    if(histObj == 0) {
        histOutValues =  new TH2F(histName.c_str(), histName.c_str(), resultsCnt, -.5, resultsCnt -0.5, 60, -1, 2);
        padOutMap->cd(1)->SetGrid();
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
        padOutMap->cd(2)->SetGrid();
        padOutMap->cd(2)->SetLogz();
        histOutVsExpected->SetStats(0);
        histOutVsExpected->Draw("colztext");
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

        cost += costFunction.get(event->classLabel, event->nnResult);

        for(unsigned int iOut = 0; iOut < event->expextedResult.size(); iOut++) {
            if(event->nnResult[iOut] >= maxOutVal) {
                maxOutVal = event->nnResult[iOut];
                maxOutNum = iOut;
            }

            if(event->classLabel == iOut) {
                histOutValues->Fill(iOut, event->nnResult[iOut]);
            }

            expectedOutNum = event->classLabel;
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


    ostringstream ostr;
    //ostr<<"iteration "<<setw(7)<<iteration<<" cost: train: "<<setw(5)<<lutNetwork.getMeanCost()<<" valid "<<setw(5)<<validSampleCost;
    ostr<<"cost: "<< cost<<" eff: "<<setw(10)<<totalEff;

    textCosts->SetLabel(ostr.str().c_str() );

    return cost;
}


void LutNetworkPrint::createCostHistory(int iterations, int printEveryIteration, double yAxisMin, double yAxisMax) {
    padCostHistory =  new TCanvas("canvasCostHist", "canvasCostHist", 1400, 800);
    padCostHistory->SetLogy();

    padCostHistory->cd()->SetGrid();

    int bins = iterations / printEveryIteration;
    trainSampleCostHist = new TH1F("trainSampleCostHist", "trainSampleCostHist", bins, -0.5, iterations - 0.5);
    trainSampleCostHist->SetStats(0);
    validSampleCostHist = new TH1F("validSampleCostHist", "validSampleCostHist", bins, -0.5, iterations - 0.5);
    validSampleCostHist->SetStats(0);
    validSampleCostHist->SetLineColor(kRed);
    trainSampleCostHist->GetYaxis()->SetRangeUser(yAxisMin, yAxisMax);
}

void LutNetworkPrint::updateCostHistory(int iteration, double trainSampleCost, double validSampleCost, std::string pngfile) {
    padCostHistory->cd();
    trainSampleCostHist->Fill(iteration, trainSampleCost);
    validSampleCostHist->Fill(iteration, validSampleCost);
    trainSampleCostHist->Draw("hist");
    validSampleCostHist->Draw("samehist");

    padCostHistory->Update();
    padCostHistory->SaveAs(pngfile.c_str());
}

template double LutNetworkPrint::printExpectedVersusNNOut(std::vector<std::pair<double, double> >& ranges, std::vector<EventInt*>& events, CostFunction& costFunction);
template double LutNetworkPrint::printExpectedVersusNNOut(std::vector<std::pair<double, double> >& ranges, std::vector<EventFloat*>& events, CostFunction& costFunction);

template double LutNetworkPrint::printExpectedVersusNNOutHotOne(std::vector<EventInt*>& events, CostFunction& costFunction);
template double LutNetworkPrint::printExpectedVersusNNOutHotOne(std::vector<EventFloat*>& events, CostFunction& costFunction);

} /* namespace lutNN */
