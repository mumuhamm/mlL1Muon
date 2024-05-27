/*
 * OmtfAnalyzer.cpp
 *
 *  Created on: Jan 15, 2020
 *      Author: kbunkow
 */

#include "lutNN/lutNN2/interface/OmtfAnalyzer.h"



#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>

#include "TCanvas.h"
#include "TFile.h"
#include "TH2F.h"
#include "TLegend.h"


namespace lutNN {

using namespace std;

double vxMuRate(double pt_GeV)
{
  if (pt_GeV == 0)
    return 0.0;
  const double lum = 2.0e34; //defoult is 1.0e34;
  const double dabseta = 1.0;
  const double dpt = 1.0;
  const double afactor = 1.0e-34*lum*dabseta*dpt;
  const double a  = 2*1.3084E6;
  const double mu=-0.725;
  const double sigma=0.4333;
  const double s2=2*sigma*sigma;

  double ptlog10;
  ptlog10 = log10(pt_GeV);
  double ex = (ptlog10-mu)*(ptlog10-mu)/s2;
  double rate = (a * exp(-ex) * afactor);
  //edm::LogError("RPCTrigger")<<ptCode<<" "<<rate;//<<<<<<<<<<<<<<<<<<<<<<<<
  return rate;
 }

double vxIntegMuRate(double pt_GeV, double dpt, double etaFrom, double etaTo) {
  //calkowanie metoda trapezow - nie do konca dobre
  double rate = 0.5 * (vxMuRate(pt_GeV) + vxMuRate(pt_GeV+dpt)) * dpt;

  rate = rate * (etaTo - etaFrom);
  //edm::LogError("RPCTrigger")<<ptCode<<" "<<rate;//<<<<<<<<<<<<<<<<<<<<<<<<
  return rate;
}

std::string findAndReplace(const std::string& str, const std::string& toReplace, const std::string& replacement) {
    assert(toReplace.size() == replacement.size());

    string newStr = str;
    auto pos = newStr.find(toReplace);
    if(pos != std::string::npos)
        newStr.replace(pos, toReplace.size(), replacement);

    return newStr;
}


MuonAlgorithm::MuonAlgorithm(float ptCut, std::string name, int color): ptCut(ptCut), name(name), color(color) {
	ostringstream histName;
	histName<<name<<"_acceptedEventsVsPtGen_ptCut"<<ptCut;
	string histNameStr = findAndReplace(histName.str(), ".", "_");

    int maxPt = 200;
    int binsCnt = 400;
	acceptedEventsVsPtGen = new TH1D(histNameStr.c_str(), histName.str().c_str(), binsCnt, 0, maxPt);
	std::cout<<"MuonAlgorithm: creating "<<histNameStr<<std::endl;

	histName.str();
	histName<<name<<"_ptGenVsPtLutNN_ptCut"<<ptCut;
	histNameStr = findAndReplace(histName.str(), ".", "_");

	ptGenVsPtLutNN = new TH2I(histNameStr.c_str(), histName.str().c_str(), 400, 0, 200, 200, 0, 200);

    histName.str();
    histName<<name<<"_goodChargeVsPt_ptCut"<<ptCut; //ptCut does not matter here, but the histograms must have unique names
    histNameStr = findAndReplace(histName.str(), ".", "_");
    goodChargeVsPt = new TH1D(histNameStr.c_str(), histName.str().c_str(), binsCnt, 0, maxPt);

    histName.str();
    histName<<name<<"_chargeGenVsChargeLutNN_ptCut"<<ptCut;
    histNameStr = findAndReplace(histName.str(), ".", "_");

    chargeGenVsChargeLutNNOnPtCut = new TH2I(histNameStr.c_str(), histName.str().c_str(), 2, -1.0001, 1.0001, 20, -1.0001, 1.0001);

	ostringstream canvasName;
	canvasName<<name<<"_ptCut"<<ptCut;
	canvas = new TCanvas(canvasName.str().c_str(), canvasName.str().c_str(), 2400, 1200);
}

void OmtfAlgorithm::fillHistos(EventFloatOmtf* eventFloatOmtf) {
    if( eventFloatOmtf->omtfQuality >= qualityCut) {
        if(eventFloatOmtf->omtfPt >= ptCut)
            acceptedEventsVsPtGen->Fill(eventFloatOmtf->muonPt);

        ptGenVsPtLutNN->Fill(eventFloatOmtf->muonPt, eventFloatOmtf->omtfPt);

        if(eventFloatOmtf->muonCharge == eventFloatOmtf->omtfCharge) {
            goodChargeVsPt->Fill(eventFloatOmtf->muonPt);
        }

        chargeGenVsChargeLutNNOnPtCut->Fill(eventFloatOmtf->muonCharge, eventFloatOmtf->omtfCharge);
    }
}

/*
void OmtfAlgorithmPtCont::fillHistos(EventFloatOmtf* eventFloatOmtf) {
	if( eventFloatOmtf->omtfQuality >= qualityCut) {
		if(eventFloatOmtf->omtfPt >= ptCut)
			acceptedEventsVsPtGen->Fill(eventFloatOmtf->muonPt);

		ptGenVsPtLutNN->Fill(eventFloatOmtf->muonPt, eventFloatOmtf->omtfPt);
	}
}
*/


void LutNNClassifierMaxPAlgorithm::fillHistos(EventFloatOmtf* eventFloatOmtf) {
	double maxOutVal = -10000;
	unsigned int maxOutNum = 0;
	//unsigned int expectedOutNum = 0;
	for(unsigned int iOut = 0; iOut < eventFloatOmtf->expextedResult.size(); iOut++) {
		if(eventFloatOmtf->nnResult[iOut] >= maxOutVal) {
			maxOutVal = eventFloatOmtf->nnResult[iOut];
			maxOutNum = iOut;
		}
		/*if(event->expextedResult[iOut] == 1) {
	                expectedOutNum = iOut;
	            }*/
	}

	double lowPtPSum = eventFloatOmtf->nnResult[0] + eventFloatOmtf->nnResult[1] +
					   eventFloatOmtf->nnResult[2] + eventFloatOmtf->nnResult[3]; //TODO choose good bins

	double lutNNPt = ptBins.at(maxOutNum/2);
	if(eventFloatOmtf->omtfQuality >= qualityCut && lowPtPSum < pThreshold) { //here must be  > ptCut, not  >= ptCut
		if(lutNNPt > ptCut)
			acceptedEventsVsPtGen->Fill(eventFloatOmtf->muonPt);

		ptGenVsPtLutNN->Fill(eventFloatOmtf->muonPt, lutNNPt);
	}
}

LutNNClassifierMaxPInterAlgorithm::LutNNClassifierMaxPInterAlgorithm(float ptCut, unsigned int qualityCut, std::string name, int color, std::vector<float>& ptBins, double pThreshold):
		LutNNClassifierMaxPAlgorithm(ptCut, qualityCut, name, color, ptBins, pThreshold) {

}

void LutNNClassifierMaxPInterAlgorithm::fillHistos(EventFloatOmtf* eventFloatOmtf) {
	double maxOutVal = -10000;
	unsigned int maxOutNum = 0;
	//unsigned int expectedOutNum = 0;
	for(unsigned int iOut = 0; iOut < eventFloatOmtf->expextedResult.size(); iOut++) {
		if(eventFloatOmtf->nnResult[iOut] >= maxOutVal) {
			maxOutVal = eventFloatOmtf->nnResult[iOut];
			maxOutNum = iOut;
		}
		/*if(event->expextedResult[iOut] == 1) {
	                expectedOutNum = iOut;
	            }*/
	}


	auto ptBinsMiddle = ptBins;

	ptBinsMiddle[0] = 3;//ptBins[0] / 2.;
	for(unsigned int iptBin = 1; iptBin < ptBins.size() -1; iptBin++) {
		ptBinsMiddle[iptBin] = (ptBins[iptBin -1] + ptBins[iptBin])/2.;
	}
	ptBinsMiddle.back() = ptBins.at(ptBins.size() -2) + 60;

	double lutNNPt = 0;
	double weight = 0;

	if(maxOutNum > 1) {
		lutNNPt += eventFloatOmtf->nnResult.at(maxOutNum -2) * ptBinsMiddle.at(maxOutNum/2 -1);
		weight  += eventFloatOmtf->nnResult.at(maxOutNum -2);
	}

	lutNNPt += eventFloatOmtf->nnResult[maxOutNum] * ptBinsMiddle.at(maxOutNum/2);
	weight  += eventFloatOmtf->nnResult[maxOutNum];

	if(maxOutNum < (eventFloatOmtf->nnResult.size() - 2) ) {
		lutNNPt += eventFloatOmtf->nnResult.at(maxOutNum +2) * ptBinsMiddle.at(maxOutNum/2 +1);
		weight  += eventFloatOmtf->nnResult.at(maxOutNum +2);
	}

	lutNNPt = lutNNPt / weight;

	double lowPtPSum = eventFloatOmtf->nnResult[0] + eventFloatOmtf->nnResult[1]; //TODO not good here, should be 0 and 2 or 1 and 3

	if(lutNNPt >= ptCut && eventFloatOmtf->omtfQuality >= qualityCut && lowPtPSum < pThreshold) {
		acceptedEventsVsPtGen->Fill(eventFloatOmtf->muonPt);

        /*if(eventFloatOmtf->muonPt < 7) {
        	cout<<name<<" ptCut "<<ptCut<<std::endl;
            cout<<*eventFloatOmtf<<endl;
        }*/
	}

	if(eventFloatOmtf->omtfQuality >= qualityCut && lowPtPSum < pThreshold)
		ptGenVsPtLutNN->Fill(eventFloatOmtf->muonPt, lutNNPt);
}

/*
void LutNNClassifierMaxPInterAlgorithm::makeHistos(TH1* ptGenNegAndPos) {
	MuonAlgorithm::makeHistos(ptGenNegAndPos);

	//cout<<"LutNNClassifierMaxPInterAlgorithm::makeHistos "<<endl;

	ostringstream histName;
	histName<<name<<"rateCumulVsPtCut_"<<ptCut;
    int maxPt = 100;
    int binsCnt = 100;
    rateCumulVsPtCut = new TH1D(histName.str().c_str(), histName.str().c_str(), binsCnt, 0, maxPt);

	for(int iBinY = 0; iBinY <= ptGenVsPtLutNN->GetYaxis()->GetNbins(); iBinY++) {
		auto ptCutGev = ptGenVsPtLutNN->GetYaxis()->GetBinLowEdge(iBinY);
		auto effVsPtGenOnPtCut1 = ptGenVsPtLutNN->ProjectionX( ( string("effVsPtGenOnPtCut_") + to_string(ptCutGev) + "_GeV").c_str(), iBinY, -1);

		effVsPtGenOnPtCut1->Divide(ptGenNegAndPos);

	    double rateTotal = 0;
	    for(int iBinX = 2; iBinX <= effVsPtGenOnPtCut1->GetXaxis()->GetNbins(); iBinX++) {
	        double eff  =      effVsPtGenOnPtCut1->GetBinContent(iBinX);
	        double rate =   eff * vxIntegMuRate(effVsPtGenOnPtCut1->GetBinLowEdge(iBinX), effVsPtGenOnPtCut1->GetXaxis()->GetBinWidth(iBinX), 0.85, 1.24);
	        rateTotal += rate;
	    }
	    rateCumulVsPtCut->Fill(ptCutGev, rateTotal);
	    //cout<<"ptCutGev "<<setw(5)<<ptCutGev<<" rateTotal "<<rateTotal<<endl;
	}

}
*/

/*
void LutNNClassifierMaxPInterAlgorithm::drawHistos() {
	MuonAlgorithm::drawHistos();
	canvas->cd(3);
	canvas->cd(3)->SetGridx();
	canvas->cd(3)->SetGridx();
	canvas->cd(3)->SetLogz();

	ptGenVsPtLutNN->Draw("col");

	canvas->cd(6);
	canvas->cd(6)->SetGridx();
	canvas->cd(6)->SetGridx();
	//canvas->cd(6)->SetLogz();
	//ptGenVsPtLutNN->DrawCopy("candle2");

	canvas->cd(6)->SetLogy();
	rateCumulVsPtCut->Draw("hist");
}
*/

void LutNNClassifierCalibrated::fillHistos(EventFloatOmtf* eventFloatOmtf) {
	auto lutNNPt = classifierToRegression.getValue(eventFloatOmtf);
	lutNNPt =  fabs(lutNNPt);
	if(ptCalibration)
		lutNNPt = ptCalibration->getValue(lutNNPt);

	if(lutNNPt >= ptCut && eventFloatOmtf->omtfQuality >= qualityCut ) { //&& lowPtPSum < pThreshold
		acceptedEventsVsPtGen->Fill(eventFloatOmtf->muonPt);

        /*if(eventFloatOmtf->muonPt < 7) {
        	cout<<name<<" ptCut "<<ptCut<<std::endl;
            cout<<*eventFloatOmtf<<endl;
        }*/
	}

	if(eventFloatOmtf->omtfQuality >= qualityCut) //&& lowPtPSum < pThreshold
		ptGenVsPtLutNN->Fill(eventFloatOmtf->muonPt, lutNNPt);
}


void LutNNClassifierPSumAlgorithm::fillHistos(EventFloatOmtf* eventFloatOmtf) {
	double pSum = 0;
	//unsigned int expectedOutNum = 0;
/*	for(unsigned int iOut = 0; iOut < eventFloatOmtf->expextedResult.size(); iOut++) {
		if(ptBins.at(iOut/2) > ptCut) {
			pSum += eventFloatOmtf->nnResult[iOut];
		}
		if(event->expextedResult[iOut] == 1) {
	                expectedOutNum = iOut;
	            }
	}
*/

    for(unsigned int iOut = 0; iOut < eventFloatOmtf->nnResult.size(); iOut++) {
        if(ptBins.at(iOut) > ptCut) {
            pSum += eventFloatOmtf->nnResult[iOut];
        }
    }
    
	if(pSum >= pThreshold && eventFloatOmtf->omtfQuality >= qualityCut) {
		acceptedEventsVsPtGen->Fill(eventFloatOmtf->muonPt);

        /*if(eventFloatOmtf->muonPt < 7 && eventFloatOmtf->omtfPt < ptCut) { //
            cout<<name<<" ptCut "<<ptCut<<std::endl;
        	cout<<*eventFloatOmtf<<endl;
        }*/
	}



}


void LutNNRegressionAlgorithm::fillHistos(EventFloatOmtf* eventFloatOmtf) {
    //double lutNNPt = exp(pow(1./fabs(eventFloatOmtf->nnResult.at(0) ) ,1) ); //expextedResult = omtfEvent.muonCharge / pow(log(omtfEvent.muonPt), 3); //TODO
    double lutNNPt = eventFloatOmtf->nnResult.at(0);

    if(ptCalibration)
        lutNNPt = ptCalibration->getValue(lutNNPt);

    if(lutNNPt >= ptCut && eventFloatOmtf->omtfQuality >= qualityCut ) {
    	 acceptedEventsVsPtGen->Fill(eventFloatOmtf->muonPt);

         if(eventFloatOmtf->muonPt < 7) {
        	 cout<<name<<" ptCut "<<ptCut<<std::endl;
             cout<<*eventFloatOmtf<<endl;
         }
     }

    double lutNNCharge = 0;
    if(eventFloatOmtf->nnResult.size() > 1)
        lutNNCharge =  eventFloatOmtf->nnResult.at(1);

    if(eventFloatOmtf->omtfQuality >= qualityCut) {//&& lowPtPSum < pThreshold //TODO add quality cut
        ptGenVsPtLutNN->Fill(eventFloatOmtf->muonPt, lutNNPt);

        if( (eventFloatOmtf->muonCharge == 1 &&  lutNNCharge > 0) ||  (eventFloatOmtf->muonCharge == -1 &&  lutNNCharge < 0) ) {
            goodChargeVsPt->Fill(eventFloatOmtf->muonPt);
        }

        chargeGenVsChargeLutNNOnPtCut->Fill(eventFloatOmtf->muonCharge, lutNNCharge);
    }
}


void MuonAlgorithm::makeHistos(TH1* ptGenNegAndPos) {
	ostringstream histName;
	histName<<name<<"_effVsPtGenOnPtCut_"<<ptCut;
    std::string histNameStr = histName.str();
    if(histNameStr.find(".") != std::string::npos) {
        histNameStr.replace(histNameStr.find("."), 1, "_");
    }
	effVsPtGenOnPtCut = (TH1D*)acceptedEventsVsPtGen->Clone(histNameStr.c_str());

	effVsPtGenOnPtCut->Divide(ptGenNegAndPos);

	effVsPtGenOnPtCut->SetLineColor(color);
    //omtfEff_20GeV->SetMarkerColor(kBlue);
    //omtfEff_20GeV->SetMarkerStyle(20);
	effVsPtGenOnPtCut->GetXaxis()->SetTitle("ptGen [GeV]");
	effVsPtGenOnPtCut->GetYaxis()->SetTitle("efficiency");

	histName<<"_rebined";
	int rebin = 4;
    effVsPtGenOnPtCut_rebined = acceptedEventsVsPtGen->Rebin(rebin, histName.str().c_str());
    auto ptGenNegAndPos_rebined = ptGenNegAndPos->Rebin(rebin, (string(ptGenNegAndPos->GetName()) + "rebined").c_str());
    effVsPtGenOnPtCut_rebined->Divide(ptGenNegAndPos_rebined);
    effVsPtGenOnPtCut_rebined->SetLineColor(color);

    //-------------------------rate--------------
    histName.str("");
    histName<<name<<"_rateVsPtGenOnPtCut_"<<ptCut;
    histNameStr = histName.str();
    if(histNameStr.find(".") != std::string::npos)
        histNameStr.replace(histNameStr.find("."), 1, "_");
    rateVsPtGenOnPtCut =  (TH1D*)acceptedEventsVsPtGen->Clone(histNameStr.c_str());
    rateVsPtGenOnPtCut->SetTitle(histName.str().c_str());
    rateVsPtGenOnPtCut->GetYaxis()->SetTitle("rate");
    rateVsPtGenOnPtCut->Reset();

    rateVsPtGenOnPtCut->SetLineColor(color);

    double rateTotal = 0;
    for(int iBin = 2; iBin <= effVsPtGenOnPtCut->GetXaxis()->GetNbins(); iBin++) {
        double eff  = effVsPtGenOnPtCut->GetBinContent(iBin);

        double rate = eff * vxIntegMuRate(effVsPtGenOnPtCut->GetBinLowEdge(iBin), effVsPtGenOnPtCut->GetXaxis()->GetBinWidth(iBin), 0.85, 1.24);

        rateVsPtGenOnPtCut->Fill(effVsPtGenOnPtCut->GetBinCenter(iBin), rate);

        rateTotal += rate;
    }

    cout<<setw(30)<<name<<" ptCut "<<ptCut<<" - rateTotal "<<rateTotal<<endl;

    histName.str("");
    histName<<name<<" ptCut "<<ptCut<<" GeV,  rate "<<setprecision(0)<< std::fixed<<rateTotal<<"; ptGen [GeV]; efficiency";
    effVsPtGenOnPtCut_rebined->SetTitle(histName.str().c_str());

    //-------------------------rate cumulative--------------

	//cout<<"LutNNClassifierMaxPInterAlgorithm::makeHistos "<<endl;

	histName.str("");
	histName<<name<<"_rateCumulVsPtCut_"<<ptCut;
    int maxPt = 100;
    int binsCnt = 100;

    histNameStr = histName.str();
    if(histNameStr.find(".") != std::string::npos)
        histNameStr.replace(histNameStr.find("."), 1, "_");

    rateCumulVsPtCut = new TH1D(histNameStr.c_str(), histName.str().c_str(), binsCnt, 0, maxPt);
    rateCumulVsPtCut->SetLineColor(color);

	for(int iBinY = 1; iBinY <= ptGenVsPtLutNN->GetYaxis()->GetNbins(); iBinY++) {
		auto ptCutGev = ptGenVsPtLutNN->GetYaxis()->GetBinLowEdge(iBinY);
		auto effVsPtGenOnPtCut1 = ptGenVsPtLutNN->ProjectionX( ( string("effVsPtGenOnPtCut_") + to_string(ptCutGev) + "_GeV").c_str(), iBinY, -1);

		effVsPtGenOnPtCut1->Divide(ptGenNegAndPos);

	    double rateTotal = 0;
	    for(int iBinX = 2; iBinX <= effVsPtGenOnPtCut1->GetXaxis()->GetNbins(); iBinX++) {
	        double eff  =      effVsPtGenOnPtCut1->GetBinContent(iBinX);
	        double rate =   eff * vxIntegMuRate(effVsPtGenOnPtCut1->GetBinLowEdge(iBinX), effVsPtGenOnPtCut1->GetXaxis()->GetBinWidth(iBinX), 0.85, 1.24);
	        rateTotal += rate;
	    }
	    rateCumulVsPtCut->Fill(ptCutGev, rateTotal);
	    //cout<<"ptCutGev "<<setw(5)<<ptCutGev<<" rateTotal "<<rateTotal<<endl;
	}

	goodChargeVsPt->Divide(ptGenNegAndPos);
	goodChargeVsPt->SetLineColor(color);
    //omtfEff_20GeV->SetMarkerColor(kBlue);
    //omtfEff_20GeV->SetMarkerStyle(20);
	goodChargeVsPt->GetXaxis()->SetTitle("ptGen [GeV]");
	goodChargeVsPt->GetYaxis()->SetTitle("good L1 charge");

}

void MuonAlgorithm::drawHistos() {
	TLegend* legend = new TLegend(0.2, 0.2, 1., 0.45);
    canvas->cd();
    canvas->Divide(4, 2);
    canvas->cd(1)->SetGridx();
    canvas->cd(1)->SetGridy();

    effVsPtGenOnPtCut_rebined->SetStats(0);
    effVsPtGenOnPtCut_rebined->Draw("hist");
    effVsPtGenOnPtCut_rebined->GetYaxis()->SetRangeUser(0.0, 1.1);
    legend->AddEntry(effVsPtGenOnPtCut_rebined, effVsPtGenOnPtCut_rebined->GetTitle(), "lep");

    for(auto& algoToCompare : algosToCompare) {
    	algoToCompare->effVsPtGenOnPtCut_rebined->Draw("hist same");
    	legend->AddEntry(algoToCompare->effVsPtGenOnPtCut_rebined, algoToCompare->effVsPtGenOnPtCut_rebined->GetTitle(), "lep");
    }

    //effVsPtGenOnPtCut_rebined->SetTitle(";ptGen [GeV];efficiency");
    legend->Draw();

    //------------------------- showing in log scale
    canvas->cd(2)->SetGridx();
    canvas->cd(2)->SetGridy();

    canvas->cd(2)->SetLogy();

	auto effVsPtGenOnPtCut_copy = effVsPtGenOnPtCut->DrawCopy("hist", "_log");
	effVsPtGenOnPtCut_copy->SetStats(0);
	effVsPtGenOnPtCut_copy->GetXaxis()->SetRangeUser(0, 30);
	effVsPtGenOnPtCut_copy->GetYaxis()->SetRangeUser(0.001, 1.1);
    for(auto& algoToCompare : algosToCompare) {
    	algoToCompare->effVsPtGenOnPtCut->Draw("hist same");
    }
    effVsPtGenOnPtCut_copy->SetTitle(";ptGen [GeV];efficiency");

    //------------------------- showing platou
    canvas->cd(5)->SetGridx();
    canvas->cd(5)->SetGridy();

    effVsPtGenOnPtCut_copy = effVsPtGenOnPtCut_rebined->DrawCopy("hist", "_log");
	effVsPtGenOnPtCut_copy->SetStats(0);
	effVsPtGenOnPtCut_copy->GetYaxis()->SetRangeUser(0.8, 1);

    for(auto& algoToCompare : algosToCompare) {
    	algoToCompare->effVsPtGenOnPtCut_rebined->Draw("hist same");
    }

    //------------------------- showing rate
    canvas->cd(6)->SetGridx();
    canvas->cd(6)->SetGridy();

    rateVsPtGenOnPtCut->SetStats(0);
    rateVsPtGenOnPtCut->Draw("hist");
    rateVsPtGenOnPtCut->GetXaxis()->SetRangeUser(0, 30);

    for(auto& algoToCompare : algosToCompare) {
    	algoToCompare->rateVsPtGenOnPtCut->Draw("hist same");
    }

    rateVsPtGenOnPtCut->SetTitle(";ptGen [GeV];rate [a.u.]");


    //------------------------- showing ptGenVsPtLutNN
	canvas->cd(3);
	canvas->cd(3)->SetGridx();
	canvas->cd(3)->SetGridx();
	canvas->cd(3)->SetLogz();

	ptGenVsPtLutNN->Draw("col");
	ptGenVsPtLutNN->SetStats(0);

	ptGenVsPtLutNN->GetXaxis()->SetRangeUser(0, 140);
	ptGenVsPtLutNN->GetYaxis()->SetRangeUser(0, 140);

	ptGenVsPtLutNN->SetTitle(";ptGen [GeV];ptL1 [GeV]");

	 //------------------------- showing rate cumulative
	canvas->cd(7);
	canvas->cd(7)->SetGridx();
	canvas->cd(7)->SetGridx();
	//canvas->cd(6)->SetLogz();
	//ptGenVsPtLutNN->DrawCopy("candle2");

	canvas->cd(7)->SetLogy();
	rateCumulVsPtCut->SetStats(0);
	rateCumulVsPtCut->GetYaxis()->SetRangeUser(50, 100000);
	rateCumulVsPtCut->Draw("hist");
	rateCumulVsPtCut->SetTitle(";L1 ptCut [GeV];rate [a.u.]");

    for(auto& algoToCompare : algosToCompare) {
    	algoToCompare->rateCumulVsPtCut->Draw("hist same");
    }

    ///charge
    canvas->cd(4);
    canvas->cd(4)->SetGridx();
    canvas->cd(4)->SetGridy();
    goodChargeVsPt->SetStats(0);
    goodChargeVsPt->Draw("hist");

    for(auto& algoToCompare : algosToCompare) {
        algoToCompare->goodChargeVsPt->Draw("hist same");
    }

    canvas->cd(8);
    canvas->cd(8)->SetGridx();

    canvas->cd(8)->SetLogz();
    canvas->cd(8)->SetLeftMargin(0.15);
    canvas->cd(8)->SetRightMargin(0.15);
    chargeGenVsChargeLutNNOnPtCut->SetTitle(";charge gen;charge L1");
    chargeGenVsChargeLutNNOnPtCut->SetStats(0);
    chargeGenVsChargeLutNNOnPtCut->Draw("colz");
}

OmtfAnalyzer::OmtfAnalyzer() {
    // TODO Auto-generated constructor stub
    for(unsigned int iLayer = 0; iLayer < 18; iLayer++ ) {
        ostringstream ostr;
        ostr<<"hitVsPt_layer_"<<iLayer<<endl;
        cout<<"creating "<<ostr.str()<<endl;
        hitVsPt.push_back(new TH2I(ostr.str().c_str(), ostr.str().c_str(), 400, -200, 200, 1024, 0, 1024));
    }
}

OmtfAnalyzer::~OmtfAnalyzer() {
    // TODO Auto-generated destructor stub
}

void OmtfAnalyzer::analyse(std::vector<EventFloat*> events, std::string dataFileName, std::string outFilePath) {
    TFile* dataRootFile = new TFile(dataFileName.c_str());

    //for the rootFileVersion==1 i.e. omtfHits_omtfAlgo0x0006_v1
    TH1* ptGenNeg = (TH1*)dataRootFile->Get("ptGenNeg");
    TH1* ptGenPos = (TH1*)dataRootFile->Get("ptGenPos");

    //for the rootFileVersion==2 i.e. CMSSW_12_3_0_pre4
    //watch out: in this version the cut on the eta was 0.85-1.24
    //TH1* ptGenNeg = (TH1*)dataRootFile->Get("simOmtfDigis/ptGenNeg");
    //TH1* ptGenPos = (TH1*)dataRootFile->Get("simOmtfDigis/ptGenPos");


    TH1* ptGenNegAndPos = (TH1*)ptGenNeg->Clone("ptGenNegAndPos");
    ptGenNegAndPos->Add(ptGenPos);

    TFile outfile( (outFilePath + ".root").c_str(), "RECREATE");
    outfile.cd();

    int maxPt = 200;
    int binsCnt = 100;


    binsCnt = 400;
    //the events
    ptGen_all = new TH1D("ptGen_allOmtf", "ptGen_allOmtf", binsCnt, 0, maxPt);
    ptGen_all_Weighted = new TH1D("ptGen_allOmtf_Weighted", "ptGen_allOmtf_Weighted", binsCnt, 0, maxPt);

    chargeGen_all = new TH1D("chargeGen_allOmtf", "chargeGen_allOmtf", 3, -1.5, 1.5);

    TH1D* etaGen = new TH1D("etaGen", "etaGen", 48*5, -2.4, 2.4);
    TH1D* etaGenOmtfCand_q1 = new TH1D("etaGenOmtfCand_q1", "etaGenOmtfCand_q1", 48*4, -2.4, 2.4);
    TH1D* etaGenOmtfCand_q4 = new TH1D("etaGenOmtfCand_q4", "etaGenOmtfCand_q4", 48*4, -2.4, 2.4);
    TH1D* etaGenOmtfCand_q8 = new TH1D("etaGenOmtfCand_q8", "etaGenOmtfCand_q8", 48*4, -2.4, 2.4);
    TH1D* etaGenOmtfCand_q12 = new TH1D("etaGenOmtfCand_q12", "etaGenOmtfCand_q12", 48*4, -2.4, 2.4);

    for(auto& event : events) {
        EventFloatOmtf* eventFloatOmtf = static_cast<EventFloatOmtf*>(event);

        //the cut on the eta is in the EventsGeneratorOmtf::readEvents, and it should be rather 0.80-1.24
        //if(eventFloatOmtf->muonPt > 0 && fabs(eventFloatOmtf->muonEta) > 0.85 && fabs(eventFloatOmtf->muonEta) < 1.24)
        //efficiency and rate analysis has no sense with the killed muons
        if(eventFloatOmtf->killed == false)
        {
            ptGen_all->Fill(eventFloatOmtf->muonPt);
            ptGen_all_Weighted->Fill(eventFloatOmtf->muonPt, eventFloatOmtf->weight);

            etaGen->Fill(eventFloatOmtf->muonEta);

            if(eventFloatOmtf->omtfPt > 0) {
                etaGenOmtfCand_q1->Fill(eventFloatOmtf->muonEta);

                if(eventFloatOmtf->omtfQuality >= 4)
                    etaGenOmtfCand_q4->Fill(eventFloatOmtf->muonEta);

                if(eventFloatOmtf->omtfQuality >= 8)
                    etaGenOmtfCand_q8->Fill(eventFloatOmtf->muonEta);

                if(eventFloatOmtf->omtfQuality >= 12)
                    etaGenOmtfCand_q12->Fill(eventFloatOmtf->muonEta);
            }

            for(auto& algo: muonAlgorithms) {
                algo->fillHistos(eventFloatOmtf);
            }

            for(unsigned int iLayer = 0; iLayer < hitVsPt.size(); iLayer++ ) {
                double ptCharge = eventFloatOmtf->muonPt *  eventFloatOmtf->muonCharge;
                hitVsPt.at(iLayer)->Fill(ptCharge,  eventFloatOmtf->inputs[iLayer]);
            }
        }
    }

    ptGen_all->Write();
    ptGen_all_Weighted->Write();
    chargeGen_all->Write();

    etaGen->Write();
    etaGenOmtfCand_q1->Write();
    etaGenOmtfCand_q4->Write();
    etaGenOmtfCand_q12->Write();

    std::string canvasName = "generalPlots";
    TCanvas* canvasGeneralPlots = new TCanvas(canvasName.c_str(), canvasName.c_str(), 2400, 1200);
    canvasGeneralPlots->Divide(2, 1);
    canvasGeneralPlots->cd(1);
    etaGen->Draw("hist");
    etaGenOmtfCand_q1->Draw("hist");
    etaGenOmtfCand_q1->SetLineColor(kBlue);
    etaGenOmtfCand_q4->Draw("same hist");
    etaGenOmtfCand_q4->SetLineColor(kGreen);
    etaGenOmtfCand_q8->Draw("same hist");
    etaGenOmtfCand_q8->SetLineColor(kMagenta);
    etaGenOmtfCand_q12->Draw("same hist");
    etaGenOmtfCand_q12->SetLineColor(kRed);
    TLegend* legend = new TLegend(0.7, 0.2, 1., 0.45);
    legend->AddEntry(etaGen, etaGen->GetTitle(), "lep");
    legend->AddEntry(etaGenOmtfCand_q1, etaGenOmtfCand_q1->GetTitle(), "lep");
    legend->AddEntry(etaGenOmtfCand_q4, etaGenOmtfCand_q4->GetTitle(), "lep");
    legend->AddEntry(etaGenOmtfCand_q8, etaGenOmtfCand_q8->GetTitle(), "lep");
    legend->AddEntry(etaGenOmtfCand_q12, etaGenOmtfCand_q12->GetTitle(), "lep");
    legend->Draw();

    canvasGeneralPlots->cd(1)->SetGridx();
    canvasGeneralPlots->cd(1)->SetGridy();
    canvasGeneralPlots->cd(2)->SetGridx();
    canvasGeneralPlots->cd(2)->SetGridy();
/*    canvasGeneralPlots->cd(3)->SetGridx();
    canvasGeneralPlots->cd(3)->SetGridy();
    canvasGeneralPlots->cd(4)->SetGridx();
    canvasGeneralPlots->cd(4)->SetGridy();*/

    canvasGeneralPlots->cd(2);
    ptGen_all_Weighted->Draw("hist");
    ptGen_all->Draw("same hist");
    ptGen_all->SetLineColor(kRed);
    ptGenNegAndPos->Draw("same hist");
    ptGenNegAndPos->SetLineColor(kGreen);
    canvasGeneralPlots->cd(2)->SetLogz();

    canvasGeneralPlots->SaveAs( (outFilePath + canvasName + ".png").c_str() );

    for(auto& algo: muonAlgorithms) {
        //TODO in the previous version the ptGen_all was used here,
        //the problem is that the ptGenNegAndPos works only if the data are read from the single file,
        //so if the events used for the analyzer are from the multiple files, the ptGen_all must be used, but it has has only for the rootFileVersion==2
        //if the ptGen_all is filled only for the not empty omtf cands, than the efficiency would be relative to the OMTF
    	algo->makeHistos(ptGenNegAndPos);

    	algo->acceptedEventsVsPtGen->Write();
    	algo->effVsPtGenOnPtCut->Write();
    	algo->effVsPtGenOnPtCut_rebined->Write();
    	algo->rateVsPtGenOnPtCut->Write();

    	algo->rateCumulVsPtCut->Write();

    	algo->ptGenVsPtLutNN->Write();
    }

    for(unsigned int iAlgo = 1; iAlgo < muonAlgorithms.size(); iAlgo++) {
        muonAlgorithms[iAlgo]->drawHistos();

        muonAlgorithms[iAlgo]->canvas->Write();

        muonAlgorithms[iAlgo]->canvas->SaveAs( (outFilePath + muonAlgorithms[iAlgo]->canvas->GetName() + ".png").c_str() );
    }

    TDirectory* subD1 = outfile.mkdir("hitVsPt");
    subD1->cd();
    for(unsigned int iLayer = 0; iLayer < hitVsPt.size(); iLayer++ ) {
        hitVsPt.at(iLayer)->Write();
    }
}
} /* namespace lutNN */
