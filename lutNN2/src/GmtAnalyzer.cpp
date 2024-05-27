/*
 * GmtAnalyzer.cpp
 *
 *  Created on: Dec 28, 2021
 *      Author: kbunkow
 */

#include "lutNN/lutNN2/interface/GmtAnalyzer.h"
#include "TFile.h"
#include "TGraph.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TGraphAsymmErrors.h"

#include <regex>

namespace lutNN {

ProbabilityVsPt::ProbabilityVsPt(std::string name, int iClass, int nnRes, int tpEvent, float etaFrom, float etaTo, bool useTttPt, bool useMargin) :
        iClass(iClass), nnRes(nnRes), tpEvent(tpEvent), etaFrom(etaFrom), etaTo(etaTo), useTttPt(useTttPt), useMargin(useMargin) {
    if(useMargin)
        hist = new TH2I(name.c_str(), name.c_str(), 100, 0, 100, 400, -100, 100);
    else
        hist = new TH2I(name.c_str(), name.c_str(), 100, 0, 100, 400, 0, 1);

    if(useTttPt)
        hist->GetXaxis()->SetTitle("tttPt (L1Pt) [GeV]");
    else
        hist->GetXaxis()->SetTitle("tpPt (genPt) [GeV]");
    //for the SoftMax the probability range can be 0 to 1
    //but for the MeanSquaredError the nnResult can be < 0 or > 1
}

void ProbabilityVsPt::fill(EventIntGmt* eventGmt) {
    if( (eventGmt->classLabel == iClass || iClass == -1) &&
        (fabs(eventGmt->tpEta) >= etaFrom) && (fabs(eventGmt->tpEta) < etaTo)   ) {
        float pt = useTttPt ? (eventGmt->tttPt  / 32.) : eventGmt->tpPt;

        double nnScore = eventGmt->nnResult.at(nnRes);
        if(useMargin)
            nnScore = eventGmt->nnResult.at(nnRes) - eventGmt->nnResult.at( nnRes == 1 ? 0 : 1 );

        if(tpEvent == -1)
            hist->Fill(pt, nnScore);
        else if(tpEvent == 0 && eventGmt->tpEvent == 0)
            hist->Fill(pt, nnScore);
        else if(tpEvent == 1 && eventGmt->tpEvent >= 1)
            hist->Fill(pt, nnScore);
    }
}

class EfficiencyVsPt {
public:
/*    EfficiencyVsPt(std::function<bool(EventIntGmt* eventGmt)> filter,
            std::string name, float ptCut, float etaFrom, float etaTo,  bool useTttPt, bool useMargin):
                filter(filter),
                etaFrom(etaFrom), etaTo(etaTo), useTttPt(useTttPt), useMargin(useMargin)
     {
        efficiency = new TEfficiency( (name + "_pt1Stub").c_str(), (name + "; " + (useTttPt ?"tttPt (L1Pt) [GeV]" : "ptGen [GeV]") + "; relative efficiency").c_str(),
                50, 0, 50 );


     }*/

    virtual ~EfficiencyVsPt() {

    }

    EfficiencyVsPt(std::string fiterFunction, bool useTttPt, bool useMargin): useTttPt(useTttPt), useMargin(useMargin) {
        init(fiterFunction);
    }

    EfficiencyVsPt(std::string name, std::string fiterFunction, float ptCut, float etaFrom, float etaTo, bool useTttPt, bool useMargin):
        ptCut(ptCut), etaFrom(etaFrom), etaTo(etaTo), useTttPt(useTttPt), useMargin(useMargin)
     {
        efficiency = new TEfficiency( (name + "_" + fiterFunction + "_EffVsPt").c_str(),
                (name + " " + fiterFunction + "; " + (useTttPt ?"tttPt (L1Pt) [GeV]" : "ptGen [GeV]") + "; relative efficiency").c_str(), 50, 0, 50 );



        std::cout<<"EfficiencyVsPt "<<name<<" "<<fiterFunction<<" ptCut "<<ptCut<<" etaFrom "<<etaFrom<<" etaTo "<<etaTo
                <<" useTttPt "<<useTttPt<<" useMargin "<<useMargin<<std::endl;

        init(fiterFunction);
     }

    virtual void init(std::string fiterFunction) {
        if(fiterFunction == "twoStubs")
            filter = [this](EventIntGmt* event) {return EfficiencyVsPt::twoStubs(event); };

        if(fiterFunction == "threeStubs")
            filter = [this](EventIntGmt* event) {return EfficiencyVsPt::threeStubs(event); };

        if(fiterFunction == "nnScoreThresh") {
            filter = [this](EventIntGmt* event) {return EfficiencyVsPt::nnScoreThresh(event); };

            nnScoreTresholds = new TH1F("nnScoreTresholds", "nnScoreTresholds", 10, 0, 10);
            /*nnScoreTresholds->Fill(2, 0.025); //for eff 0.95
            nnScoreTresholds->Fill(3, 0.025); //for eff 0.91
            nnScoreTresholds->Fill(4, 0.04); //for eff 0.96
            nnScoreTresholds->Fill(5, 0.1);
            nnScoreTresholds->Fill(6, 0.1);
            nnScoreTresholds->Fill(7, 0.1);
            nnScoreTresholds->Fill(8, 0.1);
            nnScoreTresholds->Fill(9, 0.1);
            nnScoreTresholds->Fill(10, 0.1);*/
            //nnScoreTresholds->Fill(11, 0.);

            nnScoreTresholds->Fill(2, -0.9); //for eff 0.95
            nnScoreTresholds->Fill(3, -0.5); //for eff 0.91
            nnScoreTresholds->Fill(4, -0.5); //for eff 0.96
            nnScoreTresholds->Fill(5, -0.5);
            nnScoreTresholds->Fill(6, -0.5);
            nnScoreTresholds->Fill(7, -0.5);
            nnScoreTresholds->Fill(8, -0.5);
            nnScoreTresholds->Fill(9, -0.5);
            nnScoreTresholds->Fill(10, -0.5);


/*            nnScoreTresholds->Fill(2, -0.6); //for eff 0.95
            nnScoreTresholds->Fill(3, -0.6); //for eff 0.91
            nnScoreTresholds->Fill(4, -0.6); //for eff 0.96
            nnScoreTresholds->Fill(5, -0.6);
            nnScoreTresholds->Fill(6, -0.6);
            nnScoreTresholds->Fill(7, -0.6);
            nnScoreTresholds->Fill(8, -0.6);
            nnScoreTresholds->Fill(9, -0.6);
            nnScoreTresholds->Fill(10, -0.6);*/
        }
    }

    virtual float getNNScore(EventIntGmt* eventGmt) {
        int nnRes =  1;
        double nnScore = eventGmt->nnResult.at(nnRes);
        if(useMargin)
            nnScore = eventGmt->nnResult.at(nnRes) - eventGmt->nnResult.at( nnRes == 1 ? 0 : 1 );

        return nnScore;
    }

    std::function<bool(EventIntGmt* eventGmt)> filter;

    bool twoStubs(EventIntGmt* eventGmt) {
        return eventGmt->stubCnt >= 2;
    }

    bool threeStubs(EventIntGmt* eventGmt) {
        return eventGmt->stubCnt >= 3;
    }

    bool nnScoreThresh(EventIntGmt* eventGmt) {
        bool accept = ( getNNScore(eventGmt) > nnScoreTresholds->GetBinContent( nnScoreTresholds->FindBin(eventGmt->tttPt / 32.) ) );
        //bool accept = ( getNNScore(eventGmt) > nnScoreTresholds->GetBinContent( nnScoreTresholds->FindBin(eventGmt->tpPt) ) );

       /* std::cout<<"nnScoreThresh tttPt "<<(eventGmt->tttPt/32.)<<" GeV  "
                <<" tpPt "<<eventGmt->tpPt<<" GeV "
                <<" classLabel "<<eventGmt->classLabel
                <<" bin "<<nnScoreTresholds->FindBin(eventGmt->tttPt / 32.)
                <<" threshold "<<nnScoreTresholds->GetBinContent( nnScoreTresholds->FindBin(eventGmt->tttPt / 32.) )
                <<" nnScore "<<getNNScore(eventGmt)<<" accept "<<accept<<std::endl; */

        return accept;
    }

    virtual void fill(EventIntGmt* eventGmt) {
        if( (fabs(eventGmt->tpEta) >= etaFrom) && (fabs(eventGmt->tpEta) < etaTo) )
        {
            float pt = useTttPt ? (eventGmt->tttPt  / 32.) : eventGmt->tpPt;

            efficiency->Fill(filter(eventGmt), pt);
        }
    }

    void drawCompare(TVirtualPad* canvasCompareEff, int color, bool first) {
        canvasCompareEff->cd();
        efficiency->SetLineColor(color);

        if(first) {
            efficiency->Draw("APZ");
            canvasCompareEff->Update();
            efficiency->GetPaintedGraph()->GetYaxis()->SetRangeUser(0, 1.01);
            efficiency->GetPaintedGraph()->GetXaxis()->SetRangeUser(0, 20);
        }
        else
            efficiency->Draw("PZ same");
    }

    void makeCumulativeEff(TVirtualPad* canvasCompareEff, int color, bool first) {
        auto passedHisto = efficiency->GetCopyPassedHisto();
        auto totalHisto = efficiency->GetCopyTotalHisto();

        auto passedHistoCumul = passedHisto->GetCumulative(false);
        auto totalHistoCumul = totalHisto->GetCumulative(false);

        if(TEfficiency::CheckConsistency(*passedHistoCumul, *totalHistoCumul) ) {
          canvasCompareEff->cd();
          TEfficiency* efficiency = new TEfficiency(*passedHistoCumul, *totalHistoCumul);
          //title = std::regex_replace(title, std::regex("\\muCandGenEtaMuons"), "tagging efficiency");

          efficiency->SetTitle( (std::string(passedHistoCumul->GetTitle()) + " cumulative" + ";" + ("tttPt (L1Pt) tresh [GeV]") + "; relative efficiency").c_str());
          efficiency->SetStatisticOption(TEfficiency::EStatOption::kBUniform );
          efficiency->SetPosteriorMode();
          efficiency->SetLineColor(color);

          //return efficiency;
          if(first) {
              efficiency->Draw("APZ");

              canvasCompareEff->Update();
              efficiency->GetPaintedGraph()->GetYaxis()->SetRangeUser(0, 1.01);
              efficiency->GetPaintedGraph()->GetXaxis()->SetRangeUser(0, 20);
          }
          else
              efficiency->Draw("PZ same");

          efficiency->Write();

          std::cout<<"line "<<__LINE__<<" efficiency->GetName() "<<efficiency->GetName()<<" efficiency->GetTitle() "<<efficiency->GetTitle()<<std::endl;
        }
        else  {
          std::cout<<"line "<<__LINE__<<"TEfficiency::CheckConsistency(*ptGenPtTTMuonNom, *ptGenPtTTMuonDenom) failed !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;
          //exit(1);
        }
    }

    float ptCut = 0;

    float etaFrom = 0;
    float etaTo = 0;

    bool useTttPt =  true; //if false the tpPt (gen pt) is used

    bool useMargin = false; //for hinge lost or for mean square lost
    TEfficiency*  efficiency;

    TH1F* nnScoreTresholds = nullptr;
};

class EfficiencyVsEta: public EfficiencyVsPt {
public:
    EfficiencyVsEta(std::string name, std::string fiterFunction, float ptCut, float genPtFrom, float genPtTo, bool useTttPt, bool useMargin):
        EfficiencyVsPt(fiterFunction, useTttPt, useMargin), genPtFrom(genPtFrom), genPtTo(genPtTo)
    {
        this->ptCut = ptCut;

        efficiency = new TEfficiency( (name + "_" + fiterFunction + "_EffVsEta_ptCut_" + std::to_string(((int)ptCut))).c_str(),
                (name + " " + fiterFunction + " ptCut " + std::to_string(ptCut) + " GeV; " + ("etaGen") + "; relative efficiency").c_str(), 50, -2.4, 2.4 );


        std::cout<<"EfficiencyVsEta "<<name<<" "<<fiterFunction<<" ptCut "<<ptCut<<" genPtFrom "<<genPtFrom<<" genPtTo "<<genPtTo
                <<" useTttPt "<<useTttPt<<" useMargin "<<useMargin<<std::endl;

    }

    virtual ~EfficiencyVsEta() {}

    virtual void fill(EventIntGmt* eventGmt) {
        //std::cout<<"EfficiencyVsEta::fill before if  eventGmt->tpPt "<< eventGmt->tpPt<<" tpEta "<<eventGmt->tpEta<<" filter "<<filter(eventGmt)<<std::endl;
        if( eventGmt->tpPt >= genPtFrom && eventGmt->tpPt <= genPtTo ) {
            float pt = eventGmt->tttPt  / 32.;

            if(pt >= ptCut)
                efficiency->Fill(filter(eventGmt), eventGmt->tpEta);

            //std::cout<<"EfficiencyVsEta::fill pt "<<pt<<" tpEta "<<eventGmt->tpEta<<" filter "<<filter(eventGmt)<<std::endl;
        }
    }

    float genPtFrom = 0;
    float genPtTo = 0;
};

GmtAnalyzer::GmtAnalyzer(std::string outFilePath): outfile( (outFilePath).c_str(), "RECREATE") {
    //TFile outfile( (outFilePath + outFileName).c_str(), "RECREATE");
    std::cout<<"Opening file "<<outFilePath<<std::endl;
    outfile.cd();

    gStyle->SetPadGridX(1); // grids, tickmarks
    gStyle->SetPadGridY(1);

}

GmtAnalyzer::~GmtAnalyzer() {
    outfile.Write();
}

void GmtAnalyzer::makeRocCurve(TH2* signalHist, TH2* falsesHist, float ptFromSignla, float ptToSignal, float ptFromFalse, float ptToFalse) {
    std::ostringstream ostr;
    ostr<<signalHist->GetName()<<"_ptFrom_"<<ptFromSignla<<"_ptTo_"<<ptToSignal;
    std::string name = ostr.str();
    name = std::regex_replace(name, std::regex("probabilityVsPt_"), "");

    TH1* projectionSignal = signalHist->ProjectionY((name + "_projY").c_str(), signalHist->GetXaxis()->FindBin(ptFromSignla), signalHist->GetXaxis()->FindBin(ptToSignal - 0.01));
    TH1* projectionSignalCumul = projectionSignal->GetCumulative(false);
    projectionSignalCumul->Scale(1./projectionSignal->Integral());
    projectionSignalCumul->SetTitle((name + ";probCut;eff").c_str());

    ostr.str("");
    ostr<<falsesHist->GetName()<<"_ptFrom_"<<ptFromFalse<<"_ptTo_"<<ptToFalse;
    name = ostr.str();
    name = std::regex_replace(name, std::regex("probabilityVsPt_"), "");
    TH1* projectionFalse = falsesHist->ProjectionY( (name+ "_projY").c_str(), falsesHist->GetXaxis()->FindBin(ptFromFalse), falsesHist->GetXaxis()->FindBin(ptToFalse - 0.01));
    TH1* projectionFalseCumul = projectionFalse->GetCumulative(false);
    projectionFalseCumul->Scale(1./projectionFalse->Integral());
    projectionFalseCumul->SetTitle((name + ";probCut;eff").c_str());

    auto x = new Double_t[projectionSignalCumul->GetNbinsX()];
    auto y = new Double_t[projectionFalseCumul->GetNbinsX()];

    for(int iPoint = 1; iPoint <= projectionSignalCumul->GetNbinsX(); iPoint++) {
        x[iPoint-1] = projectionFalseCumul->GetBinContent(iPoint);
        y[iPoint-1] = projectionSignalCumul->GetBinContent(iPoint);
    }

    ostr.str("");
    TGraph* rocGraph = new TGraph(projectionFalseCumul->GetNbinsX(), x, y);
    ostr<<"graph_"<<signalHist->GetName()<<"_ptFromSignal_"<<ptFromSignla<<"_ptToSignal_"<<ptToSignal<<"_ptFromFalse_"<<ptFromFalse<<"_ptToFalse_"<<ptToFalse;
    rocGraph->SetName(ostr.str().c_str());
    //ostr<<";false positive rate;muon efficiency";
    ostr<<";"<<projectionFalse->GetName()<<";"<<projectionSignal->GetName();
    rocGraph->SetTitle(ostr.str().c_str());

    projectionSignal->Write();
    projectionFalse->Write();
    projectionSignalCumul->Write();
    projectionFalseCumul->Write();
    rocGraph->Write();

    delete[] x;
    delete[] y;
}

void GmtAnalyzer::makeRocCurve(std::string signalSample, std::string falseSampl) {
    outfile.cd();
    //all
    makeRocCurve(probabilityVsPtsMap[signalSample].at(1).hist, probabilityVsPtsMap[falseSampl].at(0).hist, 2, 3, 2, 30);
    makeRocCurve(probabilityVsPtsMap[signalSample].at(1).hist, probabilityVsPtsMap[falseSampl].at(0).hist, 3, 4, 3, 30);
    makeRocCurve(probabilityVsPtsMap[signalSample].at(1).hist, probabilityVsPtsMap[falseSampl].at(0).hist, 4, 5, 4, 30);
    makeRocCurve(probabilityVsPtsMap[signalSample].at(1).hist, probabilityVsPtsMap[falseSampl].at(0).hist, 5, 6, 5, 30);
    makeRocCurve(probabilityVsPtsMap[signalSample].at(1).hist, probabilityVsPtsMap[falseSampl].at(0).hist, 6, 7, 6, 30);
    makeRocCurve(probabilityVsPtsMap[signalSample].at(1).hist, probabilityVsPtsMap[falseSampl].at(0).hist, 7, 10, 7,30);
    makeRocCurve(probabilityVsPtsMap[signalSample].at(1).hist, probabilityVsPtsMap[falseSampl].at(0).hist, 10,20,10,30);
    //makeRocCurve(probabilityVsPtsMap[signalSample].at(1).hist, probabilityVsPtsMap[falseSampl].at(0).hist, 2, 4, 1, 30);
    //makeRocCurve(probabilityVsPtsMap[signalSample].at(1).hist, probabilityVsPtsMap[falseSampl].at(0).hist, 5, 10, 1, 30);
    //makeRocCurve(probabilityVsPtsMap[signalSample].at(1).hist, probabilityVsPtsMap[falseSampl].at(0).hist, 10,15, 1, 30);
    //makeRocCurve(probabilityVsPtsMap[signalSample].at(1).hist, probabilityVsPtsMap[falseSampl].at(0).hist, 15, 20, 1, 30);

    //endcap
    makeRocCurve(probabilityVsPtsMap[signalSample].at(10).hist, probabilityVsPtsMap[falseSampl].at(9).hist, 2, 3, 2, 30);
    makeRocCurve(probabilityVsPtsMap[signalSample].at(10).hist, probabilityVsPtsMap[falseSampl].at(9).hist, 3, 4, 3, 30);
    makeRocCurve(probabilityVsPtsMap[signalSample].at(10).hist, probabilityVsPtsMap[falseSampl].at(9).hist, 4, 5, 4, 30);
    makeRocCurve(probabilityVsPtsMap[signalSample].at(10).hist, probabilityVsPtsMap[falseSampl].at(9).hist, 5, 6, 5, 30);
    makeRocCurve(probabilityVsPtsMap[signalSample].at(10).hist, probabilityVsPtsMap[falseSampl].at(9).hist, 6, 7, 6, 30);
    makeRocCurve(probabilityVsPtsMap[signalSample].at(10).hist, probabilityVsPtsMap[falseSampl].at(9).hist, 7, 10, 7,30);
    makeRocCurve(probabilityVsPtsMap[signalSample].at(10).hist, probabilityVsPtsMap[falseSampl].at(9).hist, 10,20,10,30);
//    makeRocCurve(probabilityVsPtsMap[signalSample].at(10).hist, probabilityVsPtsMap[falseSampl].at(9).hist, 2, 4, 1, 30);
//    makeRocCurve(probabilityVsPtsMap[signalSample].at(10).hist, probabilityVsPtsMap[falseSampl].at(9).hist, 5, 10, 1, 30);
//    makeRocCurve(probabilityVsPtsMap[signalSample].at(10).hist, probabilityVsPtsMap[falseSampl].at(9).hist, 10,15, 1, 30);
//    makeRocCurve(probabilityVsPtsMap[signalSample].at(10).hist, probabilityVsPtsMap[falseSampl].at(9).hist, 2, 20, 1, 30);
}

void GmtAnalyzer::analyze(std::vector<EventInt*> events, std::string sampleName, bool useTttPt, bool useMargin) {
    auto subDir = outfile.mkdir(sampleName.c_str());
    subDir->cd();

    int maxPt = 200;
    int binsCnt = 200;

    TH1* ptGen_MuEvent0 = new TH1I("ptGen_MuEvent0", "ptGen_MuEvent0;ptGen [Gev];#", binsCnt, 0, maxPt);
    TH1* ptGen_MuPU     = new TH1I("ptGen_MuPU", "ptGen_MuPU;ptGen [Gev];#", binsCnt, 0, maxPt);

    TH1* ptGen_notMuEvent0 = new TH1I("ptGen_notMuEvent0", "ptGen_notMuEvent0;ptGen [Gev];#", binsCnt, 0, maxPt);
    TH1I* ptGen_notMuPu = new TH1I("ptGen_notMuPu", "ptGen_notMuPu;ptGen [Gev];#", binsCnt, 0, maxPt);

    TH1* etaGen_MuEvent0 = new TH1I("etaGen_MuEvent0", "etaGen_MuEvent0;etaGen;#", binsCnt, -2.4, 2.4);
    TH1* etaGen_MuPU = new TH1I("etaGen_MuPU", "etaGen_MuPU;etaGen;#", binsCnt, -2.4, 2.4);

    TH1I* etaGen_notMuEvent0  = new TH1I("notMuEvent0 ", "notMuEvent0;etaGen;#", binsCnt, -2.4, 2.4);
    TH1I* etaGen_notMuPu  = new TH1I("etaGen_notMuPu ", "etaGen_notMuPu;etaGen;#", binsCnt, -2.4, 2.4);

    float overlapEndcap = 1.24; //border

    std::vector<ProbabilityVsPt>& probabilityVsPtHists = probabilityVsPtsMap[sampleName];
    int nnRes = 1;
    probabilityVsPtHists.emplace_back(sampleName + "_probabilityVsPt_allCands",      -1, nnRes, -1, 0, 2.4, useTttPt, useMargin); //name, iClass, nnRes, tpEvent, etaFrom, etaTo
    probabilityVsPtHists.emplace_back(sampleName + "_probabilityVsPt_muons_Ev0",      1, nnRes,  0, 0, 2.4, useTttPt, useMargin);
    probabilityVsPtHists.emplace_back(sampleName + "_probabilityVsPt_muons_MuPU",     1, nnRes,  1, 0, 2.4, useTttPt, useMargin);

    probabilityVsPtHists.emplace_back(sampleName + "_probabilityVsPt_allCands_barrel",      -1, nnRes, -1, 0, 0.8, useTttPt, useMargin);
    probabilityVsPtHists.emplace_back(sampleName + "_probabilityVsPt_muons_Ev0_barrel",      1, nnRes,  0, 0, 0.8, useTttPt, useMargin);
    probabilityVsPtHists.emplace_back(sampleName + "_probabilityVsPt_class_1_muons_MuPU_barrel",     1, nnRes,  1, 0, 0.8, useTttPt, useMargin);

    probabilityVsPtHists.emplace_back(sampleName + "_probabilityVsPt_allCands_overlap",      -1, nnRes, -1, 0.8, overlapEndcap, useTttPt, useMargin);
    probabilityVsPtHists.emplace_back(sampleName + "_probabilityVsPt_muons_Ev0_overlap",      1, nnRes,  0, 0.8, overlapEndcap, useTttPt, useMargin);
    probabilityVsPtHists.emplace_back(sampleName + "_probabilityVsPt_muons_MuPU_overlap",     1, nnRes,  1, 0.8, overlapEndcap, useTttPt, useMargin);

    probabilityVsPtHists.emplace_back(sampleName + "_probabilityVsPt_allCands_endcap",      -1, nnRes, -1, overlapEndcap, 2.4, useTttPt, useMargin);
    probabilityVsPtHists.emplace_back(sampleName + "_probabilityVsPt_muons_Ev0_endcap",      1, nnRes,  0, overlapEndcap, 2.4, useTttPt, useMargin);
    probabilityVsPtHists.emplace_back(sampleName + "_probabilityVsPt_muons_MuPU_endcap",     1, nnRes,  1, overlapEndcap, 2.4, useTttPt, useMargin);

    std::vector<EfficiencyVsPt> efficiencyVsPtMinBias;
    std::vector<EfficiencyVsPt> efficiencyVsPtSignal;

    std::vector<EfficiencyVsEta> efficiencyVsEtaMinBias;
    std::vector<EfficiencyVsEta> efficiencyVsEtaSignal;

    if(sampleName == "MinBias") {
        efficiencyVsPtMinBias.emplace_back(sampleName, "twoStubs", 0, 0, 2.4, true, useMargin);
        efficiencyVsPtMinBias.emplace_back(sampleName, "threeStubs", 0, 0, 2.4, true, useMargin);

        efficiencyVsPtMinBias.emplace_back(sampleName, "nnScoreThresh", 0, 0, 2.4, true, useMargin);

        efficiencyVsEtaMinBias.emplace_back(sampleName, "twoStubs", 0, 0, 50, true, useMargin);//name, fiterFunction, ptCut, genPtFrom, genPtTo, useTttPt,  useMargin
        efficiencyVsEtaMinBias.emplace_back(sampleName, "threeStubs", 0, 0, 50, true, useMargin);
        efficiencyVsEtaMinBias.emplace_back(sampleName, "nnScoreThresh", 0, 0, 50, true, useMargin);

    }
    else {
        efficiencyVsPtSignal.emplace_back(sampleName, "twoStubs", 0, 0, 2.4, false, useMargin);
        efficiencyVsPtSignal.emplace_back(sampleName, "threeStubs", 0, 0, 2.4, false, useMargin);

        efficiencyVsPtSignal.emplace_back(sampleName, "nnScoreThresh", 0, 0, 2.4, false, useMargin);

        efficiencyVsEtaSignal.emplace_back(sampleName, "twoStubs", 0, 0, 50, false, useMargin);//name, fiterFunction, ptCut, genPtFrom, genPtTo, useTttPt,  useMargin
        efficiencyVsEtaSignal.emplace_back(sampleName, "threeStubs", 0, 0, 50, false, useMargin);
        efficiencyVsEtaSignal.emplace_back(sampleName, "nnScoreThresh", 0, 0, 50, false, useMargin);

    }



    for(auto& event : events) {
        EventIntGmt* eventGmt = static_cast<EventIntGmt*>(event);
        if(eventGmt->classLabel == 1) { //muons only
            if(eventGmt->tpEvent == 0) {
                ptGen_MuEvent0->Fill(eventGmt->tpPt);
                etaGen_MuEvent0->Fill(eventGmt->tpEta);

                for(auto& eff : efficiencyVsPtSignal) {
                    eff.fill(eventGmt);
                }

                for(auto& eff : efficiencyVsEtaSignal) {
                    eff.fill(eventGmt);
                }
            }
            else {
                ptGen_MuPU->Fill(eventGmt->tpPt);
                etaGen_MuPU->Fill(eventGmt->tpEta);
            }
        }
        else {
            if(eventGmt->tpEvent == 0) {
                ptGen_notMuEvent0->Fill(eventGmt->tpPt);
                etaGen_notMuEvent0->Fill(eventGmt->tpEta);
                //std::cout<<"tpEvent == 0, tpType "<<(int)eventGmt->tpType<<std::endl;
            }
            else {
                ptGen_notMuPu->Fill(eventGmt->tpPt);
                etaGen_notMuPu->Fill(eventGmt->tpEta);
            }
        }

        for(auto& probabilityVsPt : probabilityVsPtHists) {
            probabilityVsPt.fill(eventGmt);
        }

        for(auto& eff : efficiencyVsPtMinBias) {
            eff.fill(eventGmt);
        }

        for(auto& eff : efficiencyVsEtaMinBias) {
            eff.fill(eventGmt);
        }
    }



    if(sampleName == "MinBias") {
        TCanvas* canvasCompareEff = new TCanvas("canvasCompareEffMinBias", "canvasCompareEffMinBias ", 1200, 800);
        canvasCompareEff->Divide(1, 2);
        int color = 1;
        bool first = true;
        subDir->cd();
        for(auto& eff : efficiencyVsPtMinBias) {
            eff.makeCumulativeEff(canvasCompareEff->cd(1), color, first);
            first = false;
            color++;
        }

        color = 1;
        first = true;
        subDir->cd();
        for(auto& eff : efficiencyVsEtaMinBias) {
            eff.drawCompare(canvasCompareEff->cd(2), color, first);
            first = false;
            color++;
        }

        canvasCompareEff->cd(1)->Update();
        canvasCompareEff->cd(2)->Update();

        subDir->cd();
        canvasCompareEff->Write();

        canvasCompareEff->SaveAs("canvasCompareEffMinBias.png" );
    }
    else {
        TCanvas* canvasCompareEff = new TCanvas("canvasCompareEff", "canvasCompareEff", 1200, 800);
        canvasCompareEff->Divide(1, 2);
        int color = 1;
        bool first = true;
        subDir->cd();
        for(auto& eff : efficiencyVsPtSignal) {
            eff.drawCompare(canvasCompareEff->cd(1), color, first);
            first = false;
            color++;
        }

        color = 1;
        first = true;
        subDir->cd();
        for(auto& eff : efficiencyVsEtaSignal) {
            eff.drawCompare(canvasCompareEff->cd(2), color, first);
            first = false;
            color++;
        }

        canvasCompareEff->cd(1)->Update();
        canvasCompareEff->cd(2)->Update();

        subDir->cd();
        canvasCompareEff->Write();

        canvasCompareEff->SaveAs("canvasCompareEff.png" );
    }
/*
    int pion = 211;
    int kaon = 321;
    char pionChar = pion;
    char kaonChar = kaon;

    std::cout<<"pion "<<pion<<" pionChar "<<(int)pionChar<<" kaon "<<kaon<<" kaonChar "<<(int)kaonChar<<std::endl;*/

/*    ptGen_MuEvent0->Write();
    ptGen_MuPU->Write();
    ptGen_notMuEvent0->Write();
    ptGen_notMuPu->Write();

    etaGen_MuEvent0->Write();
    etaGen_MuPU->Write();

    ptGen_notMuPu->Write();
    etaGen_notMuPu->Write();

    for(auto& probabilityVsPt : probabilityVsPtHists) {
        probabilityVsPt.hist->Write();
    }*/


/*    makeRocCurve(probabilityVsPtHists.at(1).hist, probabilityVsPtHists.at(0).hist, 2, 3);
    makeRocCurve(probabilityVsPtHists.at(1).hist, probabilityVsPtHists.at(0).hist, 3, 4);
    makeRocCurve(probabilityVsPtHists.at(1).hist, probabilityVsPtHists.at(0).hist, 4, 5);*/

}

} /* namespace lutNN */
