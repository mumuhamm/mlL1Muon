/*
 * OmtfAnalyzerPlots.cc
 *
 *  Created on: Nov 9, 2018
 *      Author: kbunkow
 */


#include "TROOT.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TString.h"
#include "TFile.h"
#include "TTree.h"
#include "TLegend.h"
#include <sstream>
#include <string>
#include <vector>

using namespace std;



bool first = true;

TCanvas* canvas = new TCanvas("canvasCompare", "canvasCompare", 1200, 800);;

void drawLut(string lutName, TFile * file);

int LutPlots() {
  gStyle->SetOptStat(0);

  canvas->Divide(1, 2);

  const char* rootFileName = "/home/kbunkow/projects/lutNN/pictures/lutNN.root";
  TFile * file = new TFile(rootFileName);
  //file->ls();

/*  TDirectory* curDir = file;
  file->cd("omtfTTAnalyzer");
  file->ls();*/

  string lutName = "layer_0_neuron_0_lut_2";
  drawLut(lutName, file);
  return 0;
}


void drawLut(string lutName, TFile * file)
{
    TH1F* lutValues = (TH1F*)file->Get( ("Values_" + lutName).c_str() );
    TH1F* lutEntries = (TH1F*)file->Get( ("Entries_" + lutName).c_str() );

    canvas->cd(1);
    lutValues->Draw();

    canvas->cd(2);
    lutEntries->Draw();


    for(int iBin = 1; iBin <= lutValues->GetNbinsX(); iBin++) {
      unsigned int iAddr = iBin - 1;
      cout<<"iAddr "<<setw(4)<<iAddr<<" value "<<setw(10)<<lutValues->GetBinContent(iBin)<<" entries "<<setw(10)<<lutEntries->GetBinContent(iBin)<<" -- ";

      unsigned int bitCnt = log2(lutValues->GetNbinsX() +1);
      for(unsigned int iBit = 0; iBit < bitCnt; iBit++) {
          unsigned int newAddr = iAddr ^ (1 << iBit);
          cout<<setw(4)<<newAddr<<" "<<setw(10)<<lutValues->GetBinContent(newAddr +1)<<", ";
      }
      cout<<endl;
    }
}


