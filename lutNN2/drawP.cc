/*
 * drawP.cc
 *
 *  Created on: Mar 3, 2020
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
#include "TEfficiency.h"


#include <sstream>
#include <string>
#include <vector>
#include <iostream>

using namespace std;
void drawP() {
	Double_t xbin[] = {
/*			-100,
			-60,
			-36,
			-26,
			-21,
			-18,
			-15,
			-12,
			-9,
			-7,
			-4,
			0,
    		4,
			7,
			9,
    		12,
			15,
			18,
			21,
			26,
			36,
			60,
			100*/

			-100,
			-60,
			-36,
			-26,
			-20,
			-16,
			-12,
			-9,
			-7,
			-4,
			0,
    		4,
			7,
			9,
    		12,
			16,
			20,
			26,
			36,
			60,
			100
	};

	vector<double> nnOut = {
			//0.000496259, 0.000483355, 0.00146641, 0.00112924,  0.0630505, 0.000492812,   0.850796, 0.00116189,   0.069685, 0.00031911, 0.000348065, 0.000316564, 0.000373579, 0.000260239, 0.000629304, 0.000281531, 0.000545358, 0.000287136, 0.00181605, 0.00147356, 0.000612165, 0.00397552
			//0.106251,    0.09303, 0.00102942, 0.00458298, 0.000904731, 0.00657112, 0.00127926, 0.00554235, 0.00153072, 0.00255858, 0.000949153,  0.0144771, 0.00119957,   0.120276, 0.00160548,   0.450564, 0.00169519,   0.176742, 0.00518899,  0.0040221,
			0.138347,   0.116486,  0.0455629, 0.000497225,   0.049326, 0.00477514,   0.259502, 0.00307722,   0.350152, 0.000959414,  0.0178794, 0.000859352, 0.00168817, 0.000577612, 0.000677979, 0.00057869, 0.000742672, 0.000507969, 0.00589089, 0.00191221,

	};

	TH1D* histP = new TH1D("nnP", "nn P", 20, xbin);

	TCanvas* canvas = new TCanvas("canvas",  "canvas", 700, 600);
	canvas->cd()->SetGridx();

	cout<<" nnOut.size() "<<nnOut.size()<<endl;

	histP->SetTitle(";pT [GeV]; P");
	histP->SetStats(0);

	for(unsigned int i = 0; i < nnOut.size(); i++) {
		int bin = 1;
		if(i%2)
			bin = i/2 + nnOut.size()/2 +1;
		else {
			bin = nnOut.size()/2 -i/2;
		}
		cout<<"i "<<i<<" bin "<<bin<<endl;
		histP->SetBinContent(bin, nnOut[i]);
	}
	histP->Draw("hist");
}

