/*
 * ClassifierToRegression.cpp
 *
 *  Created on: Feb 19, 2020
 *      Author: kbunkow
 */

#include "lutNN/lutNN2/interface/ClassifierToRegression.h"
#include "TH2F.h"
#include "TFile.h"

#include <algorithm>
#include <iostream>

namespace lutNN {

float ClassifierToRegressionBase::getCalibratedValue(float& value) {
	if(ptCalibration)
		return ptCalibration->getValue(value);

    //std::cout<<"ClassifierToRegressionBase::getCalibratedValue ptCalibration "<<ptCalibration<<std::endl;
	return 0;
}

ClassifierToRegressionMaxPLut::ClassifierToRegressionMaxPLut(std::vector<float>& ptBins, unsigned int lutSize): ClassifierToRegressionBase(), ptBins(ptBins), lutSize(lutSize),
		values(ptBins.size(), std::vector<double>(lutSize)),
		entries(ptBins.size(), std::vector<double>(lutSize))
{
}



ClassifierToRegressionMaxPLut::ClassifierToRegressionMaxPLut(std::vector<float>& ptBins, unsigned int lutSize, PtCalibration* ptCalibration):
		ClassifierToRegressionBase(ptCalibration), ptBins(ptBins), lutSize(lutSize),
		values(ptBins.size(), std::vector<double>(lutSize)),
		entries(ptBins.size(), std::vector<double>(lutSize))
{
}


ClassifierToRegressionMaxPLut::~ClassifierToRegressionMaxPLut() {
	// TODO Auto-generated destructor stub
}

//maxPBin is the class index, not the ptBin index
unsigned int ClassifierToRegressionMaxPLut::lutAddres(EventFloat* event, unsigned int& maxPBin) {

	unsigned int lutSizeBits = log2(lutSize);

	unsigned int addressL = 0;
	unsigned int addressR = 0;
	if(maxPBin > 1)
		addressL = (event->nnResult[maxPBin] - event->nnResult[maxPBin-2]) * (1<<(lutSizeBits/2)); //the difference is in range 0-1, -2 because the positive and negative charge bins ale altered

	if(maxPBin < event->nnResult.size() -2)
		addressR = (event->nnResult[maxPBin] - event->nnResult[maxPBin+2]) * (1<<(lutSizeBits/2));

	return ((addressL<<(lutSizeBits/2)) | addressR);
}

void ClassifierToRegressionMaxPLut::train(std::vector<EventFloat*> events) {
	std::cout<<"ClassifierToRegression::train"<<std::endl;

    for(auto& event : events) {
    	auto maxP = std::max_element(event->nnResult.begin(), event->nnResult.end());
    	unsigned int maxPBin = std::distance(event->nnResult.begin(), maxP);

    	//std::cout<<*(EventFloatOmtf*)event<<std::endl;
    	//std::cout<<"maxPBin "<<maxPBin<<" maxP "<<*maxP<<std::endl;

    	unsigned int lutAdd = lutAddres(event, maxPBin);

    	EventFloatOmtf* eventFloatOmtf = static_cast<EventFloatOmtf*>(event);

    	int middleBin = maxPBin/2;
    	//ptBins are the upper edges of the bins, therefore -2 here
    	int leftBin = middleBin -2; //watch out - here the leftBin is for the ptBins, not for the nnResult
    	if(leftBin < 0)
    		leftBin = 0;

    	int rightBin = middleBin +1;
    	if(rightBin >= (int)ptBins.size())
    		rightBin = ptBins.size() -1;

    	if( (ptBins.at(leftBin) <= eventFloatOmtf->muonPt) && (eventFloatOmtf->muonPt <= ptBins.at(rightBin) ) ) {
    		values.at(middleBin).at(lutAdd) += eventFloatOmtf->muonPt;
    		entries.at(middleBin).at(lutAdd) += 1; //TODO add weight if needed
    	}
    }

    TFile outfile( "ClassifierToRegression.root", "RECREATE");

    for(unsigned int iPtBin = 0; iPtBin < values.size(); iPtBin++) {
    	std::ostringstream name;
    	name<<"ClassifierToRegression_PtBin_"<<iPtBin;
    	unsigned int bins = sqrt(lutSize);
    	unsigned int lutSizeBits = log2(lutSize);

    	TH1F hist1D = TH1F(name.str().c_str(), name.str().c_str(), lutSize, 0, lutSize);
    	name<<"_2D";

    	TH2F hist2D = TH2F(name.str().c_str(), name.str().c_str(), bins, 0, bins, bins, 0, bins);
    	for(unsigned int i = 0; i < values[iPtBin].size(); i++) {
    		if(entries[iPtBin][i]) {
    			values[iPtBin][i] /= entries[iPtBin][i];

    			hist2D.Fill(i>>(lutSizeBits/2), (i & (1<<lutSizeBits/2) ) -1, values[iPtBin][i]);

    			hist1D.Fill(i, values[iPtBin][i]);
    		}
    	}

    	hist1D.Write();
    	hist2D.Write();
    }
}

float ClassifierToRegressionMaxPLut::getValue(EventFloat* event) {
	auto maxP = std::max_element(event->nnResult.begin(), event->nnResult.end());
	unsigned int maxPBin = std::distance(event->nnResult.begin(), maxP);

	unsigned int lutAdd = lutAddres(event, maxPBin);
	return values.at(maxPBin/2).at(lutAdd);
}


float ClassifierToRegressionMeanP::getValue(EventFloat* event) {
	//for the high pt the sign is often hard to determine, so we sum  the p in the two bins
	double pSumNeg = event->nnResult[event->nnResult.size() -1] + event->nnResult[event->nnResult.size() -2];
	double pSumPos = pSumNeg;

	double ptNeg = event->nnResult[event->nnResult.size() -1] * 100 + event->nnResult[event->nnResult.size() -2] * 100; //100 GeV is "middle" of the two highest pt bins
	double ptPos = ptNeg;

	for(int iOut = event->nnResult.size() -3; iOut >= 0 ; iOut--) {
		int ptBinNum = iOut/2;
		double pT = 0;
		if(ptBinNum == 0)
			pT = (ptBins.at(ptBinNum) + 3 ) /2.; //~3GeV is the lowests pt of the muon reaching the OMTF
		else
			pT = (ptBins.at(ptBinNum) + ptBins.at(ptBinNum -1) ) /2.;

		if(iOut%2) {
			pSumNeg += event->nnResult[iOut];
			ptNeg   += event->nnResult[iOut] * pT;
		}
		else {
			pSumPos += event->nnResult[iOut];
			ptPos   += event->nnResult[iOut] * pT;
		}
	}

	if(pSumNeg > pSumPos)
		return -ptNeg / pSumNeg;
	else
		return ptPos / pSumPos;
	return 0;
}



float ClassifierToRegressionPSum::getValue(EventFloat* event) {
	//for the hig pt there sign is often hard to determin, so we sum  the p in the two bins
	double pSumNeg = event->nnResult[event->nnResult.size() -1] + event->nnResult[event->nnResult.size() -2];
	double pSumPos = pSumNeg;

	if(pSumNeg >= pThreshold ) {
		if(event->nnResult[event->nnResult.size() -1] > event->nnResult[event->nnResult.size() -2]) {//the sign corresponding to the higher p is chosen
			return ptBins.at(ptBins.size() -2);
		}
		else
			return -ptBins.at(ptBins.size() -2);
	}

	for(int iOut = event->nnResult.size() -3; iOut >= 0 ; iOut--) {
		if(iOut%2)
			pSumNeg += event->nnResult[iOut];
		else
			pSumPos += event->nnResult[iOut];


		float lowerEdge = 0;
		if(iOut/2 > 0)
			lowerEdge = ptBins.at(iOut/2 - 1);

		float ptCorrection = ptBins.at(iOut/2) - lowerEdge; //ptBinSize - the ptBins is the upper edge of the bin

		if(pSumNeg >= pThreshold ) {
			ptCorrection = ptCorrection * (1. - (pSumNeg - pThreshold) / event->nnResult[iOut] );
			return -(ptBins.at(iOut/2) - ptCorrection);
		}

		if(pSumPos >= pThreshold ) {
			ptCorrection = ptCorrection * (1. - (pSumPos - pThreshold) / event->nnResult[iOut] );
			return (ptBins.at(iOut/2) - ptCorrection);
		}

	}

	return 0;
}


PtCalibration::PtCalibration(std::string name, float ptMax, float binCnt) {
    ptBinWidth = ptMax / binCnt;
	std::ostringstream histName;
	histName<<name<<"_ptGenVsPtLutNN";
	ptGenVsPtLutNN = new TH2I(histName.str().c_str(), histName.str().c_str(), binCnt, 0, ptMax, binCnt, 0, ptMax);

	histName.str("");
	histName<<name<<"_ptToPtCalib";
	ptToPtCalib = new TH1F(histName.str().c_str(), histName.str().c_str(), binCnt, 0, ptMax);
}

PtCalibration::PtCalibration(TFile* rootFile, std::string name) {
	std::ostringstream histName;
	histName.str("");
	histName<<name<<"_ptToPtCalib";
	std::cout<<"PtCalibration "<<histName.str()<<std::endl;
	ptToPtCalib = (TH1*)rootFile->Get(histName.str().c_str() );
    std::cout<<"PtCalibration ptToPtCalib "<<ptToPtCalib<<std::endl;
}

void PtCalibration::fillPtGenVsPtLutNN(std::vector<EventFloat*> events, unsigned int qualityCut) {
    for(auto& event : events) {
        EventFloatOmtf* eventFloatOmtf = static_cast<EventFloatOmtf*>(event);
        auto lutNNPt = eventFloatOmtf->nnResult[0];
        if(eventFloatOmtf->omtfQuality >= qualityCut) //&& lowPtPSum < pThreshold
            ptGenVsPtLutNN->Fill(eventFloatOmtf->muonPt, lutNNPt);
    }

    calulatePtToPtCalib();
}

void PtCalibration::fillPtGenVsPtLutNN(std::vector<EventFloat*> events, ClassifierToRegressionBase& classifierToRegression, unsigned int qualityCut) {
    for(auto& event : events) {
        auto lutNNPt = fabs(classifierToRegression.getValue(event) );

        EventFloatOmtf* eventFloatOmtf = static_cast<EventFloatOmtf*>(event);
        if(eventFloatOmtf->omtfQuality >= qualityCut) //&& lowPtPSum < pThreshold
            ptGenVsPtLutNN->Fill(eventFloatOmtf->muonPt, lutNNPt);
    }

    calulatePtToPtCalib();
}

void PtCalibration::calulatePtToPtCalib() {
    double effTreshold = 0.8;

    int plateauOffset = 20 / ptBinWidth;

    for(unsigned int iPt1 = ptGenVsPtLutNN->GetYaxis()->GetNbins() -plateauOffset; iPt1 >= 1; iPt1--) {
    	//ptGenVsPtLutNN->GetYaxis()->getbin

    	double accpeptedEvCntOnPlateau = ptGenVsPtLutNN->Integral(iPt1 + plateauOffset,  -1, iPt1, -1);
    	double allEvCntOnPlateau       = ptGenVsPtLutNN->Integral(iPt1 + plateauOffset,  -1, 0,   -1);

    	double plateauEff = accpeptedEvCntOnPlateau / allEvCntOnPlateau;

    	double previousBinEff = 0;
        for(int iPt2 = iPt1; iPt2 <  ptGenVsPtLutNN->GetXaxis()->GetNbins(); iPt2++) {
        	//ptGenVsPtLutNN->GetYaxis()->getbin

        	double accpeptedEvCntInBin = ptGenVsPtLutNN->Integral(iPt2,  iPt2, iPt1, -1);
        	double allEvCntInBin       = ptGenVsPtLutNN->Integral(iPt2,  iPt2,  -1,  -1);

        	double binEff = accpeptedEvCntInBin / allEvCntInBin;

        	if(binEff > (effTreshold * plateauEff) ) {
        		double prevBinContent = ptToPtCalib->GetBinContent(iPt1, ptGenVsPtLutNN->GetXaxis()->GetBinLowEdge(iPt2));
        		if(prevBinContent != 0) {
        			std::cout<<"PtCalibration::train: iPt1 "<<iPt1<<"iPt2 "<<iPt2<<" prevBinContent "<<prevBinContent<<" !!!!!!!!!!!!!!!!!!";
        		}

        		if( (binEff - (effTreshold * plateauEff) ) < ( (effTreshold * plateauEff) - previousBinEff) ) {
        			ptToPtCalib->SetBinContent(iPt1, ptGenVsPtLutNN->GetXaxis()->GetBinLowEdge(iPt2));
        		}
        		else
        			ptToPtCalib->SetBinContent(iPt1, ptGenVsPtLutNN->GetXaxis()->GetBinLowEdge(iPt2-1));
        		break;
        	}

        	previousBinEff = binEff;
        }
    }

    float last = 0;
    for(int iPt1 = 20; iPt1 < ptGenVsPtLutNN->GetYaxis()->GetNbins(); iPt1++) {
    	if(last < ptToPtCalib->GetBinContent(iPt1))
    		last = ptToPtCalib->GetBinContent(iPt1);
    	else if(ptToPtCalib->GetBinContent(iPt1) == 0)
    		ptToPtCalib->SetBinContent(iPt1, last);
    }


    ptGenVsPtLutNN->Write();
	ptToPtCalib->Write();
}

float PtCalibration::getValue(float lutNNPt) {
	auto bin = ptToPtCalib->GetXaxis()->FindBin(lutNNPt);
	return  ptToPtCalib->GetBinContent(bin);
}

};
