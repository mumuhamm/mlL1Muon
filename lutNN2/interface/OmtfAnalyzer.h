/*
 * OmtfAnalyzer.h
 *
 *  Created on: Jan 15, 2020
 *      Author: kbunkow
 */

#ifndef INTERFACE_OMTFANALYZER_H_
#define INTERFACE_OMTFANALYZER_H_

#include <string>
#include <vector>

#include "lutNN/lutNN2/interface/Event.h"
#include "lutNN/lutNN2/interface/EventsGeneratorOmtf.h"
#include "lutNN/lutNN2/interface/ClassifierToRegression.h"
#include "TH2.h"
#include "TCanvas.h"


namespace lutNN {

class MuonAlgorithm {
public:
	MuonAlgorithm(float ptCut, std::string name, int color);

	virtual ~MuonAlgorithm() {};

	virtual void fillHistos(EventFloatOmtf* eventFloatOmtf) = 0;

	virtual void makeHistos(TH1* ptGenNegAndPos);

	virtual void drawHistos();


	std::vector<MuonAlgorithm*> algosToCompare;

protected:
	float ptCut = 0;
	std::string name;
	int color = kBlack;

	TH1* acceptedEventsVsPtGen = nullptr;

	TH1* effVsPtGenOnPtCut = nullptr;
	TH1* effVsPtGenOnPtCut_rebined = nullptr;

	TH1* rateVsPtGenOnPtCut = nullptr;


	TH2* ptGenVsPtLutNN = nullptr;

	TH1* rateCumulVsPtCut = nullptr;

    TH1* goodChargeVsPt = nullptr;

    TH2* chargeGenVsChargeLutNNOnPtCut = nullptr;

	TCanvas* canvas = nullptr;

	friend class OmtfAnalyzer;
};

class OmtfAlgorithm: public MuonAlgorithm {
public:
	OmtfAlgorithm(float ptCut, unsigned int qualityCut, std::string name, int color): MuonAlgorithm(ptCut, name, color), qualityCut(qualityCut) {}

	virtual ~OmtfAlgorithm() {};

	virtual void fillHistos(EventFloatOmtf* eventFloatOmtf);

protected:
	unsigned int qualityCut = 0;
};


/*class OmtfAlgorithmPtCont: public OmtfAlgorithm {
public:
	OmtfAlgorithmPtCont(float ptCut, unsigned int qualityCut, std::string name, int color): OmtfAlgorithmPtCont(ptCut, qualityCut, name, color) {}

	virtual ~OmtfAlgorithmPtCont() {};

	virtual void fillHistos(EventFloatOmtf* eventFloatOmtf);

protected:
};*/



class LutNNClassifierMaxPAlgorithm: public MuonAlgorithm {
public:
	LutNNClassifierMaxPAlgorithm(float ptCut, unsigned int qualityCut, std::string name, int color, std::vector<float>& ptBins, double pThreshold = 1):
		MuonAlgorithm(ptCut, name, color), ptBins(ptBins), qualityCut(qualityCut), pThreshold(pThreshold) {} //TODO avoid copying of the ptBins
	virtual ~LutNNClassifierMaxPAlgorithm() {};

	virtual void fillHistos(EventFloatOmtf* eventFloatOmtf);

protected:
	std::vector<float> ptBins;

	unsigned int qualityCut = 0;

	double pThreshold = 1;
};

class LutNNClassifierMaxPInterAlgorithm: public LutNNClassifierMaxPAlgorithm {
public:
	LutNNClassifierMaxPInterAlgorithm(float ptCut, unsigned int qualityCut, std::string name, int color, std::vector<float>& ptBins, double pThreshold); //TODO avoid copying of the ptBins
	virtual ~LutNNClassifierMaxPInterAlgorithm() {};

	virtual void fillHistos(EventFloatOmtf* eventFloatOmtf);

	//virtual void makeHistos(TH1* ptGenNegAndPos);

	//virtual void drawHistos();

protected:

};



class LutNNClassifierCalibrated: public MuonAlgorithm {
public:
	LutNNClassifierCalibrated(float ptCut, unsigned int qualityCut, std::string name, int color, std::vector<float>& ptBins,
			ClassifierToRegressionBase& classifierToRegression, PtCalibration* ptCalibration = nullptr):
		MuonAlgorithm(ptCut, name, color), ptBins(ptBins), qualityCut(qualityCut),
		classifierToRegression(classifierToRegression), ptCalibration(ptCalibration) {} //TODO avoid copying of the ptBins

	virtual ~LutNNClassifierCalibrated() {};

	virtual void fillHistos(EventFloatOmtf* eventFloatOmtf);

protected:
	std::vector<float> ptBins;

	unsigned int qualityCut = 0;


	ClassifierToRegressionBase& classifierToRegression;

	PtCalibration* ptCalibration = nullptr;
};


class LutNNClassifierPSumAlgorithm: public MuonAlgorithm {
public:
	//pTreshold  probability threshold
	LutNNClassifierPSumAlgorithm(float ptCut, unsigned int qualityCut, std::string name, int color, std::vector<float>& ptBins, double pThreshold): MuonAlgorithm(ptCut, name, color), ptBins(ptBins), pThreshold(pThreshold), qualityCut(qualityCut) {} //TODO avoid copying of the ptBins
	virtual ~LutNNClassifierPSumAlgorithm() {};

	virtual void fillHistos(EventFloatOmtf* eventFloatOmtf);

private:
	std::vector<float> ptBins;
	double pThreshold = 0;
	unsigned int qualityCut = 0;
};

class LutNNRegressionAlgorithm: public MuonAlgorithm {
public:
	LutNNRegressionAlgorithm(float ptCut,  unsigned int qualityCut, std::string name, int color, PtCalibration* ptCalibration = nullptr):
	    MuonAlgorithm(ptCut, name, color),
	qualityCut(qualityCut), ptCalibration(ptCalibration) {}
	virtual ~LutNNRegressionAlgorithm() {};

	virtual void fillHistos(EventFloatOmtf* eventFloatOmtf);

    unsigned int qualityCut = 0;

private:
    PtCalibration* ptCalibration = nullptr;
};

enum OmtfAnalysisType {
    classifier,
    regression
};

class OmtfAnalyzer {
public:
    OmtfAnalyzer();
    virtual ~OmtfAnalyzer();

    std::vector<std::unique_ptr<MuonAlgorithm> > muonAlgorithms;

    void analyse(std::vector<EventFloat*> events, std::string dataFileName, std::string outFilePath);

    void printAlgoResults();

    TH1* ptGenNegAndPos = nullptr; //all events in the data sample, the histograms are in the root dataFileName

    TH1* ptGen_all = nullptr; //all events recorded in the root file dataFileName i.e. the events for which there was any OMTF candidate - for some events the candidate might not be present
    TH1* ptGen_all_Weighted = nullptr;
    TH1* chargeGen_all = nullptr;

    std::vector<TH2*> hitVsPt;
};

double vxMuRate(double pt_GeV);
double vxIntegMuRate(double pt_GeV, double dpt, double etaFrom, double etaTo);





} /* namespace lutNN */

#endif /* INTERFACE_OMTFANALYZER_H_ */
