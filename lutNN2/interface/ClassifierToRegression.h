/*
 * ClassifierToRegression.h
 *
 *  Created on: Feb 19, 2020
 *      Author: kbunkow
 */

#ifndef INTERFACE_CLASSIFIERTOREGRESSION_H_
#define INTERFACE_CLASSIFIERTOREGRESSION_H_

#include "lutNN/lutNN2/interface/Event.h"
#include "lutNN/lutNN2/interface/EventsGeneratorOmtf.h"
#include <vector>
#include "TH2.h"
#include "TH1.h"
#include "TFile.h"
#include <memory.h>

namespace lutNN {

class PtCalibration;

class ClassifierToRegressionBase {
public:
	ClassifierToRegressionBase(): ptCalibration(nullptr) {}

	ClassifierToRegressionBase(PtCalibration* ptCalibration): ptCalibration(ptCalibration) {}

	virtual ~ClassifierToRegressionBase() {}

	virtual float getValue(EventFloat* event) = 0;

	//takes the ownership of the ptCalibration
	void setPtCalibration(PtCalibration* ptCalibration) {
		this->ptCalibration.reset(ptCalibration);
	}

	virtual float getCalibratedValue(float& value);

private:
	std::unique_ptr<PtCalibration> ptCalibration;
};

class ClassifierToRegressionMaxPLut: public ClassifierToRegressionBase {
public:
	ClassifierToRegressionMaxPLut(std::vector<float>& ptBins, unsigned int lutSize);

	//takes the ownership of the ptCalibration
	ClassifierToRegressionMaxPLut(std::vector<float>& ptBins, unsigned int lutSize, PtCalibration* ptCalibration);

	virtual ~ClassifierToRegressionMaxPLut();

	//maxPBin is the class index, not the ptBin index
	virtual unsigned int lutAddres(EventFloat* event, unsigned int& maxPBin);

	virtual void train(std::vector<EventFloat*> events);

	virtual float getValue(EventFloat* event);

protected:
	std::vector<float>& ptBins;//watch out the ptBins are two times smaller than the number of classes in the nn

	unsigned int lutSize = 1;

	std::vector<std::vector<double> > values; //[ptBin][complex lutAddres]
	std::vector<std::vector<double> > entries; //entries can have weights

};

class ClassifierToRegressionMeanP: public ClassifierToRegressionBase {
public:
	ClassifierToRegressionMeanP(std::vector<float>& ptBins): ptBins(ptBins) {};

	//takes the ownership of the ptCalibration
	ClassifierToRegressionMeanP(std::vector<float>& ptBins, PtCalibration* ptCalibration): ClassifierToRegressionBase(ptCalibration), ptBins(ptBins) {};

	virtual ~ClassifierToRegressionMeanP() {};


	virtual float getValue(EventFloat* event);

protected:
	std::vector<float>& ptBins;//watch out the ptBins are two times smaller than the number of classes in the nn
};

class ClassifierToRegressionPSum: public ClassifierToRegressionBase {
public:
	ClassifierToRegressionPSum(std::vector<float>& ptBins, double pThreshold): ptBins(ptBins), pThreshold(pThreshold) {};

	//takes the ownership of the ptCalibration
	ClassifierToRegressionPSum(std::vector<float>& ptBins, double pThreshold, PtCalibration* ptCalibration): ClassifierToRegressionBase(ptCalibration), ptBins(ptBins), pThreshold(pThreshold) {};

	virtual ~ClassifierToRegressionPSum() {};


	virtual float getValue(EventFloat* event);

protected:
	std::vector<float>& ptBins;//watch out the ptBins are two times smaller than the number of classes in the nn

	double pThreshold = 1.;
};


class PtCalibration { //
public:
	//to be used for training
	PtCalibration(std::string name, float ptMax, float binCnt) ;


	PtCalibration(TFile* rootFile, std::string name) ;

	virtual ~PtCalibration() {}

	//version for regression
	void fillPtGenVsPtLutNN(std::vector<EventFloat*> events, unsigned int qualityCut);

	//version for classifier
	void fillPtGenVsPtLutNN(std::vector<EventFloat*> events, ClassifierToRegressionBase& classifierToRegression, unsigned int qualityCut);

	void calulatePtToPtCalib();

	float getValue(float lutNNPt);

	const auto& getPtToPtCalib() const {
	    return ptToPtCalib;
	}

private:
	TH2* ptGenVsPtLutNN = nullptr;

	TH1* ptToPtCalib = nullptr;

	double ptBinWidth = 1;
};

};

#endif /* INTERFACE_CLASSIFIERTOREGRESSION_H_ */
