/*
 * Utlis.h
 *
 *  Created on: Jul 3, 2018
 *      Author: kbunkow
 */

#ifndef UTLIS_H_
#define UTLIS_H_

#include "TCanvas.h"
#include "TH1F.h"

#include "lutNN/interface/Node.h"
#include "lutNN/interface/LutNetwork.h"

namespace lutNN {

class Utlis {
public:
    Utlis();
    virtual ~Utlis();
};


void printLuts(TPad& padLuts, LutNetwork& lutNetwork, unsigned int maxNodesPerLayer);

void printLuts2(TPad& padLuts, TPad& padGradients, LutNetwork& lutNetwork, unsigned int maxNodesPerLayer);

void printGradinets(TPad& padGradients, LutNetwork& lutNetwork, unsigned int maxHistsPerLayer);

void printOutMap(TPad& padOutMap,  LutNetwork& lutNetwork, unsigned int& lutSize);

double printExpectedVersusNNOut(TPad& padOutMap, std::vector<std::pair<double, double> >& ranges, std::vector<Event>& events);

double printExpectedVersusNNOutHotOne(TPad& padOutMap,  std::vector<std::pair<double, double> >& ranges, std::vector<Event*>& events, CostFunction& costFunction);

void createCostHistory(int iterations, int printEveryIteration);

void updateCostHistory(TPad& padRateHistory, int iteration, double trainSampleCost, double validSampleCost);

class EventsGeneratorBase {
public:
    EventsGeneratorBase(std::default_random_engine& generator): miniBatchBegin(events.begin()), generator(generator) {};

    virtual ~EventsGeneratorBase() {};

    //virtual void generateEvents(std::vector<Event*>& events, int mode) = 0;

    virtual std::vector<Event*>& getEvents() = 0;

    virtual void getRandomEvents(std::vector<Event*>& events) = 0;

    virtual void shuffle();

    virtual void getNextMiniBatch(std::vector<Event*>& events);

protected:
    std::vector<Event*> events;

    std::vector<Event*>::iterator miniBatchBegin;

    std::default_random_engine& generator;
};

} /* namespace lutNN */


#endif /* UTLIS_H_ */
