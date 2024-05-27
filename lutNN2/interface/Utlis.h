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

#include "lutNN/lutNN2/interface/Node.h"
#include "lutNN/lutNN2/interface/Event.h"
#include "TPaveLabel.h"
#include "LutNetworkBase.h"

namespace lutNN {

class Utlis {
public:
    Utlis();
    virtual ~Utlis();
};


class LutNetworkPrint {
public:
    LutNetworkPrint();

    virtual ~LutNetworkPrint();

    TCanvas* createCanvasLutsAndOutMap(unsigned int lutLayerCnt, unsigned int maxNodesPerLayer, int padOutMapSubPpadsCnt);

    //void printLuts(TPad& padLuts, LutNetwork& lutNetwork, unsigned int maxNodesPerLayer);

    void printLuts2(LutNetworkBase& lutNetwork, unsigned int maxNodesPerLayer);

    //void printGradinets(TPad& padGradients, LutNetwork& lutNetwork, unsigned int maxHistsPerLayer);

    void printOutMap(LutNetworkBase& lutNetwork, unsigned int& lutSize);

    template <typename EventType>
    double printExpectedVersusNNOut(std::vector<std::pair<double, double> >& ranges, std::vector<EventType*>& events, CostFunction& costFunction);

    template <typename EventType>
    double printExpectedVersusNNOutHotOne(std::vector<EventType*>& events, CostFunction& costFunction);

    void createCostHistory(int iterations, int printEveryIteration, double yAxisMin = 0.01, double yAxisMax = 10);

    void updateCostHistory(int iteration, double trainSampleCost, double validSampleCost, std::string pngfile);

private:

    TVirtualPad* padCostHistory  = nullptr;
    TH1F* trainSampleCostHist = nullptr;
    TH1F* validSampleCostHist = nullptr;

    TCanvas* canvasLutNN = nullptr;
    TVirtualPad* padLuts = nullptr;
    TVirtualPad* padOutMap = nullptr;

    TPaveLabel* textCosts = nullptr;

    unsigned int lutLayerCnt = 0;

};

void printLut(std::string name, TVirtualPad* pad, LutNode* lutNode, unsigned int iLayer, bool logY);

void printLuts3(LutNetworkBase& lutNetwork, unsigned int lutsPerCanvas);

} /* namespace lutNN */


#endif /* UTLIS_H_ */
