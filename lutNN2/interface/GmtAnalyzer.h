/*
 * GmtAnalyzer.h
 *
 *  Created on: Dec 28, 2021
 *      Author: kbunkow
 */

#ifndef INTERFACE_GMTANALYZER_H_
#define INTERFACE_GMTANALYZER_H_


#include <string>
#include <vector>

#include "lutNN/lutNN2/interface/Event.h"
#include "lutNN/lutNN2/interface/EventsGeneratorGmt.h"

#include "TH1.h"
#include "TH2.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TEfficiency.h"

namespace lutNN {

class ProbabilityVsPt {
public:
    //useMargin - if true the difference between two nn outputs is used as the posterior probability measurement
    ProbabilityVsPt(std::string name, int iClass, int nnRes, int tpEvent, float etaFrom, float etaTo, bool useTttPt, bool useMargin);

    void fill(EventIntGmt* eventGmt);
//private:
    TH2* hist = nullptr;
    int iClass = -1;
    int nnRes = -1;
    int tpEvent = -1;
    float etaFrom = 0;
    float etaTo = 0;

    bool useTttPt =  true; //if false the tpPt (gen pt) is used

    bool useMargin = false; //for hinge lost or for mean square lost
};

class GmtAnalyzer {
public:
    GmtAnalyzer(std::string outFilePath);
    virtual ~GmtAnalyzer();

    void analyze(std::vector<EventInt*> events, std::string sampleName, bool useTttPt, bool useMargin);

    void makeRocCurve(TH2* signalHist, TH2* falsesHist, float ptFromSignla, float ptToSignal, float ptFromFalse, float ptToFalse);

    void makeRocCurve(std::string signalSample, std::string falseSample);
private:
    TFile outfile;

    std::map<std::string, std::vector<ProbabilityVsPt> > probabilityVsPtsMap;

/*    TH1* ptGen_MuEvent0 = nullptr;
    TH1* etaGen_MuEvent0 = nullptr;

    TH1* ptGen_MuPU = nullptr;
    TH1* etaGen_MuPU = nullptr;


    TH1* ptGen_notMuEvent0 = nullptr;*/
};

} /* namespace lutNN */

#endif /* INTERFACE_GMTANALYZER_H_ */
