/*
 * EventsGeneratoOmtf.h
 *
 *  Created on: Dec 12, 2019
 *      Author: kbunkow
 */

#ifndef SRC_EVENTSGENERATOOMTF_H_
#define SRC_EVENTSGENERATOOMTF_H_

#include "lutNN/lutNN2/interface/EventsGeneratorBase.h"
#include "TH1.h"

#include <functional>

namespace lutNN {

struct OmtfEvent {
public:
	//TODO this is the version for the rootFileVersion==1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    double muonPt = 0, muonEta = 0, muonPhi = 0;
    int muonCharge = 0;
    
    int omtfCharge = 0;
    int omtfProcessor = 0;
    int omtfScore = 0;

    double omtfPt = 0, omtfEta = 0, omtfPhi = 0;
        
    unsigned int omtfQuality = 0;
    unsigned int omtfRefLayer = 0;
        
   	//TODO this is the version for the rootFileVersion==2 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   	/*
   	float muonPt = 0, muonEta = 0, muonPhi = 0;
    char muonCharge = 0;
    
    char omtfCharge = 0;
    char omtfProcessor = 0;
    short omtfScore = 0;

    float omtfPt = 0, omtfEta = 0, omtfPhi = 0;
        
    char omtfQuality = 0;
    char omtfRefLayer = 0;
   	*/
   	    
        
    unsigned int eventNum = 0;

    short muonEvent = -1;

    float muonDxy = 0;
    float muonRho = 0;

    short omtfHwEta = 0;

    char omtfRefHitNum = 0;

    unsigned int omtfFiredLayers = 0;

    bool killed = false;

    float omtfPtCont = 0;

  struct Hit {
    union {
      unsigned long rawData;
      struct {
        /*char layer = 0;
        char quality = 0;
        char z = 0;
        short eta = 0;
        short phiDist = 0;*/

        //new

        char layer = 0;
        char quality = 0;
        char z = 0;
        char valid = 0;
        short eta = 0;
        short phiDist = 0;
      };
    };

    Hit(unsigned long rawData): rawData(rawData) {

    }
  };

  std::vector<unsigned long>* hits = nullptr;

};


class EventFloatOmtf: public EventFloat {
public:
    EventFloatOmtf(unsigned int inputCnt, unsigned int ouptutCnt, float defInVal = 0) : EventFloat(inputCnt, ouptutCnt, defInVal) { }

    virtual ~EventFloatOmtf() {};

    float muonPt = 0;
    float muonEta = 0;
    signed char muonCharge = 0;

    float omtfPt = 0;
    signed char omtfCharge = 0;
    unsigned char omtfQuality = 0;

    char omtfRefLayer = 0;

    bool killed = false;

    friend std::ostream & operator << (std::ostream &out, const EventFloatOmtf& event);
};

class EventsGeneratorOmtf: public EventsGeneratorBase<EventFloat> {
private:
    unsigned int inputCnt = 1;
    unsigned int outputCnt = 1;

    unsigned int omtfGoodEvents = 0;

    std::vector<float> ptBins;

public:

    //for pt regression, outputCnt should be 1
    EventsGeneratorOmtf(std::default_random_engine& generator, unsigned int outputCnt);

    //for classification
    EventsGeneratorOmtf(std::default_random_engine& generator, unsigned int outputCnt, std::vector<float>& ptBins);
    virtual ~EventsGeneratorOmtf();


    void printEvent(EventFloat* event, std::ostream& ostr = std::cout);

    //return true if the hit is valid
    static bool omtfHitToEventInput(OmtfEvent::Hit& hit, EventFloat* event, unsigned int omtfRefLayer, bool print);

    //generates one cell in event->inputs[] for every pixel
    virtual void readEvents(std::string dataFileName, int rootFileVersion, Long64_t maxEvents = -1 );

    virtual void setEventWeight(TH1* ptGenNeg, TH1* ptGenPos);

    virtual void getRandomEvents(std::vector<EventFloat*>& events) {};

    virtual std::vector<EventFloat*>& getEvents() {
        return events;
    }

    unsigned int getInputCnt() const {
        return inputCnt;
    }

    std::function<void (OmtfEvent& omtfEvent, EventFloatOmtf* event)> setExpectedResult;
    void setExpectedResultClassifier(OmtfEvent& omtfEvent, EventFloatOmtf* event);
    void setExpectedResultPtRegression(OmtfEvent& omtfEvent, EventFloatOmtf* event);

    const unsigned int minHitsCnt = 3;
    const unsigned int maxHitCnt = 18; //layer cnt

    const float noHitVal = 1023.;// / (maxHitsCnt - minHitsCnt); //TODO take t from one place

};



} /* namespace lutNN */

#endif /* SRC_EVENTSGENERATOOMTF_H_ */
