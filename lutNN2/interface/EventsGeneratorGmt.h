/*
 * EventsGeneratorGMT.h
 *
 *  Created on: Dec 15, 2021
 *      Author: kbunkow
 */

#ifndef INTERFACE_EVENTSGENERATORGMT_H_
#define INTERFACE_EVENTSGENERATORGMT_H_

#include "lutNN/lutNN2/interface/EventsGeneratorBase.h"

#include "TH1.h"

#include <functional>

namespace lutNN {

const unsigned int tfLayersCnt = 5;
struct TrackMatchedMuonRecord {
  //from trackingParticle
  unsigned int eventNum = 0;
  short tpEvent = -1;

  float tpPt = 0, tpEta = 0, tpPhi = 0, tpBeta = 0;
  short tpCharge = 0;
  int tpType = 0;

  //0 - no match, 1 - very loose, 2 - loose , 3
  char matching = 0;

  //from ttTrack
  short tttCurvature = 0;
  char tttCharge = 0;
  unsigned short tttPt = 0;
  short tttEta = 0;
  short tttPhi = 0;
  short tttZ0 = 0;
  short tttD0 = 0;

  //from TTTrack_TrackWord
  char tttChi2rphi = 0; //4 bits
  char tttChi2rz = 0; //4 bits
  char tttBendChi2 = 0; //3 bits
  char tttQualityMVA = 0; //3 bits
  char tttOtherMVA = 0; //6 bits


  //from GMT
  uint gmtBeta = 0;
  bool isGlobal = 0;
  uint quality = 0;

  unsigned short hitsValid = 0;

  std::vector<unsigned char>* deltaCoords1 = nullptr;
  std::vector<unsigned char>* deltaCoords2 = nullptr;

  std::vector<unsigned char>* deltaEtas1 = nullptr;
  std::vector<unsigned char>* deltaEtas2 = nullptr;

  std::vector<char>* stubTiming = nullptr;
  std::vector<char>* stubType = nullptr;


  //std::vector<unsigned char> stubTiming2;

  TrackMatchedMuonRecord()/*:
    deltaCoords1(tfLayersCnt), deltaCoords2(tfLayersCnt),
    deltaEtas1(tfLayersCnt), deltaEtas2(tfLayersCnt),
    stubTiming(tfLayersCnt), stubType(tfLayersCnt) */
  {}
};


class EventIntGmt: public EventInt {
public:
    EventIntGmt(unsigned int inputCnt, unsigned int ouptutCnt, float defInVal = 0) : EventInt(inputCnt, ouptutCnt, defInVal) { }

    virtual ~EventIntGmt() {};

    friend std::ostream & operator << (std::ostream &out, const EventIntGmt& event);

    short tpEvent = -1;

    float tpPt = 0, tpEta = 0;// tpPhi = 0, tpBeta = 0;
    char tpCharge = 0;
    int tpType = 0;

    unsigned short tttPt = 0;

    unsigned char stubCnt = 0; //combined stub is one muon station,

    unsigned short noHitCnt = 0;
};


class EventsGeneratorGmt: public EventsGeneratorBase<EventInt> {
public:
    EventsGeneratorGmt(std::default_random_engine& generator, unsigned int outputCnt, bool useHitsValid, bool addNoHitCntToInputs);

    virtual ~EventsGeneratorGmt();


    void printEvent(EventFloat* event, std::ostream& ostr = std::cout);

    bool gmtRecordToEventInput(TrackMatchedMuonRecord& record, EventIntGmt* event, bool print);

    virtual void readEvents(std::string dataFileName, Long64_t maxEvents = -1 );

    virtual void getRandomEvents(std::vector<EventInt*>& events) {};

    virtual std::vector<EventInt*>& getEvents() {
        return events;
    }

    unsigned int getInputCnt() const {
        return inputCnt;
    }

private:
    unsigned int inputCnt = 1;
    unsigned int outputCnt = 1;

    bool useHitsValid = false;

    bool addNoHitCntToInputs =  false;
};

} /* namespace lutNN */

#endif /* INTERFACE_EVENTSGENERATORGMT_H_ */
