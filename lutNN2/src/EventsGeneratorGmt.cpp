/*
 * EventsGeneratorGMT.cpp
 *
 *  Created on: Dec 15, 2021
 *      Author: kbunkow
 */

#include "lutNN/lutNN2/interface/EventsGeneratorGmt.h"

#include "lutNN/lutNN2/interface/LutNetworkBase.h"

#include "TFile.h"
#include "TTree.h"

#include <iomanip>
#include <iostream>
#include <bitset>

#include "boost/dynamic_bitset.hpp"

namespace lutNN {

EventsGeneratorGmt::EventsGeneratorGmt(std::default_random_engine& generator, unsigned int outputCnt, bool useHitsValid, bool addNoHitCntToInputs):
  EventsGeneratorBase<EventInt>(generator), outputCnt(outputCnt), useHitsValid(useHitsValid), addNoHitCntToInputs(addNoHitCntToInputs)    {

  inputCnt = 28; //TODO
  //inputCnt = 29; //TODO
}

EventsGeneratorGmt::~EventsGeneratorGmt() {
    // TODO Auto-generated destructor stub
}

using namespace std;

bool EventsGeneratorGmt::gmtRecordToEventInput(TrackMatchedMuonRecord& record, EventIntGmt* event, bool print) {
    unsigned int inputNum = 0;

    event->tpEvent = record.tpEvent;
    event->tpPt = record.tpPt;
    event->tpEta = record.tpEta;
    event->tpCharge = record.tpCharge;
    event->tpType = record.tpType;

    event->tttPt = record.tttPt;

    event->inputs.at(inputNum++) = record.tttCurvature >> 9 & 0x3f; //BITSTTCURV = 15, so we keep only 6 TODO increase
    //event->inputs.at(inputNum++) = record.tttCurvature >> 3 & 0x3f; //BITSTTCURV = 15, so we keep only 6 TODO increase TODO !!!!!!!!!!!!!!!!!!!!!!!!!, for inputCnt = 29;

    //event->inputs.at(inputNum++) = record.tttBendChi2 & 0x7; // 3 bits TODO

    event->inputs.at(inputNum++) = record.tttEta >> 7 & 0x3f; //  BITSETA = 13, s0 we keep only 6 TODO
    event->inputs.at(inputNum++) = record.tttPhi >> 7 & 0x3f; //  BITSPHI = 13, s0 we keep only 6 TODO

    //record.hitsValid stores the bit denoting that given hit is valid
    //the hit cab be invalidated during the ghost busting
    //if there is not hit fitting to the track, the deltaCoords* or deltaEtas1 are 0

    event->noHitCnt = 0;
    //inputs are 6 bits, with the stubType at the bit no 5 (from 0)
    for(unsigned int iLayer = 0; iLayer < record.deltaCoords1->size(); iLayer++) {
        if(!useHitsValid || (record.hitsValid & (1 << (iLayer * 2))) ) {
            int stubType = ( (record.stubType->at(iLayer) == 1) && record.deltaCoords1->at(iLayer) ) ? (1<<5) : 0; //to fix the bug in DataDumper::process present up to t207
            event->inputs.at(inputNum) = (int)record.deltaCoords1->at(iLayer) | stubType; //deltaCoords1 is 5 bits, last bit is sign
        }
        else
            event->inputs.at(inputNum) = 0;

        if(event->inputs.at(inputNum) == 0)
            event->noHitCnt++;

        inputNum++;
    }

    for(unsigned int iLayer = 0; iLayer < record.deltaCoords2->size(); iLayer++) {
        if(!useHitsValid || (record.hitsValid & (2 << (iLayer * 2))) ) {
            int stubType = ( (record.stubType->at(iLayer) == 1) && record.deltaCoords2->at(iLayer) ) ? (1<<5) : 0; //to fix the bug in DataDumper::process present up to t207
            event->inputs.at(inputNum) = (int)record.deltaCoords2->at(iLayer) | stubType;//deltaCoords1 is 5 bits, last bit is sign
        }
        else
            event->inputs.at(inputNum) = 0;

        if(event->inputs.at(inputNum) == 0)
            event->noHitCnt++;

        inputNum++;
    }

    for(unsigned int iLayer = 0; iLayer < record.deltaEtas1->size(); iLayer++) {
        if(!useHitsValid || (record.hitsValid & (1 << (iLayer * 2))) ) {
            int stubType = ( (record.stubType->at(iLayer) == 1) && record.deltaEtas1->at(iLayer) ) ? (1<<5) : 0; //to fix the bug in DataDumper::process present up to t207
            event->inputs.at(inputNum) = (int)record.deltaEtas1->at(iLayer) | stubType;//deltaEtas1 is 5 bits, last bit is sign
        }
        else
            event->inputs.at(inputNum) = 0;

        inputNum++;
    }
    for(unsigned int iLayer = 0; iLayer < record.deltaEtas2->size(); iLayer++) {
        if(!useHitsValid || (record.hitsValid & (2 << (iLayer * 2))) ) {
            int stubType = ( (record.stubType->at(iLayer) == 1) && record.deltaEtas2->at(iLayer) ) ? (1<<5) : 0; //to fix the bug in DataDumper::process present up to t207
            event->inputs.at(inputNum) = (int)record.deltaEtas2->at(iLayer) | stubType;//deltaEtas1 is 5 bits, last bit is sign //TODO shouldn't it unsigned?
        }
        else
            event->inputs.at(inputNum) = 0;

        inputNum++;
    }


    for(unsigned int iLayer = 0; iLayer < record.stubTiming->size(); iLayer++) {
        int timing = (int)record.stubTiming->at(iLayer);// TODO  change when the timing in the hardware scale is available !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //assuming the timing is in ns, -12.5..+12.5 for the BX = 0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

        unsigned int timingHw = round(timing * 8./25.) + 4; //so we convert 1 BX into the range 0...8, with 4 corresponding to the timing = 0, timingHw = 0 is not hit
        if(timingHw < 0) {
            cout<<"EventsGeneratorGmt::gmtRecordToEventInput timingHw "<<timingHw<<" !!!!!!!!!"<<std::endl;
            timingHw = 0;
        }
        else if(timingHw > 31) {
            cout<<"EventsGeneratorGmt::gmtRecordToEventInput timingHw "<<timingHw<<" !!!!!!!!!"<<std::endl;
            timingHw = 31;
        }

        if(!useHitsValid || ( (record.hitsValid & (3 << (iLayer * 2)) ) != 0 ) ) {
            //TODO correct once the data > t207 are used!!!!!!!!!!!!
            if(record.deltaCoords1->at(iLayer) || record.deltaCoords2->at(iLayer)) {
                int stubType = ( (record.stubType->at(iLayer) == 1) ) ? (1<<5) : 0; //to fix the bug in DataDumper::process present up to t207
                event->inputs.at(inputNum) = timingHw | stubType; //stubTiming is 5 bits
            }
            else
                event->inputs.at(inputNum) = 0;
        }
        else
            event->inputs.at(inputNum) = 0;

        inputNum++;

        if(record.deltaCoords1->at(iLayer) || record.deltaCoords2->at(iLayer))
            event->stubCnt++;
    }

    if(addNoHitCntToInputs)
        event->inputs.at(inputNum) = event->noHitCnt;


    if(abs(record.tpType) == 13) //muon
        event->classLabel = 1;
    else if(abs(record.tpType) == 1000015) // stau - HSCP
        event->classLabel = 2;
    else
        event->classLabel = 0;

    if(print) {
        inputNum = 0;
        cout<<"Event:"<<endl;
        cout<<"curv "<<event->inputs.at(inputNum++)<<endl;
        cout<<"eta  "<<event->inputs.at(inputNum++)<<endl;
        cout<<"phi  "<<event->inputs.at(inputNum++)<<endl;
        cout<<"classLabel  "<<event->classLabel<<endl;
        cout<<"noHitCnt "<<event->noHitCnt<<endl;
        cout<<"stubCnt "<<(unsigned int)(event->stubCnt)<<endl;
        for(; inputNum < event->inputs.size(); inputNum++) {
            if(addNoHitCntToInputs && inputNum == (event->inputs.size() -1))
                cout<<" inputNum "<<inputNum<<" NoHitCnt "<<dec<<event->inputs.at(inputNum)<<endl;
            else
            cout<<" inputNum "<<inputNum<<" layer "<<((inputNum-3)%5)<<" val 0x"<<hex<<event->inputs.at(inputNum)<<dec<<endl;
        }
        cout<<endl;

    }

    return true;
}

void EventsGeneratorGmt::readEvents(std::string dataFileName, Long64_t maxEvents) {
    TFile* rootFile = new TFile(dataFileName.c_str());
    TDirectory* dir = (TDirectory*)rootFile->Get("gmtMuons/GmtDataDumper");
    TTree* rootTree = (TTree*)dir->Get("GmtMuonTree");

    TrackMatchedMuonRecord record;

    rootTree->SetBranchAddress("eventNum", &record.eventNum);
    rootTree->SetBranchAddress("tpEvent", &record.tpEvent);

    rootTree->SetBranchAddress("tpPt", &record.tpPt);
    rootTree->SetBranchAddress("tpEta", &record.tpEta);
    rootTree->SetBranchAddress("tpPhi", &record.tpPhi);
    rootTree->SetBranchAddress("tpBeta", &record.tpBeta);

    //rootTree->SetBranchAddress("tpCharge", &record.tpCharge); charge is in type
    rootTree->SetBranchAddress("tpType", &record.tpType);

    rootTree->SetBranchAddress("matching", &record.matching);

    rootTree->SetBranchAddress("tttCurvature", &record.tttCurvature);
    rootTree->SetBranchAddress("tttCharge", &record.tttCharge);
    rootTree->SetBranchAddress("tttPt", &record.tttPt);
    rootTree->SetBranchAddress("tttEta", &record.tttEta);

    rootTree->SetBranchAddress("tttPhi", &record.tttPhi);
    rootTree->SetBranchAddress("tttZ0", &record.tttZ0);
    rootTree->SetBranchAddress("tttD0", &record.tttD0);

    rootTree->SetBranchAddress("tttChi2rphi", &record.tttChi2rphi);
    rootTree->SetBranchAddress("tttChi2rz", &record.tttChi2rz);
    rootTree->SetBranchAddress("tttBendChi2", &record.tttBendChi2);
    //rootTree->Branch("tttQualityMVA", &record.tttQualityMVA); lokks that is not set in the tracking trigger
    //rootTree->Branch("tttOtherMVA", &record.tttOtherMVA); is not set in the tracking trigger

    rootTree->SetBranchAddress("gmtBeta", &record.gmtBeta);

    rootTree->SetBranchAddress("isGlobal", &record.isGlobal);

    rootTree->SetBranchAddress("quality", &record.quality);

    rootTree->SetBranchAddress("hitsValid", &record.hitsValid);

    rootTree->SetBranchAddress("deltaCoords1", &record.deltaCoords1);
    rootTree->SetBranchAddress("deltaCoords2", &record.deltaCoords2);

    rootTree->SetBranchAddress("deltaEtas1", &record.deltaEtas1);
    rootTree->SetBranchAddress("deltaEtas2", &record.deltaEtas2);


    rootTree->SetBranchAddress("stubTiming", &record.stubTiming);

    rootTree->SetBranchAddress("stubType", &record.stubType);

    //read all entries and fill the histograms
    Long64_t nentries = rootTree->GetEntries();
    if(maxEvents == -1)
        maxEvents = nentries;
    else if(nentries < maxEvents)
        maxEvents = nentries;

    //events.resize(nentries, nullptr);
    events.reserve(maxEvents);


    //int goodEvents = 0;

    int printEvery = 1000;

    unsigned int lastEventNum = -1;
    unsigned int lhcEventCnt = 0;

    for (Long64_t i = 0; i < maxEvents; i++) {
        rootTree->GetEntry(i);

        if(i%printEvery == 0) {
            cout<<"gmtDataDumper record "<<i<<": "<<endl;
            cout<<"tp Pt "<<record.tpPt<<" Eta "<<record.tpEta<<" Phi "<<record.tpPhi<<" Type "<<record.tpType<<endl;
            cout<<"curvature "<<record.tttCurvature<<" eta "<<record.tttEta<<" phi "<<record.tttPhi<<endl;
            cout<<"hitsValid 0b"<<std::bitset<10>(record.hitsValid)<<endl;
            for(unsigned int iLayer = 0; iLayer < record.deltaCoords1->size(); iLayer++) {
                cout<<"tfLayer "<<iLayer
                        <<" deltaCoords1 "<<setw(3)<<(int)record.deltaCoords1->at(iLayer)
                        <<" deltaCoords2 "<<setw(3)<<(int)record.deltaCoords2->at(iLayer)
                        <<" deltaEtas1 "<<setw(3)<<(int)record.deltaEtas1->at(iLayer)
                        <<" deltaEtas2 "<<setw(3)<<(int)record.deltaEtas2->at(iLayer)
                        <<" stubTiming "<<setw(3)<<(int)record.stubTiming->at(iLayer)
                        <<" stubType "<<(int)record.stubType->at(iLayer)<<endl ;
            }
            cout<<endl;
        }

        EventIntGmt* event = new EventIntGmt(inputCnt + addNoHitCntToInputs, outputCnt);

        bool print = false;
        if(i%printEvery == 0)
            print = true;
        else
            print = false;

        gmtRecordToEventInput(record, event, print) ;

        events.push_back(event);

        if(lastEventNum != record.eventNum) {
            lastEventNum = record.eventNum;
            lhcEventCnt++;
        }
    }

    miniBatchBegin = events.end(); //to force shuffling at the beginning
    delete rootFile;

    cout<<"EventsGeneratorOmtf::readEvents events.size() "<<events.size()<<endl;
    cout<<"EventsGeneratorOmtf::readEvents lhcEventCnt "<<lhcEventCnt<<endl;
}


} /* namespace lutNN */
