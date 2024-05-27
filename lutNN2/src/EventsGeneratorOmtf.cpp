/*
 * EventsGeneratoOmtf.cpp
 *
 *  Created on: Dec 12, 2019
 *      Author: kbunkow
 */

#include "lutNN/lutNN2/interface/EventsGeneratorOmtf.h"
#include "TFile.h"
#include "TTree.h"

#include <iomanip>
#include <iostream>
#include <filesystem>

#include "boost/dynamic_bitset.hpp"

namespace lutNN {

using namespace std;

EventsGeneratorOmtf::EventsGeneratorOmtf(std::default_random_engine& generator, unsigned int outputCnt):
            EventsGeneratorBase<EventFloat>(generator), outputCnt(outputCnt) {
   setExpectedResult = std::bind(&EventsGeneratorOmtf::setExpectedResultPtRegression, this, std::placeholders::_1, std::placeholders::_2);

    inputCnt = maxHitCnt + 1; //last input is for noHitCnt
}

EventsGeneratorOmtf::EventsGeneratorOmtf(std::default_random_engine& generator, unsigned int outputCnt, std::vector<float>& ptBins):
         EventsGeneratorBase<EventFloat>(generator), outputCnt(outputCnt), ptBins(ptBins) {
    if(outputCnt != (ptBins.size()) ) // * 2
        throw std::runtime_error("EventsGeneratorOmtf()  outputCnt != (ptBins.size() * 2)");

    setExpectedResult = std::bind(&EventsGeneratorOmtf::setExpectedResultClassifier, this, std::placeholders::_1, std::placeholders::_2);
    inputCnt = maxHitCnt + 1; //last input is for noHitCnt
}

EventsGeneratorOmtf::~EventsGeneratorOmtf() {
    // TODO Auto-generated destructor stub
}

bool EventsGeneratorOmtf::omtfHitToEventInput(OmtfEvent::Hit& hit, EventFloat* event, unsigned int omtfRefLayer, bool print) {
	float offset = (omtfRefLayer<<7) + 64;

	if(hit.valid) {
		if( (hit.layer == 1 || hit.layer == 3 || hit.layer == 5) && hit.quality < 4) ///TODO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
			return false;

		int rangeFactor = 2; //rangeFactor scales the hit.phiDist such that the event->inputs is smaller then 63
		if(hit.layer == 1) {
			rangeFactor = 8;
		}
		/*else if(hit.layer == 8 || hit.layer == 17) {
        rangeFactor = 4;
    }*/
		else if(hit.layer == 3) {
			rangeFactor = 4;
		}
		else if(hit.layer == 9) {
			rangeFactor = 1;
		}
		/*else {
        rangeFactor = 2;
    }
		 */

		rangeFactor *= 2; //TODO !!!!!!!!!!!!!!!!!!!



		if(abs(hit.phiDist) >= (63 * rangeFactor) ) {
			cout   //<<" muonPt "<<omtfEvent.muonPt<<" omtfPt "<<omtfEvent.omtfPt
			<<" RefLayer "<<omtfRefLayer<<" layer "
			<<int(hit.layer)<<" hit.phiDist "<<hit.phiDist
			<<" valid "<<((short)hit.valid)<<" !!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
			hit.phiDist = copysign(63 * rangeFactor, hit.phiDist);
		}

		event->inputs.at(hit.layer) = (float)hit.phiDist / (float)rangeFactor + offset;

		if(event->inputs.at(hit.layer) >= 1022) //the last address i.e. 1023 is reserved for the no-hit value, so interpolation between the 1022 and 1023 has no sense
			event->inputs.at(hit.layer) = 1022;

		if(print || event->inputs.at(hit.layer) < 0) {
			cout//<<"rawData "<<hex<<setw(16)<<hit.rawData
			<<" layer "<<dec<<int(hit.layer);
			cout<<" phiDist "<<hit.phiDist<<" inputVal "<<event->inputs.at(hit.layer)<<" hit.z "<<int(hit.z)<<" valid "<<((short)hit.valid)<<" quality "<<(short)hit.quality<<" omtfRefLayer "<<omtfRefLayer;
			if(event->inputs.at(hit.layer) < 0)
			    cout<<" event->inputs.at(hit.layer) < 0 !!!!!!!!!!!!!!!!!"<<endl;
			cout<<endl;
		}

		if(event->inputs[hit.layer] >= 1024) { //TODO should be the size of the LUT of the first layer
			cout<<" event->inputs[hit.layer] >= 1024 !!!!!!!!!!!!!!!!!"<<endl;
		}
		return true;
	}

	return false;

}


void EventsGeneratorOmtf::readEvents(std::string dataFileName, int rootFileVersion, Long64_t maxEvents) {
    if(std::filesystem::exists(dataFileName))
        std::cout<<"EventsGeneratorOmtf::readEvents reading file "<<dataFileName<<std::endl;
    else {
        std::cout<<"EventsGeneratorOmtf::readEvents file "<<dataFileName<<" does not exist "<<std::endl;
        return;
    }

    TFile* rootFile = new TFile(dataFileName.c_str());
    
    TTree* rootTree = nullptr;
    if(rootFileVersion == 1) {
    	rootTree = (TTree*)rootFile->Get("OMTFHitsTree");	
    }
    else if(rootFileVersion >= 2) {
    	TDirectory* dir = (TDirectory*)rootFile->Get("simOmtfDigis");
	    rootTree = (TTree*)dir->Get("OMTFHitsTree");
	}
	else {
		std::cout<<"EventsGeneratorOmtf::readEvents file "<<dataFileName<<" the rootFileVersion "<<rootFileVersion<<" is not correct"<<std::endl;
        return;
	}

    OmtfEvent omtfEvent;

    rootTree->SetBranchAddress("muonPt", &omtfEvent.muonPt);
    rootTree->SetBranchAddress("muonEta", &omtfEvent.muonEta);
    rootTree->SetBranchAddress("muonPhi", &omtfEvent.muonPhi);
    rootTree->SetBranchAddress("muonCharge", &omtfEvent.muonCharge);

    rootTree->SetBranchAddress("omtfPt", &omtfEvent.omtfPt);
    rootTree->SetBranchAddress("omtfEta", &omtfEvent.omtfEta);
    rootTree->SetBranchAddress("omtfPhi", &omtfEvent.omtfPhi);
    rootTree->SetBranchAddress("omtfCharge", &omtfEvent.omtfCharge);

    rootTree->SetBranchAddress("omtfScore", &omtfEvent.omtfScore);
    rootTree->SetBranchAddress("omtfQuality", &omtfEvent.omtfQuality);
    rootTree->SetBranchAddress("omtfRefLayer", &omtfEvent.omtfRefLayer);
    rootTree->SetBranchAddress("omtfProcessor", &omtfEvent.omtfProcessor);

	bool isOmtfPtCont = false;
	if(rootFileVersion >= 2) {
		rootTree->SetBranchAddress("eventNum", &omtfEvent.eventNum);
	    rootTree->SetBranchAddress("muonEvent", &omtfEvent.muonEvent);
	    
	    rootTree->SetBranchAddress("muonDxy", &omtfEvent.muonDxy);
    	rootTree->SetBranchAddress("muonRho", &omtfEvent.muonRho);
	    
    	rootTree->SetBranchAddress("omtfRefHitNum", &omtfEvent.omtfRefHitNum);
    	
	    rootTree->SetBranchAddress("omtfHwEta", &omtfEvent.omtfHwEta);
    	
    	rootTree->SetBranchAddress("killed", &omtfEvent.killed);

	    rootTree->SetBranchAddress("omtfFiredLayers", &omtfEvent.omtfFiredLayers); 
	    
	    isOmtfPtCont = rootTree->GetListOfBranches()->FindObject("omtfPtCont");
	    if(isOmtfPtCont) {
    		rootTree->SetBranchAddress("omtfPtCont", &omtfEvent.omtfPtCont);
	    }
	}

    rootTree->SetBranchAddress("hits", &omtfEvent.hits); //TODO

    //read all entries and fill the histograms
    Long64_t nentries = rootTree->GetEntries();
    if(maxEvents == -1)
        maxEvents = nentries;
    else if(nentries < maxEvents)
        maxEvents = nentries;

    //events.resize(nentries, nullptr);
    events.reserve(maxEvents);


    omtfGoodEvents = 0;

    int printEvery = 1000000;

    int lastQuality = 0;
    for (Long64_t i = 0; i < maxEvents; i++) {
        rootTree->GetEntry(i);

        if(rootFileVersion >= 2) {
            //for rootFileVersion == 1 killed is always false - the killed muons were not saved
            //in the rootFileVersion == 2 the muons killed in the GhostBusterPreferRefDt are saved
            //(the muons rejected by the CandidateSimMuonMatcher::ghostBust are not included here
            //the question is if it has a sense to used them for the training, as they by construction worse,
            //and are not included in the OMTF output, so do not affect the performance
            //on the other hand including the ghostbusted muons will increase the statistics for some ref layers
            if(omtfEvent.killed ) continue; ///TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            //the quality of the killed candidates is 0, so we are taking the quality of the last not killed candidate
            //it is not very good though, because the killed muon usually has lower quality
            if(omtfEvent.killed == false) {
                lastQuality = omtfEvent.omtfQuality;
            }

            /*if(omtfEvent.muonEvent != 0 ) //TODO!!!!!!!!!!!!!!!!!!!!
                continue;*/

            /*if(omtfEvent.omtfPt == 0 ) //|| omtfEvent.killed TODO!!!!!!!!!!!!!!!!!!!!
                continue;

            if(omtfEvent.muonPt == 0) //TODO this removes the candidates without matched muon, !!!!!!!!!!!!!!!!!!
                continue;*/

            //if(omtfEvent.muonPt == 0) continue; //TODO this removes the fake candidates from training

            //TODO in the fwVersion() >= 8 omtfQuality 1 are candidates with one station only, so should be dropped here
            if(lastQuality <= 1) continue;
        }

        //in the omtfHits_omtfAlgo0x0006_v1 there was a cut on eta 0.8-1.24 in the /afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_10_x_x_l1tOfflinePhase2/CMSSW_10_6_1_patch2/src/L1Trigger/L1TMuonBayes/src/OmtfPatternGeneration/DataROOTDumper2.cc
        //in the DataROOTDumper2 in the CMSSW_12_3_0_pre4 this cut is dropped,
        //in any case we should train the OMTF to cover the region from 0.8, so that it performs relatively well also is in the region of overlay with the BMTF
        if(fabs(omtfEvent.muonEta) < 0.80 || fabs(omtfEvent.muonEta) > 1.24)
            continue;


        //events with less hits then min cannot be processed by the network
        if( (omtfEvent.hits->size() < minHitsCnt) ) //<<<<<<<<<<<<<<<TODO || (omtfEvent.omtfQuality < 12)
            continue;

        /*
        if ( omtfEvent.omtfRefLayer == 1) {//<<<<<<< TOOD !!!!!!!!!!!!!!!!!!!!!
        }
        else
            continue; */

        //if(omtfEvent.muonCharge == -1)
        //    continue;

        if(i%printEvery == 0) {
            boost::dynamic_bitset<> firedLayers(18, omtfEvent.omtfFiredLayers);

            cout<<" i "<<i<<" eventNum "<<omtfEvent.eventNum<<" muonPt "<<omtfEvent.muonPt<<" muonCharge "<<(int)omtfEvent.muonCharge
            <<" omtfPt " <<omtfEvent.omtfPt
            <<" RefLayer "<<(int)omtfEvent.omtfRefLayer
            <<" omtfRefHitNum "<<(int)omtfEvent.omtfRefHitNum
            <<" quality "<<(int)omtfEvent.omtfQuality<<" killed "<<omtfEvent.killed
            //<<" omtfPtCont "<<omtfEvent.omtfPtCont
			<<" hits->size() "<<omtfEvent.hits->size()<<"  firedLayers "<<firedLayers<<endl;;
        }

        bool print = false;
        /*if(firedLayers.count() != omtfEvent.hits->size() ) {//|| omtfEvent.omtfQuality > 12
            cout<<"event "<<i<<" muonPt "<<omtfEvent.muonPt<<" muonCharge "<<omtfEvent.muonCharge<<" RefLayer "<<omtfEvent.omtfRefLayer<<" quality "<<omtfEvent.omtfQuality
            		<<" hits->size() "<<omtfEvent.hits->size()<<"  firedLayers "<<firedLayers<<" <<<<<"<<endl;
            print = true;
        }*/

        //float offset = (omtfEvent.omtfRefLayer<<7) + 64;

        //EventFloatOmtf* event = new EventFloatOmtf(inputCnt, outputCnt, offset + 63);//TODO!!!!!!!!!!!!!!!!!!!! for version without separton of distributions for differnt hit count
        EventFloatOmtf* event = new EventFloatOmtf(inputCnt, outputCnt, noHitVal);
        event->muonPt = omtfEvent.muonPt;
        event->muonEta = omtfEvent.muonEta;
        event->muonCharge = omtfEvent.muonCharge;
        event->omtfPt = omtfEvent.omtfPt;
        event->omtfCharge = omtfEvent.omtfCharge;
        event->omtfQuality = omtfEvent.omtfQuality;
        event->omtfRefLayer = omtfEvent.omtfRefLayer;
        event->killed = omtfEvent.killed;

        if(isOmtfPtCont) {
        	event->omtfPt = omtfEvent.omtfPtCont;
        }

        int hitCnt = 0;
        for(auto& hitRaw : *(omtfEvent.hits)) {
            OmtfEvent::Hit hit(hitRaw);
            hitCnt += omtfHitToEventInput(hit, event, omtfEvent.omtfRefLayer,  i%printEvery == 0 || print);
        }

        event->inputs.back() = maxHitCnt - hitCnt; //event->hitCnt;

        setExpectedResult(omtfEvent, event);

        if(i%printEvery == 0)
            cout<<"event "<<(*event)<<endl<<endl;

        if(hitCnt >= 3 )
            events.push_back(event);
        else
            delete event;

        //events.push_back(event);
    }


    cout<<"omtf efficiency "<<omtfGoodEvents/(double)events.size()<<std::endl;
    miniBatchBegin = events.end(); //to force shuffling at the beginning
    delete rootFile;

    cout<<"EventsGeneratorOmtf::readEvents events.size() "<<events.size()<<endl;
}


void EventsGeneratorOmtf::setExpectedResultClassifier(OmtfEvent& omtfEvent, EventFloatOmtf* event) {
    unsigned int expectedOutNum = 0;
    for(unsigned int iPt = 0; iPt < ptBins.size(); iPt++) {
        if(ptBins[iPt] > omtfEvent.muonPt) {
            /*expectedOutNum = (2 * iPt) + (omtfEvent.muonCharge == 1 ? 1 : 0);
            event->expextedResult.at(expectedOutNum) = 1;*/
            expectedOutNum = iPt;
            event->classLabel = iPt;
            break;
        }
    }

    double ptCut = 20.;
/*
    unsigned int omtfOutNum = 0;
    if(omtfEvent.omtfPt >= ptCut) {
        if(omtfEvent.omtfCharge == 1)
            omtfOutNum = 2;
        else
            omtfOutNum = 3;
    }
    else {
        if(omtfEvent.omtfCharge == 1)
            omtfOutNum = 0;
        else
            omtfOutNum = 1;
    }

    if(expectedOutNum == omtfOutNum)
        omtfGoodEvents++;*/

    if(omtfEvent.omtfPt >= ptCut && omtfEvent.muonPt >= ptCut && omtfEvent.omtfCharge == omtfEvent.muonCharge ) {
        omtfGoodEvents++;
    }
}

void EventsGeneratorOmtf::setExpectedResultPtRegression(OmtfEvent& omtfEvent, EventFloatOmtf* event) {
    double ptCut = 20.;

    double expextedResult = 0;
    if(omtfEvent.muonPt < 0.01) {
        cout    <<" event "<<setw(10)<<omtfEvent.eventNum<<" muonEvent "<<omtfEvent.muonEvent
                <<" omtfEvent.muonPt "<<omtfEvent.muonPt<<" muonCharge "<<(int)omtfEvent.muonCharge<<" omtfPt "<<omtfEvent.omtfPt
                <<" omtfEta "<<omtfEvent.omtfEta <<" omtfHwEta "<<omtfEvent.omtfHwEta<<" omtfRefHitNum "<<(int)omtfEvent.omtfRefHitNum
                <<" !!!!!1111!!!!!!!!!!!!!!!!!!!!!"<<endl;
        cout<<"event "<<(*event)<<endl<<endl;
    }
    /*if( omtfEvent.muonPt > 200)
        expextedResult = 0;//omtfEvent.muonCharge / log(200.);
    else
        expextedResult = omtfEvent.muonCharge / log(omtfEvent.muonPt) - omtfEvent.muonCharge / log(200.);*/

    //expextedResult = omtfEvent.muonCharge / pow(log(omtfEvent.muonPt), 1.);
    //expextedResult = 1./ pow(log(omtfEvent.muonPt), 1.);
    expextedResult = omtfEvent.muonPt;

    event->expextedResult.at(0) = expextedResult;
    event->expextedResult.at(1) = omtfEvent.muonCharge; //TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if(outputCnt > 2)
        event->expextedResult.at(2) = (omtfEvent.muonPt > 0 ? 1 : 0);

    if( (omtfEvent.muonPt >= ptCut && omtfEvent.omtfPt >= ptCut) ||
        (omtfEvent.muonPt <  ptCut && omtfEvent.omtfPt <  ptCut) ) {
        omtfGoodEvents++;
    }
}


void EventsGeneratorOmtf::setEventWeight(TH1* ptGenNeg, TH1* ptGenPos) {
    if(ptGenNeg ==  nullptr || ptGenPos ==  nullptr) {
        ptGenPos = new TH1I("ptGenPos_setEventWeight", "ptGenPos_setEventWeight", 1000, 0, 200);
        ptGenNeg = new TH1I("ptGenNeg_setEventWeight", "ptGenNeg_setEventWeight", 1000, 0, 200);
    }

    for(auto& event : events) {
        EventFloatOmtf* eventFloatOmtf = static_cast<EventFloatOmtf*>(event);

        if(eventFloatOmtf->muonCharge > 0) {
            ptGenPos->Fill(eventFloatOmtf->muonPt);
        }
        else {
            ptGenNeg->Fill(eventFloatOmtf->muonPt);
        }
    }


    for(auto& event : events) {
        EventFloatOmtf* eventFloatOmtf = static_cast<EventFloatOmtf*>(event);

        if(eventFloatOmtf->muonCharge > 0) {
            eventFloatOmtf->weight  = 1. / ptGenPos->GetBinContent(ptGenPos->FindBin(eventFloatOmtf->muonPt) ); //FindBin should return 0 or fNbins+1 in case of underflow or overflow, so should be good
        }
        else {
            eventFloatOmtf->weight  = 1. / ptGenNeg->GetBinContent(ptGenNeg->FindBin(eventFloatOmtf->muonPt) );
        }

        /*if(eventFloatOmtf->muonPt <= 60) {
            eventFloatOmtf->weight *= 1. + (60. - eventFloatOmtf->muonPt) * 0.2;
        }*/

        /*if(eventFloatOmtf->muonPt <= 18) {
            eventFloatOmtf->weight *= 1. + (18. - eventFloatOmtf->muonPt) * 1;
        }*/

        /*if(eventFloatOmtf->muonPt <= 15) {
            eventFloatOmtf->weight *= 1. + (15. - eventFloatOmtf->muonPt) * 4;
        }*/

        /*if(eventFloatOmtf->muonPt <= 7) {
        	eventFloatOmtf->weight *= 1. + (7. - eventFloatOmtf->muonPt) * 1.1;
        }*/

        /*if(eventFloatOmtf->muonPt > 205) {
            cout<<"muonPt "<<eventFloatOmtf->muonPt<<" "<<eventFloatOmtf->weight<<endl;
        }*/

        eventFloatOmtf->weight  = 1.; //TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
        if(eventFloatOmtf->muonPt <= 7) {
            eventFloatOmtf->weight *= 1. + (7- eventFloatOmtf->muonPt) * 3.;
        }
        if(eventFloatOmtf->muonPt <= 5) {
            eventFloatOmtf->weight *= 1. + (5. - eventFloatOmtf->muonPt) * 1.1;
        }

        //eventFloatOmtf->weight *= (1./pow(eventFloatOmtf->muonPt, 0.5) + 1.);
    }

    cout<<"EventsGeneratorOmtf::setEventWeight "<<std::endl;
}

std::ostream & operator << (std::ostream &out, const EventFloatOmtf& event) {
    out<<"EventFloatOmtf "
    <<"muonPt "<<setw(10)<<event.muonPt
    <<" muonEta "<<setw(10)<<event.muonEta
    <<" muonCharge "<<(int)event.muonCharge
    <<" omtfPt "<<setw(10)<<event.omtfPt
    <<" omtfCharge "<<(int)event.omtfCharge
	<<" omtfQuality "<<(unsigned int)event.omtfQuality
	<<" omtfRefLayer "<<(int)event.omtfRefLayer
    <<" weight "<<event.weight
    <<endl;
    cout<<"expextedResult ";

    for(unsigned int iOut = 0; iOut < event.expextedResult.size(); iOut++) {
        cout<<setw(10)<<setw(11)<<event.expextedResult[iOut]<<" ";
        if(event.expextedResult.size() == 1)
        	cout<<setw(10)<<exp(pow(1./fabs(event.expextedResult[iOut] ) ,1) );
    }
    out<<endl;
    cout<<"      nnResult ";

    double maxP =  -1000000;
    double maxPClass = -1;
    for(unsigned int iOut = 0; iOut < event.nnResult.size(); iOut++) {
        cout<<setw(10)<<setw(10)<<event.nnResult[iOut]<<", ";
        if(event.nnResult.size() == 1)
        	cout<<setw(10)<<exp(pow(1./fabs(event.nnResult[iOut] ) ,1) ) ;

        if(maxP < event.nnResult[iOut]) {
        	maxP = event.nnResult[iOut];
        	maxPClass = iOut;
        }
    }
    cout<<endl<<" maxPClass "<<maxPClass<<" maxP "<<maxP<<endl;

    unsigned int iLayer = 0;
    for(auto& hit : event.inputs) {
        if(iLayer == 18)
            out<<setw(2)<<iLayer++<<"      "<<hit<<" -  missing hits count"<<endl;
        else if(hit != event.noHitVal)
            out<<setw(2)<<iLayer++<<"      "<<hit<<endl;
        else
            out<<setw(2)<<iLayer++<<" "<<hit<<endl;
    }
    return out;
}
} /* namespace lutNN */
